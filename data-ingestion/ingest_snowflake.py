import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
from pyspark.sql.functions import coalesce

#Snowflake connetion options
sfOptions = {
    "sfURL": "KRWLMZY-HU90233.snowflakecomputing.com",
    "sfUser": "amoghayalgi",
    "sfPassword": "Ammu2025Heeya461",
    "sfDatabase": "RETAIL_DB",
    "sfWarehouse": "RETAIL_WH"
}

spark = SparkSession.builder \
        .appName("RetailDataIngestion") \
        .config("spark.jars.packages","net.snowflake:snowflake-jdbc:3.13.30,net.snowflake:spark-snowflake_2.12:2.16.0-spark_3.4")\
        .getOrCreate()

#Ingest raw CSVs to RAW schema
sfOptions_raw = sfOptions.copy()
sfOptions_raw["sfSchema"] = "RAW"

stores_df = spark.read.csv("retail-data/stores.csv",header=True,inferSchema=True)
stores_df.write.format("snowflake")\
               .options(**sfOptions_raw)\
               .option("dbtable","STG_STORES")\
               .mode("overwrite")\
               .save()

features_df = spark.read.csv("retail-data/features.csv",header=True,inferSchema=True)
features_df = features_df.withColumnRenamed("IsHoliday", "IsHoliday_feat")
features_df.write.format("snowflake")\
                 .options(**sfOptions_raw)\
                 .option("dbtable","STG_FEATURES")\
                 .mode("overwrite")\
                 .save()

sales_df = spark.read.csv("retail-data/sales.csv", header=True, inferSchema=True)
sales_df.write.format("snowflake")\
              .options(**sfOptions_raw)\
              .option("dbtable","STG_SALES")\
              .mode("overwrite")\
              .save()

joined_df = sales_df.join(features_df, ["Store","Date"], "left").join(stores_df, ["Store"], "left")
joined_df = joined_df.drop("IsHoliday_feat")

#Total Markdown
for i in range(1, 6):
    col_name = f"MarkDown{i}"
    if col_name in joined_df.columns:
        joined_df = joined_df.withColumn(col_name, F.col(col_name).cast("float"))
        joined_df = joined_df.withColumn(col_name, F.coalesce(F.col(col_name), F.lit(0.0)))

#F.array() converts the list of columns into a single column
#[F.col("MarkDown1"), F.col("MarkDown2"), F.col("MarkDown3")] -> [MarkDown1, MarkDown2, MarkDown3]
#* converts F.array([F.col("MarkDown1"), F.col("MarkDown2")]) -> F.array(F.col("MarkDown1"), F.col("MarkDown2"))
joined_df = joined_df.withColumn("Total_Markdown",F.col("MarkDown1") + F.col("MarkDown2") + F.col("MarkDown3") + F.col("MarkDown4") + F.col("MarkDown5"))


# Extract Week, Month, Year from Date
joined_df = joined_df.withColumn(
    "Date",
    coalesce(
        F.to_date(F.trim(F.col("Date")), "dd/MM/yyyy"),
        F.to_date(F.trim(F.col("Date")), "yyyy-MM-dd"),
        F.to_date(F.trim(F.col("Date")), "MM/dd/yyyy")
    )
)
joined_df = joined_df.withColumn("Week", F.weekofyear("Date")) \
	                .withColumn("Month", F.month("Date")) \
	                .withColumn("Year", F.year("Date"))

#Lag features and rolling averages for weekly sales
window_spec = Window.partitionBy("Store", "Date").orderBy("Date")
joined_df = joined_df.withColumn("Weekly_Sales_Lag1", F.lag("Weekly_Sales", 1).over(window_spec))
joined_df = joined_df.withColumn("Weekly_Sales_Rolling4", F.avg("Weekly_Sales").over(window_spec.rowsBetween(-3, 0)))

# Encode IsHoliday (assume boolean or Y/N)
joined_df = joined_df.withColumn("IsHoliday", F.col("IsHoliday").cast("string"))
joined_df = joined_df.withColumn(
    "IsHolidayEncoded",
    F.when(F.upper(F.col("IsHoliday")) == "TRUE", 1).otherwise(0)
)

#Normalize economic indicators (CPI, Unemployment, Fuel_Price)
econ_cols = [col for col in ["CPI", "Unemployment", "Fuel_Price"] if col in joined_df.columns]
for col in econ_cols:
    joined_df = joined_df.withColumn(col, F.col(col).cast("float"))
if econ_cols:
    assembler = VectorAssembler(inputCols=econ_cols, outputCol="econ_features")
    joined_df = assembler.transform(joined_df)
    scaler = StandardScaler(inputCol="econ_features", outputCol="econ_features_scaled", withMean=True, withStd=True)
    scaler_model = scaler.fit(joined_df)
    joined_df = scaler_model.transform(joined_df)

# Handle NAs (example: fill with 0 for numeric, 'Unknown' for string)
for col_name, dtype in joined_df.dtypes:
    if dtype == "string":
        joined_df = joined_df.fillna({col_name: "Unknown"})
    elif dtype in ["int", "double", "float", "bigint"]:
        joined_df = joined_df.fillna({col_name: 0})

# Assume econ_cols = ["CPI", "Unemployment", "Fuel_Price"]
for i, col_name in enumerate(econ_cols):
    # Define a UDF to extract the i-th element from the vector
    extract_element = udf(lambda v: float(v[i]) if v is not None else None, FloatType())
    joined_df = joined_df.withColumn(f"{col_name}_scaled", extract_element("econ_features_scaled"))

# Now drop the vector columns before saving
joined_df = joined_df.drop("econ_features", "econ_features_scaled")
	
# --- Write cleaned data to STAGING schema ---
sfOptions_staging = sfOptions.copy()
sfOptions_staging["sfSchema"] = "STAGING"
	
joined_df.write.format("snowflake") \
	    .options(**sfOptions_staging) \
	    .option("dbtable", "RETAIL_FEATURES") \
	    .mode("overwrite") \
	    .save()
	
print("Raw and cleaned data ingested successfully!")
	
spark.stop()
