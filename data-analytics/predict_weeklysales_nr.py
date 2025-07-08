from pyspark.sql import SparkSession, functions as F, Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from datetime import datetime, timedelta

# 1. Start Spark session and set up Snowflake options
sfOptions = {
    "sfURL": "KRWLMZY-HU90233.snowflakecomputing.com",
    "sfUser": "amoghayalgi",
    "sfPassword": "Ammu2025Heeya461",
    "sfDatabase": "RETAIL_DB",
    "sfSchema": "STAGING",
    "sfWarehouse": "RETAIL_WH"
}

spark = SparkSession.builder \
    .appName("RetailSalesPrediction") \
    .master("local[*]") \
    .config("spark.jars.packages", "net.snowflake:snowflake-jdbc:3.13.30,net.snowflake:spark-snowflake_2.12:2.16.0-spark_3.4") \
    .getOrCreate()

# 2. Read feature-engineered data from Snowflake
df = spark.read.format("snowflake").options(**sfOptions).option("dbtable", "RETAIL_FEATURES").load()

# Add lag2 and lag3 for weekly sales
window_spec = Window.partitionBy("STORE", "DEPT").orderBy("DATE")
df = df.withColumn("WEEKLY_SALES_LAG2", F.lag("WEEKLY_SALES", 2).over(window_spec))
df = df.withColumn("WEEKLY_SALES_LAG3", F.lag("WEEKLY_SALES", 3).over(window_spec))

# 3. Split into train/test - use data up to 2012-11-02 for training
split_date = "2012-11-02"
train_df = df.filter(F.col("DATE") < split_date)

# 4. Generate future dates for prediction period
start_date = datetime(2012, 11, 2)
end_date = datetime(2013, 10, 26)
future_dates = []
current_date = start_date
while current_date <= end_date:
    future_dates.append(current_date.strftime('%Y-%m-%d'))
    current_date += timedelta(days=7)

# Get unique store-dept combinations from training data
store_dept_combinations = train_df.select("STORE", "DEPT").distinct()

# Create future prediction dataframe
future_rows = []
for store_dept in store_dept_combinations.collect():
    store = store_dept["STORE"]
    dept = store_dept["DEPT"]
    for date in future_dates:
        future_rows.append((store, dept, date))

future_df = spark.createDataFrame(future_rows, ["STORE", "DEPT", "DATE"])

# Get last known values for each store-dept
last_known = train_df.groupBy("STORE", "DEPT").agg(
    F.last("WEEKLY_SALES_LAG1").alias("WEEKLY_SALES_LAG1"),
    F.last("WEEKLY_SALES_LAG2").alias("WEEKLY_SALES_LAG2"),
    F.last("WEEKLY_SALES_LAG3").alias("WEEKLY_SALES_LAG3"),
    F.last("WEEKLY_SALES_ROLLING4").alias("WEEKLY_SALES_ROLLING4"),
    F.last("TOTAL_MARKDOWN").alias("TOTAL_MARKDOWN"),
    F.last("TEMPERATURE").alias("TEMPERATURE"),
    F.last("FUEL_PRICE").alias("FUEL_PRICE"),
    F.last("CPI_SCALED").alias("CPI_SCALED"),
    F.last("UNEMPLOYMENT_SCALED").alias("UNEMPLOYMENT_SCALED"),
    F.last("FUEL_PRICE_SCALED").alias("FUEL_PRICE_SCALED"),
    F.last("SIZE").alias("SIZE"),
    F.last("ISHOLIDAYENCODED").alias("ISHOLIDAYENCODED")
)

# Join last known values to future_df
future_df = future_df.join(last_known, ["STORE", "DEPT"], "left")

# Add date-based features
future_df = future_df.withColumn("DATE", F.to_date("DATE"))
future_df = future_df.withColumn("WEEK", F.dayofweek("DATE"))
future_df = future_df.withColumn("MONTH", F.month("DATE"))
future_df = future_df.withColumn("YEAR", F.year("DATE"))

# Prepare feature columns
feature_cols = [
    "STORE", "DEPT", "ISHOLIDAYENCODED", "TOTAL_MARKDOWN", "TEMPERATURE", "FUEL_PRICE",
    "CPI_SCALED", "UNEMPLOYMENT_SCALED", "FUEL_PRICE_SCALED", "SIZE", "WEEK", "MONTH", "YEAR",
    "WEEKLY_SALES_LAG1", "WEEKLY_SALES_ROLLING4", "WEEKLY_SALES_LAG2", "WEEKLY_SALES_LAG3"
]

# Cast features to double for both train and future data
for col_name in feature_cols:
    train_df = train_df.withColumn(col_name, F.col(col_name).cast("double"))
    future_df = future_df.withColumn(col_name, F.col(col_name).cast("double"))

# Fill nulls with means (from training data)
mean_cols = ["SIZE", "TOTAL_MARKDOWN", "TEMPERATURE", "FUEL_PRICE", "CPI_SCALED", "UNEMPLOYMENT_SCALED", "FUEL_PRICE_SCALED"]
means = train_df.agg({col: "mean" for col in mean_cols}).collect()[0].asDict()
fill_dict = {
    "WEEKLY_SALES_LAG1": 0,
    "WEEKLY_SALES_LAG2": 0,
    "WEEKLY_SALES_LAG3": 0,
    "WEEKLY_SALES_ROLLING4": 0,
    "SIZE": means["avg(SIZE)"],
    "TOTAL_MARKDOWN": means["avg(TOTAL_MARKDOWN)"],
    "TEMPERATURE": means["avg(TEMPERATURE)"],
    "FUEL_PRICE": means["avg(FUEL_PRICE)"],
    "CPI_SCALED": means["avg(CPI_SCALED)"],
    "UNEMPLOYMENT_SCALED": means["avg(UNEMPLOYMENT_SCALED)"],
    "FUEL_PRICE_SCALED": means["avg(FUEL_PRICE_SCALED)"]
}
train_df = train_df.fillna(fill_dict)
future_df = future_df.fillna(fill_dict)

# Assemble features for both train and future data
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
train_df = assembler.transform(train_df)
future_df = assembler.transform(future_df)

# Train your best model (GBT, RF, or LR)
gbt = GBTRegressor(featuresCol="features", labelCol="WEEKLY_SALES", maxIter=50)
gbt_model = gbt.fit(train_df)

# Predict in bulk (vectorized, fast)
future_predictions = gbt_model.transform(future_df)

# Write predictions to Snowflake (or handle as needed)
sfOptions_pred = sfOptions.copy()
sfOptions_pred["sfSchema"] = "ANALYTICS"

future_predictions.select("STORE", "DEPT", "DATE", "prediction") \
    .withColumnRenamed("prediction", "PREDICTED_WEEKLY_SALES") \
    .write.format("snowflake") \
    .options(**sfOptions_pred) \
    .option("dbtable", "PREDICTED_WEEKLY_SALES_NR") \
    .mode("overwrite") \
    .save()

print("Predictions for 2012-11-02 to 2013-10-26 written to Snowflake!")
print(f"Total predictions generated: {future_predictions.count()}")

spark.stop()