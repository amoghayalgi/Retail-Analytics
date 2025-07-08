from pyspark.sql import SparkSession, functions as F, Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor, LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
from datetime import datetime, timedelta
import pandas as pd
  
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

print(f"Training data: {train_df.count()} rows (up to {split_date})")
print(f"Note: No historical data exists after {split_date} - we will generate future predictions")

# 4. Generate future dates for prediction period
start_date = datetime(2012, 11, 2)
end_date = datetime(2013, 10, 26)
future_dates = []
current_date = start_date
while current_date <= end_date:
    future_dates.append(current_date.strftime('%Y-%m-%d'))
    current_date += timedelta(days=7)  # Weekly predictions

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

# 5. Feature engineering for future predictions with recursive predictions
def create_future_features_with_recursion(df, train_df, model):
    """Create features for future predictions with recursive lag updates"""
    
    # Get the last known values for each store-dept combination
    last_known = train_df.groupBy("STORE", "DEPT").agg(
        F.last("WEEKLY_SALES").alias("last_sales"),
        F.last("WEEKLY_SALES_LAG1").alias("last_lag1"),
        F.last("WEEKLY_SALES_LAG2").alias("last_lag2"),
        F.last("WEEKLY_SALES_LAG3").alias("last_lag3"),
        F.last("WEEKLY_SALES_ROLLING4").alias("last_rolling4"),
        F.last("TOTAL_MARKDOWN").alias("last_markdown"),
        F.last("TEMPERATURE").alias("last_temperature"),
        F.last("FUEL_PRICE").alias("last_fuel_price"),
        F.last("CPI_SCALED").alias("last_cpi"),
        F.last("UNEMPLOYMENT_SCALED").alias("last_unemployment"),
        F.last("FUEL_PRICE_SCALED").alias("last_fuel_price_scaled"),
        F.last("SIZE").alias("last_size"),
        F.last("ISHOLIDAYENCODED").alias("last_holiday")
    )
    
    # Join with future dates to get last known values for each store-dept
    df = df.join(last_known, ["STORE", "DEPT"], "left")
    
    # Add date-based features
    df = df.withColumn("DATE", F.to_date("DATE"))
    df = df.withColumn("WEEK", F.dayofweek("DATE"))
    df = df.withColumn("MONTH", F.month("DATE"))
    df = df.withColumn("YEAR", F.year("DATE"))
    
    # Sort by store, dept, and date for recursive predictions
    df = df.orderBy("STORE", "DEPT", "DATE")
    
    # Convert to pandas for recursive processing
    pandas_df = df.toPandas()
    
    # Process each store-dept combination recursively
    updated_predictions = []
    
    for (store, dept), group in pandas_df.groupby(['STORE', 'DEPT']):
        group = group.sort_values('DATE').reset_index(drop=True)
        
        # Initialize with last known values for this store-dept combination
        current_lag1 = group.iloc[0]['last_lag1']
        current_lag2 = group.iloc[0]['last_lag2'] 
        current_lag3 = group.iloc[0]['last_lag3']
        current_rolling4 = group.iloc[0]['last_rolling4']
        
        for idx, row in group.iterrows():
            # Prepare features for this date using the last known values from join
            features = {
                'STORE': row['STORE'],
                'DEPT': row['DEPT'],
                'DATE': row['DATE'],
                'WEEK': row['WEEK'],
                'MONTH': row['MONTH'],
                'YEAR': row['YEAR'],
                'ISHOLIDAYENCODED': row['last_holiday'],
                'TOTAL_MARKDOWN': row['last_markdown'],
                'TEMPERATURE': row['last_temperature'],
                'FUEL_PRICE': row['last_fuel_price'],
                'CPI_SCALED': row['last_cpi'],
                'UNEMPLOYMENT_SCALED': row['last_unemployment'],
                'FUEL_PRICE_SCALED': row['last_fuel_price_scaled'],
                'SIZE': row['last_size'],
                'WEEKLY_SALES_LAG1': current_lag1,
                'WEEKLY_SALES_LAG2': current_lag2,
                'WEEKLY_SALES_LAG3': current_lag3,
                'WEEKLY_SALES_ROLLING4': current_rolling4
            }
            
            # Create feature vector and predict
            feature_vector = [features[col] for col in feature_cols]
            
            # Convert to Spark DataFrame for prediction
            from pyspark.ml.linalg import Vectors
            feature_vector_spark = Vectors.dense(feature_vector)
            
            # Create a single-row DataFrame for prediction
            prediction_df = spark.createDataFrame([(feature_vector_spark,)], ["features"])
            prediction_result = model.transform(prediction_df)
            prediction = prediction_result.select("prediction").collect()[0]["prediction"]
            
            # Update lag features for next iteration
            current_lag3 = current_lag2
            current_lag2 = current_lag1
            current_lag1 = prediction
            current_rolling4 = (current_lag1 + current_lag2 + current_lag3 + prediction) / 4
            
            # Store the result
            features['PREDICTED_SALES'] = prediction
            updated_predictions.append(features)
    
    # Convert back to Spark DataFrame
    result_df = spark.createDataFrame(updated_predictions)
    return result_df

# Create future features with recursive predictions
# Note: We'll call this after model training

# 6. Prepare training data with proper feature engineering
feature_cols = [
    "STORE", "DEPT", "ISHOLIDAYENCODED", "TOTAL_MARKDOWN", "TEMPERATURE", "FUEL_PRICE",
    "CPI_SCALED", "UNEMPLOYMENT_SCALED", "FUEL_PRICE_SCALED", "SIZE", "WEEK", "MONTH", "YEAR",
    "WEEKLY_SALES_LAG1", "WEEKLY_SALES_ROLLING4", "WEEKLY_SALES_LAG2", "WEEKLY_SALES_LAG3"
]

# Cast features to double and handle nulls for training data only
for col_name in feature_cols:
    train_df = train_df.withColumn(col_name, F.col(col_name).cast("double"))

# Fill nulls with appropriate values
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

print("Training data prepared:")
print(f"Train rows: {train_df.count()}")

# 7. Assemble features for training data only
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
train_df = assembler.transform(train_df)

# 8. Train models on historical data
print("Training models...")
rf = RandomForestRegressor(featuresCol="features", labelCol="WEEKLY_SALES", numTrees=50)
rf_model = rf.fit(train_df)

gbt = GBTRegressor(featuresCol="features", labelCol="WEEKLY_SALES", maxIter=50)
gbt_model = gbt.fit(train_df)

lr = LinearRegression(featuresCol="features", labelCol="WEEKLY_SALES")
lr_model = lr.fit(train_df)

# 9. Evaluate models using simple train-test split for quick comparison
print("Evaluating models using train-test split...")

# Create a simple train-test split for evaluation
from pyspark.ml.evaluation import RegressionEvaluator

# Use last 20% of data for testing
train_eval_df, test_eval_df = train_df.randomSplit([0.8, 0.2], seed=42)

# Evaluate Random Forest
rf_predictions = rf_model.transform(test_eval_df)
rf_rmse = RegressionEvaluator(labelCol="WEEKLY_SALES", predictionCol="prediction", metricName="rmse").evaluate(rf_predictions)
rf_r2 = RegressionEvaluator(labelCol="WEEKLY_SALES", predictionCol="prediction", metricName="r2").evaluate(rf_predictions)

# Evaluate GBT
gbt_predictions = gbt_model.transform(test_eval_df)
gbt_rmse = RegressionEvaluator(labelCol="WEEKLY_SALES", predictionCol="prediction", metricName="rmse").evaluate(gbt_predictions)
gbt_r2 = RegressionEvaluator(labelCol="WEEKLY_SALES", predictionCol="prediction", metricName="r2").evaluate(gbt_predictions)

# Evaluate Linear Regression
lr_predictions = lr_model.transform(test_eval_df)
lr_rmse = RegressionEvaluator(labelCol="WEEKLY_SALES", predictionCol="prediction", metricName="rmse").evaluate(lr_predictions)
lr_r2 = RegressionEvaluator(labelCol="WEEKLY_SALES", predictionCol="prediction", metricName="r2").evaluate(lr_predictions)

# Compare models and select the best one
model_scores = [
    ("Random Forest", rf_rmse, rf_r2, rf_model),
    ("GBT", gbt_rmse, gbt_r2, gbt_model),
    ("Linear Regression", lr_rmse, lr_r2, lr_model)
]

print("\nModel Performance (RMSE / R²):")
for name, rmse, r2, model in model_scores:
    print(f"{name}: RMSE={rmse:.2f}, R²={r2:.3f}")

# Select best model (lowest RMSE)
best_model_name, best_rmse, best_r2, best_model = min(model_scores, key=lambda x: x[1])
print(f"\nBest model: {best_model_name} (RMSE: {best_rmse:.2f}, R²: {best_r2:.3f})")

# 10. Generate recursive predictions for future dates
print("Generating recursive future predictions...")
future_predictions = create_future_features_with_recursion(future_df, train_df, best_model)

# 11. Write predictions to Snowflake
sfOptions_pred = sfOptions.copy()
sfOptions_pred["sfSchema"] = "ANALYTICS"

future_predictions.select("STORE", "DEPT", "DATE", "PREDICTED_SALES") \
    .withColumnRenamed("PREDICTED_SALES", "PREDICTED_WEEKLY_SALES") \
    .write.format("snowflake") \
    .options(**sfOptions_pred) \
    .option("dbtable", "PREDICTED_WEEKLY_SALES_FUTURE") \
    .mode("overwrite") \
    .save()

print(f"{best_model_name} predictions for 2012-11-02 to 2013-10-26 written to Snowflake!")
print(f"Prediction period: {len(future_dates)} weeks")
print(f"Total predictions generated: {future_predictions.count()}")

spark.stop()