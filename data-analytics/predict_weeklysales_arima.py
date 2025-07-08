import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, DateType, DoubleType
from pyspark.sql.functions import pandas_udf
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from snowflake.connector.pandas_tools import write_pandas
import snowflake.connector
from pmdarima import auto_arima
import time

# 1. Spark session and Snowflake connection options
spark = SparkSession.builder \
    .appName("ARIMA Forecast") \
    .config("spark.jars.packages", "net.snowflake:snowflake-jdbc:3.13.30,net.snowflake:spark-snowflake_2.12:2.16.0-spark_3.4") \
    .getOrCreate()

sfOptions = {
    "sfURL": "KRWLMZY-HU90233.snowflakecomputing.com",
    "sfUser": "amoghayalgi",
    "sfPassword": "Ammu2025Heeya461",
    "sfDatabase": "RETAIL_DB",
    "sfSchema": "STAGING",
    "sfWarehouse": "RETAIL_WH"
}

# 2. Read sales data from Snowflake into Spark
sales_df = spark.read.format("snowflake").options(**sfOptions).option("dbtable", "RETAIL_FEATURES").load()

# 3. Define schema and Pandas UDF for ARIMA forecasting
schema = StructType([
    StructField("STORE", IntegerType()),
    StructField("DEPT", IntegerType()),
    StructField("DATE", DateType()),
    StructField("PREDICTED_WEEKLY_SALES", DoubleType())
])

from pyspark.sql.functions import PandasUDFType
@pandas_udf(schema, PandasUDFType.GROUPED_MAP)
#@pandas_udf(schema, "grouped_map")
def arima_forecast_udf(pdf: pd.DataFrame) -> pd.DataFrame:
    # Ensure DATE is datetime and set as index with weekly frequency
    pdf = pdf.sort_values("DATE")
    pdf["DATE"] = pd.to_datetime(pdf["DATE"])
    pdf = pdf.set_index("DATE").asfreq("W-FRI")
    # Fill missing sales with forward fill (or interpolate as needed)
    y = pdf["WEEKLY_SALES"].astype(float).fillna(method='ffill')
    # Use only last 2 years for training
    cutoff = pd.to_datetime("2011-07-01") - pd.DateOffset(years=2)
    y = y[y.index >= cutoff]
    results = []
    # Skip groups with too little data or too many zeros
    if len(y) < 30 or (y == 0).sum() > 0.5 * len(y):
        print(f"Skipping STORE={pdf['STORE'].iloc[0]}, DEPT={pdf['DEPT'].iloc[0]}: not enough data or too many zeros.")
        return pd.DataFrame(results)
    try:
        start = time.time()
        model = auto_arima(
            y, seasonal=True, m=52, suppress_warnings=True, error_action='ignore',
            max_p=2, max_q=2, max_d=1, max_P=1, max_Q=1, max_D=1, stepwise=True
        )
        future_dates = pd.date_range("2011-07-01", "2013-10-26", freq="W-FRI")
        forecast = model.predict(n_periods=len(future_dates))
        print(f"STORE={pdf['STORE'].iloc[0]}, DEPT={pdf['DEPT'].iloc[0]}, time: {time.time()-start:.2f}s, forecast dates: {future_dates[0]} to {future_dates[-1]} (auto_arima)")
    except Exception as e:
        print(f"auto_arima failed for store {pdf['STORE'].iloc[0]}, dept {pdf['DEPT'].iloc[0]}: {e}. Falling back to ARIMA(2,1,2)")
        try:
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(y, order=(2,1,2))
            model_fit = model.fit()
            future_dates = pd.date_range("2011-07-01", "2013-10-26", freq="W-FRI")
            forecast = model_fit.forecast(steps=len(future_dates))
        except Exception as e2:
            print(f"ARIMA(2,1,2) also failed for store {pdf['STORE'].iloc[0]}, dept {pdf['DEPT'].iloc[0]}: {e2}")
            return pd.DataFrame(results)
    for date, pred in zip(future_dates, forecast):
        results.append({
            "STORE": int(pdf["STORE"].iloc[0]),
            "DEPT": int(pdf["DEPT"].iloc[0]),
            "DATE": date.date(),
            "PREDICTED_WEEKLY_SALES": float(pred)
        })
    if results:
        print(f"Results count: {len(results)}; First date: {results[0]['DATE']}; Last date: {results[-1]['DATE']}")
    else:
        print("No results returned for this group.")
    return pd.DataFrame(results)

# 4. Filter your Spark DataFrame for the full range needed for evaluation
filtered_df = sales_df.filter((sales_df.DATE >= "2010-07-01") & (sales_df.DATE <= "2013-10-26"))

# 5. Group by store and dept, and apply the UDF
forecast_df = filtered_df.groupby("STORE", "DEPT").apply(arima_forecast_udf)

# 6. Write the forecast results to Snowflake (ANALYTICS schema)
forecast_df.write.format("snowflake") \
    .options(**sfOptions) \
    .option("dbtable", "PREDICTED_WEEKLY_SALES_ARIMA") \
    .option("dbschema", "ANALYTICS") \
    .mode("overwrite") \
    .save()

print("Wrote ARIMA forecast results to Snowflake ANALYTICS.PREDICTED_WEEKLY_SALES_ARIMA.")

# 7. Model evaluation: collect to pandas and evaluate per group
pdf = filtered_df.select("STORE", "DEPT", "DATE", "WEEKLY_SALES").toPandas()
eval_results = []

def safe_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

for (store, dept), group in pdf.groupby(["STORE", "DEPT"]):
    group = group.sort_values("DATE")
    group = group.dropna(subset=["WEEKLY_SALES", "DATE"])
    group = group.set_index("DATE")
    y = group["WEEKLY_SALES"].astype(float)
    # --- Evaluation: hold out last 13 weeks as test ---
    if len(y) > 20:
        train, test = y[:-13], y[-13:]
        try:
            model = ARIMA(train, order=(2,1,2))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=len(test))
            rmse = np.sqrt(mean_squared_error(test, forecast))
            mape = safe_mape(test.values, forecast)
            eval_results.append({
                "STORE": store,
                "DEPT": dept,
                "RMSE": rmse,
                "MAPE": mape
            })
        except Exception as e:
            print(f"Eval ARIMA failed for store {store}, dept {dept}: {e}")

# 8. Write evaluation results to Snowflake
if eval_results:
    eval_df = pd.DataFrame(eval_results)
    conn = snowflake.connector.connect(
        user=sfOptions["sfUser"],
        password=sfOptions["sfPassword"],
        account=sfOptions["sfURL"].split('.')[0],
        warehouse=sfOptions["sfWarehouse"],
        database=sfOptions["sfDatabase"],
        schema="ANALYTICS"
    )
    write_pandas(conn, eval_df, "ARIMA_EVAL_RESULTS", auto_create_table=True)
    print("Wrote evaluation results to Snowflake ANALYTICS.ARIMA_EVAL_RESULTS.")

spark.stop()
