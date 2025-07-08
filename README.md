# Retail-Analytics

Retail-Analytics is an end-to-end data analytics and forecasting project designed to predict weekly sales for a retail chain at the store and department level. The project leverages Snowflake for data warehousing, dbt for data modeling, PySpark for scalable machine learning, and Power BI for business intelligence and visualization. The workflow includes data ingestion, feature engineering, model training and evaluation, future sales prediction, and interactive reporting.

Project Workflow: Step-by-Step

1. Data Ingestion and Storage

   -Raw sales, store, and external data (e.g., weather, markdowns, economic indicators) are ingested and stored in Snowflake tables.

2. Data Modeling with dbt

    -dbt is used to create staging and fact models, including:

       a. Staging tables for raw and cleaned data.

       b. Fact tables for actual sales, predicted sales, and combined actual vs. predicted sales.

       c. Schema documentation for all models.

3. Feature Engineering in PySpark

   -Feature engineering is performed in PySpark, including:

       a. Creation of lag features (e.g., previous weeks’ sales).

       b. Rolling averages and holiday encoding.

       c. Scaling and imputation of missing values.

4. Model Training and Selection

   -Multiple regression models are trained using PySpark MLlib:

       a. Random Forest, Gradient Boosted Trees (GBT), and Linear Regression.

       b. Models are evaluated using a train-test split (or cross-validation if desired).

       c. The best model is selected based on RMSE and R² metrics.

5. Future Sales Prediction

   -Future dates (2012-11-02 to 2013-10-26) are generated for all store-department combinations.

   -The best model is used to predict weekly sales for these future dates.

   -Predictions are written back to Snowflake in a dedicated table (PREDICTED_WEEKLY_SALES_FUTURE).

6. Power BI Reporting

   -Data is loaded from Snowflake into Power BI.

   Visualizations include:

        a. Actual vs. predicted sales over time.

        b. Department/store-level performance.

        c. KPI cards for total/average sales, top departments, and percentage changes.

        d. DAX measures for custom calculations (e.g., percentage increase, top department).

10. GitHub Version Control

     -The entire project directory is version-controlled and uploaded to GitHub for collaboration and reproducibility.
