-- models/staging/stg_sales.sql

WITH base AS (
    SELECT
        CAST(STORE AS INTEGER) AS store,
        CAST(DEPT AS INTEGER) AS dept,
        CAST(DATE AS DATE) AS date,
        CAST(WEEKLY_SALES AS FLOAT) AS weekly_sales,
        -- Standardize isholiday to boolean (1/0)
        CASE 
            WHEN UPPER(ISHOLIDAY) IN ('TRUE', '1', 'Y') THEN 1
            ELSE 0
        END AS isholiday
    FROM STAGING.RETAIL_FEATURES
    WHERE WEEKLY_SALES IS NOT NULL
      AND STORE IS NOT NULL
      AND DEPT IS NOT NULL
      AND DATE IS NOT NULL
)

SELECT
    store,
    dept,
    date,
    weekly_sales,
    isholiday
FROM base
QUALIFY ROW_NUMBER() OVER (PARTITION BY store, dept, date ORDER BY weekly_sales DESC) = 1