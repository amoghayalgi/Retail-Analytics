{{ config(materialized='table') }}

SELECT
    year,
    month,
    SUM(actual_weekly_sales) AS total_sales
FROM {{ ref('fct_sales_analytics') }}
GROUP BY year, month
ORDER BY year, month