{{ config(materialized='table') }}

SELECT
    store,
    SUM(actual_weekly_sales) AS total_sales
FROM {{ ref('fct_sales_analytics') }}
GROUP BY store