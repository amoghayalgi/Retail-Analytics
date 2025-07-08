{{ config(materialized='table') }}

SELECT
    date,
    EXTRACT(DAYOFWEEK FROM date) AS day_of_week,
    EXTRACT(WEEK FROM date) AS week_number,
    EXTRACT(MONTH FROM date) AS month,
    EXTRACT(QUARTER FROM date) AS quarter,
    EXTRACT(YEAR FROM date) AS year,
    CASE WHEN EXTRACT(DAYOFWEEK FROM date) IN (1, 7) THEN 1 ELSE 0 END AS is_weekend
FROM (
    SELECT DISTINCT date FROM {{ ref('stg_sales') }}
)