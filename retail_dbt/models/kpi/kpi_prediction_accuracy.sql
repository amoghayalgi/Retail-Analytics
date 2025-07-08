{{ config(materialized='table') }}

SELECT
    store,
    dept,
    year,
    month,
    COUNT(*) AS total_predictions,
    AVG(ABS(prediction_error_percentage)) AS avg_absolute_error_percentage,
    AVG(prediction_error_percentage) AS avg_error_percentage,
    STDDEV(prediction_error_percentage) AS stddev_error_percentage,
    COUNT(CASE WHEN ABS(prediction_error_percentage) <= 5 THEN 1 END) AS excellent_predictions,
    COUNT(CASE WHEN ABS(prediction_error_percentage) <= 10 THEN 1 END) AS good_predictions,
    COUNT(CASE WHEN ABS(prediction_error_percentage) <= 20 THEN 1 END) AS fair_predictions,
    COUNT(CASE WHEN ABS(prediction_error_percentage) > 20 THEN 1 END) AS poor_predictions
FROM {{ ref('fct_sales_analytics') }}
WHERE prediction_error_percentage IS NOT NULL
GROUP BY store, dept, year, month