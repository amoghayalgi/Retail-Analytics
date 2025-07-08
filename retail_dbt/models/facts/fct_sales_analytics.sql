{{ config(materialized='table') }}

SELECT
    s.store,
    s.dept,
    s.date,
    s.weekly_sales AS actual_weekly_sales,
    COALESCE(p.predicted_weekly_sales, 0) AS predicted_weekly_sales,
    s.isholiday,
    f.temperature,
    f.fuel_price,
    f.cpi,
    f.unemployment,
    st.type AS store_type,
    st.size AS store_size,
    -- Pre-calculated metrics
    CASE WHEN st.size > 0 THEN s.weekly_sales / st.size ELSE NULL END AS actual_sales_per_sqft,
    CASE WHEN st.size > 0 AND p.predicted_weekly_sales IS NOT NULL
         THEN p.predicted_weekly_sales / st.size ELSE NULL END AS predicted_sales_per_sqft,
    CASE WHEN s.weekly_sales > 0 AND p.predicted_weekly_sales IS NOT NULL
         THEN ((p.predicted_weekly_sales - s.weekly_sales) / s.weekly_sales) * 100
         ELSE NULL END AS prediction_error_percentage,
    CASE WHEN p.predicted_weekly_sales IS NOT NULL
         THEN p.predicted_weekly_sales - s.weekly_sales
         ELSE NULL END AS prediction_error_absolute,
    EXTRACT(YEAR FROM s.date) AS year,
    EXTRACT(MONTH FROM s.date) AS month,
    EXTRACT(DAYOFWEEK FROM s.date) AS day_of_week,
    CASE WHEN UPPER(s.isholiday) = 'TRUE' THEN 1 ELSE 0 END AS is_holiday_flag
FROM {{ ref('stg_sales') }} s
LEFT JOIN {{ ref('stg_features') }} f
    ON s.store = f.store AND s.date = f.date
LEFT JOIN {{ ref('stg_stores') }} st
    ON s.store = st.store
LEFT JOIN ANALYTICS.PREDICTED_WEEKLY_SALES p
    ON s.store = p.store
    AND s.dept = p.dept
    AND s.date = p.date