{{ config(materialized='table') }}

SELECT
    store AS store_id,
    type AS store_type,
    size AS store_size
FROM {{ ref('stg_stores') }}