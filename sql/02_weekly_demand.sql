-- Roll daily sales up to weekly demand per store-item (one SKU-location = one series).
-- Replenishment is planned weekly, so weekly is the planning grain.
-- Weeks start on Monday; partial weeks at the edges of the data are dropped.

CREATE OR REPLACE TABLE weekly_demand AS
WITH by_week AS (
    SELECT
        store,
        item,
        CAST(DATE_TRUNC('week', date) AS DATE) AS week_start,
        SUM(sales)                             AS demand,
        COUNT(*)                               AS days_in_week
    FROM raw_sales
    GROUP BY 1, 2, 3
)
SELECT store, item, week_start, demand
FROM by_week
WHERE days_in_week = 7
ORDER BY store, item, week_start;
