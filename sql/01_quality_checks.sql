-- Data quality checks on the raw daily sales file.
-- Every check should return zero problem rows before modeling.

-- 1. Coverage: date range, stores, items, row count
SELECT
    MIN(date)              AS first_day,
    MAX(date)              AS last_day,
    COUNT(DISTINCT store)  AS stores,
    COUNT(DISTINCT item)   AS items,
    COUNT(*)               AS rows
FROM raw_sales;

-- 2. Duplicate store-item-day rows
SELECT COUNT(*) AS duplicate_rows
FROM (
    SELECT date, store, item
    FROM raw_sales
    GROUP BY 1, 2, 3
    HAVING COUNT(*) > 1
);

-- 3. Missing days: every store-item should have one row per calendar day
SELECT COUNT(*) AS series_with_gaps
FROM (
    SELECT store, item,
           COUNT(*) AS days,
           DATE_DIFF('day', MIN(date), MAX(date)) + 1 AS expected_days
    FROM raw_sales
    GROUP BY 1, 2
)
WHERE days <> expected_days;

-- 4. Negative or null sales
SELECT COUNT(*) AS bad_sales_rows
FROM raw_sales
WHERE sales IS NULL OR sales < 0;
