-- Forecast features built with window functions.
--
-- Every feature only looks back 3+ weeks. With a 2-week supplier lead time and a
-- weekly review, an order placed today must cover the next 3 weeks, so the
-- forecast for any of those weeks can only use demand we have already seen.
-- This keeps the backtest honest: no feature leaks future demand.

CREATE OR REPLACE TABLE features AS
WITH w AS (
    SELECT
        *,
        ROW_NUMBER() OVER s AS week_idx
    FROM weekly_demand
    WINDOW s AS (PARTITION BY store, item ORDER BY week_start)
)
SELECT
    store,
    item,
    week_start,
    week_idx,
    demand,

    -- recent demand
    LAG(demand, 3) OVER s                                                       AS lag_3,
    LAG(demand, 4) OVER s                                                       AS lag_4,
    LAG(demand, 5) OVER s                                                       AS lag_5,
    LAG(demand, 6) OVER s                                                       AS lag_6,

    -- rolling averages (windows end 3 weeks back)
    AVG(demand) OVER (s ROWS BETWEEN 6 PRECEDING AND 3 PRECEDING)               AS ma_4,
    AVG(demand) OVER (s ROWS BETWEEN 10 PRECEDING AND 3 PRECEDING)              AS ma_8,
    AVG(demand) OVER (s ROWS BETWEEN 15 PRECEDING AND 3 PRECEDING)              AS ma_13,

    -- same period last year (seasonality) and its 4-week context
    LAG(demand, 51) OVER s                                                      AS lag_51,
    LAG(demand, 52) OVER s                                                      AS lag_52,
    LAG(demand, 53) OVER s                                                      AS lag_53,
    AVG(demand) OVER (s ROWS BETWEEN 58 PRECEDING AND 55 PRECEDING)             AS ma_4_last_year,

    -- calendar
    WEEKOFYEAR(week_start)                                                      AS week_of_year,
    MONTH(week_start)                                                           AS month,
    YEAR(week_start)                                                            AS year
FROM w
WINDOW s AS (PARTITION BY store, item ORDER BY week_start)
ORDER BY store, item, week_start;
