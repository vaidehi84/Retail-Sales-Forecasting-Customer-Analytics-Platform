-- Business KPI SQL queries for executive reporting.

-- 1. Executive KPI snapshot
SELECT
    SUM(sales) AS total_revenue,
    SUM(profit) AS total_profit,
    ROUND(SUM(profit) * 100.0 / NULLIF(SUM(sales), 0), 2) AS profit_margin_pct,
    ROUND(SUM(sales) / NULLIF(COUNT(DISTINCT order_id), 0), 2) AS average_order_value,
    COUNT(DISTINCT customer_id) AS unique_customers,
    COUNT(DISTINCT order_id) AS total_orders
FROM superstore_orders;

-- 2. Top customers by lifetime value
SELECT
    customer_id,
    customer_name,
    segment,
    region,
    SUM(sales) AS lifetime_sales,
    SUM(profit) AS lifetime_profit,
    COUNT(DISTINCT order_id) AS order_count,
    MAX(order_date) AS last_order_date
FROM superstore_orders
GROUP BY customer_id, customer_name, segment, region
ORDER BY lifetime_sales DESC
LIMIT 25;

-- 3. Repeat customer rate
WITH customer_orders AS (
    SELECT customer_id, COUNT(DISTINCT order_id) AS orders
    FROM superstore_orders
    GROUP BY customer_id
)
SELECT
    ROUND(SUM(CASE WHEN orders > 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS repeat_customer_rate_pct
FROM customer_orders;

-- 4. Regional revenue and profitability
SELECT
    region,
    state,
    SUM(sales) AS revenue,
    SUM(profit) AS profit,
    ROUND(SUM(profit) * 100.0 / NULLIF(SUM(sales), 0), 2) AS margin_pct,
    COUNT(DISTINCT order_id) AS orders
FROM superstore_orders
GROUP BY region, state
ORDER BY revenue DESC;

-- 5. Customer retention proxy by purchase frequency
SELECT
    segment,
    CASE
        WHEN COUNT(DISTINCT order_id) >= 5 THEN 'High Frequency'
        WHEN COUNT(DISTINCT order_id) BETWEEN 2 AND 4 THEN 'Repeat'
        ELSE 'One-Time'
    END AS frequency_group,
    COUNT(DISTINCT customer_id) AS customers
FROM superstore_orders
GROUP BY segment, frequency_group
ORDER BY segment, customers DESC;
