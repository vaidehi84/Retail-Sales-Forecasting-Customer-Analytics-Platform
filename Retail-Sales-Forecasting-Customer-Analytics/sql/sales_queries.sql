-- Retail Sales Forecasting & Customer Analytics Platform
-- Core sales analytics queries for a Superstore-style table named superstore_orders.

-- 1. Monthly revenue and profit trend
SELECT
    DATE_TRUNC('month', order_date) AS sales_month,
    SUM(sales) AS monthly_revenue,
    SUM(profit) AS monthly_profit,
    COUNT(DISTINCT order_id) AS total_orders
FROM superstore_orders
GROUP BY DATE_TRUNC('month', order_date)
ORDER BY sales_month;

-- 2. Month-over-month sales growth
WITH monthly_sales AS (
    SELECT DATE_TRUNC('month', order_date) AS sales_month, SUM(sales) AS revenue
    FROM superstore_orders
    GROUP BY DATE_TRUNC('month', order_date)
)
SELECT
    sales_month,
    revenue,
    LAG(revenue) OVER (ORDER BY sales_month) AS previous_month_revenue,
    ROUND(
        (revenue - LAG(revenue) OVER (ORDER BY sales_month)) * 100.0
        / NULLIF(LAG(revenue) OVER (ORDER BY sales_month), 0),
        2
    ) AS mom_growth_pct
FROM monthly_sales
ORDER BY sales_month;

-- 3. Top 20 products by revenue
SELECT product_name, category, sub_category, SUM(sales) AS revenue, SUM(profit) AS profit
FROM superstore_orders
GROUP BY product_name, category, sub_category
ORDER BY revenue DESC
LIMIT 20;

-- 4. Loss-making products needing margin action
SELECT product_name, category, sub_category, SUM(sales) AS revenue, SUM(profit) AS profit
FROM superstore_orders
GROUP BY product_name, category, sub_category
HAVING SUM(profit) < 0
ORDER BY profit ASC;

-- 5. Category and sub-category performance
SELECT category, sub_category, SUM(sales) AS revenue, SUM(profit) AS profit, SUM(quantity) AS units_sold
FROM superstore_orders
GROUP BY category, sub_category
ORDER BY revenue DESC;
