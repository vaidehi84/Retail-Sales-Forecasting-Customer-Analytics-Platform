"""Customer analytics and segmentation helpers."""
from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create customer-level RFM-style features."""
    max_date = df["Order Date"].max()
    customer = (
        df.groupby(["Customer ID", "Customer Name", "Segment", "Region"])
        .agg(
            total_sales=("Sales", "sum"),
            total_profit=("Profit", "sum"),
            order_count=("Order ID", "nunique"),
            quantity=("Quantity", "sum"),
            last_order=("Order Date", "max"),
            categories=("Category", "nunique"),
        )
        .reset_index()
    )
    customer["recency_days"] = (max_date - customer["last_order"]).dt.days
    customer["avg_order_value"] = customer["total_sales"] / customer["order_count"].replace(0, pd.NA)
    customer["profit_margin"] = customer["total_profit"] / customer["total_sales"].replace(0, pd.NA)
    return customer.fillna(0)


def segment_customers(df: pd.DataFrame, n_clusters: int = 4) -> tuple[pd.DataFrame, KMeans, StandardScaler]:
    """Segment customers with K-Means clustering."""
    customer = build_customer_features(df)
    features = [
        "total_sales",
        "total_profit",
        "order_count",
        "quantity",
        "recency_days",
        "avg_order_value",
    ]
    scaler = StandardScaler().fit(customer[features])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(
        scaler.transform(customer[features])
    )
    customer["cluster"] = kmeans.labels_
    cluster_order = customer.groupby("cluster")["total_sales"].mean().sort_values().index.tolist()
    readable = {
        cluster_order[0]: "At-Risk Buyers",
        cluster_order[1]: "Growth Accounts",
        cluster_order[2]: "Value Loyalists",
        cluster_order[3]: "Premium Customers",
    }
    customer["customer_segment"] = customer["cluster"].map(readable)
    return customer, kmeans, scaler


def high_value_customers(customer_df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Return highest value customers by sales and profit."""
    return customer_df.sort_values(["total_sales", "total_profit"], ascending=False).head(top_n)


def repeat_customer_rate(df: pd.DataFrame) -> float:
    """Share of customers with more than one order."""
    orders = df.groupby("Customer ID")["Order ID"].nunique()
    if len(orders) == 0:
        return 0.0
    return float((orders > 1).mean())
