"""Shared utility functions for reports, KPIs, and file paths."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_processed_data(path: str | Path | None = None) -> pd.DataFrame:
    data_path = Path(path) if path else project_root() / "data/processed/superstore_processed.csv"
    if not data_path.exists():
        from src.preprocessing import clean_superstore_data

        clean_superstore_data(project_root() / "data/raw/train.csv", data_path)
    return pd.read_csv(data_path, parse_dates=["Order Date", "Ship Date"])


def calculate_kpis(df: pd.DataFrame) -> dict:
    total_revenue = float(df["Sales"].sum())
    total_profit = float(df["Profit"].sum())
    orders = int(df["Order ID"].nunique())
    customers = int(df["Customer ID"].nunique())
    previous_month = df.groupby("Order YearMonth")["Sales"].sum().tail(2)
    monthly_growth = 0.0
    if len(previous_month) == 2 and previous_month.iloc[0] != 0:
        monthly_growth = ((previous_month.iloc[1] - previous_month.iloc[0]) / previous_month.iloc[0]) * 100
    return {
        "total_revenue": total_revenue,
        "total_profit": total_profit,
        "profit_margin": (total_profit / total_revenue * 100) if total_revenue else 0,
        "average_order_value": (total_revenue / orders) if orders else 0,
        "monthly_growth": float(monthly_growth),
        "orders": orders,
        "customers": customers,
    }


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def executive_summary(df: pd.DataFrame) -> pd.DataFrame:
    kpis = calculate_kpis(df)
    top_category = df.groupby("Category")["Sales"].sum().idxmax()
    top_region = df.groupby("Region")["Sales"].sum().idxmax()
    loss_products = df.groupby("Product Name")["Profit"].sum().lt(0).sum()
    return pd.DataFrame(
        [
            {
                "Insight Area": "Revenue",
                "Business Insight": f"Total revenue is ${kpis['total_revenue']:,.0f} across {kpis['orders']:,} orders.",
            },
            {
                "Insight Area": "Profitability",
                "Business Insight": f"Estimated profit margin is {kpis['profit_margin']:.1f}%, highlighting margin quality of the sales mix.",
            },
            {
                "Insight Area": "Category",
                "Business Insight": f"{top_category} is the highest revenue category and should anchor inventory and campaign planning.",
            },
            {
                "Insight Area": "Region",
                "Business Insight": f"{top_region} contributes the highest sales and is a priority market for growth experiments.",
            },
            {
                "Insight Area": "Product Risk",
                "Business Insight": f"{loss_products} products show negative estimated profit and need pricing, discount, or sourcing review.",
            },
        ]
    )
