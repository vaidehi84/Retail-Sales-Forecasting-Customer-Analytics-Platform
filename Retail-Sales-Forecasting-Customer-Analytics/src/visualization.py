"""Reusable Plotly visualizations for the Streamlit dashboard."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

TEMPLATE = "plotly_white"
COLORWAY = ["#2563eb", "#14b8a6", "#f97316", "#7c3aed", "#334155", "#dc2626"]


def monthly_sales_chart(df: pd.DataFrame) -> go.Figure:
    monthly = df.groupby("Order YearMonth", as_index=False).agg(
        Sales=("Sales", "sum"), Profit=("Profit", "sum")
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=monthly["Order YearMonth"],
            y=monthly["Sales"],
            mode="lines+markers",
            name="Sales",
            line=dict(color=COLORWAY[0], width=3),
        )
    )
    fig.add_trace(
        go.Bar(
            x=monthly["Order YearMonth"],
            y=monthly["Profit"],
            name="Profit",
            marker_color=COLORWAY[1],
            opacity=0.65,
        )
    )
    fig.update_layout(
        template=TEMPLATE,
        title="Monthly Sales and Profit Trend",
        hovermode="x unified",
        legend_orientation="h",
    )
    return fig


def category_performance_chart(df: pd.DataFrame) -> go.Figure:
    data = df.groupby("Category", as_index=False).agg(
        Sales=("Sales", "sum"), Profit=("Profit", "sum"), Quantity=("Quantity", "sum")
    )
    return px.bar(
        data,
        x="Category",
        y=["Sales", "Profit"],
        barmode="group",
        template=TEMPLATE,
        color_discrete_sequence=COLORWAY,
        title="Category-wise Sales and Profit",
    )


def regional_sales_chart(df: pd.DataFrame) -> go.Figure:
    data = df.groupby("Region", as_index=False).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
    return px.scatter(
        data,
        x="Sales",
        y="Profit",
        size="Sales",
        color="Region",
        template=TEMPLATE,
        color_discrete_sequence=COLORWAY,
        title="Region-wise Sales vs Profit",
    )


def top_products_chart(df: pd.DataFrame, n: int = 10) -> go.Figure:
    data = (
        df.groupby("Product Name", as_index=False)
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
        .sort_values("Sales", ascending=False)
        .head(n)
    )
    return px.bar(
        data.sort_values("Sales"),
        x="Sales",
        y="Product Name",
        color="Profit",
        orientation="h",
        template=TEMPLATE,
        color_continuous_scale="Tealgrn",
        title=f"Top {n} Products by Sales",
    )


def forecast_chart(monthly: pd.DataFrame, forecast: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=monthly["Order Date"],
            y=monthly["Sales"],
            mode="lines+markers",
            name="Actual Sales",
            line=dict(color=COLORWAY[0], width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["Order Date"],
            y=forecast["Forecast Sales"],
            mode="lines+markers",
            name="Forecast Sales",
            line=dict(color=COLORWAY[2], width=3, dash="dash"),
        )
    )
    fig.update_layout(
        template=TEMPLATE,
        title="Future Monthly Revenue Forecast",
        hovermode="x unified",
        legend_orientation="h",
    )
    return fig


def customer_segment_chart(customer_df: pd.DataFrame) -> go.Figure:
    data = customer_df.groupby("customer_segment", as_index=False).agg(
        customers=("Customer ID", "count"),
        sales=("total_sales", "sum"),
        profit=("total_profit", "sum"),
    )
    return px.treemap(
        data,
        path=["customer_segment"],
        values="sales",
        color="profit",
        color_continuous_scale="Blues",
        template=TEMPLATE,
        title="Customer Segment Contribution",
    )


def state_profit_chart(df: pd.DataFrame) -> go.Figure:
    data = (
        df.groupby("State", as_index=False)
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
        .sort_values("Profit", ascending=False)
        .head(20)
    )
    return px.bar(
        data,
        x="State",
        y="Profit",
        color="Sales",
        template=TEMPLATE,
        color_continuous_scale="Viridis",
        title="Top State Profit Distribution",
    )
