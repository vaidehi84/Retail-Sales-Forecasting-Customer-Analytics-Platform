from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.customer_analysis import high_value_customers, repeat_customer_rate, segment_customers
from src.forecasting_model import (
    forecast_future_sales,
    load_forecast_artifacts,
    prepare_monthly_features,
    train_sales_models,
)
from src.preprocessing import clean_superstore_data
from src.utils import calculate_kpis, dataframe_to_csv_bytes, executive_summary, load_processed_data
from src.visualization import (
    category_performance_chart,
    customer_segment_chart,
    forecast_chart,
    monthly_sales_chart,
    regional_sales_chart,
    state_profit_chart,
    top_products_chart,
)

st.set_page_config(
    page_title="Retail Sales Forecasting & Customer Analytics",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main {background-color: #f8fafc;}
    div[data-testid="stSidebar"] {background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);}
    div[data-testid="stSidebar"] * {color: #f8fafc !important;}
    .metric-card {padding: 18px; border-radius: 8px; background: #ffffff; border: 1px solid #e2e8f0; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);}
    .metric-label {font-size: 0.82rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: .02em;}
    .metric-value {font-size: 1.55rem; font-weight: 800; color: #0f172a; margin-top: 5px;}
    .metric-help {font-size: 0.78rem; color: #64748b; margin-top: 4px;}
    .section-title {font-size: 1.25rem; font-weight: 800; color: #0f172a; margin: 12px 0 8px 0;}
    .small-muted {color:#64748b; font-size:.92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    processed_path = PROJECT_ROOT / "data/processed/superstore_processed.csv"
    raw_path = PROJECT_ROOT / "data/raw/train.csv"
    if not processed_path.exists():
        clean_superstore_data(raw_path, processed_path)
    return load_processed_data(processed_path)


@st.cache_resource(show_spinner=False)
def get_models(df: pd.DataFrame):
    models_dir = PROJECT_ROOT / "models"
    model_path = models_dir / "sales_forecast_model.pkl"
    scaler_path = models_dir / "scaler.pkl"
    if not model_path.exists() or not scaler_path.exists():
        train_sales_models(df, models_dir)
    return load_forecast_artifacts(models_dir)


def money(value: float) -> str:
    return f"${value:,.0f}"


def metric_card(label: str, value: str, help_text: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


df = get_data()

st.sidebar.title("Retail Analytics")
st.sidebar.caption("Forecasting, customers, product and regional intelligence")
page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Dashboard",
        "Sales Forecasting",
        "Customer Analytics",
        "Product Analytics",
        "Regional Analysis",
        "Reports",
    ],
)

region_filter = st.sidebar.multiselect(
    "Region", sorted(df["Region"].unique()), default=sorted(df["Region"].unique())
)
category_filter = st.sidebar.multiselect(
    "Category", sorted(df["Category"].unique()), default=sorted(df["Category"].unique())
)
min_date, max_date = df["Order Date"].min().date(), df["Order Date"].max().date()
date_range = st.sidebar.date_input(
    "Order date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
)

filtered = df[df["Region"].isin(region_filter) & df["Category"].isin(category_filter)].copy()
if len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered = filtered[(filtered["Order Date"] >= start_date) & (filtered["Order Date"] <= end_date)]

if filtered.empty:
    st.warning("No records match the selected filters. Adjust the sidebar filters to continue.")
    st.stop()

st.title("Retail Sales Forecasting & Customer Analytics Platform")
st.caption("Business intelligence dashboard built on the Kaggle Superstore dataset")

if page == "Executive Dashboard":
    kpis = calculate_kpis(filtered)
    cols = st.columns(5)
    with cols[0]:
        metric_card("Revenue", money(kpis["total_revenue"]), "Total sales value")
    with cols[1]:
        metric_card("Profit Margin", f"{kpis['profit_margin']:.1f}%", money(kpis["total_profit"]))
    with cols[2]:
        metric_card("Avg Order Value", money(kpis["average_order_value"]), f"{kpis['orders']:,} orders")
    with cols[3]:
        metric_card("Monthly Growth", f"{kpis['monthly_growth']:.1f}%", "Latest month vs previous")
    with cols[4]:
        metric_card("Customers", f"{kpis['customers']:,}", "Unique buyers")

    left, right = st.columns([1.4, 1])
    with left:
        st.plotly_chart(monthly_sales_chart(filtered), use_container_width=True)
    with right:
        st.plotly_chart(regional_sales_chart(filtered), use_container_width=True)
    st.plotly_chart(top_products_chart(filtered, 12), use_container_width=True)

elif page == "Sales Forecasting":
    model_bundle, scaler = get_models(df)
    monthly = prepare_monthly_features(filtered)
    periods = st.slider("Forecast horizon in months", min_value=3, max_value=18, value=6)
    model_name = st.selectbox(
        "Forecasting model",
        ["random_forest", "linear_regression"],
        format_func=lambda value: value.replace("_", " ").title(),
    )
    forecast = forecast_future_sales(monthly, model_bundle, scaler, periods=periods, model_name=model_name)
    st.plotly_chart(forecast_chart(monthly, forecast), use_container_width=True)

    st.markdown("<div class='section-title'>Model Evaluation</div>", unsafe_allow_html=True)
    metrics = pd.DataFrame(model_bundle["metrics"]).T.reset_index().rename(columns={"index": "Model"})
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    st.download_button("Download forecast CSV", dataframe_to_csv_bytes(forecast), "sales_forecast.csv", "text/csv")

elif page == "Customer Analytics":
    customer_df, _, _ = segment_customers(filtered)
    repeat_rate = repeat_customer_rate(filtered) * 100
    cols = st.columns(4)
    with cols[0]:
        metric_card("Repeat Customer Rate", f"{repeat_rate:.1f}%", "Customers with 2+ orders")
    with cols[1]:
        metric_card("Customer Segments", f"{customer_df['customer_segment'].nunique()}", "K-Means clusters")
    with cols[2]:
        metric_card("Top Customer Sales", money(customer_df["total_sales"].max()), "Highest account value")
    with cols[3]:
        metric_card("Median AOV", money(customer_df["avg_order_value"].median()), "Per customer")
    st.plotly_chart(customer_segment_chart(customer_df), use_container_width=True)
    st.markdown("<div class='section-title'>High-Value Customers</div>", unsafe_allow_html=True)
    top_customers = high_value_customers(customer_df, 20)
    st.dataframe(
        top_customers[
            [
                "Customer Name",
                "Segment",
                "Region",
                "customer_segment",
                "total_sales",
                "total_profit",
                "order_count",
                "recency_days",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download customer segments",
        dataframe_to_csv_bytes(customer_df),
        "customer_segments.csv",
        "text/csv",
    )

elif page == "Product Analytics":
    st.plotly_chart(category_performance_chart(filtered), use_container_width=True)
    left, right = st.columns(2)
    with left:
        subcategory = (
            filtered.groupby("Sub-Category", as_index=False)
            .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"), Quantity=("Quantity", "sum"))
            .sort_values("Sales", ascending=False)
        )
        st.markdown("<div class='section-title'>Sub-Category Performance</div>", unsafe_allow_html=True)
        st.dataframe(subcategory, use_container_width=True, hide_index=True)
    with right:
        loss_products = (
            filtered.groupby("Product Name", as_index=False)
            .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
            .query("Profit < 0")
            .sort_values("Profit")
        )
        st.markdown("<div class='section-title'>Loss-Making Products</div>", unsafe_allow_html=True)
        st.dataframe(loss_products.head(25), use_container_width=True, hide_index=True)
    st.plotly_chart(top_products_chart(filtered, 15), use_container_width=True)

elif page == "Regional Analysis":
    st.plotly_chart(regional_sales_chart(filtered), use_container_width=True)
    st.plotly_chart(state_profit_chart(filtered), use_container_width=True)
    regional = (
        filtered.groupby(["Region", "State"], as_index=False)
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"), Orders=("Order ID", "nunique"))
        .sort_values("Sales", ascending=False)
    )
    st.dataframe(regional, use_container_width=True, hide_index=True)
    st.download_button(
        "Download regional performance",
        dataframe_to_csv_bytes(regional),
        "regional_performance.csv",
        "text/csv",
    )

elif page == "Reports":
    summary = executive_summary(filtered)
    st.markdown("<div class='section-title'>Business Insights Summary</div>", unsafe_allow_html=True)
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.download_button(
        "Download insights CSV",
        dataframe_to_csv_bytes(summary),
        "business_insights_summary.csv",
        "text/csv",
    )

    pdf_path = PROJECT_ROOT / "reports/business_insights.pdf"
    if pdf_path.exists():
        st.download_button(
            "Download PDF report",
            pdf_path.read_bytes(),
            "business_insights.pdf",
            "application/pdf",
        )
    st.markdown(
        "<p class='small-muted'>The PDF report is generated from the processed dataset and included in the repository for recruiter-friendly review.</p>",
        unsafe_allow_html=True,
    )
