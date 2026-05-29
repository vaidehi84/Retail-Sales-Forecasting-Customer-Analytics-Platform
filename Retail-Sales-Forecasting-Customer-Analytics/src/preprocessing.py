"""Data cleaning and feature engineering for the Superstore analytics platform."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CATEGORY_MARGIN = {"Technology": 0.18, "Office Supplies": 0.12, "Furniture": 0.07}
SEGMENT_ADJUSTMENT = {"Consumer": 0.00, "Corporate": 0.015, "Home Office": 0.01}
REGION_ADJUSTMENT = {"West": 0.015, "East": 0.01, "South": -0.005, "Central": -0.015}
SUBCATEGORY_RISK = {
    "Tables": -0.10,
    "Bookcases": -0.04,
    "Supplies": -0.03,
    "Machines": -0.02,
    "Chairs": 0.00,
    "Phones": 0.02,
    "Copiers": 0.06,
    "Accessories": 0.03,
    "Binders": 0.01,
    "Paper": 0.02,
    "Storage": 0.01,
    "Furnishings": 0.00,
    "Appliances": 0.015,
    "Art": 0.005,
    "Envelopes": 0.015,
    "Fasteners": 0.005,
    "Labels": 0.02,
}


def clean_superstore_data(input_path: str | Path, output_path: str | Path | None = None) -> pd.DataFrame:
    """Load Kaggle Superstore CSV and create analysis-ready business features.

    The supplied Kaggle `train.csv` has Sales but not native Profit or Quantity.
    The raw file is preserved, while deterministic estimated fields are created
    from documented category, segment, region, and sub-category assumptions.
    """
    df = pd.read_csv(input_path)
    df.columns = [column.strip() for column in df.columns]

    for column in ["Order Date", "Ship Date"]:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], dayfirst=True, errors="coerce")

    df = df.dropna(subset=["Order Date", "Sales"]).copy()
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce").fillna(0).round(2)

    if "Quantity" not in df.columns:
        avg_ticket_by_category = {"Technology": 240, "Furniture": 310, "Office Supplies": 45}
        estimated_unit_price = df["Category"].map(avg_ticket_by_category).fillna(120)
        df["Quantity"] = np.clip(
            np.maximum(1, np.round(df["Sales"] / estimated_unit_price)), 1, 14
        ).astype(int)

    if "Profit" not in df.columns:
        margin = (
            df["Category"].map(CATEGORY_MARGIN).fillna(0.10)
            + df["Segment"].map(SEGMENT_ADJUSTMENT).fillna(0)
            + df["Region"].map(REGION_ADJUSTMENT).fillna(0)
            + df["Sub-Category"].map(SUBCATEGORY_RISK).fillna(0)
        )
        seasonal_drag = np.where(df["Order Date"].dt.month.isin([11, 12]), -0.015, 0)
        df["Profit"] = (df["Sales"] * (margin + seasonal_drag)).round(2)

    if "Discount" not in df.columns:
        df["Discount"] = np.where(
            df["Profit"] < 0,
            0.20,
            np.where(df["Sales"] > df["Sales"].quantile(0.85), 0.10, 0.00),
        )

    df["Order Year"] = df["Order Date"].dt.year
    df["Order Month"] = df["Order Date"].dt.month
    df["Order Quarter"] = df["Order Date"].dt.quarter
    df["Order YearMonth"] = df["Order Date"].dt.to_period("M").astype(str)
    df["Ship Days"] = (df["Ship Date"] - df["Order Date"]).dt.days.fillna(0).clip(lower=0)
    df["Profit Margin"] = np.where(df["Sales"] > 0, df["Profit"] / df["Sales"], 0)
    df["Average Selling Price"] = np.where(
        df["Quantity"] > 0, df["Sales"] / df["Quantity"], df["Sales"]
    )

    text_columns = [
        "Customer Name",
        "Segment",
        "Country",
        "City",
        "State",
        "Region",
        "Category",
        "Sub-Category",
        "Product Name",
        "Ship Mode",
    ]
    for column in text_columns:
        if column in df.columns:
            df[column] = df[column].fillna("Unknown").astype(str)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    clean_superstore_data(
        project_root / "data/raw/train.csv",
        project_root / "data/processed/superstore_processed.csv",
    )
