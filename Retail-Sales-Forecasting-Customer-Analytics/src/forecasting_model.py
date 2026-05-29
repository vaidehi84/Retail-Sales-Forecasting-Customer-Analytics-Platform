"""Sales forecasting models for monthly revenue prediction."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "month_index",
    "month",
    "quarter",
    "year",
    "lag_1",
    "lag_2",
    "rolling_3",
    "Profit",
    "Quantity",
    "Orders",
]


def prepare_monthly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate order-level data into monthly forecasting features."""
    monthly = (
        df.groupby(pd.Grouper(key="Order Date", freq="MS"))
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum"),
            Quantity=("Quantity", "sum"),
            Orders=("Order ID", "nunique"),
        )
        .reset_index()
        .sort_values("Order Date")
    )
    monthly["month_index"] = np.arange(len(monthly))
    monthly["month"] = monthly["Order Date"].dt.month
    monthly["quarter"] = monthly["Order Date"].dt.quarter
    monthly["year"] = monthly["Order Date"].dt.year
    monthly["lag_1"] = monthly["Sales"].shift(1)
    monthly["lag_2"] = monthly["Sales"].shift(2)
    monthly["rolling_3"] = monthly["Sales"].rolling(3).mean()
    return monthly.dropna().reset_index(drop=True)


def train_sales_models(df: pd.DataFrame, models_dir: str | Path) -> dict:
    """Train Linear Regression and Random Forest forecasting models."""
    monthly = prepare_monthly_features(df)
    X = monthly[FEATURE_COLUMNS]
    y = monthly["Sales"]
    split = max(1, int(len(monthly) * 0.8))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler().fit(X_train)
    linear_model = LinearRegression().fit(scaler.transform(X_train), y_train)
    forest_model = RandomForestRegressor(
        n_estimators=300, random_state=42, min_samples_leaf=2
    ).fit(X_train, y_train)

    eval_X = X_test if len(X_test) else X_train
    eval_y = y_test if len(y_test) else y_train
    metrics = {
        "linear_regression": _regression_metrics(
            eval_y, linear_model.predict(scaler.transform(eval_X))
        ),
        "random_forest": _regression_metrics(eval_y, forest_model.predict(eval_X)),
    }
    bundle = {
        "linear_regression": linear_model,
        "random_forest": forest_model,
        "metrics": metrics,
        "feature_columns": FEATURE_COLUMNS,
        "last_month": str(monthly["Order Date"].max().date()),
        "training_rows": int(len(monthly)),
        "source_dataset": "data/raw/train.csv",
    }
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    dump(bundle, models_path / "sales_forecast_model.pkl")
    dump(scaler, models_path / "scaler.pkl")
    return bundle


def forecast_future_sales(
    monthly: pd.DataFrame,
    model_bundle: dict,
    scaler: StandardScaler,
    periods: int = 6,
    model_name: str = "random_forest",
) -> pd.DataFrame:
    """Generate recursive monthly forecasts from the latest monthly history."""
    history = monthly.copy().sort_values("Order Date").reset_index(drop=True)
    forecasts = []
    model = model_bundle[model_name]

    for _ in range(periods):
        next_date = history["Order Date"].max() + pd.offsets.MonthBegin(1)
        recent = history.tail(3)
        row = {
            "Order Date": next_date,
            "month_index": int(history["month_index"].max() + 1),
            "month": next_date.month,
            "quarter": next_date.quarter,
            "year": next_date.year,
            "lag_1": float(history.iloc[-1]["Sales"]),
            "lag_2": float(history.iloc[-2]["Sales"]),
            "rolling_3": float(recent["Sales"].mean()),
            "Profit": float(recent["Profit"].mean()),
            "Quantity": float(recent["Quantity"].mean()),
            "Orders": float(recent["Orders"].mean()),
        }
        X_next = pd.DataFrame([row])[FEATURE_COLUMNS]
        if model_name == "linear_regression":
            prediction = float(model.predict(scaler.transform(X_next))[0])
        else:
            prediction = float(model.predict(X_next)[0])
        row["Sales"] = max(0, prediction)
        row["Forecast Sales"] = max(0, prediction)
        forecasts.append(row)
        history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)

    return pd.DataFrame(forecasts)[["Order Date", "Forecast Sales"]]


def load_forecast_artifacts(models_dir: str | Path) -> tuple[dict, StandardScaler]:
    """Load saved forecasting model bundle and scaler."""
    models_path = Path(models_dir)
    return load(models_path / "sales_forecast_model.pkl"), load(models_path / "scaler.pkl")


def _regression_metrics(actual: pd.Series, predicted: np.ndarray) -> dict:
    return {
        "MAE": round(float(mean_absolute_error(actual, predicted)), 2),
        "RMSE": round(float(mean_squared_error(actual, predicted, squared=False)), 2),
        "R2": round(float(r2_score(actual, predicted)), 4) if len(actual) > 1 else 0.0,
    }
