# Retail Sales Forecasting & Customer Analytics Platform

A production-ready Business Analytics and Machine Learning project built on the Kaggle Superstore sales dataset. The platform combines executive KPI reporting, customer segmentation, product analytics, regional performance analysis, and monthly revenue forecasting in a deployable Streamlit dashboard.

This repository is designed to be resume-worthy, ATS-friendly, and aligned with analytics consulting workflows used in companies such as Mu Sigma, Fractal, Deloitte, and Accenture.

## Business Problem

Retail leaders need a reliable way to understand revenue performance, identify profitable customer segments, monitor category and regional trends, and forecast future sales. Raw order data is often scattered across spreadsheets and does not directly answer business questions such as:

- Which products and categories drive revenue and profit?
- Which customers are high value or at risk?
- Which regions need growth or margin improvement?
- What monthly sales trend should leadership expect next?

This project converts Superstore transaction data into a decision-support analytics application.

## Key Features

- Executive KPI dashboard with revenue, profit margin, average order value, monthly growth, orders, and customer count
- Interactive filters by region, category, and order date
- Monthly sales forecasting using Linear Regression and Random Forest Regressor
- Customer segmentation using K-Means clustering and RFM-style features
- High-value customer analysis and repeat customer insights
- Category, sub-category, top-product, and loss-making product analysis
- Region and state-level performance reporting
- Downloadable CSV reports and included PDF business report
- SQL query library for business KPI and sales analysis
- Streamlit Community Cloud compatible deployment structure

## Tech Stack

| Layer | Tools |
|---|---|
| Frontend | Streamlit |
| Analytics | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualization | Plotly, Matplotlib |
| Model Persistence | Joblib |
| Query Layer | SQL |
| Deployment | Streamlit Community Cloud |

## Dataset

The project uses the Kaggle Superstore `train.csv` file provided by the user and stores it at:

```text
data/raw/train.csv
```

The specific Kaggle file used here contains `Sales` but does not include native `Profit` and `Quantity` fields. To support a complete business analytics workflow, the preprocessing pipeline preserves the raw dataset and creates deterministic estimated `Profit`, `Quantity`, and `Discount` fields using documented category, segment, region, and sub-category business assumptions.

The processed analysis file is stored at:

```text
data/processed/superstore_processed.csv
```

## Machine Learning Workflow

1. Load Kaggle Superstore orders from `data/raw/train.csv`
2. Clean dates, text columns, and numeric fields
3. Engineer business features including order month, quarter, shipping days, profit margin, and average selling price
4. Aggregate monthly sales history
5. Train forecasting models:
   - Linear Regression
   - Random Forest Regressor
6. Evaluate models using MAE, RMSE, and R2
7. Save trained artifacts with Joblib:
   - `models/sales_forecast_model.pkl`
   - `models/scaler.pkl`
8. Build customer-level features and K-Means segmentation

## KPIs Analyzed

- Total revenue
- Total profit
- Profit margin
- Average order value
- Monthly sales growth
- Total orders
- Unique customers
- Repeat customer rate
- Category and sub-category sales
- Product profitability
- Regional and state performance

## Project Structure

```text
Retail-Sales-Forecasting-Customer-Analytics/
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample_data.csv
├── notebooks/
│   ├── data_cleaning.ipynb
│   ├── exploratory_analysis.ipynb
│   ├── model_training.ipynb
│   └── customer_segmentation.ipynb
├── src/
│   ├── preprocessing.py
│   ├── forecasting_model.py
│   ├── customer_analysis.py
│   ├── visualization.py
│   └── utils.py
├── dashboard/
│   └── app.py
├── models/
│   ├── sales_forecast_model.pkl
│   └── scaler.pkl
├── sql/
│   ├── sales_queries.sql
│   └── business_kpi_queries.sql
├── reports/
│   ├── business_insights.pdf
│   └── dashboard_screenshots/
├── screenshots/
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

```bash
git clone <your-repository-url>
cd Retail-Sales-Forecasting-Customer-Analytics
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For macOS or Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard opens in your browser and loads the processed dataset and saved models automatically. If processed files or model artifacts are missing, the app can regenerate them from `data/raw/train.csv`.

## Deployment on Streamlit Community Cloud

1. Push this project to GitHub.
2. Go to [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Select the GitHub repository.
4. Set the app entry point to:

```text
dashboard/app.py
```

5. Deploy.

## Screenshots

Add dashboard screenshots in either location:

```text
reports/dashboard_screenshots/
screenshots/
```

Recommended screenshots:

- Executive Dashboard
- Sales Forecasting
- Customer Analytics
- Product Analytics
- Regional Analysis

## Business Impact

This solution helps retail stakeholders:

- Monitor revenue and profitability trends in one place
- Forecast future monthly sales for planning and inventory decisions
- Identify premium customers and at-risk accounts
- Detect loss-making products requiring pricing or discount review
- Compare regional and state-level performance
- Download business-ready reports for leadership presentations

## Future Improvements

- Add Prophet, XGBoost, or LightGBM forecasting models
- Integrate a live database such as PostgreSQL or Snowflake
- Add authentication for stakeholder-specific dashboards
- Add automated model retraining workflows
- Add live PDF generation from active Streamlit filters
- Include inventory optimization and promotion response modeling

## Resume Keywords

Business Analytics, Retail Analytics, Sales Forecasting, Customer Segmentation, Machine Learning, Streamlit Dashboard, KPI Reporting, SQL Analytics, Random Forest, Linear Regression, K-Means Clustering, Data Visualization, Python, Pandas, Scikit-learn, Plotly.
