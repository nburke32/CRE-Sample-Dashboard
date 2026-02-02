"""
Shared helpers for Home.py and Predictive_Modeling.py.
"""

import pandas as pd


def get_latest_valid(df: pd.DataFrame, column: str):
    """
    Get most recent non-null value for *column* and a ~1 month prior value for delta.

    Returns (latest_row, prev_row) where prev_row may be None.
    Both rows are full DataFrame rows (pd.Series).
    """
    valid = df[df[column].notna()].sort_values("date")
    if valid.empty:
        return None, None
    latest_row = valid.iloc[-1]
    prev_data = valid[valid["date"] <= (latest_row["date"] - pd.Timedelta(days=25))]
    prev_row = prev_data.iloc[-1] if not prev_data.empty else None
    return latest_row, prev_row


def format_indicator_name(name: str) -> str:
    """Map FRED series keys to human-readable labels."""
    labels = {
        "treasury_10y": "10-Year Treasury",
        "treasury_2y": "2-Year Treasury",
        "mortgage_30y": "30-Year Mortgage",
        "fed_funds": "Fed Funds Rate",
        "housing_starts": "Housing Starts",
        "building_permits": "Building Permits",
        "construction_spending": "Total Construction",
        "commercial_construction": "Commercial Construction",
        "retail_sales": "Retail Sales",
        "consumer_sentiment": "Consumer Sentiment",
        "unemployment_national": "Unemployment Rate",
        "payrolls": "Nonfarm Payrolls",
        "industrial_production": "Industrial Production",
        "cre_loans": "CRE Loans Outstanding",
        "cre_delinquency": "CRE Delinquency Rate",
        "cpi": "Consumer Price Index",
        "gdp": "GDP",
    }
    return labels.get(name, name.replace("_", " ").title())
