import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from data.config import STREAMLIT_CACHE_TTL
from data.storage import DataStorage
from helpers import get_latest_valid

st.set_page_config(
    page_title="Commercial Real Estate Dashboard",
    page_icon="🏢",
    layout="wide",
)

st.html("""
    <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
    </style>
""")

with st.sidebar:
    st.image("assets/logo.jpg", width=280)
    st.markdown("---")
    st.markdown("### Quick Links")
    st.page_link("Home.py", label="🏠 Home")
    st.page_link("pages/1_📊_Market_Analytics.py", label="📊 Market Analytics")
    st.page_link("pages/2_🗺️_Map_View.py", label="🗺️ Map View")
    st.page_link("pages/3_🤖_SEC_Chatbot.py", label="🤖 SEC Chatbot")
    st.page_link("pages/4_🔮_Predictive_Modeling.py", label="🔮 Predictive Modeling")
    st.markdown("---")
    st.caption("v0.1.0")

st.title("🏢 Commercial Real Estate Dashboard")
st.markdown("Welcome back! Here's your overview.")

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    ### 📊 Market Analytics
    Commercial trends, cap rates, and market insights.
    """)
    st.page_link("pages/1_📊_Market_Analytics.py", label="Go to Analytics →")

with col2:
    st.markdown("""
    ### 🗺️ Map View
    Explore commercial properties on an interactive map.
    """)
    st.page_link("pages/2_🗺️_Map_View.py", label="Open Map →")

with col3:
    st.markdown("""
    ### 🤖 SEC Chatbot
    Query SEC filings and regulatory documents.
    """)
    st.page_link("pages/3_🤖_SEC_Chatbot.py", label="Chat with SEC →")

with col4:
    st.markdown("""
    ### 🔮 Predictive Modeling
    Forecast prices, cap rates, and NOI.
    """)
    st.page_link("pages/4_🔮_Predictive_Modeling.py", label="View Forecasts →")

st.markdown("---")

st.subheader("📊 Key Economic Indicators")

@st.cache_data(ttl=STREAMLIT_CACHE_TTL)
def load_national_data():
    """Load national economic data from storage."""
    try:
        storage = DataStorage()
        return storage.load_dataframe("fred_national")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

national_df = load_national_data()

if not national_df.empty:
    m1, m2, m3, m4 = st.columns(4)

    metric_dates = []

    with m1:
        latest, prev = get_latest_valid(national_df, "treasury_10y")
        if latest is not None:
            delta = latest["treasury_10y"] - prev["treasury_10y"] if prev is not None else None
            st.metric(
                "10Y Treasury",
                f"{latest['treasury_10y']:.2f}%",
                f"{delta:+.2f}%" if delta else None,
                delta_color="inverse",
                help="10-Year Treasury yield - Key benchmark for CRE cap rates"
            )
            metric_dates.append(latest["date"])
        else:
            st.metric("10Y Treasury", "N/A")

    with m2:
        latest, prev = get_latest_valid(national_df, "mortgage_30y")
        if latest is not None:
            delta = latest["mortgage_30y"] - prev["mortgage_30y"] if prev is not None else None
            st.metric(
                "30Y Mortgage",
                f"{latest['mortgage_30y']:.2f}%",
                f"{delta:+.2f}%" if delta else None,
                delta_color="inverse",
                help="30-Year fixed mortgage rate - Impacts residential real estate"
            )
            metric_dates.append(latest["date"])
        else:
            st.metric("30Y Mortgage", "N/A")

    with m3:
        latest, prev = get_latest_valid(national_df, "unemployment_national")
        if latest is not None:
            delta = latest["unemployment_national"] - prev["unemployment_national"] if prev is not None else None
            st.metric(
                "Unemployment",
                f"{latest['unemployment_national']:.1f}%",
                f"{delta:+.1f}%" if delta else None,
                delta_color="inverse",
                help="National unemployment rate - Labor market health indicator"
            )
            metric_dates.append(latest["date"])
        else:
            st.metric("Unemployment", "N/A")

    with m4:
        latest, prev = get_latest_valid(national_df, "cre_delinquency")
        if latest is not None:
            delta = latest["cre_delinquency"] - prev["cre_delinquency"] if prev is not None else None
            st.metric(
                "CRE Delinquency",
                f"{latest['cre_delinquency']:.2f}%",
                f"{delta:+.2f}%" if delta else None,
                delta_color="inverse",
                help="Commercial real estate loan delinquency rate - Direct CRE stress indicator"
            )
            metric_dates.append(latest["date"])
        else:
            st.metric("CRE Delinquency", "N/A")

    if metric_dates:
        min_date = min(metric_dates)
        max_date = max(metric_dates)
        if min_date == max_date:
            st.caption(f"📅 Data as of {max_date.strftime('%B %d, %Y')}")
        else:
            st.caption(f"📅 Data from {min_date.strftime('%b %Y')} to {max_date.strftime('%b %Y')} (metrics update on different schedules)")
else:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("10Y Treasury", "N/A")
    m2.metric("30Y Mortgage", "N/A")
    m3.metric("Unemployment", "N/A")
    m4.metric("CRE Delinquency", "N/A")
    st.warning("⚠️ Economic data not loaded. Visit Predictive Modeling page to refresh data.")
