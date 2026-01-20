import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.storage import DataStorage

st.set_page_config(
    page_title="Commercial Real Estate Dashboard",
    page_icon="🏢",
    layout="wide",
)

# Hide default Streamlit navigation
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar
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

# Main content
st.title("🏢 Commercial Real Estate Dashboard")
st.markdown("Welcome back! Here's your overview.")

st.markdown("---")

# Navigation cards
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

# Real economic metrics
st.subheader("📊 Key Economic Indicators")

# Load national economic data
@st.cache_data(ttl=3600)
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
    # Helper function to get latest valid value for a column
    def get_latest_valid(df, column):
        """Get the most recent non-null value for a column."""
        valid_data = df[df[column].notna()].sort_values("date")
        if not valid_data.empty:
            latest_row = valid_data.iloc[-1]
            # Get previous value for delta (1 month before)
            prev_data = valid_data[valid_data["date"] <= (latest_row["date"] - pd.Timedelta(days=25))]
            prev_row = prev_data.iloc[-1] if not prev_data.empty else None
            return latest_row, prev_row
        return None, None

    m1, m2, m3, m4 = st.columns(4)

    # Track dates for caption
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

    # Show date range in caption
    if metric_dates:
        min_date = min(metric_dates)
        max_date = max(metric_dates)
        if min_date == max_date:
            st.caption(f"📅 Data as of {max_date.strftime('%B %d, %Y')}")
        else:
            st.caption(f"📅 Data from {min_date.strftime('%b %Y')} to {max_date.strftime('%b %Y')} (metrics update on different schedules)")
else:
    # Fallback if no data available
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("10Y Treasury", "N/A")
    m2.metric("30Y Mortgage", "N/A")
    m3.metric("Unemployment", "N/A")
    m4.metric("CRE Delinquency", "N/A")
    st.warning("⚠️ Economic data not loaded. Visit Predictive Modeling page to refresh data.")
