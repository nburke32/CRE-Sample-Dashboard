import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.nyc_opendata_fetcher import fetch_nyc_property_sales
from data.storage import DataStorage

st.set_page_config(page_title="Market Analytics", page_icon="📊", layout="wide")

st.title("📊 Market Analytics")
st.markdown("Commercial real estate transaction trends and market insights from NYC property sales data.")
st.markdown("---")

# =============================================================================
# REAL NYC PROPERTY DATA
# =============================================================================
@st.cache_data(ttl=86400)  # 24 hours
def load_nyc_transactions(force_refresh: bool = False):
    """
    Load NYC commercial property sales data.
    Data source: NYC OpenData (Citywide Rolling Calendar Sales)

    Returns DataFrame with columns:
    - sale_date: Transaction date
    - borough: NYC borough (Manhattan, Brooklyn, Queens, Bronx, Staten Island)
    - property_type: Office, Retail, Industrial, etc.
    - sale_price: Transaction price
    - gross_square_feet: Building square footage
    - price_per_sqft: Calculated price per square foot
    - address: Property address
    - neighborhood: NYC neighborhood
    """
    df = fetch_nyc_property_sales(force_refresh=force_refresh)

    if df.empty:
        st.warning("⚠️ No NYC property data available. Click 'Refresh Data' in the sidebar.")
        return pd.DataFrame()

    # Rename columns to match expected format
    df = df.rename(columns={
        'sale_date': 'date',
        'borough': 'market',
        'sale_price': 'price',
        'gross_square_feet': 'sqft'
    })

    # Add property_name from address
    if 'address' in df.columns:
        df['property_name'] = df['address'].fillna('Unknown Address')
    else:
        df['property_name'] = 'NYC Commercial Property'

    return df

df = load_nyc_transactions()

# =============================================================================
# SIDEBAR FILTERS
# =============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    # Data refresh button
    if st.button("🔄 Refresh NYC Data", use_container_width=True):
        st.cache_data.clear()
        df = load_nyc_transactions(force_refresh=True)
        st.success("✅ Data refreshed!")
        st.rerun()

    # Data status
    storage = DataStorage()
    if storage.dataset_exists("nyc_property_sales"):
        last_updated = storage.get_last_updated("nyc_property_sales")
        if last_updated:
            st.caption(f"📅 Last updated: {last_updated.strftime('%m/%d/%Y %H:%M')}")

    st.markdown("---")
    st.markdown("### 🔍 Filters")

# Only show filters if data is available
if not df.empty:
    with st.sidebar:
        # Date range
        date_range = st.date_input(
            "Date Range",
            value=(df['date'].min(), df['date'].max()),
            min_value=df['date'].min(),
            max_value=df['date'].max()
        )

        # Property type filter
        property_types = st.multiselect(
            "Property Type",
            options=sorted(df['property_type'].unique()),
            default=df['property_type'].unique()
        )

        # Market filter (Borough)
        markets = st.multiselect(
            "Borough",
            options=sorted(df['market'].unique()),
            default=df['market'].unique()
        )

        # Price range
        price_min, price_max = st.slider(
            "Price Range ($M)",
            min_value=0,
            max_value=int(df['price'].max() / 1_000_000) + 10,
            value=(0, int(df['price'].max() / 1_000_000) + 10)
        )

        st.markdown("---")
        st.markdown("### 📊 Data Source")
        st.caption("**NYC OpenData**")
        st.caption(f"{len(df):,} transactions")
        st.caption("Last 24 months")
        st.caption("[View Dataset →](https://data.cityofnewyork.us/dataset/NYC-Citywide-Rolling-Calendar-Sales/usep-8jbt)")

# Apply filters only if data exists
if not df.empty:
    filtered_df = df[
        (df['date'].dt.date >= date_range[0]) &
        (df['date'].dt.date <= date_range[1]) &
        (df['property_type'].isin(property_types)) &
        (df['market'].isin(markets)) &
        (df['price'] >= price_min * 1_000_000) &
        (df['price'] <= price_max * 1_000_000)
    ]
else:
    filtered_df = df
    st.stop()

# =============================================================================
# KEY METRICS
# =============================================================================
st.markdown("### 📈 Key Metrics")

m1, m2, m3, m4 = st.columns(4)

total_volume = filtered_df['price'].sum()
total_transactions = len(filtered_df)
avg_deal_size = filtered_df['price'].mean()
avg_price_psf = filtered_df['price_per_sqft'].mean() if 'price_per_sqft' in filtered_df.columns else 0

m1.metric("Total Volume", f"${total_volume/1e9:.2f}B")
m2.metric("Transactions", f"{total_transactions:,}")
m3.metric("Avg Deal Size", f"${avg_deal_size/1e6:.1f}M")
m4.metric("Avg $/SF", f"${avg_price_psf:.0f}" if avg_price_psf > 0 else "N/A")

st.markdown("---")

# =============================================================================
# TRANSACTION VOLUME OVER TIME
# =============================================================================
st.markdown("### 📊 Transaction Volume Over Time")

# Aggregate by month
monthly_df = filtered_df.copy()
monthly_df['month'] = monthly_df['date'].dt.to_period('M').dt.to_timestamp()
monthly_agg = monthly_df.groupby('month').agg({
    'price': 'sum',
    'property_name': 'count'
}).reset_index()
monthly_agg.columns = ['month', 'volume', 'count']

col1, col2 = st.columns(2)

with col1:
    fig_volume = px.bar(
        monthly_agg,
        x='month',
        y='volume',
        title='Monthly Transaction Volume',
        labels={'month': 'Month', 'volume': 'Volume ($)'}
    )
    fig_volume.update_layout(yaxis_tickformat='$,.0f')
    st.plotly_chart(fig_volume, use_container_width=True)

with col2:
    fig_count = px.line(
        monthly_agg,
        x='month',
        y='count',
        title='Monthly Transaction Count',
        labels={'month': 'Month', 'count': 'Transactions'},
        markers=True
    )
    st.plotly_chart(fig_count, use_container_width=True)

st.markdown("---")

# =============================================================================
# MARKET BREAKDOWN
# =============================================================================
st.markdown("### 🏙️ Market Breakdown")

col1, col2 = st.columns(2)

with col1:
    market_volume = filtered_df.groupby('market')['price'].sum().reset_index()
    market_volume.columns = ['market', 'volume']
    
    fig_market = px.pie(
        market_volume,
        values='volume',
        names='market',
        title='Volume by Market',
        hole=0.4
    )
    st.plotly_chart(fig_market, use_container_width=True)

with col2:
    type_volume = filtered_df.groupby('property_type')['price'].sum().reset_index()
    type_volume.columns = ['property_type', 'volume']
    
    fig_type = px.pie(
        type_volume,
        values='volume',
        names='property_type',
        title='Volume by Property Type',
        hole=0.4
    )
    st.plotly_chart(fig_type, use_container_width=True)

st.markdown("---")

# Note: Cap rate data not available in NYC OpenData
# We'd need NOI (Net Operating Income) to calculate cap rates
# This could be added in the future with supplementary data sources

st.markdown("---")

# =============================================================================
# PRICE PER SQUARE FOOT ANALYSIS
# =============================================================================
st.markdown("### 💰 Price Per Square Foot")

col1, col2 = st.columns(2)

with col1:
    # Price/SF by market
    psf_by_market = filtered_df.groupby('market')['price_per_sqft'].mean().reset_index()
    psf_by_market = psf_by_market.sort_values('price_per_sqft', ascending=True)
    
    fig_psf = px.bar(
        psf_by_market,
        x='price_per_sqft',
        y='market',
        orientation='h',
        title='Average $/SF by Market',
        labels={'price_per_sqft': '$/SF', 'market': 'Market'}
    )
    fig_psf.update_layout(xaxis_tickprefix='$')
    st.plotly_chart(fig_psf, use_container_width=True)

with col2:
    # Scatter: Price vs $/SF (filter out NaN sqft for visualization)
    scatter_df = filtered_df.dropna(subset=['sqft', 'price_per_sqft'])

    if not scatter_df.empty:
        fig_scatter = px.scatter(
            scatter_df,
            x='price',
            y='price_per_sqft',
            color='property_type',
            size='sqft',
            hover_data=['property_name', 'market'],
            title='Sale Price vs $/SF by Property Type',
            labels={'price': 'Sale Price', 'price_per_sqft': '$/SF', 'sqft': 'Square Feet'}
        )
        fig_scatter.update_layout(
            xaxis_tickprefix='$',
            yaxis_tickprefix='$'
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("No properties with square footage data available for visualization")

st.markdown("---")

# =============================================================================
# TRANSACTION TABLE
# =============================================================================
st.markdown("### 📋 Recent Transactions")

# Format for display
display_df = filtered_df.copy()
display_df = display_df.sort_values('date', ascending=False)
display_df['price_display'] = display_df['price'].apply(lambda x: f"${x/1e6:.1f}M")
display_df['sqft_display'] = display_df['sqft'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")
display_df['psf_display'] = display_df['price_per_sqft'].apply(lambda x: f"${x:.0f}" if pd.notna(x) else "N/A")
display_df['date_display'] = display_df['date'].dt.strftime('%Y-%m-%d')

st.dataframe(
    display_df[['date_display', 'property_name', 'property_type', 'market',
                'price_display', 'sqft_display', 'psf_display']].rename(columns={
        'date_display': 'Date',
        'property_name': 'Property',
        'property_type': 'Type',
        'market': 'Borough',
        'price_display': 'Price',
        'sqft_display': 'Sq Ft',
        'psf_display': '$/SF'
    }),
    use_container_width=True,
    hide_index=True
)

# =============================================================================
# EXPORT
# =============================================================================
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Export to CSV",
        data=csv,
        file_name=f"transactions_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

with col2:
    st.markdown(f"**{len(filtered_df)}** transactions shown")

# =============================================================================
# DATA SOURCING NOTE
# =============================================================================
st.markdown("---")
with st.expander("ℹ️ About Data Sources"):
    st.markdown("""
    ### NYC Open Data Approach

    This dashboard currently uses **NYC OpenData** (via the Socrata API) to provide real commercial property
    transaction data for New York City. This approach was chosen because:

    - **Free & Accessible**: NYC OpenData provides comprehensive property sales records at no cost
    - **Official Source**: Data comes directly from the NYC Department of Finance
    - **Real-Time**: Updates are published regularly as transactions are recorded
    - **Detailed**: Includes sale prices, square footage, property types, and locations

    ### Expanding to National Coverage

    To provide **nationwide commercial real estate transaction data**, the dashboard would need to integrate
    with premium data providers such as:

    **Real Capital Analytics (RCA)**
    - Comprehensive commercial property transaction database
    - Covers $2.5M+ transactions across all major property types
    - Provides cap rates, NOI, and detailed market analytics
    - API access available with enterprise subscription

    **Green Street Advisors**
    - Institutional-grade CRE data and analytics
    - Property valuations and market fundamentals
    - REIT analysis and pricing data
    - Subscription-based data feeds

    **CoStar / LoopNet**
    - Largest commercial real estate database
    - Property listings, sales comps, and market data
    - API access with premium subscription

    **Implementation Note**: Integrating these sources would require:
    1. Enterprise subscriptions and API credentials
    2. Additional data fetcher modules (similar to `nyc_opendata_fetcher.py`)
    3. Data normalization across multiple sources
    4. Enhanced caching and storage infrastructure

    For demonstration and development purposes, NYC OpenData provides an excellent foundation
    that showcases the dashboard's analytical capabilities with real transaction data.
    """)

