import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

st.set_page_config(page_title="Market Analytics", page_icon="📊", layout="wide")

st.title("📊 Market Analytics")
st.markdown("Commercial real estate transaction trends and market insights.")
st.markdown("---")

# =============================================================================
# SAMPLE DATA - Replace with your actual data source (Snowflake, API, etc.)
# =============================================================================
@st.cache_data
def load_sample_transactions():
    """
    Load transaction data. Replace this with your actual data source.
    Expected columns: date, property_name, property_type, market, 
                      price, sqft, cap_rate, noi, price_per_sqft
    """
    np.random.seed(42)
    n = 150
    
    property_types = ['Office', 'Retail', 'Industrial', 'Multifamily', 'Mixed Use']
    markets = ['Manhattan', 'Brooklyn', 'Los Angeles', 'Chicago', 'Miami', 'Austin', 'Seattle']
    
    dates = pd.date_range(end=datetime.now(), periods=n, freq='W')
    
    data = {
        'date': dates,
        'property_name': [f"Property {i+1}" for i in range(n)],
        'property_type': np.random.choice(property_types, n),
        'market': np.random.choice(markets, n),
        'price': np.random.uniform(5_000_000, 150_000_000, n),
        'sqft': np.random.uniform(20_000, 500_000, n),
        'cap_rate': np.random.uniform(4.5, 8.5, n),
        'noi': np.random.uniform(500_000, 10_000_000, n),
    }
    
    df = pd.DataFrame(data)
    df['price_per_sqft'] = df['price'] / df['sqft']
    return df

df = load_sample_transactions()

# =============================================================================
# SIDEBAR FILTERS
# =============================================================================
with st.sidebar:
    st.markdown("### 🔍 Filters")
    
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
        options=df['property_type'].unique(),
        default=df['property_type'].unique()
    )
    
    # Market filter
    markets = st.multiselect(
        "Market",
        options=df['market'].unique(),
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
    st.markdown("### 📤 Upload Data")
    uploaded_file = st.file_uploader(
        "Upload transaction data",
        type=["csv", "xlsx"],
        help="Replace sample data with your own"
    )
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, parse_dates=['date'])
            else:
                df = pd.read_excel(uploaded_file, parse_dates=['date'])
            st.success(f"✅ Loaded {len(df)} transactions")
        except Exception as e:
            st.error(f"Error loading file: {e}")

# Apply filters
filtered_df = df[
    (df['date'].dt.date >= date_range[0]) &
    (df['date'].dt.date <= date_range[1]) &
    (df['property_type'].isin(property_types)) &
    (df['market'].isin(markets)) &
    (df['price'] >= price_min * 1_000_000) &
    (df['price'] <= price_max * 1_000_000)
]

# =============================================================================
# KEY METRICS
# =============================================================================
st.markdown("### 📈 Key Metrics")

m1, m2, m3, m4, m5 = st.columns(5)

total_volume = filtered_df['price'].sum()
avg_cap_rate = filtered_df['cap_rate'].mean()
avg_price_psf = filtered_df['price_per_sqft'].mean()
total_transactions = len(filtered_df)
avg_deal_size = filtered_df['price'].mean()

m1.metric("Total Volume", f"${total_volume/1e9:.2f}B")
m2.metric("Transactions", f"{total_transactions:,}")
m3.metric("Avg Deal Size", f"${avg_deal_size/1e6:.1f}M")
m4.metric("Avg Cap Rate", f"{avg_cap_rate:.2f}%")
m5.metric("Avg $/SF", f"${avg_price_psf:.0f}")

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

# =============================================================================
# CAP RATE ANALYSIS
# =============================================================================
st.markdown("### 📉 Cap Rate Analysis")

col1, col2 = st.columns(2)

with col1:
    # Cap rate by property type
    cap_by_type = filtered_df.groupby('property_type')['cap_rate'].mean().reset_index()
    cap_by_type = cap_by_type.sort_values('cap_rate', ascending=True)
    
    fig_cap = px.bar(
        cap_by_type,
        x='cap_rate',
        y='property_type',
        orientation='h',
        title='Average Cap Rate by Property Type',
        labels={'cap_rate': 'Cap Rate (%)', 'property_type': 'Property Type'}
    )
    fig_cap.update_layout(xaxis_ticksuffix='%')
    st.plotly_chart(fig_cap, use_container_width=True)

with col2:
    # Cap rate trend over time
    monthly_cap = monthly_df.groupby('month')['cap_rate'].mean().reset_index()
    
    fig_cap_trend = px.line(
        monthly_cap,
        x='month',
        y='cap_rate',
        title='Cap Rate Trend',
        labels={'month': 'Month', 'cap_rate': 'Avg Cap Rate (%)'},
        markers=True
    )
    fig_cap_trend.update_layout(yaxis_ticksuffix='%')
    st.plotly_chart(fig_cap_trend, use_container_width=True)

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
    # Scatter: Price vs Cap Rate
    fig_scatter = px.scatter(
        filtered_df,
        x='cap_rate',
        y='price_per_sqft',
        color='property_type',
        size='price',
        hover_data=['property_name', 'market'],
        title='Cap Rate vs $/SF by Property Type',
        labels={'cap_rate': 'Cap Rate (%)', 'price_per_sqft': '$/SF'}
    )
    fig_scatter.update_layout(
        xaxis_ticksuffix='%',
        yaxis_tickprefix='$'
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# =============================================================================
# TRANSACTION TABLE
# =============================================================================
st.markdown("### 📋 Recent Transactions")

# Format for display
display_df = filtered_df.copy()
display_df = display_df.sort_values('date', ascending=False)
display_df['price_display'] = display_df['price'].apply(lambda x: f"${x/1e6:.1f}M")
display_df['sqft_display'] = display_df['sqft'].apply(lambda x: f"{x:,.0f}")
display_df['cap_rate_display'] = display_df['cap_rate'].apply(lambda x: f"{x:.2f}%")
display_df['psf_display'] = display_df['price_per_sqft'].apply(lambda x: f"${x:.0f}")
display_df['date_display'] = display_df['date'].dt.strftime('%Y-%m-%d')

st.dataframe(
    display_df[['date_display', 'property_name', 'property_type', 'market', 
                'price_display', 'sqft_display', 'cap_rate_display', 'psf_display']].rename(columns={
        'date_display': 'Date',
        'property_name': 'Property',
        'property_type': 'Type',
        'market': 'Market',
        'price_display': 'Price',
        'sqft_display': 'Sq Ft',
        'cap_rate_display': 'Cap Rate',
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
