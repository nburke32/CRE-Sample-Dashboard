import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Map View", page_icon="🗺️", layout="wide")

st.title("🗺️ Map View")
st.markdown("Explore commercial properties on an interactive map.")
st.markdown("---")


def find_column(df, possible_names):
    """Find a column matching any of the possible names (case-insensitive)."""
    df_cols_lower = {col.lower().strip(): col for col in df.columns}
    for name in possible_names:
        if name.lower() in df_cols_lower:
            return df_cols_lower[name.lower()]
    return None


def parse_uploaded_file(uploaded_file):
    """Parse uploaded CSV or Excel file into a DataFrame."""
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(('.xlsx', '.xls')):
        return pd.read_excel(uploaded_file)
    return None


# Sidebar - File upload
with st.sidebar:
    st.markdown("### 📤 Upload Data")
    uploaded_file = st.file_uploader(
        "Upload spreadsheet with coordinates",
        type=["csv", "xlsx", "xls"],
        help="File should contain latitude and longitude columns"
    )
    
    st.markdown("---")
    st.markdown("### 📋 Expected Columns")
    st.markdown("""
    **Latitude:** `lat`, `latitude`, `y`  
    **Longitude:** `lng`, `lon`, `long`, `longitude`, `x`
    
    *Optional:* `name`, `address`, `price`, `type`
    """)

# Initialize data
df = None
lat_col = None
lng_col = None

if uploaded_file:
    df = parse_uploaded_file(uploaded_file)
    
    if df is not None:
        # Find lat/lng columns
        lat_col = find_column(df, ['lat', 'latitude', 'y'])
        lng_col = find_column(df, ['lng', 'lon', 'long', 'longitude', 'x'])
        
        if lat_col and lng_col:
            # Clean data - drop rows with missing coordinates
            df = df.dropna(subset=[lat_col, lng_col])
            
            with st.sidebar:
                st.success(f"✅ Loaded {len(df)} locations")
                st.markdown(f"**Lat column:** `{lat_col}`")
                st.markdown(f"**Lng column:** `{lng_col}`")
        else:
            missing = []
            if not lat_col:
                missing.append("latitude")
            if not lng_col:
                missing.append("longitude")
            st.sidebar.error(f"❌ Could not find {' or '.join(missing)} column(s)")
            df = None

# Determine map center and zoom
if df is not None and lat_col and lng_col:
    center_lat = df[lat_col].mean()
    center_lng = df[lng_col].mean()
    zoom = 10
else:
    # Default to NYC
    center_lat = 40.7128
    center_lng = -74.0060
    zoom = 12

# Create map
m = folium.Map(location=[center_lat, center_lng], zoom_start=zoom)

# Add markers if data is loaded
if df is not None and lat_col and lng_col:
    # Find optional columns for popups
    name_col = find_column(df, ['name', 'property', 'title', 'address'])
    price_col = find_column(df, ['price', 'value', 'amount'])
    type_col = find_column(df, ['type', 'category', 'property_type'])
    
    for idx, row in df.iterrows():
        # Build popup content
        popup_parts = []
        if name_col and pd.notna(row.get(name_col)):
            popup_parts.append(f"<b>{row[name_col]}</b>")
        if price_col and pd.notna(row.get(price_col)):
            popup_parts.append(f"Price: {row[price_col]}")
        if type_col and pd.notna(row.get(type_col)):
            popup_parts.append(f"Type: {row[type_col]}")
        
        popup_html = "<br>".join(popup_parts) if popup_parts else f"Location {idx + 1}"
        
        # Tooltip (hover text)
        tooltip = row[name_col] if name_col and pd.notna(row.get(name_col)) else f"Location {idx + 1}"
        
        folium.Marker(
            location=[row[lat_col], row[lng_col]],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=tooltip,
            icon=folium.Icon(color='blue', icon='building', prefix='fa')
        ).add_to(m)

# Display map
st_folium(m, use_container_width=True, height=600)

# Show data table below map
if df is not None:
    st.markdown("---")
    st.markdown("### 📊 Uploaded Data")
    st.dataframe(df, use_container_width=True)

