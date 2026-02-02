"""
3D Map View — Two tabs:
  Tab 1: Property Transactions (pydeck ColumnLayer from geocoded NYC sales)
  Tab 2: Overture + Transactions (building footprints matched to transactions)
"""


import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pydeck as pdk
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.config import CACHE_TTL_SECONDS, DEFAULT_BEARING, DEFAULT_PITCH

st.set_page_config(page_title="Map View", page_icon="🗺️", layout="wide")

st.title("🗺️ 3D Map View")
st.markdown("NYC commercial property transactions and building footprints.")
st.markdown("---")


# =============================================================================
# CONSTANTS
# =============================================================================

TYPE_COLORS = {
    "Office": [31, 119, 180],
    "Retail": [255, 127, 14],
    "Industrial": [44, 160, 44],
    "Hotel": [214, 39, 40],
    "Multifamily": [148, 103, 189],
    "Parking": [140, 140, 140],
    "Mixed Use": [227, 119, 194],
    "Other": [127, 127, 127],
}

# Unicode colored circles that actually render in Streamlit
TYPE_LEGEND = {
    "Office": "\U0001F535",       # blue circle
    "Retail": "\U0001F7E0",       # orange circle
    "Industrial": "\U0001F7E2",   # green circle
    "Hotel": "\U0001F534",        # red circle
    "Multifamily": "\U0001F7E3",  # purple circle
    "Parking": "\u26AA",          # white circle
    "Mixed Use": "\U0001F7E1",    # yellow circle (closest available)
    "Other": "\u2B1C",            # white square
}


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_transactions():
    """Load geocoded transactions from parquet cache. Returns None if not cached."""
    from data.geocoder import NYCGeocoder
    return NYCGeocoder().get_geocoded_transactions()


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_matched_buildings():
    """Load Overture matched buildings from parquet cache. Returns None if not cached."""
    from data.overture_fetcher import OvertureFetcher
    return OvertureFetcher().get_matched()


def find_column(df, possible_names):
    """Find a column matching any of the possible names (case-insensitive)."""
    df_cols_lower = {col.lower().strip(): col for col in df.columns}
    for name in possible_names:
        if name.lower() in df_cols_lower:
            return df_cols_lower[name.lower()]
    return None


def parse_uploaded_file(uploaded_file):
    """Parse uploaded CSV or Excel file into a DataFrame."""
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    return None


def render_legend(type_legend, cols=4):
    """Render property type legend using unicode emoji."""
    legend_cols = st.columns(cols)
    for i, (ptype, symbol) in enumerate(type_legend.items()):
        legend_cols[i % cols].markdown(f"{symbol} **{ptype}**")


def format_currency(val):
    """Format a number as currency string."""
    if pd.isna(val) or val == 0:
        return "—"
    if val >= 1e9:
        return f"${val / 1e9:.1f}B"
    if val >= 1e6:
        return f"${val / 1e6:.1f}M"
    if val >= 1e3:
        return f"${val / 1e3:.0f}K"
    return f"${val:,.0f}"


# =============================================================================
# SIDEBAR
# =============================================================================

selected_borough = "All"
selected_type = "All"
price_range = (0, 999_999_999)

with st.sidebar:
    st.markdown("### \u2699\ufe0f Data")

    # Load cached data
    transactions_df = load_transactions()
    buildings_gdf = load_matched_buildings()

    # Status
    if transactions_df is not None and not transactions_df.empty:
        geo_count = transactions_df["latitude"].notna().sum()
        st.caption(f"\U0001F4CD {int(geo_count):,} / {len(transactions_df):,} transactions geocoded")
    else:
        st.caption("\U0001F4CD No transaction data cached")

    if buildings_gdf is not None:
        matched_count = int(buildings_gdf["has_transaction"].sum()) if "has_transaction" in buildings_gdf.columns else 0
        st.caption(f"\U0001F3E2 {len(buildings_gdf):,} buildings ({matched_count:,} matched)")
    else:
        st.caption("\U0001F3E2 No building data cached")

    # Refresh buttons — separate so geocoding doesn't re-run every time
    r1, r2 = st.columns(2)
    with r1:
        if st.button("\U0001F4CD Transactions", use_container_width=True,
                     help="Re-geocode all property sales (~7 min)"):
            from data.geocoder import NYCGeocoder
            from data.nyc_opendata_fetcher import fetch_nyc_property_sales

            with st.spinner("Fetching sales..."):
                sales_df = fetch_nyc_property_sales(force_refresh=True)
            if not sales_df.empty:
                geocoder = NYCGeocoder()
                progress_bar = st.progress(0, text="Geocoding...")

                def update_progress(current, total):
                    progress_bar.progress(min(current / total, 1.0), text=f"Geocoding {current}/{total}")

                geocoder.geocode_transactions(sales_df, force_refresh=True, progress_callback=update_progress)
                progress_bar.empty()
            st.cache_data.clear()
            st.rerun()

    with r2:
        if st.button("\U0001F3D7\ufe0f Buildings", use_container_width=True,
                     help="Re-download Overture buildings (~1 min)"):
            txn_data = load_transactions()
            if txn_data is not None and not txn_data.empty:
                from data.overture_fetcher import OvertureFetcher

                with st.spinner("Downloading from Overture Maps..."):
                    OvertureFetcher().fetch_matched(txn_data, force_refresh=True)
                st.cache_data.clear()
                st.rerun()
            else:
                st.error("Geocode transactions first.")

    if transactions_df is None or transactions_df.empty:
        st.info("Click **Transactions** to geocode property sales.")

    st.markdown("---")

    st.markdown("### \U0001F4E4 Upload Custom Data")
    uploaded_file = st.file_uploader(
        "CSV/Excel with lat/lng",
        type=["csv", "xlsx", "xls"],
        help="File should have latitude and longitude columns",
    )

    st.markdown("---")

    st.markdown("### \U0001F39B\ufe0f View Controls")

    def _reset_view():
        st.session_state.pitch = DEFAULT_PITCH
        st.session_state.bearing = DEFAULT_BEARING

    if "pitch" not in st.session_state:
        st.session_state.pitch = DEFAULT_PITCH
    if "bearing" not in st.session_state:
        st.session_state.bearing = DEFAULT_BEARING

    pitch = st.slider("Pitch (tilt)", 0, 60, step=5, key="pitch")
    bearing = st.slider("Bearing (rotation)", -180, 180, step=5, key="bearing")

    st.button("Reset to Defaults", use_container_width=True, on_click=_reset_view)

    # Filters
    has_transactions = transactions_df is not None and not transactions_df.empty
    if has_transactions:
        st.markdown("---")
        st.markdown("### \U0001F50D Filters")

        if "borough" in transactions_df.columns:
            boroughs = ["All"] + sorted(transactions_df["borough"].dropna().unique().tolist())
            selected_borough = st.selectbox("Borough", boroughs)

        if "property_type" in transactions_df.columns:
            types = ["All"] + sorted(transactions_df["property_type"].dropna().unique().tolist())
            selected_type = st.selectbox("Property Type", types)

        if "sale_price" in transactions_df.columns:
            min_price = int(transactions_df["sale_price"].min())
            max_price = int(transactions_df["sale_price"].max())
            # Build price steps with formatted labels for comma grouping
            step = max(10_000, (max_price - min_price) // 200)
            price_options = list(range(min_price, max_price + 1, step))
            if price_options[-1] != max_price:
                price_options.append(max_price)
            price_labels = {v: f"${v:,}" for v in price_options}
            price_range = st.select_slider(
                "Sale Price Range",
                options=price_options,
                value=(price_options[0], price_options[-1]),
                format_func=lambda v: price_labels.get(v, f"${v:,}"),
            )


# =============================================================================
# FILTER TRANSACTIONS (shared by both tabs)
# =============================================================================

txn = pd.DataFrame()
if has_transactions:
    txn = transactions_df.dropna(subset=["latitude", "longitude"]).copy()

    if "borough" in txn.columns and selected_borough != "All":
        txn = txn[txn["borough"] == selected_borough]
    if "property_type" in txn.columns and selected_type != "All":
        txn = txn[txn["property_type"] == selected_type]
    if "sale_price" in txn.columns:
        txn = txn[
            (txn["sale_price"] >= price_range[0])
            & (txn["sale_price"] <= price_range[1])
        ]

if not txn.empty:
    txn["color"] = txn["property_type"].map(lambda t: TYPE_COLORS.get(t, [127, 127, 127]))
    max_height = 800
    txn["elevation"] = (
        txn["sale_price"]
        .clip(upper=txn["sale_price"].quantile(0.95))
        .pipe(lambda s: s / s.max() * max_height)
        .fillna(10)
    )

    # Pre-format values for pydeck tooltip (it can't format numbers)
    txn["price_fmt"] = txn["sale_price"].apply(lambda v: f"${v:,.0f}" if pd.notna(v) else "—")
    txn["ppsf_fmt"] = txn["price_per_sqft"].apply(lambda v: f"${v:,.0f}" if pd.notna(v) else "—")

upload_df = None
if uploaded_file:
    upload_df = parse_uploaded_file(uploaded_file)
    if upload_df is not None:
        lat_col = find_column(upload_df, ["lat", "latitude", "y"])
        lng_col = find_column(upload_df, ["lng", "lon", "long", "longitude", "x"])
        if lat_col and lng_col:
            upload_df = upload_df.dropna(subset=[lat_col, lng_col])
            upload_df = upload_df.rename(columns={lat_col: "latitude", lng_col: "longitude"})
        else:
            upload_df = None

if not txn.empty:
    center_lat, center_lng = txn["latitude"].mean(), txn["longitude"].mean()
else:
    center_lat, center_lng = 40.7128, -74.0060


# =============================================================================
# TABS
# =============================================================================

tab1, tab2 = st.tabs(["\U0001F4CD Property Transactions", "\U0001F3D7\ufe0f Overture + Transactions"])


# -------------------------------------------------------------------------
# TAB 1: TRANSACTION COLUMNS
# -------------------------------------------------------------------------
with tab1:
    st.markdown("#### NYC Commercial Property Sales")

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("\U0001F4CD Mapped", f"{len(txn):,}" if not txn.empty else "\u2014")
    if not txn.empty and "sale_price" in txn.columns:
        c2.metric("\U0001F4B0 Volume", format_currency(txn["sale_price"].sum()))
    else:
        c2.metric("\U0001F4B0 Volume", "\u2014")
    if not txn.empty and "price_per_sqft" in txn.columns:
        med = txn["price_per_sqft"].median()
        c3.metric("\U0001F4D0 Median $/SF", f"${med:,.0f}" if pd.notna(med) else "\u2014")
    else:
        c3.metric("\U0001F4D0 Median $/SF", "\u2014")
    c4.metric("\U0001F4E4 Uploaded", f"{len(upload_df):,}" if upload_df is not None else "\u2014")

    layers_t1 = []
    if not txn.empty:
        layers_t1.append(pdk.Layer(
            "ColumnLayer", data=txn,
            get_position=["longitude", "latitude"],
            get_elevation="elevation", elevation_scale=1, radius=30,
            get_fill_color="color", pickable=True, auto_highlight=True,
        ))
    if upload_df is not None:
        layers_t1.append(pdk.Layer(
            "ScatterplotLayer", data=upload_df,
            get_position=["longitude", "latitude"],
            get_radius=50, get_fill_color=[220, 50, 50, 200], pickable=True,
        ))

    deck1 = pdk.Deck(
        layers=layers_t1,
        initial_view_state=pdk.ViewState(
            latitude=center_lat, longitude=center_lng,
            zoom=11, pitch=pitch, bearing=bearing,
        ),
        map_style="dark",
        tooltip={"text": "{address}\n{borough} \u2014 {property_type}\nPrice: {price_fmt}\n$/SF: {ppsf_fmt}"},
    )
    st.pydeck_chart(deck1, use_container_width=True, height=600)

    if not txn.empty and "property_type" in txn.columns:
        active_types = txn["property_type"].dropna().unique()
        active_legend = {k: v for k, v in TYPE_LEGEND.items() if k in active_types}
        if active_legend:
            render_legend(active_legend, cols=min(len(active_legend), 4))
    else:
        render_legend(TYPE_LEGEND)
    st.caption("Column height = sale price (capped at 95th percentile)")

    # Data table
    if not txn.empty:
        with st.expander("\U0001F4CA Transaction Data", expanded=False):
            table_df = txn.sort_values("sale_price", ascending=False).copy()

            # Pre-format numeric columns as strings (Streamlit sprintf doesn't support comma grouping)
            if "sale_price" in table_df.columns:
                table_df["Sale Price"] = table_df["sale_price"].apply(lambda v: f"${v:,.0f}" if pd.notna(v) else "")
            if "price_per_sqft" in table_df.columns:
                table_df["$/SF"] = table_df["price_per_sqft"].apply(lambda v: f"${v:,.0f}" if pd.notna(v) else "")
            if "gross_square_feet" in table_df.columns:
                table_df["Gross SF"] = table_df["gross_square_feet"].apply(lambda v: f"{v:,.0f}" if pd.notna(v) else "")

            display_cols = [
                c for c in [
                    "address", "borough", "neighborhood", "property_type",
                    "Sale Price", "$/SF", "Gross SF", "sale_date",
                ] if c in table_df.columns
            ]
            table_df = table_df[display_cols]

            col_config = {
                "address": st.column_config.TextColumn("Address", width="large"),
                "borough": st.column_config.TextColumn("Borough"),
                "neighborhood": st.column_config.TextColumn("Neighborhood"),
                "property_type": st.column_config.TextColumn("Type"),
                "sale_date": st.column_config.DateColumn("Sale Date", format="MM/DD/YYYY"),
            }

            st.dataframe(table_df, column_config=col_config, use_container_width=True, hide_index=True)


# -------------------------------------------------------------------------
# TAB 2: OVERTURE + TRANSACTIONS (case study)
# -------------------------------------------------------------------------
with tab2:
    st.markdown("#### Building Footprints + Transaction Matching")

    if buildings_gdf is None:
        st.info("No Overture building data cached. Click **Buildings** in the sidebar "
                "to download building footprints matched to transaction locations.")
    else:
        # Metrics — compute from transaction side (how many txns land in a building)
        matched_buildings = int(buildings_gdf["has_transaction"].sum()) if "has_transaction" in buildings_gdf.columns else 0
        total_txn = len(txn) if not txn.empty else 0

        # Compute transaction-side match via spatial join with 5m buffer
        matched_txn_count = 0
        if not txn.empty and not buildings_gdf.empty:
            from shapely.geometry import Point as _Pt
            _pts = gpd.GeoDataFrame(
                txn[["latitude", "longitude"]],
                geometry=[_Pt(xy) for xy in zip(txn["longitude"], txn["latitude"])],
                crs="EPSG:4326",
            )
            _bld_m = buildings_gdf.to_crs(epsg=32618).copy()
            _bld_m["geometry"] = _bld_m.geometry.buffer(5)
            _bld_buffered = _bld_m.to_crs("EPSG:4326")
            _tj = gpd.sjoin(_pts, _bld_buffered[["geometry"]], how="left", predicate="within")
            matched_txn_count = len(_tj.dropna(subset=["index_right"]).index.unique())

        unmatched_txn = max(0, total_txn - matched_txn_count)
        match_pct = (matched_txn_count / total_txn * 100) if total_txn > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("\U0001F3E2 Buildings", f"{len(buildings_gdf):,}")
        c2.metric("\u2705 Matched Txns", f"{matched_txn_count:,}")
        c3.metric("\U0001F4CA Match Rate", f"{match_pct:.1f}%")
        c4.metric("\u274C Unmatched Txns", f"{unmatched_txn:,}")

        # Build GeoJSON for buildings
        features = []
        for _, row in buildings_gdf.iterrows():
            has_txn = bool(row.get("has_transaction", False))
            name = row.get("name", "") or ""
            subtype = row.get("subtype", "") or ""
            height = row.get("height", 10) or 10
            feat = {
                "type": "Feature",
                "geometry": row.geometry.__geo_interface__,
                "properties": {
                    "name": name if name else "Unknown",
                    "subtype": subtype.title() if subtype else "\u2014",
                    "height": round(float(height), 1),
                    "has_transaction": "Yes" if has_txn else "No",
                    "fill_color": [46, 204, 113, 160] if has_txn else [100, 120, 140, 60],
                    "line_color": [46, 204, 113, 220] if has_txn else [140, 170, 200, 100],
                },
            }
            features.append(feat)

        buildings_geojson = {"type": "FeatureCollection", "features": features}

        layers_t2 = [
            pdk.Layer(
                "GeoJsonLayer", data=buildings_geojson,
                stroked=True, filled=True, extruded=True, opacity=0.7,
                get_elevation="properties.height",
                elevation_scale=1,
                get_fill_color="properties.fill_color",
                get_line_color="properties.line_color",
                line_width_min_pixels=1, pickable=True,
            )
        ]

        if not txn.empty:
            layers_t2.append(pdk.Layer(
                "ScatterplotLayer", data=txn,
                get_position=["longitude", "latitude"],
                get_radius=15, get_fill_color="color",
                pickable=True, opacity=0.9,
            ))

        deck2 = pdk.Deck(
            layers=layers_t2,
            initial_view_state=pdk.ViewState(
                latitude=center_lat, longitude=center_lng,
                zoom=13, pitch=pitch, bearing=bearing,
            ),
            map_style="dark",
            tooltip={"text": "{name}\n{subtype}\nHeight: {height}m\nMatched: {has_transaction}"},
        )
        st.pydeck_chart(deck2, use_container_width=True, height=600)

        # Legend
        leg1, leg2, leg3 = st.columns(3)
        leg1.markdown("\U0001F7E2 **Matched building**")
        leg2.markdown("\u26AA **Nearby building**")
        leg3.markdown("\U0001F534 **Transaction point**")
        st.caption(
            f"{len(buildings_gdf):,} buildings within 50m of a transaction. "
            "Match = transaction point within building footprint (5m buffer for geocoding offset)."
        )

        # ---- Matching process breakdown ----
        st.markdown("---")

        total_txn_all = len(transactions_df) if transactions_df is not None else 0
        geocoded_count = int(transactions_df["latitude"].notna().sum()) if transactions_df is not None else 0
        geocode_pct = (geocoded_count / total_txn_all * 100) if total_txn_all > 0 else 0

        with st.expander("\U0001F50D Matching Process & Code", expanded=False):
            st.markdown("#### Step 1 \u2014 Source Data")
            st.markdown(
                f"- **Transactions**: {total_txn_all:,} NYC commercial property sales from NYC OpenData (Socrata API)\n"
                "- **Buildings**: 1.67M footprints from Overture Maps Foundation (S3 GeoParquet)"
            )

            st.markdown("#### Step 2 \u2014 Geocoding")
            st.markdown(
                "Each transaction address geocoded via NYC GeoSearch (Pelias). "
                f"Result: {geocoded_count:,} / {total_txn_all:,} resolved ({geocode_pct:.1f}%)"
            )
            st.code("""
import requests

def geocode(address, borough, zipcode):
    url = "https://geosearch.planninglabs.nyc/v2/search"
    resp = requests.get(url, params={"text": f"{address}, {borough}, NY {zipcode}"})
    coords = resp.json()["features"][0]["geometry"]["coordinates"]
    return coords[1], coords[0]  # lat, lng
""".strip(), language="python")

            st.markdown("#### Step 3 \u2014 Proximity Filter")
            st.markdown(
                "All 1.67M Overture buildings projected to UTM Zone 18N (meters). "
                f"50m buffer around each geocoded transaction point. "
                f"Buildings intersecting the buffer kept \u2192 {len(buildings_gdf):,} buildings."
            )
            st.code("""
import overturemaps, geopandas as gpd

# Stream all NYC buildings from Overture S3
reader = overturemaps.record_batch_reader("building", bbox=NYC_BBOX)
gdf = gpd.GeoDataFrame.from_arrow(reader.read_all()).set_crs("EPSG:4326")

# Project to meters, buffer transactions, filter buildings
gdf_m = gdf.to_crs(epsg=32618)
points_m = txn_points.to_crs(epsg=32618)
hull = points_m.geometry.union_all().buffer(50)
nearby = gdf_m[gdf_m.geometry.intersects(hull)].to_crs("EPSG:4326")
""".strip(), language="python")

            st.markdown("#### Step 4 \u2014 Spatial Join (5m buffer)")
            st.markdown(
                "Building footprints buffered 5m to account for geocoding offset. "
                f"Result: {matched_txn_count:,} / {total_txn:,} transactions matched ({match_pct:.1f}%)."
            )
            st.code("""
# Buffer building footprints 5m for geocoding tolerance
nearby_m = nearby.to_crs(epsg=32618)
nearby_m["geometry"] = nearby_m.geometry.buffer(5)
nearby_buffered = nearby_m.to_crs("EPSG:4326")

# Spatial join: which transactions fall within a building?
joined = gpd.sjoin(
    txn_points, nearby_buffered[["geometry"]],
    how="left", predicate="within"
)
matched = joined.dropna(subset=["index_right"]).index.unique()
""".strip(), language="python")

            st.markdown(f"**Unmatched transactions** ({unmatched_txn:,}) are typically:")
            st.markdown(
                "- Parking lots or vacant land with no building footprint\n"
                "- Geocoding placed point on adjacent parcel or intersection\n"
                "- Property sold as land/air rights without a physical structure"
            )

        with st.expander("\U0001F4CB Data Sources", expanded=False):
            st.markdown(
                "| Source | Provider | Access |\n"
                "|--------|----------|--------|\n"
                "| Building footprints | [Overture Maps Foundation](https://overturemaps.org/) | S3 GeoParquet, open |\n"
                "| Property transactions | [NYC OpenData](https://data.cityofnewyork.us/) | Socrata API, open |\n"
                "| Geocoding | [NYC GeoSearch](https://geosearch.planninglabs.nyc/) | Pelias API, free, no key |"
            )
