import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.config import METROS, REIT_TICKERS
from data.storage import DataStorage
from data.fred_fetcher import FREDFetcher
from data.yfinance_fetcher import YFinanceFetcher

st.set_page_config(page_title="Predictive Modeling", page_icon="🔮", layout="wide")

st.title("🔮 Predictive Modeling")
st.markdown("Forecast CRE market indicators using real economic data from FRED and market sentiment from REITs.")

# =============================================================================
# DATA LOADING & CACHING
# =============================================================================

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_fred_data(force_refresh: bool = False):
    """Load FRED data (national + metros)."""
    try:
        fetcher = FREDFetcher()
        return fetcher.fetch_all(force_refresh=force_refresh)
    except Exception as e:
        st.error(f"Error loading FRED data: {e}")
        return {"national": pd.DataFrame(), "metros": pd.DataFrame()}

@st.cache_data(ttl=3600)
def load_reit_data(force_refresh: bool = False):
    """Load REIT price data."""
    try:
        fetcher = YFinanceFetcher()
        return fetcher.fetch_reit_prices(force_refresh=force_refresh)
    except Exception as e:
        st.error(f"Error loading REIT data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_reit_sentiment():
    """Get current REIT sector sentiment."""
    try:
        fetcher = YFinanceFetcher()
        return fetcher.get_current_sentiment()
    except Exception:
        return pd.DataFrame()

@st.cache_data
def run_prophet_forecast(_metro_df: pd.DataFrame, indicator: str, periods: int, metro_code: str):
    """Run Prophet forecast for a metro indicator."""
    from models.prophet_forecast import forecast_metro_indicator
    return forecast_metro_indicator(_metro_df, indicator, periods)

@st.cache_data
def calculate_market_scores(_metro_df: pd.DataFrame, _national_df: pd.DataFrame, _reit_df: pd.DataFrame, max_sentiment_adj: float = 0.20):
    """Calculate market strength scores with multiplicative sentiment."""
    from models.market_scoring import score_all_metros
    return score_all_metros(_metro_df, _national_df, _reit_df, max_sentiment_adj=max_sentiment_adj)

@st.cache_data
def get_methodology():
    """Get scoring methodology details for display."""
    from models.market_scoring import get_score_methodology
    return get_score_methodology()

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Settings")

    # Initialize force_refresh state
    if "force_refresh" not in st.session_state:
        st.session_state["force_refresh"] = False

    # Data refresh - use session state to trigger force refresh
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.session_state["force_refresh"] = True
        st.cache_data.clear()
        st.rerun()

    # Get force_refresh value and reset it after use
    force_refresh = st.session_state.get("force_refresh", False)
    if force_refresh:
        st.session_state["force_refresh"] = False  # Reset after this run

    st.markdown("---")

    # Check data status
    storage = DataStorage()
    data_status = []
    stale_datasets = []

    # Define staleness thresholds by data type
    staleness_thresholds = {
        "reit_prices": 3,      # Market data: 3 days (allows for weekends)
        "fred_national": 30,   # Economic indicators: monthly updates
        "fred_metros": 30      # Economic indicators: monthly updates
    }

    for dataset in ["fred_national", "fred_metros", "reit_prices"]:
        if storage.dataset_exists(dataset):
            updated = storage.get_last_updated(dataset)
            if updated:
                age_days = (datetime.now() - updated).days
                data_status.append(f"✅ {dataset}: {updated.strftime('%m/%d %H:%M')}")

                # Check if data is stale based on type-specific threshold
                threshold = staleness_thresholds.get(dataset, 7)
                if age_days > threshold:
                    data_type = "REIT sentiment" if dataset == "reit_prices" else "Economic indicators"
                    stale_datasets.append((dataset, age_days, threshold, data_type))
            else:
                data_status.append(f"✅ {dataset}")
        else:
            data_status.append(f"⏳ {dataset}: Not loaded")

    st.markdown("### 📊 Data Status")
    for status in data_status:
        st.caption(status)

    # Show staleness warnings if needed
    if stale_datasets:
        for dataset, age_days, threshold, data_type in stale_datasets:
            st.warning(f"⚠️ {dataset} is {age_days} days old (>{threshold} day threshold). {data_type} may be inaccurate. Consider refreshing data.")

    st.markdown("---")

    st.markdown("### 🎯 Analysis Type")
    analysis_type = st.selectbox(
        "Select Analysis",
        ["Economic Overview", "Market Rankings", "Metro Forecast", "REIT Sentiment", "Value Opportunities"]
    )

    st.markdown("---")

    if analysis_type == "Metro Forecast":
        st.markdown("### 📈 Forecast Settings")
        forecast_periods = st.slider("Months Ahead", 3, 24, 12)
        forecast_indicator = st.selectbox(
            "Indicator to Forecast",
            ["hpi", "unemployment"],
            format_func=lambda x: {"hpi": "House Price Index", "unemployment": "Unemployment Rate"}[x]
        )

        st.markdown("### 📍 Metro Selection")

        # Filter metros based on available data for selected indicator
        # This happens after data is loaded, so we'll do the filtering in the main section

    elif analysis_type == "Market Rankings":
        st.markdown("### ⚖️ Sentiment Settings")
        max_sentiment_pct = st.slider(
            "Max Sentiment Impact",
            min_value=0,
            max_value=50,
            value=20,
            step=5,
            help="Maximum magnitude (ceiling) of sentiment adjustment. REIT momentum determines the direction:\n"
                 "• Positive REIT momentum = scores multiplied by (1 + X%)\n"
                 "• Negative REIT momentum = scores multiplied by (1 - X%)\n"
                 "• Set to 0% to disable sentiment adjustment entirely.\n"
                 "Recommended range: 10-30%."
        )
        st.caption(f"⚖️ Sentiment range: ±{max_sentiment_pct}% maximum (direction set by REIT market momentum)")

# =============================================================================
# MAIN CONTENT
# =============================================================================

# Load data (pass force_refresh from session state if set)
with st.spinner("Loading economic data..." if not force_refresh else "Refreshing economic data from FRED..."):
    fred_data = load_fred_data(force_refresh=force_refresh)
    national_df = fred_data.get("national", pd.DataFrame())
    metros_df = fred_data.get("metros", pd.DataFrame())

with st.spinner("Loading REIT data..." if not force_refresh else "Refreshing REIT data from YFinance..."):
    reit_df = load_reit_data(force_refresh=force_refresh)

# Check if data is available
if metros_df.empty:
    st.warning("⚠️ No metro data available. Click 'Refresh Data' in the sidebar to fetch data from FRED.")
    st.info("This will fetch economic indicators for 20 major CRE markets.")
    st.stop()

st.markdown("---")

# =============================================================================
# METRO FORECAST VIEW
# =============================================================================

if analysis_type == "Metro Forecast":
    # Filter metros that have data for the selected indicator
    available_metros = []
    for metro_code in METROS.keys():
        metro_data_check = metros_df[metros_df["metro_code"] == metro_code]
        if not metro_data_check.empty and forecast_indicator in metro_data_check.columns:
            if metro_data_check[forecast_indicator].notna().sum() > 0:
                available_metros.append(metro_code)

    # Show metro selector with only available metros
    if not available_metros:
        st.error(f"No metros have {forecast_indicator} data available.")
        st.stop()

    selected_metro = st.selectbox(
        "Select Metro",
        options=available_metros,
        format_func=lambda x: f"{x} - {METROS[x]['name']}",
        key="metro_forecast_selector"
    )

    st.caption(f"Showing {len(available_metros)} of {len(METROS)} top metros by population with {forecast_indicator} data")

    st.markdown(f"### 📍 {METROS[selected_metro]['name']} Forecast")

    # Get metro data
    metro_data = metros_df[metros_df["metro_code"] == selected_metro].copy()

    if metro_data.empty:
        st.warning(f"No data available for {selected_metro}")
        st.stop()

    # Check if indicator is available
    if forecast_indicator not in metro_data.columns or metro_data[forecast_indicator].isna().all():
        st.warning(f"No {forecast_indicator} data available for {selected_metro}. Try a different indicator.")
        available = [c for c in ["hpi", "unemployment"] if c in metro_data.columns and not metro_data[c].isna().all()]
        st.info(f"Available indicators: {', '.join(available) if available else 'None'}")
        st.stop()

    # Run forecast
    with st.spinner("Running Prophet forecast..."):
        try:
            forecast_result, metrics = run_prophet_forecast(
                metro_data, forecast_indicator, forecast_periods, selected_metro
            )
        except Exception as e:
            st.error(f"Forecast error: {e}")
            st.info("Prophet may not be installed. Run: pip install prophet")
            forecast_result = pd.DataFrame()
            metrics = {}

    if not forecast_result.empty:
        # Split historical and forecast
        historical_end = metro_data["date"].max()
        historical_fit = forecast_result[forecast_result["date"] <= historical_end]
        future = forecast_result[forecast_result["date"] > historical_end]

        # Get actual (non-interpolated) historical data
        actual_data = metro_data[["date", forecast_indicator]].dropna()

        # Create forecast chart
        fig = go.Figure()

        # Interpolated/fitted historical line (shows what Prophet learned)
        fig.add_trace(go.Scatter(
            x=historical_fit["date"],
            y=historical_fit["forecast"],
            mode="lines",
            name="Fitted (Interpolated)",
            line=dict(color="#636EFA", width=1, dash="dot"),
            opacity=0.6
        ))

        # Actual historical values (real data points only)
        fig.add_trace(go.Scatter(
            x=actual_data["date"],
            y=actual_data[forecast_indicator],
            mode="markers",
            name="Actual Data",
            marker=dict(color="#636EFA", size=8)
        ))

        # Future forecast
        fig.add_trace(go.Scatter(
            x=future["date"],
            y=future["forecast"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#00CC96", width=2, dash="dash"),
            marker=dict(size=5)
        ))

        # Confidence interval for future
        fig.add_trace(go.Scatter(
            x=pd.concat([future["date"], future["date"][::-1]]),
            y=pd.concat([future["upper_bound"], future["lower_bound"][::-1]]),
            fill="toself",
            fillcolor="rgba(0, 204, 150, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% CI"
        ))

        indicator_labels = {
            "hpi": "House Price Index",
            "employment": "Employment (thousands)",
            "unemployment": "Unemployment Rate (%)"
        }

        fig.update_layout(
            title=f"{indicator_labels[forecast_indicator]} - {forecast_periods} Month Forecast",
            xaxis_title="Date",
            yaxis_title=indicator_labels[forecast_indicator],
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Metrics
        col1, col2, col3, col4 = st.columns(4)

        if metrics:
            # Determine precision and delta color based on indicator
            if forecast_indicator == "unemployment":
                precision = 2  # More precision for unemployment (small changes matter)
                delta_color = "inverse"  # Rising unemployment = bad = red
                # For unemployment, show absolute change in percentage points
                absolute_change = metrics['forecast_end'] - metrics['current_value']
                delta_str = f"{absolute_change:+.2f} pp"  # percentage points
            else:
                precision = 1
                delta_color = "normal"  # Rising HPI = good = green
                delta_str = f"{metrics['pct_change']:+.1f}%"

            with col1:
                st.metric("Current Value", f"{metrics['current_value']:.{precision}f}")
            with col2:
                st.metric(
                    f"{forecast_periods}mo Forecast",
                    f"{metrics['forecast_end']:.{precision}f}",
                    delta_str,
                    delta_color=delta_color
                )
            with col3:
                st.metric("Lower Bound (95% CI)", f"{metrics['forecast_lower']:.{precision}f}")
            with col4:
                st.metric("Upper Bound (95% CI)", f"{metrics['forecast_upper']:.{precision}f}")

        # Add explanatory note for HPI
        if forecast_indicator == "hpi" and metrics:
            current_hpi = metrics.get('current_value', 437)
            base_hpi = 100
            multiplier = current_hpi / base_hpi
            pct_increase = (current_hpi - base_hpi) / base_hpi * 100

            # Calculate realistic example prices
            example_price_1995 = 160000
            example_price_today = example_price_1995 * multiplier

            with st.expander("📊 Understanding the House Price Index (HPI)", expanded=False):
                st.markdown(f"""
                **What is HPI?**

                The House Price Index is a **relative index**, not a dollar amount. It measures price changes over time compared to a baseline.

                **How to Read It:**

                - **Base Period = 100** (1995 Q1 for FHFA data)
                - **HPI = {current_hpi:.0f}** means prices are {multiplier:.2f}× higher than the base period
                - This represents a {pct_increase:.0f}% increase from the baseline

                **What Matters Most:**

                Focus on the **growth rate** (e.g., {metrics.get('pct_change', 0):+.1f}%) rather than the absolute number. The growth rate indicates market strength and momentum.

                **Example:**

                If a home cost **\${example_price_1995:,.0f}** in 1995 (HPI = 100), at today's HPI of {current_hpi:.0f}, that same home would cost approximately **\${example_price_today:,.0f}**.
                """)

        # Add explanatory note for Unemployment Rate
        if forecast_indicator == "unemployment" and metrics:
            current_rate = metrics.get('current_value', 0)
            forecast_rate = metrics.get('forecast_end', 0)
            change = metrics.get('pct_change', 0)

            with st.expander("📊 Understanding Unemployment Rate", expanded=False):
                st.markdown(f"""
                **What is the Unemployment Rate?**

                The unemployment rate measures the percentage of the labor force that is actively seeking employment but unable to find work. It's a key indicator of economic health and labor market conditions.

                **How to Read It:**

                - **Current Rate = {current_rate:.1f}%** means {current_rate:.1f}% of the labor force is unemployed
                - **Forecasted Rate = {forecast_rate:.1f}%** ({change:+.1f}% change over forecast period)
                - **~4-5%** is generally considered "full employment" (natural rate)
                - **Below 4%** indicates a very tight labor market
                - **Above 6%** typically signals economic weakness

                **Why It Matters for CRE:**

                - **Office & Retail**: Low unemployment → more employed workers → higher demand for office space and consumer spending
                - **Multifamily**: Employment levels directly impact rental demand and tenant quality
                - **Economic Indicator**: Rising unemployment often precedes recessions, affecting all CRE sectors

                **What Matters Most:**

                Focus on the **direction and rate of change** rather than the absolute level. Rapidly rising unemployment signals economic distress, while declining unemployment indicates economic strength.

                **Interpreting the Forecast:**

                {"📉 **Declining unemployment** suggests strengthening labor market conditions, which generally supports CRE demand." if change < 0 else "📈 **Rising unemployment** may indicate softening economic conditions, potentially weakening CRE fundamentals." if change > 0 else "➡️ **Stable unemployment** suggests steady labor market conditions."}
                """)

        # Data Limitations expander
        with st.expander("⚠️ Data Coverage & Limitations", expanded=False):
            st.markdown("""
            **Metro Coverage**

            This dashboard uses publicly available data from FRED (Federal Reserve Economic Data) and covers 20 major U.S. metropolitan areas selected by:
            - **Population size**: Top 20 MSAs by population (2020 Census)
            - **Economic significance**: Major commercial real estate markets
            - **Data availability**: FRED API coverage

            **Current Data Status**

            - ✅ **13 metros with HPI data**: NYC, LAX, CHI, DFW, HOU, WAS, MIA, PHI, ATL, PHX, BOS, SFO, MSP
            - ✅ **14 metros with unemployment data**: All above + SEA
            - ❌ **6 metros with limited/no data**: DEN, SAN, AUS, NSH, CLT, RDU

            **Why Some Metros Are Missing**

            1. **FRED API Limitations**: Not all metros have FHFA HPI series in FRED
            2. **Data Lag**: Smaller/newer metros may have shorter history or delayed reporting
            3. **Series Changes**: Some metros use different identifier codes or consolidated series

            **Design Approach: Population-First vs. Data-First**

            **Our Approach (Population-First):**
            - Selected top 20 metros by population (2020 Census)
            - Assumed FRED would have data for major markets
            - Result: 6 metros (30%) have missing/incomplete data

            **Alternative Approach (Data-First):**
            - Query FRED API to discover available FHFA HPI series
            - Select top N metros from those with complete data
            - Prioritize by population + CRE significance among available metros
            - Result: 100% data coverage, but potentially excluding some top-20 markets

            **Trade-offs:**
            - Population-first shows data gaps transparently (our choice)
            - Data-first ensures functionality but may miss important markets
            - Production systems typically use data-first to guarantee reliability

            The population-first approach better demonstrates real-world data engineering challenges: requirements don't always match data availability.

            **What We'd Want vs. What We Have**

            | Data Type | Ideal | Current | Gap |
            |-----------|-------|---------|-----|
            | Metro Coverage | All top 50 MSAs | 14-20 MSAs | Smaller markets unavailable |
            | HPI Frequency | Monthly | Quarterly | 2-month lag between updates |
            | Update Speed | Real-time | 1-2 quarters lag | FHFA reporting delay |
            | REIT Data | Daily | Daily | ✅ No gap |
            | Unemployment | Monthly | Monthly | ✅ No gap |

            **Portfolio Transparency Note**

            This is a **portfolio demonstration project** using free, publicly available data. In a work setting, I would:
            1. Negotiate paid API access for comprehensive coverage
            2. Build data pipelines to handle missing data gracefully
            3. Document data quality metrics and coverage reports
            4. Set up monitoring for data freshness and completeness

            The focus here is demonstrating **analytical methodology** and **system architecture**, not perfect data coverage.
            """)

    else:
        st.warning("Could not generate forecast. Check that Prophet is installed.")

# =============================================================================
# MARKET RANKINGS VIEW
# =============================================================================

elif analysis_type == "Market Rankings":
    st.markdown("### 🏆 Market Strength Rankings")
    st.markdown("Metros ranked using **multiplicative sentiment adjustment**: backward-looking fundamentals adjusted by forward-looking REIT momentum.")

    # Get user-selected max sentiment adjustment (from sidebar slider)
    max_sentiment_adj = max_sentiment_pct / 100.0  # Convert % to decimal

    with st.spinner("Calculating market scores..."):
        try:
            scores, features, sentiment = calculate_market_scores(metros_df, national_df, reit_df, max_sentiment_adj=max_sentiment_adj)
        except Exception as e:
            st.error(f"Error calculating scores: {e}")
            scores = pd.DataFrame()
            sentiment = {}

    if not scores.empty:
        # Show data coverage info
        scored_metros = len(scores)
        total_metros = len(METROS)
        if scored_metros < total_metros:
            missing = set(METROS.keys()) - set(scores["metro_code"].unique())
            st.info(f"ℹ️ Scoring {scored_metros} of {total_metros} metros. Missing: {', '.join(sorted(missing))} (insufficient data for growth rate calculations)")

        # Sentiment indicator
        composite_sentiment = sentiment.get("composite", 0)
        sentiment_pct = composite_sentiment * max_sentiment_pct  # User-selected max adjustment

        st.markdown("#### 📊 Current Market Sentiment")

        # Main sentiment metric
        sent_metric_col1, sent_metric_col2 = st.columns([1, 3])
        with sent_metric_col1:
            if sentiment_pct >= 0:
                st.metric("REIT Sentiment", "Positive", f"+{sentiment_pct:.1f}%")
            else:
                st.metric("REIT Sentiment", "Negative", f"{sentiment_pct:.1f}%")

        with sent_metric_col2:
            # Sentiment gauge
            sentiment_desc = (
                "Strong bullish signal from REITs" if sentiment_pct > 10 else
                "Moderately positive market outlook" if sentiment_pct > 3 else
                "Neutral market sentiment" if sentiment_pct > -3 else
                "Moderately negative outlook" if sentiment_pct > -10 else
                "Strong bearish signal from REITs"
            )
            st.info(f"**Formula:** Final Score = Base Score × (1 {sentiment_pct:+.1f}%)")
            st.caption(sentiment_desc)
            st.caption("ℹ️ This composite sentiment is applied uniformly to ALL metros. Scores are relative rankings (0-100 scale).")

        st.markdown("---")

        # Top/Bottom markets with enhanced info
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🟢 Top 5 Markets")
            top_cols = ["rank", "metro_code", "metro_name", "strength_score", "base_score", "region"]
            available_top_cols = [c for c in top_cols if c in scores.columns]
            top_5 = scores.head(5)[available_top_cols].copy()
            top_5["strength_score"] = top_5["strength_score"].round(1)
            if "base_score" in top_5.columns:
                top_5["base_score"] = top_5["base_score"].round(1)
            st.dataframe(top_5, hide_index=True, use_container_width=True)

        with col2:
            st.markdown("#### 🔴 Bottom 5 Markets")
            bottom_5 = scores.tail(5)[available_top_cols].copy()
            bottom_5["strength_score"] = bottom_5["strength_score"].round(1)
            if "base_score" in bottom_5.columns:
                bottom_5["base_score"] = bottom_5["base_score"].round(1)
            st.dataframe(bottom_5, hide_index=True, use_container_width=True)

        # Chart tabs
        # Sort by rank descending (rank 1 at top of horizontal chart) to maintain tie-breaking order
        chart_df = scores.sort_values("rank", ascending=False).copy()

        chart_tab1, chart_tab2 = st.tabs(["Final Scores", "Score Breakdown"])

        with chart_tab1:
            # Clean final score chart with color gradient
            fig1 = go.Figure()

            fig1.add_trace(go.Bar(
                y=chart_df["metro_code"],
                x=chart_df["strength_score"],
                orientation="h",
                marker=dict(
                    color=chart_df["strength_score"],
                    colorscale="RdYlGn",
                    showscale=True,
                    colorbar=dict(title="Score")
                ),
                text=chart_df["strength_score"].round(1),
                textposition="outside"
            ))

            fig1.update_layout(
                title="Market Strength Scores (Final)",
                xaxis_title="Score",
                yaxis_title="Metro",
                height=650,
                showlegend=False
            )
            st.plotly_chart(fig1, use_container_width=True)

        with chart_tab2:
            # Stacked bar showing base score + sentiment contribution
            # Calculate sentiment_contribution if not present (for cache compatibility)
            if "base_score" in chart_df.columns and "sentiment_contribution" not in chart_df.columns:
                sentiment_adj = chart_df["sentiment_adjustment"].iloc[0] if "sentiment_adjustment" in chart_df.columns else 0
                chart_df["sentiment_contribution"] = chart_df["base_score"] * sentiment_adj

            if "base_score" in chart_df.columns:
                fig2 = go.Figure()

                # Get sentiment direction
                sentiment_adj = chart_df["sentiment_adjustment"].iloc[0] if "sentiment_adjustment" in chart_df.columns else 0
                is_positive = sentiment_adj >= 0

                if is_positive:
                    # Positive sentiment: show base (blue) + boost (green) stacked
                    fig2.add_trace(go.Bar(
                        y=chart_df["metro_code"],
                        x=chart_df["base_score"],
                        name="Base Score (Fundamentals)",
                        orientation="h",
                        marker_color="steelblue",
                        text=chart_df["base_score"].round(1),
                        textposition="inside",
                        insidetextanchor="middle"
                    ))
                    fig2.add_trace(go.Bar(
                        y=chart_df["metro_code"],
                        x=chart_df["sentiment_contribution"],
                        name="Sentiment Adjustment (uncapped)",
                        orientation="h",
                        marker_color="forestgreen",
                        text=chart_df["sentiment_contribution"].apply(lambda x: f"+{x:.1f}"),
                        textposition="outside"
                    ))
                    fig2.update_layout(barmode="stack", legend=dict(traceorder="normal"))
                else:
                    # Negative sentiment: show final score (blue) + drag amount (red) to show what was lost
                    # Final score is base_score + sentiment_contribution (where contribution is negative)
                    final_score = chart_df["base_score"] + chart_df["sentiment_contribution"]
                    drag_amount = chart_df["sentiment_contribution"].abs()

                    fig2.add_trace(go.Bar(
                        y=chart_df["metro_code"],
                        x=final_score,
                        name="Final Score (after drag)",
                        orientation="h",
                        marker_color="steelblue",
                        text=final_score.round(1),
                        textposition="inside",
                        insidetextanchor="middle"
                    ))
                    fig2.add_trace(go.Bar(
                        y=chart_df["metro_code"],
                        x=drag_amount,
                        name="Sentiment Adjustment (uncapped)",
                        orientation="h",
                        marker_color="indianred",
                        marker_opacity=0.7,
                        text=drag_amount.apply(lambda x: f"-{x:.1f}"),
                        textposition="inside",
                        insidetextanchor="middle"
                    ))
                    fig2.update_layout(barmode="stack")

                fig2.update_layout(
                    title="Score Breakdown: Base + Sentiment Adjustment",
                    xaxis_title="Score",
                    yaxis_title="Metro",
                    height=650,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    bargap=0.15
                )
                st.plotly_chart(fig2, use_container_width=True)

                # Show the actual sentiment adjustment info
                avg_contribution = chart_df["sentiment_contribution"].mean()
                if is_positive:
                    st.caption(
                        f"**Blue** = Base Score (fundamentals). **Green** = Sentiment Boost ({sentiment_adj*100:+.1f}%). "
                        f"Total bar length = Final Score."
                    )
                else:
                    st.caption(
                        f"**Blue** = Final Score (after sentiment). **Red** = Sentiment Drag ({sentiment_adj*100:.1f}% lost). "
                        f"Blue + Red = Base Score (what it would be without negative sentiment)."
                    )
            else:
                st.warning("Base score data not available for breakdown chart.")

        # Regional breakdown
        if "region" in scores.columns:
            st.markdown("#### 🗺️ Regional Analysis")

            region_summary = scores.groupby("region").agg({
                "strength_score": ["mean", "min", "max", "count"],
                "base_score": "mean"
            }).round(1)
            region_summary.columns = ["Avg Score", "Min", "Max", "Count", "Avg Base"]
            region_summary = region_summary.sort_values("Avg Score", ascending=False)

            col1, col2 = st.columns([1, 2])

            with col1:
                st.dataframe(region_summary, use_container_width=True)

            with col2:
                fig = px.box(
                    scores,
                    x="region",
                    y="strength_score",
                    color="region",
                    title="Score Distribution by Region",
                    labels={"strength_score": "Strength Score", "region": "Region"}
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        # Feature breakdown
        with st.expander("📊 Score Components (Backward-Looking Fundamentals)"):
            feature_cols = [
                "metro_code", "metro_name", "region", "base_score", "strength_score",
                "hpi_growth_12m", "hpi_growth_3m", "employment_growth_12m",
                "employment_growth_3m", "unemployment_rate", "unemployment_change_12m",
                "population_growth_1y"
            ]
            available_cols = [c for c in feature_cols if c in scores.columns]
            if available_cols:
                display_df = scores[available_cols].copy()
                # Round numeric columns
                for col in display_df.columns:
                    if display_df[col].dtype in ["float64", "int64"]:
                        display_df[col] = display_df[col].round(2)
                st.dataframe(display_df, hide_index=True, use_container_width=True)

        # Data quality indicator
        if "imputed_count" in scores.columns:
            imputed_metros = scores[scores["imputed_count"] > 0]
            if not imputed_metros.empty:
                with st.expander("⚠️ Data Quality Notes"):
                    st.warning(f"⚠️ **Ranking Reliability Issue**: {len(imputed_metros)} metros have imputed (estimated) values for missing data")

                    st.markdown("""
                    **What this means:**
                    - Missing fundamental data (HPI growth, population growth, etc.) was filled with median values from other metros
                    - Scores for these metros are based partly on real data and partly on estimates
                    - Rankings may be **inaccurate** - metros with imputed data are artificially pushed toward average scores
                    - The final ranking might be correct, but it's not based on complete real data

                    **Why this happens:**
                    - FRED data coverage is incomplete for some metros (quarterly HPI lags, annual population updates)
                    - Imputation keeps all top-20 metros visible, but transparency about data quality is critical

                    **Recommendation:**
                    - Treat rankings for metros with high imputation counts (2-3+) as **directional only**
                    - Focus on metros with 0 imputed values for most reliable comparisons
                    """)

                    st.dataframe(
                        imputed_metros[["metro_code", "metro_name", "imputed_count"]],
                        hide_index=True,
                        use_container_width=True
                    )

        # Methodology explanation
        with st.expander("📖 Scoring Methodology"):
            methodology = get_methodology()

            st.markdown("**Model:** Rule-Based Scoring (NOT machine learning)")
            st.markdown("Weighted scoring of fundamentals with REIT sentiment adjustment")
            st.markdown("")

            st.markdown(f"**Approach:** {methodology['approach']}")
            st.markdown(f"**Formula:** `{methodology['formula']}`")
            st.markdown(f"**Max Sentiment Adjustment:** ±{max_sentiment_pct:.0f}% *(adjustable in sidebar)*")

            st.markdown("##### Backward-Looking Weights (Base Score)")
            for feature, weight in methodology["base_weights"].items():
                direction = "↑ Higher is better" if weight > 0 else "↓ Lower is better"
                st.caption(f"  • {feature}: {abs(weight)*100:.0f}% ({direction})")

            st.markdown("##### Forward-Looking Weights (Sentiment)")
            for sector, weight in methodology["sentiment_weights"].items():
                st.caption(f"  • {sector.replace('_', ' ').title()}: {weight*100:.0f}%")

            # Data notes
            if "data_notes" in methodology:
                st.markdown("##### Data Sources & Notes")
                for note in methodology["data_notes"]:
                    st.caption(f"  • {note}")

        # Enhanced sentiment breakdown at bottom with time-travel
        st.markdown("---")
        st.markdown("### 🎯 REIT Sentiment Deep Dive")

        with st.expander("🔍 Sentiment Calculation Breakdown & Time Travel", expanded=False):
            st.markdown("**Explore how individual REIT sector returns combine into the composite sentiment**")
            st.markdown("This composite is applied uniformly to ALL metros, not metro-specific.")

            # Time travel controls
            st.markdown("#### ⏰ Time Travel: Rewind REIT Momentum")

            if reit_df is not None and not reit_df.empty:
                latest_date = reit_df["date"].max()
                earliest_date = reit_df["date"].min()

                # Initialize session state for reset counter (forces widget recreation)
                if "time_travel_reset_counter" not in st.session_state:
                    st.session_state["time_travel_reset_counter"] = 0

                time_travel_col1, time_travel_col2 = st.columns([3, 1])

                with time_travel_col2:
                    if st.button("↻ Reset to Current", use_container_width=True):
                        st.session_state["time_travel_reset_counter"] += 1
                        st.rerun()

                with time_travel_col1:
                    # Use unique key that changes on reset to force widget recreation
                    selected_end_date = st.date_input(
                        "Select end date for 30-day momentum window",
                        value=latest_date,
                        min_value=earliest_date + timedelta(days=30),
                        max_value=latest_date,
                        key=f"sentiment_time_travel_picker_{st.session_state['time_travel_reset_counter']}",
                        help="Choose any historical date to see how sentiment was calculated at that time"
                    )

                # Convert to datetime
                selected_end_datetime = pd.Timestamp(selected_end_date)
                lookback_start_date = selected_end_datetime - timedelta(days=30)

                st.caption(f"📅 Analyzing REIT momentum from **{lookback_start_date.date()}** to **{selected_end_datetime.date()}**")

                # Recalculate sentiment for selected time window
                from models.market_scoring import CRE_SECTOR_WEIGHTS

                time_travel_reit = reit_df[
                    (reit_df["date"] >= lookback_start_date) &
                    (reit_df["date"] <= selected_end_datetime)
                ]

                if not time_travel_reit.empty:
                    # Calculate sector returns for time window
                    sector_breakdown = []
                    total_weighted_return = 0
                    total_weight = 0
                    positive_sectors = []
                    negative_sectors = []

                    for sector_key, weight in CRE_SECTOR_WEIGHTS.items():
                        sector_data = time_travel_reit[time_travel_reit["sector"] == sector_key]
                        if not sector_data.empty:
                            avg_daily = sector_data["pct_change"].mean()
                            cumulative_return = avg_daily * min(30, len(sector_data))
                            weighted_contribution = cumulative_return * weight
                            total_weighted_return += weighted_contribution
                            total_weight += weight

                            sector_breakdown.append({
                                "Sector": sector_key.replace("_", " ").title(),
                                "Weight": weight,
                                "Weight_Display": f"{weight*100:.0f}%",
                                "30d Return": cumulative_return,
                                "Return_Display": f"{cumulative_return:+.2f}%",
                                "Contribution": weighted_contribution,
                                "Contribution_Display": f"{weighted_contribution:+.2f}%"
                            })

                            if cumulative_return >= 0:
                                positive_sectors.append(sector_key.replace("_", " ").title())
                            else:
                                negative_sectors.append(sector_key.replace("_", " ").title())

                    if sector_breakdown:
                        st.markdown("---")

                        # Header with composite metric in upper right
                        breakdown_header_col1, breakdown_header_col2 = st.columns([2, 1])

                        with breakdown_header_col1:
                            st.markdown("#### 📊 Visual Sector Breakdown")

                        with breakdown_header_col2:
                            composite_contribution = total_weighted_return
                            # Display composite total as a metric
                            if composite_contribution >= 0:
                                st.metric("🎯 Composite Total", f"+{composite_contribution:.2f}%", help="Sum of all weighted sector contributions")
                            else:
                                st.metric("🎯 Composite Total", f"{composite_contribution:.2f}%", help="Sum of all weighted sector contributions")

                        # Create gauge chart for each sector
                        import plotly.graph_objects as go

                        fig = go.Figure()

                        # Add individual sector bars
                        for i, sector_data in enumerate(sector_breakdown):
                            contribution = sector_data["Contribution"]
                            sector_name = sector_data["Sector"]
                            weight = sector_data["Weight"]

                            # Create horizontal bar showing contribution
                            color = "green" if contribution >= 0 else "red"
                            fig.add_trace(go.Bar(
                                y=[sector_name],
                                x=[contribution],
                                orientation='h',
                                marker_color=color,
                                text=f"{contribution:+.2f}%",
                                textposition='outside',
                                name=sector_name,
                                hovertemplate=f"<b>{sector_name}</b><br>Weight: {weight*100:.0f}%<br>Contribution: {contribution:+.2f}%<extra></extra>"
                            ))

                        fig.update_layout(
                            title="Individual Sector Contributions",
                            xaxis_title="Contribution (%)",
                            yaxis_title="",
                            showlegend=False,
                            height=400,
                            xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='gray')
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Sector summary
                        st.markdown("#### 📋 Sector Details")
                        breakdown_display = pd.DataFrame([{
                            "Sector": s["Sector"],
                            "Weight": s["Weight_Display"],
                            "30d Return": s["Return_Display"],
                            "Contribution": s["Contribution_Display"]
                        } for s in sector_breakdown])
                        st.dataframe(breakdown_display, use_container_width=True, hide_index=True)

                        # Mixed sector explanation
                        st.markdown("---")
                        st.markdown("#### 🔄 Understanding Mixed Positive/Negative Sectors")

                        pos_count = len(positive_sectors)
                        neg_count = len(negative_sectors)

                        if pos_count > 0 and neg_count > 0:
                            st.info(f"""
                            **Mixed Market Detected:** {pos_count} positive sectors, {neg_count} negative sectors

                            - **Positive:** {', '.join(positive_sectors)}
                            - **Negative:** {', '.join(negative_sectors)}

                            When sectors are mixed, positive and negative returns **offset each other** in the weighted average.
                            The composite sentiment reflects the **net effect** based on sector weights.
                            """)
                        elif neg_count == 0:
                            st.success(f"**All sectors positive** during this period ({pos_count} sectors)")
                        else:
                            st.error(f"**All sectors negative** during this period ({neg_count} sectors)")

                        st.markdown("---")
                        st.markdown("#### 🧮 Calculation Steps")

                        # Step 1: Weighted average
                        avg_return = total_weighted_return / total_weight if total_weight > 0 else 0
                        st.caption(f"**Step 1 - Weighted Average Return:** {avg_return:+.2f}%")
                        st.caption(f"  → Sum of all contributions: {total_weighted_return:+.2f}%")
                        st.caption(f"  → Divided by total weight: {total_weight:.2f}")

                        # Step 2: Normalization
                        normalized = np.clip(avg_return / 10.0, -1.0, 1.0)
                        st.caption(f"**Step 2 - Normalize to [-1, +1]:** {normalized:+.3f}")
                        st.caption(f"  → Divide by 10% (typical monthly REIT range): {avg_return:+.2f}% ÷ 10% = {normalized:+.3f}")
                        st.caption(f"  → Clipped to [-1, +1] range")

                        # Step 3: Apply max adjustment
                        final_sentiment_pct = normalized * max_sentiment_pct
                        st.caption(f"**Step 3 - Apply Max Adjustment:** {final_sentiment_pct:+.1f}%")
                        st.caption(f"  → Multiply by user slider setting: {normalized:+.3f} × {max_sentiment_pct:.0f}% = {final_sentiment_pct:+.1f}%")

                        st.markdown("---")
                        st.success(f"""
                        ✨ **Final Result:** All metro scores are multiplied by **(1 {final_sentiment_pct:+.1f}%)**

                        - A metro with base_score of 80 becomes: 80 × (1 {final_sentiment_pct:+.1f}%) = {80 * (1 + final_sentiment_pct/100):.1f}
                        - This adjustment is **uniform across all metros** (not metro-specific)
                        - Negative sentiment reduces all scores proportionally
                        - Positive sentiment boosts all scores proportionally
                        """)

                        # Key insights
                        st.markdown("#### 💡 Key Insights")
                        st.markdown("""
                        **How the mechanism works:**
                        - **Sector weights** determine importance (e.g., Broad Market 25%, Industrial 20%)
                        - **Positive sectors** contribute positive momentum to composite
                        - **Negative sectors** contribute negative momentum to composite
                        - **Mixed markets** produce net sentiment based on weighted balance
                        - **Slider setting** (0-50%) controls maximum magnitude, not direction
                        - **REIT momentum** determines direction (positive/negative), not the slider
                        - Setting slider to **0% disables sentiment** entirely
                        """)
                    else:
                        st.warning("No sector data available for selected time window.")
                else:
                    st.warning("No REIT data available for selected time window. Try a more recent date.")
            else:
                st.warning("No REIT data loaded. Click 'Refresh Data' in the sidebar.")

    else:
        st.warning("Could not calculate market scores. Try refreshing the data.")

# =============================================================================
# VALUE OPPORTUNITIES VIEW
# =============================================================================

elif analysis_type == "Value Opportunities":
    st.markdown("### 💎 Value Opportunities")
    st.markdown("When sentiment is negative, which metros have the strongest fundamentals? And within each metro, which sector is best positioned?")

    # Calculate scores with default sentiment adjustment
    with st.spinner("Analyzing market fundamentals..."):
        try:
            scores, features, sentiment = calculate_market_scores(metros_df, national_df, reit_df, max_sentiment_adj=0.20)
        except Exception as e:
            st.error(f"Error calculating scores: {e}")
            scores = pd.DataFrame()
            sentiment = {}

    if not scores.empty and reit_df is not None and not reit_df.empty:
        # Get composite sentiment
        composite_sentiment = sentiment.get("composite", 0)

        # Current sentiment indicator
        st.markdown("#### 📊 Current Market Sentiment")
        if composite_sentiment < 0:
            st.error(f"**Negative Sentiment: {composite_sentiment*100:.1f}%** — Value opportunity conditions present")
        elif composite_sentiment < 0.05:
            st.warning(f"**Neutral Sentiment: {composite_sentiment*100:.1f}%** — Mixed signals")
        else:
            st.success(f"**Positive Sentiment: +{composite_sentiment*100:.1f}%** — Risk-on environment")

        st.markdown("---")

        # Rank metros by BASE score (fundamentals only, ignoring sentiment)
        st.markdown("#### 🏙️ Metros Ranked by Fundamentals")
        st.markdown("*Sorted by base score (economic strength without sentiment adjustment)*")

        # Prepare display data
        value_df = scores[["metro_code", "metro_name", "base_score", "strength_score", "sentiment_contribution", "region"]].copy()
        value_df["sentiment_drag"] = value_df["sentiment_contribution"]
        value_df = value_df.sort_values("base_score", ascending=False)

        # Display top metros by fundamentals
        col1, col2 = st.columns([2, 1])

        with col1:
            # Bar chart of base scores
            fig = px.bar(
                value_df.head(10),
                x="base_score",
                y="metro_name",
                orientation="h",
                color="base_score",
                color_continuous_scale="Greens",
                title="Top 10 Metros by Fundamental Strength"
            )
            fig.update_layout(
                yaxis=dict(categoryorder="total ascending"),
                showlegend=False,
                coloraxis_showscale=False,
                height=400
            )
            fig.update_traces(
                hovertemplate="<b>%{y}</b><br>Base Score: %{x:.1f}<extra></extra>"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("##### Interpretation")
            st.markdown("""
            **Base Score** = Pure fundamentals:
            - HPI growth (house prices)
            - Employment trends
            - Unemployment rate
            - Population growth

            During negative sentiment, high base scores indicate metros with strong underlying economics that may be undervalued by the market.
            """)

        # Sentiment-adjusted Full Metro Rankings
        st.caption("Adjust how heavily REIT-based sentiment influences the Adjusted Score below. "
                   "This only affects the Full Metro Rankings breakdown — the fundamentals chart above is unchanged.")
        value_sentiment_pct = st.slider(
            "Sentiment Weight",
            min_value=0,
            max_value=100,
            value=60,
            step=5,
            format="%d%%",
            help="Controls the max sentiment adjustment applied to base scores. "
                 "Higher values amplify the impact of REIT momentum on the Adjusted Score."
        )
        # Scale so 100% on slider = max_sentiment_adj of 2.0 (full swing)
        value_sentiment_adj = (value_sentiment_pct / 100.0) * 2.0

        # Re-score with user-selected sentiment weight
        fmr_scores, _, _ = calculate_market_scores(metros_df, national_df, reit_df, max_sentiment_adj=value_sentiment_adj)
        fmr_df = fmr_scores[["metro_code", "metro_name", "base_score", "strength_score", "sentiment_contribution", "region"]].copy()
        fmr_df = fmr_df.sort_values("base_score", ascending=False)

        with st.expander("📋 Full Metro Rankings", expanded=True):
            display_df = fmr_df.copy()
            display_df.columns = ["Code", "Metro", "Base Score", "Adjusted Score", "Sentiment Impact", "Region"]
            display_df = display_df[["Code", "Metro", "Region", "Base Score", "Adjusted Score", "Sentiment Impact"]]
            display_df["Base Score"] = display_df["Base Score"].round(1)
            display_df["Adjusted Score"] = display_df["Adjusted Score"].round(1)
            display_df["Sentiment Impact"] = display_df["Sentiment Impact"].apply(lambda x: f"{x:+.1f}" if pd.notna(x) else "N/A")
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Sector breakdown
        st.markdown("#### 🏢 Sector Performance by REIT Momentum")
        st.markdown("*Which property types are showing relative strength?*")

        # Calculate sector returns from REIT data
        recent_date = reit_df["date"].max()
        lookback_date = recent_date - timedelta(days=30)
        recent_reit = reit_df[reit_df["date"] >= lookback_date]

        if not recent_reit.empty:
            # Group by sector and calculate average return
            sector_perf = recent_reit.groupby("sector").agg(
                avg_return=("pct_change", "mean"),
                ticker_count=("ticker", "nunique")
            ).reset_index()
            sector_perf["cumulative_30d"] = sector_perf["avg_return"] * 30  # Approximate 30-day return
            sector_perf = sector_perf.sort_values("cumulative_30d", ascending=False)

            # Rename sectors for display
            sector_names = {
                "broad": "Broad Market",
                "office": "Office",
                "industrial": "Industrial/Logistics",
                "retail_openair": "Retail (Open Air)",
                "retail_mall": "Retail (Malls)",
                "multifamily": "Multifamily",
                "data_center": "Data Centers",
                "homebuilders": "Homebuilders"
            }
            sector_perf["sector_name"] = sector_perf["sector"].map(sector_names).fillna(sector_perf["sector"])

            # Bar chart
            colors = ["green" if x > 0 else "red" for x in sector_perf["cumulative_30d"]]
            fig = px.bar(
                sector_perf,
                x="sector_name",
                y="cumulative_30d",
                color="cumulative_30d",
                color_continuous_scale=["red", "lightgray", "green"],
                color_continuous_midpoint=0,
                title="30-Day Sector Momentum"
            )
            fig.update_layout(
                xaxis_title="",
                yaxis_title="Cumulative Return (%)",
                showlegend=False,
                coloraxis_showscale=False,
                height=350
            )
            fig.update_traces(
                hovertemplate="<b>%{x}</b><br>30-Day Return: %{y:.1f}%<extra></extra>"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Sector recommendations based on current conditions
            st.markdown("#### 💡 Sector Insights")

            best_sectors = sector_perf[sector_perf["cumulative_30d"] > 0]["sector_name"].tolist()
            worst_sectors = sector_perf[sector_perf["cumulative_30d"] < 0]["sector_name"].tolist()

            insight_col1, insight_col2 = st.columns(2)

            with insight_col1:
                if best_sectors:
                    st.success(f"**Relative Strength:** {', '.join(best_sectors[:3])}")
                else:
                    st.warning("No sectors showing positive momentum")

            with insight_col2:
                if worst_sectors:
                    st.error(f"**Under Pressure:** {', '.join(worst_sectors[:3])}")
                else:
                    st.info("No sectors showing significant weakness")

        # Value thesis summary
        st.markdown("---")
        st.markdown("#### 📝 Value Thesis Summary")

        if composite_sentiment < 0:
            # Get top 3 metros by fundamentals
            top_metros = value_df.head(3)["metro_name"].tolist()
            # Get strongest sector
            if not recent_reit.empty:
                strongest_sector = sector_perf.iloc[0]["sector_name"] if not sector_perf.empty else "N/A"

                st.info(f"""
                **Current conditions suggest value opportunities in:**
                - **Top Metros:** {', '.join(top_metros)}
                - **Strongest Sector:** {strongest_sector}

                *Thesis: Strong fundamentals + negative sentiment = potential entry point for long-term investors*
                """)
        else:
            st.info("""
            **Current sentiment is neutral/positive.**

            Value opportunities are typically more compelling during periods of negative sentiment,
            when strong fundamentals may be overlooked by the market.
            """)

    else:
        st.warning("Could not calculate scores. Try refreshing the data.")

# =============================================================================
# REIT SENTIMENT VIEW
# =============================================================================

elif analysis_type == "REIT Sentiment":
    st.markdown("### 📈 REIT Sector Sentiment")
    st.markdown("Real-time market sentiment from publicly traded REITs by CRE sector.")

    if reit_df.empty:
        st.warning("No REIT data available. Click 'Refresh Data' to fetch.")
        st.stop()

    # Sector performance
    sentiment = get_reit_sentiment()

    if not sentiment.empty:
        # Sector cards
        cols = st.columns(4)
        for i, (_, row) in enumerate(sentiment.iterrows()):
            col = cols[i % 4]
            with col:
                return_1m = row.get("return_1m", 0)
                if pd.notna(return_1m):
                    delta_color = "normal" if return_1m >= 0 else "inverse"
                    st.metric(
                        row["sector"].replace("_", " ").title(),
                        f"${row['current_avg_price']:.2f}",
                        f"{return_1m:+.1f}% (1M)",
                        delta_color=delta_color
                    )

        st.markdown("---")

    # Sector selector
    sectors = list(reit_df["sector"].unique())
    selected_sector = st.selectbox("Select Sector", sectors, format_func=lambda x: x.replace("_", " ").title())

    sector_data = reit_df[reit_df["sector"] == selected_sector]

    # Price chart
    fig = px.line(
        sector_data,
        x="date",
        y="close",
        color="ticker",
        title=f"{selected_sector.replace('_', ' ').title()} REIT Prices",
        labels={"close": "Price ($)", "date": "Date", "ticker": "Ticker"}
    )
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Ticker details - Show all contributing REITs
    with st.expander("📋 All Contributing REIT Tickers"):
        st.markdown("**All REITs used in this dashboard for market sentiment analysis:**")
        st.markdown("---")

        # Group by sector
        sectors_grouped = {}
        for ticker, info in REIT_TICKERS.items():
            sector = info["sector"]
            if sector not in sectors_grouped:
                sectors_grouped[sector] = []
            sectors_grouped[sector].append((ticker, info["name"]))

        # Display by sector
        for sector, tickers in sorted(sectors_grouped.items()):
            st.markdown(f"**{sector.replace('_', ' ').title()}**")
            for ticker, name in sorted(tickers):
                st.markdown(f"- **{ticker}**: {name}")
            st.markdown("")

# =============================================================================
# ECONOMIC OVERVIEW VIEW
# =============================================================================

elif analysis_type == "Economic Overview":
    st.markdown("### 🌐 National Economic Indicators")
    st.markdown("Key economic indicators affecting CRE markets.")

    if national_df.empty:
        st.warning("No national data available. Click 'Refresh Data' to fetch.")
        st.stop()

    # Get latest values and calculate changes
    sorted_df = national_df.sort_values("date")

    def get_latest_valid(column):
        """Get most recent non-null value and a ~1 month prior value for delta."""
        valid = sorted_df[sorted_df[column].notna()]
        if valid.empty:
            return None, None
        latest_row = valid.iloc[-1]
        prev_data = valid[valid["date"] <= (latest_row["date"] - pd.Timedelta(days=25))]
        prev_row = prev_data.iloc[-1] if not prev_data.empty else None
        return latest_row, prev_row

    # For backward compat: latest row for date reference
    latest = sorted_df.iloc[-1]
    prev_fallback = sorted_df[sorted_df["date"] <= (latest["date"] - pd.Timedelta(days=25))]
    prev = prev_fallback.iloc[-1] if not prev_fallback.empty else latest

    # =============================================================================
    # TOP METRICS ROW - Key CRE Indicators
    # =============================================================================
    st.markdown("#### 📊 Key Rates & Indicators")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        cur, prv = get_latest_valid("treasury_10y")
        if cur is not None:
            delta = cur["treasury_10y"] - prv["treasury_10y"] if prv is not None else None
            st.metric(
                "10Y Treasury",
                f"{cur['treasury_10y']:.2f}%",
                f"{delta:+.2f}%" if delta else None,
                delta_color="inverse"
            )
    with col2:
        cur, prv = get_latest_valid("mortgage_30y")
        if cur is not None:
            delta = cur["mortgage_30y"] - prv["mortgage_30y"] if prv is not None else None
            st.metric(
                "30Y Mortgage",
                f"{cur['mortgage_30y']:.2f}%",
                f"{delta:+.2f}%" if delta else None,
                delta_color="inverse"
            )
    with col3:
        cur, prv = get_latest_valid("fed_funds")
        if cur is not None:
            delta = cur["fed_funds"] - prv["fed_funds"] if prv is not None else None
            st.metric(
                "Fed Funds",
                f"{cur['fed_funds']:.2f}%",
                f"{delta:+.2f}%" if delta else None,
                delta_color="inverse"
            )
    with col4:
        cur, prv = get_latest_valid("unemployment_national")
        if cur is not None:
            delta = cur["unemployment_national"] - prv["unemployment_national"] if prv is not None else None
            st.metric(
                "Unemployment",
                f"{cur['unemployment_national']:.1f}%",
                f"{delta:+.1f}%" if delta else None,
                delta_color="inverse"
            )
    with col5:
        cur, prv = get_latest_valid("cre_delinquency")
        if cur is not None:
            delta = cur["cre_delinquency"] - prv["cre_delinquency"] if prv is not None else None
            st.metric(
                "CRE Delinquency",
                f"{cur['cre_delinquency']:.2f}%",
                f"{delta:+.2f}%" if delta else None,
                delta_color="inverse"
            )

    st.markdown("---")

    # =============================================================================
    # TABBED SECTIONS FOR DIFFERENT INDICATOR CATEGORIES
    # =============================================================================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💰 Interest Rates",
        "🏗️ Construction",
        "🛒 Consumer & Retail",
        "📈 Labor & Production",
        "🏦 CRE Credit"
    ])

    # Helper function for cleaner indicator labels
    def format_indicator_name(name):
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
            "gdp": "GDP"
        }
        return labels.get(name, name.replace("_", " ").title())

    # TAB 1: Interest Rates
    with tab1:
        st.markdown("##### Interest Rate Environment")
        st.caption("Interest rates directly impact CRE cap rates, financing costs, and property valuations.")

        rate_cols = ["treasury_10y", "treasury_2y", "mortgage_30y", "fed_funds"]
        available_rates = [c for c in rate_cols if c in national_df.columns]

        if available_rates:
            # Rate metrics row
            rate_metrics = st.columns(len(available_rates))
            for i, col_name in enumerate(available_rates):
                with rate_metrics[i]:
                    cur, prv = get_latest_valid(col_name)
                    if cur is not None:
                        current = cur[col_name]
                        delta = current - prv[col_name] if prv is not None else None
                        st.metric(
                            format_indicator_name(col_name),
                            f"{current:.2f}%",
                            f"{delta:+.2f}%" if delta else None,
                            delta_color="inverse"
                        )

            # Rate chart - melt then drop NaN to handle different frequencies
            rate_df = national_df[["date"] + available_rates].copy()
            rate_df = rate_df.melt(id_vars=["date"], var_name="indicator", value_name="rate")
            rate_df = rate_df.dropna(subset=["rate"])  # Drop NaN after melt so each series keeps its data
            rate_df["indicator"] = rate_df["indicator"].apply(format_indicator_name)

            fig = px.line(
                rate_df,
                x="date",
                y="rate",
                color="indicator",
                title="Interest Rate Trends",
                labels={"rate": "Rate (%)", "date": "Date", "indicator": ""}
            )
            fig.update_layout(hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig, use_container_width=True)

            # Yield curve spread
            if "treasury_10y" in national_df.columns and "treasury_2y" in national_df.columns:
                spread_df = national_df[["date", "treasury_10y", "treasury_2y"]].dropna()
                spread_df["yield_spread"] = spread_df["treasury_10y"] - spread_df["treasury_2y"]

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=spread_df["date"],
                    y=spread_df["yield_spread"],
                    fill="tozeroy",
                    name="10Y-2Y Spread",
                    line=dict(color="#636EFA")
                ))
                fig2.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Inversion")
                fig2.update_layout(
                    title="Yield Curve Spread (10Y - 2Y Treasury)",
                    xaxis_title="Date",
                    yaxis_title="Spread (%)",
                    hovermode="x unified"
                )
                st.plotly_chart(fig2, use_container_width=True)
                st.caption("Negative spread (yield curve inversion) historically signals recession risk.")
        else:
            st.info("No interest rate data available. Click 'Refresh Data' to fetch.")

    # TAB 2: Construction
    with tab2:
        st.markdown("##### Construction & Development Activity")
        st.caption("Construction spending and permits indicate CRE supply pipeline and developer confidence. "
                   "Note: Commercial construction is an aggregate national figure from the U.S. Census Bureau — "
                   "it does not break down by asset class or geography. For property-level pipeline data, "
                   "sources like CoStar or Dodge Data would be needed.")

        construction_cols = ["construction_spending", "commercial_construction", "housing_starts", "building_permits"]
        available_construction = [c for c in construction_cols if c in national_df.columns]

        if available_construction:
            # Construction metrics
            const_metrics = st.columns(min(len(available_construction), 4))
            for i, col_name in enumerate(available_construction[:4]):
                with const_metrics[i]:
                    cur, prv = get_latest_valid(col_name)
                    if cur is not None:
                        current = cur[col_name]
                        previous = prv[col_name] if prv is not None else None
                        delta_pct = ((current - previous) / previous * 100) if previous is not None and pd.notna(previous) and previous != 0 else None
                        # Format based on indicator type
                        if col_name in ["construction_spending", "commercial_construction"]:
                            val_str = f"${current/1000:.0f}B" if current > 1000 else f"${current:.0f}M"
                        else:
                            val_str = f"{current:.0f}K"
                        st.metric(
                            format_indicator_name(col_name),
                            val_str,
                            f"{delta_pct:+.1f}%" if delta_pct else None
                        )

            # Split into two charts
            col1, col2 = st.columns(2)

            # Construction spending chart
            spending_cols = ["construction_spending", "commercial_construction"]
            available_spending = [c for c in spending_cols if c in national_df.columns]
            if available_spending:
                with col1:
                    spend_df = national_df[["date"] + available_spending].dropna(subset=available_spending, how="all")
                    spend_df = spend_df.melt(id_vars=["date"], var_name="indicator", value_name="value")
                    spend_df["indicator"] = spend_df["indicator"].apply(format_indicator_name)
                    spend_df["value"] = spend_df["value"] / 1000  # Convert to billions

                    fig = px.line(
                        spend_df,
                        x="date",
                        y="value",
                        color="indicator",
                        title="Construction Spending",
                        labels={"value": "Billions ($)", "date": "Date", "indicator": ""}
                    )
                    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02))
                    st.plotly_chart(fig, use_container_width=True)

            # Housing permits/starts chart
            housing_cols = ["housing_starts", "building_permits"]
            available_housing = [c for c in housing_cols if c in national_df.columns]
            if available_housing:
                with col2:
                    housing_df = national_df[["date"] + available_housing].dropna(subset=available_housing, how="all")
                    housing_df = housing_df.melt(id_vars=["date"], var_name="indicator", value_name="value")
                    housing_df["indicator"] = housing_df["indicator"].apply(format_indicator_name)

                    fig = px.line(
                        housing_df,
                        x="date",
                        y="value",
                        color="indicator",
                        title="Housing Supply Pipeline",
                        labels={"value": "Units (thousands, SAAR)", "date": "Date", "indicator": ""}
                    )
                    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02))
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No construction data available. Click 'Refresh Data' to fetch.")

    # TAB 3: Consumer & Retail
    with tab3:
        st.markdown("##### Consumer Health & Retail Activity")
        st.caption("Consumer spending drives demand for retail CRE. Sentiment indicates future spending trends.")

        consumer_cols = ["retail_sales", "consumer_sentiment", "cpi"]
        available_consumer = [c for c in consumer_cols if c in national_df.columns]

        if available_consumer:
            # Consumer metrics
            cons_metrics = st.columns(len(available_consumer))
            for i, col_name in enumerate(available_consumer):
                with cons_metrics[i]:
                    cur, prv = get_latest_valid(col_name)
                    if cur is not None:
                        current = cur[col_name]
                        previous = prv[col_name] if prv is not None else None
                        if col_name == "retail_sales":
                            val_str = f"${current/1000:.0f}B"
                            delta_pct = ((current - previous) / previous * 100) if previous is not None and pd.notna(previous) and previous != 0 else None
                            delta_str = f"{delta_pct:+.1f}%" if delta_pct else None
                        elif col_name == "consumer_sentiment":
                            val_str = f"{current:.1f}"
                            delta = current - previous if previous is not None and pd.notna(previous) else None
                            delta_str = f"{delta:+.1f}" if delta else None
                        else:  # CPI
                            val_str = f"{current:.1f}"
                            delta_pct = ((current - previous) / previous * 100) if previous is not None and pd.notna(previous) and previous != 0 else None
                            delta_str = f"{delta_pct:+.1f}%" if delta_pct else None
                        st.metric(format_indicator_name(col_name), val_str, delta_str)

            col1, col2 = st.columns(2)

            # Retail sales chart
            if "retail_sales" in national_df.columns:
                with col1:
                    retail_df = national_df[["date", "retail_sales"]].dropna()
                    retail_df["retail_sales"] = retail_df["retail_sales"] / 1000  # Billions

                    fig = px.area(
                        retail_df,
                        x="date",
                        y="retail_sales",
                        title="Advance Retail Sales",
                        labels={"retail_sales": "Billions ($)", "date": "Date"}
                    )
                    fig.update_traces(fillcolor="rgba(99, 110, 250, 0.3)", line_color="#636EFA")
                    fig.update_layout(hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)

            # Consumer sentiment chart
            if "consumer_sentiment" in national_df.columns:
                with col2:
                    sent_df = national_df[["date", "consumer_sentiment"]].dropna()

                    fig = px.line(
                        sent_df,
                        x="date",
                        y="consumer_sentiment",
                        title="U of Michigan Consumer Sentiment",
                        labels={"consumer_sentiment": "Index", "date": "Date"}
                    )
                    # Add recession reference bands (historical average ~85)
                    fig.add_hline(y=80, line_dash="dash", line_color="orange", annotation_text="Caution")
                    fig.add_hline(y=60, line_dash="dash", line_color="red", annotation_text="Pessimistic")
                    fig.update_layout(hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No consumer data available. Click 'Refresh Data' to fetch.")

    # TAB 4: Labor & Production
    with tab4:
        st.markdown("##### Labor Market & Industrial Production")
        st.caption("Employment drives office and multifamily demand. Industrial production affects logistics/warehouse CRE.")

        labor_cols = ["unemployment_national", "payrolls", "industrial_production"]
        available_labor = [c for c in labor_cols if c in national_df.columns]

        if available_labor:
            # Labor metrics
            labor_metrics = st.columns(len(available_labor))
            for i, col_name in enumerate(available_labor):
                with labor_metrics[i]:
                    cur, prv = get_latest_valid(col_name)
                    if cur is not None:
                        current = cur[col_name]
                        previous = prv[col_name] if prv is not None else None
                        if col_name == "unemployment_national":
                            val_str = f"{current:.1f}%"
                            delta = current - previous if previous is not None and pd.notna(previous) else None
                            delta_str = f"{delta:+.1f}%" if delta else None
                            st.metric(format_indicator_name(col_name), val_str, delta_str, delta_color="inverse")
                        elif col_name == "payrolls":
                            val_str = f"{current/1000:.1f}M"
                            delta = (current - previous) if previous is not None and pd.notna(previous) else None
                            delta_str = f"{delta:+.0f}K" if delta else None
                            st.metric(format_indicator_name(col_name), val_str, delta_str)
                        else:  # industrial_production
                            val_str = f"{current:.1f}"
                            delta_pct = ((current - previous) / previous * 100) if previous is not None and pd.notna(previous) and previous != 0 else None
                            delta_str = f"{delta_pct:+.1f}%" if delta_pct else None
                            st.metric(format_indicator_name(col_name), val_str, delta_str)

            col1, col2 = st.columns(2)

            # Payrolls chart
            if "payrolls" in national_df.columns:
                with col1:
                    pay_df = national_df[["date", "payrolls"]].dropna()
                    pay_df["payrolls"] = pay_df["payrolls"] / 1000  # Millions

                    fig = px.area(
                        pay_df,
                        x="date",
                        y="payrolls",
                        title="Total Nonfarm Payrolls",
                        labels={"payrolls": "Millions", "date": "Date"}
                    )
                    fig.update_traces(fillcolor="rgba(0, 204, 150, 0.3)", line_color="#00CC96")
                    fig.update_layout(hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)

            # Industrial production chart
            if "industrial_production" in national_df.columns:
                with col2:
                    ind_df = national_df[["date", "industrial_production"]].dropna()

                    fig = px.line(
                        ind_df,
                        x="date",
                        y="industrial_production",
                        title="Industrial Production Index",
                        labels={"industrial_production": "Index (2017=100)", "date": "Date"}
                    )
                    fig.update_layout(hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("The Industrial Production Index (Federal Reserve) measures real output of U.S. manufacturing, "
                               "mining, and utilities, indexed to 2017 = 100. Rising production typically signals increased "
                               "demand for warehouse, distribution, and logistics CRE.")

            # Unemployment chart (if not already shown)
            if "unemployment_national" in national_df.columns:
                unemp_df = national_df[["date", "unemployment_national"]].dropna()

                fig = px.area(
                    unemp_df,
                    x="date",
                    y="unemployment_national",
                    title="National Unemployment Rate",
                    labels={"unemployment_national": "Rate (%)", "date": "Date"}
                )
                fig.update_traces(fillcolor="rgba(239, 85, 59, 0.3)", line_color="#EF553B")
                fig.add_hline(y=5, line_dash="dash", line_color="gray", annotation_text="~Full Employment")
                fig.update_layout(hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No labor market data available. Click 'Refresh Data' to fetch.")

    # TAB 5: CRE Credit Conditions
    with tab5:
        st.markdown("##### Commercial Real Estate Credit")
        st.caption("Lending conditions and delinquency rates signal CRE market health and financing availability.")

        cre_cols = ["cre_loans", "cre_delinquency"]
        available_cre = [c for c in cre_cols if c in national_df.columns]

        if available_cre:
            # CRE metrics
            cre_metrics = st.columns(len(available_cre))
            for i, col_name in enumerate(available_cre):
                with cre_metrics[i]:
                    cur, prv = get_latest_valid(col_name)
                    if cur is not None:
                        current = cur[col_name]
                        previous = prv[col_name] if prv is not None else None
                        if col_name == "cre_loans":
                            val_str = f"${current:,.0f}B"
                            delta_pct = ((current - previous) / previous * 100) if previous is not None and pd.notna(previous) and previous != 0 else None
                            delta_str = f"{delta_pct:+.1f}%" if delta_pct else None
                            st.metric(format_indicator_name(col_name), val_str, delta_str)
                        else:  # delinquency
                            val_str = f"{current:.2f}%"
                            delta = current - previous if previous is not None and pd.notna(previous) else None
                            delta_str = f"{delta:+.2f}%" if delta else None
                            st.metric(format_indicator_name(col_name), val_str, delta_str, delta_color="inverse")

            col1, col2 = st.columns(2)

            # CRE loans outstanding
            if "cre_loans" in national_df.columns:
                with col1:
                    loans_df = national_df[["date", "cre_loans"]].dropna()
                    fig = px.area(
                        loans_df,
                        x="date",
                        y="cre_loans",
                        title="CRE Loans Outstanding (All Commercial Banks)",
                        labels={"cre_loans": "Billions ($)", "date": "Date"}
                    )
                    fig.update_traces(fillcolor="rgba(99, 110, 250, 0.3)", line_color="#636EFA")
                    fig.update_layout(hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)

            # CRE delinquency
            if "cre_delinquency" in national_df.columns:
                with col2:
                    delq_df = national_df[["date", "cre_delinquency"]].dropna()

                    fig = px.area(
                        delq_df,
                        x="date",
                        y="cre_delinquency",
                        title="CRE Loan Delinquency Rate",
                        labels={"cre_delinquency": "Rate (%)", "date": "Date"}
                    )
                    fig.update_traces(fillcolor="rgba(239, 85, 59, 0.3)", line_color="#EF553B")
                    # Add warning threshold
                    fig.add_hline(y=2, line_dash="dash", line_color="orange", annotation_text="Elevated Risk")
                    fig.add_hline(y=5, line_dash="dash", line_color="red", annotation_text="Distress")
                    fig.update_layout(hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)

            st.info("💡 **Interpretation:** Rising delinquency rates may signal distressed asset opportunities, "
                    "while declining CRE loan growth can indicate tighter lending standards.")
            st.caption("⚠️ CRE loans shown reflect only commercial bank balance sheets (FDIC-reported). "
                       "Significant CRE debt held by CMBS, life insurers, debt funds, and private lenders is not captured here. "
                       "Additionally, the CRE market faces substantial near-term maturity walls — "
                       "many CRE firms are actively working through refinancing as large volumes of loans come due.")
        else:
            st.info("No CRE credit data available. Click 'Refresh Data' to fetch.")

st.markdown("---")

# =============================================================================
# FOOTER
# =============================================================================

with st.expander("ℹ️ About This Dashboard"):
    st.markdown("""
    **Data Sources:**
    - **FRED** (Federal Reserve Economic Data): Employment, housing prices, interest rates, economic indicators
    - **YFinance**: REIT prices for sector sentiment analysis

    **Methodology by Tab:**
    - **Market Rankings** — Composite scoring from FRED fundamentals (HPI growth, unemployment, population) with REIT sentiment adjustment
    - **Metro Forecast** — Prophet time-series model for individual metro HPI and unemployment projections
    - **REIT Sentiment** — Sector-level price momentum from publicly traded REITs, weighted by CRE relevance
    - **Value Opportunities** — Fundamentals-first ranking with adjustable sentiment overlay to identify potential entry points
    - **Economic Overview** — National economic indicators tracked across interest rates, construction, consumer health, labor, and CRE credit

    **Metros Covered:** 20 major CRE markets across 6 regions (Northeast, Southeast, Midwest, Southwest, West, Mid-Atlantic)

    **Note:** This uses publicly available economic data. For direct CRE metrics (cap rates, vacancy, rent growth),
    consider integrating CoStar, CBRE, or similar commercial data providers.
    """)
