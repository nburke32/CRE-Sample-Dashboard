# CRE Analytics Dashboard

A professional portfolio project showcasing commercial real estate market analysis and predictive modeling capabilities. This interactive dashboard demonstrates proficiency in data engineering, financial modeling, and visualization.

## Overview

This Streamlit-based dashboard provides comprehensive commercial real estate market intelligence through:

- **Real-time Market Analytics**: NYC commercial property transaction data via NYC OpenData API
- **Economic Indicators**: Federal Reserve economic data (FRED API) integration
- **Predictive Modeling**: Market forecasting using Prophet and custom scoring algorithms
- **REIT Sentiment Analysis**: Multi-sector CRE sentiment tracking with time-travel capabilities

## Key Features

### 📊 Market Analytics
- 3,800+ real commercial property transactions from NYC OpenData
- Interactive filtering by borough, property type, price range, and date
- Transaction volume trends and price per square foot analysis
- Data export capabilities

### 📈 Economic Overview
- Live economic indicators: 10Y Treasury, 30Y Mortgage Rate, Unemployment, CRE Delinquency
- Historical trend analysis with automated data refresh
- Month-over-month change tracking

### 🔮 Predictive Modeling
- **Market Rankings**: Custom scoring system incorporating economic fundamentals and REIT sentiment
- **Metro Forecasting**: Prophet-based market predictions with confidence intervals
- **REIT Sentiment Deep Dive**: Sector-weighted sentiment analysis with visual breakdown
- **Value Opportunities**: Identify undervalued markets based on composite scoring

## Technical Stack

**Languages & Frameworks:**
- Python 3.x
- Streamlit (web framework)
- Pandas, NumPy (data manipulation)
- Plotly (interactive visualizations)
- Prophet (time series forecasting)

**Data Sources:**
- NYC OpenData (Socrata API)
- Federal Reserve Economic Data (FRED API)
- Custom REIT sector data

**Infrastructure:**
- Parquet-based caching for performance
- Environment-based configuration (.env)
- Modular architecture with dedicated fetchers and storage layers

## Project Structure

```
streamlit-dashboard/
├── Home.py                     # Landing page with economic overview
├── pages/
│   ├── 1_📊_Market_Analytics.py
│   ├── 4_🔮_Predictive_Modeling.py
│   └── ...
├── data/
│   ├── fred_fetcher.py         # FRED API integration
│   ├── nyc_opendata_fetcher.py # NYC OpenData integration
│   └── storage.py              # Data caching layer
├── models/
│   ├── prophet_forecast.py     # Time series forecasting
│   └── market_scoring.py       # Custom scoring algorithms
└── .streamlit/
    └── config.toml             # Theme and configuration
```

## Installation & Setup

1. Clone the repository:
```bash
git clone https://github.com/nburke32/CRE-Sample-Dashboard.git
cd streamlit-dashboard
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure API credentials in `.env`:
```bash
FRED_API_KEY=your_fred_api_key
NYC_OPENDATA_APP_TOKEN=your_nyc_token
```

4. Run the dashboard:
```bash
streamlit run Home.py
```

## Notable Implementation Details

- **Mixed-frequency Data Handling**: Robust handling of economic indicators with different update schedules (monthly vs quarterly)
- **Session State Management**: Advanced Streamlit patterns for widget resets and time-travel functionality
- **Error Handling**: Fallback mechanisms for API failures and missing data
- **Performance Optimization**: 24-hour caching strategy for commercial property data with manual refresh option
- **Data Quality**: Outlier filtering, null handling, and data validation for NYC property records

## Future Enhancements

This dashboard serves as a foundation for expanding into:
- **National Coverage**: Integration with Real Capital Analytics, CoStar, or Green Street Advisors
- **Cap Rate Analysis**: NOI data integration for yield calculations
- **Map Visualizations**: Geographic property distribution and heat maps
- **User-Specific API Keys**: Secure chatbot integration with user-provided credentials

## Development Notes

This is an ongoing portfolio project built with AI assistance (Claude Code). It demonstrates rapid prototyping capabilities while maintaining production-quality code standards. Most of my professional work exists in private/enterprise repositories.

## Contact

**Nolan Burke**
GitHub: [@nburke32](https://github.com/nburke32)
Repository: [CRE-Sample-Dashboard](https://github.com/nburke32/CRE-Sample-Dashboard)

## License

MIT

---

*Built with Streamlit • Powered by NYC OpenData & FRED*
