# CRE Analytics Dashboard

A professional portfolio project showcasing commercial real estate market analysis and predictive modeling capabilities. This interactive dashboard demonstrates proficiency in data engineering, financial modeling, and visualization.

## Overview

This Streamlit-based dashboard provides comprehensive commercial real estate market intelligence through:

- **Real-time Market Analytics**: NYC commercial property transaction data via NYC OpenData API
- **Economic Indicators**: Federal Reserve economic data (FRED API) integration
- **Predictive Modeling**: Market forecasting using Prophet and custom scoring algorithms
- **REIT Sentiment Analysis**: Multi-sector CRE sentiment tracking with time-travel capabilities
- **SEC Filing Chatbot**: AI-powered Q&A on 10-K/10-Q filings via SEC Edgar API + Claude

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

### 🤖 SEC Filing Chatbot
- Natural language Q&A on SEC filings (10-K annual, 10-Q quarterly)
- 9 curated companies: REITs (PLD, EQIX, DLR, SPG, O) + Tech (AAPL, MSFT, AMZN, GOOGL)
- SEC Edgar API integration with rate limiting and local caching
- Claude AI analysis with model selection, cost tracking, and response time logging

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
- SEC Edgar API (10-K/10-Q filings)
- Yahoo Finance (REIT price data)
- Anthropic Claude API (filing analysis)

**Infrastructure:**
- Parquet-based caching for performance
- Environment-based configuration (.env)
- Modular architecture with dedicated fetchers and storage layers

## Project Structure

```
streamlit-dashboard/
├── Home.py                        # Landing page with economic overview
├── pages/
│   ├── 1_📊_Market_Analytics.py   # NYC property transaction analysis
│   ├── 2_🗺️_Map_View.py           # Interactive map visualization
│   ├── 3_🤖_SEC_Chatbot.py        # SEC filing Q&A with Claude AI
│   └── 4_🔮_Predictive_Modeling.py # Market forecasting and scoring
├── data/
│   ├── fred_fetcher.py            # FRED API integration
│   ├── nyc_opendata_fetcher.py    # NYC OpenData integration
│   ├── sec_fetcher.py             # SEC Edgar API integration
│   ├── yfinance_fetcher.py        # REIT price data
│   └── storage.py                 # Data caching layer
├── models/
│   ├── prophet_forecast.py        # Time series forecasting
│   ├── market_scoring.py          # Custom scoring algorithms
│   └── sec_chatbot.py             # Claude AI chatbot engine
└── .streamlit/
    └── config.toml                # Theme and configuration
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

## Security & Production Considerations

This project uses `.env` files and Streamlit secrets for local development, which is fine for a personal portfolio demo. Every production environment is different, but here are the kinds of things that change when moving beyond local development:

**Secrets Management:**
- **Local/Personal**: `.env` files, Streamlit `secrets.toml` (gitignored)
- **Enterprise Development**: OS keyring, Azure Key Vault, or team-managed secret stores — even on a developer's machine, credentials shouldn't sit in plaintext files
- **Production**: Azure Key Vault, AWS Secrets Manager, GCP Secret Manager — secrets are injected at runtime, never stored alongside code

**CI/CD & Deployment:**
- GitHub Actions or similar pipelines for automated testing and deployment
- Secrets passed as encrypted environment variables in the pipeline, not committed
- Container-based deployment (Docker) with secrets mounted at runtime

**Access Control:**
- API keys scoped with minimal permissions and spend limits
- User authentication and session management (not needed for a portfolio demo, critical in production)
- Audit logging for compliance-sensitive data like SEC filings

**What This Project Does Right (Even as a Demo):**
- `.env` and `secrets.toml` are gitignored — credentials never hit the repo
- `.env.example` and `secrets.toml.example` document what's needed without exposing values
- API usage is logged with cost tracking
- SEC rate limits are enforced in code

The point is that security practices scale with the environment. A personal project, a team repo, and a production deployment each have different requirements — but the habit of keeping secrets out of source control applies to all of them.

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
