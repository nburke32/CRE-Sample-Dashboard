"""
Configuration for data fetching and storage.
Contains metro definitions, FRED series IDs, and REIT tickers.
"""

import os
from pathlib import Path

# =============================================================================
# STORAGE CONFIGURATION
# =============================================================================

# Storage backend: "parquet" (local) or "snowflake" (cloud)
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "parquet")

# Local parquet storage path
DATA_STORE_PATH = Path(__file__).parent.parent / "data_store"

# =============================================================================
# TOP 20 CRE METROS
# =============================================================================
# Major commercial real estate markets in the US
# Each has verified FRED series IDs for unemployment, employment, and HPI

METROS = {
    "NYC": {
        "name": "New York-Newark-Jersey City",
        "unemployment": "NEWY636URN",
        "hpi": "ATNHPIUS35614Q",
        "population": "NYTPOP",
    },
    "LAX": {
        "name": "Los Angeles-Long Beach-Anaheim",
        "unemployment": "LOSA106URN",
        "hpi": "ATNHPIUS31084Q",
        "population": "LNAPOP",
    },
    "CHI": {
        "name": "Chicago-Naperville-Elgin",
        "unemployment": "CHIC917URN",
        "hpi": "ATNHPIUS16984Q",
        "population": "CHIPOP",
    },
    "DFW": {
        "name": "Dallas-Fort Worth-Arlington",
        "unemployment": "DALL148URN",
        "hpi": "ATNHPIUS19124Q",
        "population": "DFWPOP",
    },
    "HOU": {
        "name": "Houston-The Woodlands-Sugar Land",
        "unemployment": "HOUS448URN",
        "hpi": "ATNHPIUS26420Q",
        "population": "HTNPOP",
    },
    "WAS": {
        "name": "Washington-Arlington-Alexandria",
        "unemployment": "WASH911URN",
        "hpi": "ATNHPIUS47894Q",
        "population": "WSHPOP",
    },
    "MIA": {
        "name": "Miami-Fort Lauderdale-Pompano Beach",
        "unemployment": "MIAM112URN",
        "hpi": "ATNHPIUS33124Q",
        "population": "MIMPOP",
    },
    "PHI": {
        "name": "Philadelphia-Camden-Wilmington",
        "unemployment": "PHIL942URN",
        "hpi": "ATNHPIUS37964Q",
        "population": "PCWPOP",
    },
    "ATL": {
        "name": "Atlanta-Sandy Springs-Alpharetta",
        "unemployment": "ATLA013URN",
        "hpi": "ATNHPIUS12060Q",
        "population": "ATLPOP",
    },
    "PHX": {
        "name": "Phoenix-Mesa-Chandler",
        "unemployment": "PHOE004URN",
        "hpi": "ATNHPIUS38060Q",
        "population": "PHXPOP",
    },
    "BOS": {
        "name": "Boston-Cambridge-Newton",
        "unemployment": "BOST625URN",
        "hpi": "ATNHPIUS14454Q",
        "population": "BOSPOP",
    },
    "SFO": {
        "name": "San Francisco-Oakland-Berkeley",
        "unemployment": "SANF806URN",
        "hpi": "ATNHPIUS41884Q",
        "population": "SFCPOP",
    },
    "SEA": {
        "name": "Seattle-Tacoma-Bellevue",
        "unemployment": "SEAT653URN",
        "hpi": "ATNHPIUS42644Q",
        "population": "STWPOP",
    },
    "DEN": {
        "name": "Denver-Aurora-Lakewood",
        "unemployment": "DENV708URN",
        "hpi": "ATNHPIUS19740Q",
        "population": "DNVPOP",
    },
    "SAN": {
        "name": "San Diego-Chula Vista-Carlsbad",
        "unemployment": "SAND706URN",
        "hpi": "ATNHPIUS41740Q",
        "population": "SDIPOP",
    },
    "AUS": {
        "name": "Austin-Round Rock-Georgetown",
        "unemployment": "AUST448URN",
        "hpi": "ATNHPIUS12420Q",
        "population": "AUSPOP",
    },
    "NSH": {
        "name": "Nashville-Davidson-Murfreesboro",
        "unemployment": "NASH947URN",
        "hpi": "ATNHPIUS34980Q",
        "population": "NVLPOP",
    },
    "CLT": {
        "name": "Charlotte-Concord-Gastonia",
        "unemployment": "CHAR737URN",
        "hpi": "ATNHPIUS16740Q",
        "population": "CGRPOP",
    },
    "RDU": {
        "name": "Raleigh-Cary",
        "unemployment": "RALE537URN",
        "hpi": "ATNHPIUS39580Q",
        "population": "RCYPOP",
    },
    "MSP": {
        "name": "Minneapolis-St. Paul-Bloomington",
        "unemployment": "MINN427URN",
        "hpi": "ATNHPIUS33460Q",
        "population": "MSPPOP",
    },
}

# =============================================================================
# FRED SERIES - NATIONAL INDICATORS
# =============================================================================

FRED_SERIES = {
    "national": {
        # Interest Rates (critical for CRE valuations)
        "treasury_10y": "DGS10",                    # 10-Year Treasury Rate (cap rate driver)
        "treasury_2y": "DGS2",                      # 2-Year Treasury (yield curve)
        "fed_funds": "FEDFUNDS",                    # Federal Funds Rate
        "mortgage_30y": "MORTGAGE30US",             # 30-Year Mortgage Rate

        # Inflation & Growth
        "cpi": "CPIAUCSL",                          # Consumer Price Index (inflation)
        "gdp": "GDP",                               # Gross Domestic Product (quarterly)

        # Labor Market
        "unemployment_national": "UNRATE",          # National Unemployment Rate
        "payrolls": "PAYEMS",                       # Total Nonfarm Payrolls (thousands)

        # Consumer Health (retail CRE demand driver)
        "retail_sales": "RSAFS",                    # Advance Retail Sales (millions)
        "consumer_sentiment": "UMCSENT",            # U of Michigan Consumer Sentiment

        # Construction & Development
        "housing_starts": "HOUST",                  # Housing Starts (thousands, SAAR)
        "building_permits": "PERMIT",               # Building Permits (thousands, SAAR)
        "construction_spending": "TTLCONS",         # Total Construction Spending (millions)
        "commercial_construction": "TLCOMCONS",     # Commercial Construction Spending (millions)

        # CRE Credit Conditions
        "cre_loans": "CREACBM027NBOG",              # Commercial Real Estate Loans, All Banks (millions)
        "cre_delinquency": "DRCRELEXFACBS",         # CRE Loan Delinquency Rate (%)

        # Industrial Production (industrial CRE demand)
        "industrial_production": "INDPRO",          # Industrial Production Index
    },
}

# =============================================================================
# REIT TICKERS - CRE SECTOR SENTIMENT
# =============================================================================
# REITs provide real-time market sentiment for CRE sectors

REIT_TICKERS = {
    # Broad Market REITs
    "VNQ": {"name": "Vanguard Real Estate ETF", "sector": "broad"},
    "IYR": {"name": "iShares U.S. Real Estate ETF", "sector": "broad"},

    # Office - Gateway/Coastal Markets
    "BXP": {"name": "Boston Properties", "sector": "office"},
    "VNO": {"name": "Vornado Realty Trust", "sector": "office"},
    "SLG": {"name": "SL Green Realty", "sector": "office"},

    # Office - Sunbelt Markets
    "CUZ": {"name": "Cousins Properties", "sector": "office"},
    "HIW": {"name": "Highwoods Properties", "sector": "office"},

    # Industrial/Logistics
    "PLD": {"name": "Prologis", "sector": "industrial"},

    # Retail - Malls
    "SPG": {"name": "Simon Property Group", "sector": "retail_mall"},

    # Retail - Open Air (grocery-anchored, net lease, shopping centers)
    "O": {"name": "Realty Income", "sector": "retail_openair"},
    "REG": {"name": "Regency Centers", "sector": "retail_openair"},
    "KIM": {"name": "Kimco Realty", "sector": "retail_openair"},

    # Multifamily/Residential
    "EQR": {"name": "Equity Residential", "sector": "multifamily"},
    "AVB": {"name": "AvalonBay Communities", "sector": "multifamily"},
    "MAA": {"name": "Mid-America Apartment", "sector": "multifamily"},

    # Data Centers
    "EQIX": {"name": "Equinix", "sector": "data_center"},
    "DLR": {"name": "Digital Realty", "sector": "data_center"},

    # Homebuilders (leading indicator)
    "XHB": {"name": "SPDR Homebuilders ETF", "sector": "homebuilders"},
    "DHI": {"name": "D.R. Horton", "sector": "homebuilders"},
    "LEN": {"name": "Lennar", "sector": "homebuilders"},
}

# =============================================================================
# DATA REFRESH SETTINGS
# =============================================================================

# How often to refresh data (in hours)
REFRESH_INTERVAL_HOURS = 24

# Historical data lookback (in years)
HISTORICAL_YEARS = 10
