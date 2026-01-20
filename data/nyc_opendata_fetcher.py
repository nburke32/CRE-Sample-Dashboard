"""
NYC OpenData API client for fetching commercial property sales data.
Uses Socrata Open Data API (SODA) to access NYC property transaction records.
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from sodapy import Socrata
from dotenv import load_dotenv

from .storage import DataStorage

# Load environment variables
load_dotenv()

# NYC OpenData configuration
NYC_OPENDATA_DOMAIN = "data.cityofnewyork.us"
PROPERTY_SALES_DATASET_ID = "usep-8jbt"  # NYC Citywide Rolling Calendar Sales


class NYCOpenDataFetcher:
    """
    Fetcher for NYC OpenData property sales via Socrata API.

    Datasets:
    - NYC Citywide Rolling Calendar Sales (usep-8jbt)
    - Includes commercial and residential property transactions
    """

    def __init__(self):
        self.app_token = os.getenv("NYC_OPENDATA_APP_TOKEN")
        self.app_secret = os.getenv("NYC_OPENDATA_APP_SECRET")
        self.storage = DataStorage()

        # Initialize Socrata client
        # Note: sodapy only uses app_token for authentication, not the secret
        # The secret is used for OAuth flows, which we don't need for read-only access
        if self.app_token:
            try:
                self.client = Socrata(
                    NYC_OPENDATA_DOMAIN,
                    self.app_token,
                    timeout=30
                )
            except Exception as e:
                print(f"⚠️ Error initializing with app token: {e}")
                print("Falling back to unauthenticated access (throttled)")
                self.client = Socrata(NYC_OPENDATA_DOMAIN, None, timeout=30)
        else:
            self.client = Socrata(NYC_OPENDATA_DOMAIN, None, timeout=30)
            print("⚠️ No NYC OpenData app token found. API requests will be throttled.")

    def fetch_property_sales(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 50000,
        commercial_only: bool = True,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Fetch NYC property sales data.

        Args:
            start_date: Start date in YYYY-MM-DD format (default: 18 months ago)
            end_date: End date in YYYY-MM-DD format (default: today)
            limit: Maximum records to fetch (default: 50000)
            commercial_only: Filter for commercial properties only (tax_class = 4)
            force_refresh: Force refresh from API even if cached

        Returns:
            DataFrame with property sales data
        """
        # Check cache first
        if not force_refresh and self.storage.dataset_exists("nyc_property_sales"):
            print("Loading NYC property sales from cache...")
            cached_df = self.storage.load_dataframe("nyc_property_sales")
            if cached_df is not None and not cached_df.empty:
                return cached_df

        print("Fetching NYC property sales from Socrata API...")

        # Default date range: last 24 months (commercial sales are less frequent)
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

        # Build SoQL query
        where_clauses = [
            f"sale_date >= '{start_date}'",
            f"sale_date <= '{end_date}'",
            "sale_price > 0"  # Exclude zero-dollar sales (gifts, errors)
        ]

        if commercial_only:
            # Tax class 4 = Commercial & Industrial
            where_clauses.append("tax_class_at_time_of_sale = 4")

        where_query = " AND ".join(where_clauses)

        try:
            # Fetch data using Socrata client
            results = self.client.get(
                PROPERTY_SALES_DATASET_ID,
                where=where_query,
                limit=limit,
                order="sale_date DESC"
            )

            if not results:
                print("⚠️ No results returned from API")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame.from_records(results)

            # Data cleaning and type conversions
            df = self._clean_property_data(df)

            # Save to cache
            self.storage.save_dataframe(df, "nyc_property_sales")
            print(f"✅ Fetched {len(df)} property sales records")

            return df

        except Exception as e:
            print(f"❌ Error fetching NYC property sales: {e}")
            return pd.DataFrame()

    def _clean_property_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize property sales data."""
        if df.empty:
            return df

        # Convert date columns
        if "sale_date" in df.columns:
            df["sale_date"] = pd.to_datetime(df["sale_date"])

        # Convert numeric columns
        numeric_cols = {
            "sale_price": float,
            "gross_square_feet": float,
            "land_square_feet": float,
            "residential_units": float,
            "commercial_units": float,
            "total_units": float,
            "lot": float
        }

        for col, dtype in numeric_cols.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Calculate price per square foot
        if "sale_price" in df.columns and "gross_square_feet" in df.columns:
            df["price_per_sqft"] = df["sale_price"] / df["gross_square_feet"]
            # Remove outliers (likely data errors)
            df.loc[df["price_per_sqft"] > 5000, "price_per_sqft"] = None
            df.loc[df["price_per_sqft"] < 10, "price_per_sqft"] = None

        # Standardize borough names
        if "borough" in df.columns:
            borough_map = {
                "1": "Manhattan",
                "2": "Bronx",
                "3": "Brooklyn",
                "4": "Queens",
                "5": "Staten Island"
            }
            df["borough"] = df["borough"].map(borough_map).fillna(df["borough"])

        # Map building class to readable categories
        if "building_class_category" in df.columns:
            df["property_type"] = df["building_class_category"].apply(
                self._categorize_property_type
            )

        # Remove clearly invalid sales (likely data errors)
        if "sale_price" in df.columns:
            # Filter out suspiciously low sales (< $10k likely errors/non-arms-length)
            df = df[df["sale_price"] >= 10000].copy()

        # Drop rows with missing critical data
        df = df.dropna(subset=["sale_price", "sale_date"])

        return df

    def _categorize_property_type(self, building_class_category: str) -> str:
        """Map building class categories to simplified property types."""
        if pd.isna(building_class_category):
            return "Other"

        category_lower = building_class_category.lower()

        if "office" in category_lower:
            return "Office"
        elif "retail" in category_lower or "store" in category_lower:
            return "Retail"
        elif "warehouse" in category_lower or "industrial" in category_lower or "factory" in category_lower:
            return "Industrial"
        elif "hotel" in category_lower:
            return "Hotel"
        elif "apartment" in category_lower or "family" in category_lower:
            return "Multifamily"
        elif "garage" in category_lower or "parking" in category_lower:
            return "Parking"
        else:
            return "Mixed Use"

    def fetch_building_data(
        self,
        limit: int = 10000,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Fetch building characteristics data (for enrichment).

        Dataset: Property Data (Buildings Information System)
        Note: This is a large dataset, so we limit records by default.

        Args:
            limit: Maximum records to fetch
            force_refresh: Force refresh from API

        Returns:
            DataFrame with building characteristics
        """
        # Check cache first
        if not force_refresh and self.storage.dataset_exists("nyc_building_data"):
            print("Loading NYC building data from cache...")
            cached_df = self.storage.load_dataframe("nyc_building_data")
            if cached_df is not None and not cached_df.empty:
                return cached_df

        print("Fetching NYC building data from Socrata API...")

        # Building Information System dataset ID
        bis_dataset_id = "e98g-f8hy"

        try:
            # Fetch recent building data
            results = self.client.get(
                bis_dataset_id,
                limit=limit,
                where="1=1"  # Get all records (up to limit)
            )

            if not results:
                print("⚠️ No building data returned")
                return pd.DataFrame()

            df = pd.DataFrame.from_records(results)

            # Save to cache
            self.storage.save_dataframe(df, "nyc_building_data")
            print(f"✅ Fetched {len(df)} building records")

            return df

        except Exception as e:
            print(f"❌ Error fetching building data: {e}")
            return pd.DataFrame()

    def get_last_updated(self, dataset_name: str) -> Optional[datetime]:
        """Get last update timestamp for a cached dataset."""
        return self.storage.get_last_updated(dataset_name)

    def __del__(self):
        """Close Socrata client connection."""
        if hasattr(self, 'client'):
            self.client.close()


# Convenience functions for easy imports
def fetch_nyc_property_sales(force_refresh: bool = False) -> pd.DataFrame:
    """Convenience function to fetch NYC property sales."""
    fetcher = NYCOpenDataFetcher()
    return fetcher.fetch_property_sales(force_refresh=force_refresh)


def fetch_nyc_building_data(force_refresh: bool = False) -> pd.DataFrame:
    """Convenience function to fetch NYC building data."""
    fetcher = NYCOpenDataFetcher()
    return fetcher.fetch_building_data(force_refresh=force_refresh)
