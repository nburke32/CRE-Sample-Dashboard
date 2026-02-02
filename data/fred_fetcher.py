"""
FRED API data fetcher for economic indicators.
Fetches national and metro-level data relevant to CRE analysis.
"""

import os
from datetime import datetime, timedelta

import pandas as pd
from dotenv import load_dotenv

from .config import FRED_FETCH_TIMEOUT, FRED_SERIES, HISTORICAL_YEARS, METROS
from .storage import DataStorage

load_dotenv()


class FREDFetcher:
    """
    Fetches economic data from FRED API.
    Handles both national and metro-specific series.
    """

    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        if not self.api_key:
            raise ValueError(
                "FRED API key not found. Set FRED_API_KEY in .env file or pass to constructor."
            )
        self.storage = DataStorage()

    def _fetch_series(self, series_id: str, start_date: str) -> pd.DataFrame | None:
        """
        Fetch a single FRED series.

        Args:
            series_id: FRED series ID
            start_date: Start date in YYYY-MM-DD format

        Returns:
            DataFrame with date and value columns, or None if error
        """
        import requests

        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start_date,
        }

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=FRED_FETCH_TIMEOUT)
            response.raise_for_status()
            data = response.json()

            if "observations" not in data:
                return None

            df = pd.DataFrame(data["observations"])
            if df.empty:
                return None

            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df[["date", "value"]].dropna()

            return df

        except Exception as e:
            print(f"Error fetching {series_id}: {e}")
            return None

    def fetch_national_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """Fetch all national economic indicators."""
        if not force_refresh and self.storage.dataset_exists("fred_national"):
            last_updated = self.storage.get_last_updated("fred_national")
            if last_updated and (datetime.now() - last_updated).days < 1:
                return self.storage.load_dataframe("fred_national")

        start_date = (datetime.now() - timedelta(days=365 * HISTORICAL_YEARS)).strftime(
            "%Y-%m-%d"
        )

        all_data = []

        for indicator, series_id in FRED_SERIES["national"].items():
            print(f"Fetching {indicator}...")
            df = self._fetch_series(series_id, start_date)

            if df is not None and not df.empty:
                df["indicator"] = indicator
                df["series_id"] = series_id
                all_data.append(df)

        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)

        result_wide = result.pivot_table(
            index="date", columns="indicator", values="value", aggfunc="first"
        ).reset_index()

        self.storage.save_dataframe(result_wide, "fred_national")
        return result_wide

    def fetch_metro_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """Fetch economic indicators for all metros."""
        if not force_refresh and self.storage.dataset_exists("fred_metros"):
            last_updated = self.storage.get_last_updated("fred_metros")
            if last_updated and (datetime.now() - last_updated).days < 1:
                return self.storage.load_dataframe("fred_metros")

        start_date = (datetime.now() - timedelta(days=365 * HISTORICAL_YEARS)).strftime(
            "%Y-%m-%d"
        )

        all_metro_data = []

        indicators = ["unemployment", "hpi", "population"]

        for metro_code, metro_info in METROS.items():
            print(f"Fetching data for {metro_info['name']}...")
            metro_data = []

            for indicator in indicators:
                series_id = metro_info.get(indicator)
                if not series_id:
                    continue

                df = self._fetch_series(series_id, start_date)

                if df is not None and not df.empty:
                    df["indicator"] = indicator
                    metro_data.append(df)

            if metro_data:
                metro_df = pd.concat(metro_data, ignore_index=True)

                metro_wide = metro_df.pivot_table(
                    index="date", columns="indicator", values="value", aggfunc="first"
                ).reset_index()

                metro_wide["metro_code"] = metro_code
                metro_wide["metro_name"] = metro_info["name"]

                all_metro_data.append(metro_wide)

        if not all_metro_data:
            return pd.DataFrame()

        result = pd.concat(all_metro_data, ignore_index=True)

        cols = ["date", "metro_code", "metro_name"] + [
            c for c in result.columns if c not in ["date", "metro_code", "metro_name"]
        ]
        result = result[cols]

        self.storage.save_dataframe(result, "fred_metros")
        return result

    def fetch_all(self, force_refresh: bool = False) -> dict[str, pd.DataFrame]:
        """Fetch all FRED data (national + metros)."""
        return {
            "national": self.fetch_national_data(force_refresh),
            "metros": self.fetch_metro_data(force_refresh),
        }

    def get_metro_data(self, metro_code: str) -> pd.DataFrame | None:
        """Get data for a single metro from cache."""
        df = self.storage.load_dataframe("fred_metros", filters={"metro_code": metro_code})
        return df

    def get_national_data(self) -> pd.DataFrame | None:
        """Get national data from cache."""
        return self.storage.load_dataframe("fred_national")
