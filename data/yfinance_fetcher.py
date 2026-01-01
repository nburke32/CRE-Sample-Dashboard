"""
YFinance data fetcher for REIT and homebuilder sentiment.
Provides real-time market sentiment indicators for CRE sectors.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from .config import REIT_TICKERS, HISTORICAL_YEARS
from .storage import DataStorage


class YFinanceFetcher:
    """
    Fetches REIT and homebuilder stock data from Yahoo Finance.
    Used as a sentiment/leading indicator for CRE markets.
    """

    def __init__(self):
        self.storage = DataStorage()

    def fetch_reit_prices(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch historical prices for all REIT tickers.

        Returns:
            DataFrame with columns: date, ticker, name, sector, close, volume, pct_change
        """
        import yfinance as yf

        # Check cache
        if not force_refresh and self.storage.dataset_exists("reit_prices"):
            last_updated = self.storage.get_last_updated("reit_prices")
            if last_updated and (datetime.now() - last_updated).total_seconds() < 4 * 3600:
                return self.storage.load_dataframe("reit_prices")

        start_date = datetime.now() - timedelta(days=365 * HISTORICAL_YEARS)
        tickers = list(REIT_TICKERS.keys())

        print(f"Fetching REIT data for {len(tickers)} tickers...")

        # Download all at once for efficiency
        data = yf.download(
            tickers,
            start=start_date,
            end=datetime.now(),
            group_by="ticker",
            auto_adjust=True,
            progress=False,
        )

        all_data = []

        for ticker in tickers:
            try:
                if len(tickers) > 1:
                    ticker_data = data[ticker].copy()
                else:
                    ticker_data = data.copy()

                ticker_data = ticker_data.reset_index()
                ticker_data = ticker_data.rename(columns={"Date": "date", "Close": "close", "Volume": "volume"})
                ticker_data = ticker_data[["date", "close", "volume"]].dropna()

                ticker_data["ticker"] = ticker
                ticker_data["name"] = REIT_TICKERS[ticker]["name"]
                ticker_data["sector"] = REIT_TICKERS[ticker]["sector"]
                ticker_data["pct_change"] = ticker_data["close"].pct_change() * 100

                all_data.append(ticker_data)

            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                continue

        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)

        # Reorder columns
        result = result[["date", "ticker", "name", "sector", "close", "volume", "pct_change"]]

        self.storage.save_dataframe(result, "reit_prices")
        return result

    def fetch_sector_indices(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Calculate sector-level indices from individual REIT prices.

        Returns:
            DataFrame with columns: date, sector, avg_price, total_volume, avg_pct_change
        """
        prices = self.fetch_reit_prices(force_refresh)

        if prices.empty:
            return pd.DataFrame()

        # Group by date and sector
        sector_data = (
            prices.groupby(["date", "sector"])
            .agg(
                avg_close=("close", "mean"),
                total_volume=("volume", "sum"),
                avg_pct_change=("pct_change", "mean"),
                ticker_count=("ticker", "count"),
            )
            .reset_index()
        )

        # Normalize to index (100 = first date for each sector)
        sector_data = sector_data.sort_values(["sector", "date"])

        def normalize_to_index(group):
            first_price = group["avg_close"].iloc[0]
            group["sector_index"] = (group["avg_close"] / first_price) * 100
            return group

        sector_data = sector_data.groupby("sector").apply(normalize_to_index).reset_index(drop=True)

        self.storage.save_dataframe(sector_data, "reit_sector_indices")
        return sector_data

    def get_current_sentiment(self) -> pd.DataFrame:
        """
        Get current market sentiment summary.

        Returns:
            DataFrame with current sector performance metrics
        """
        prices = self.storage.load_dataframe("reit_prices")

        if prices is None or prices.empty:
            prices = self.fetch_reit_prices()

        if prices.empty:
            return pd.DataFrame()

        # Get latest date
        latest_date = prices["date"].max()

        # Calculate performance metrics
        results = []

        for sector in prices["sector"].unique():
            sector_data = prices[prices["sector"] == sector].sort_values("date")

            current = sector_data[sector_data["date"] == latest_date]

            if current.empty:
                continue

            # Calculate returns over different periods
            dates_1w = latest_date - timedelta(days=7)
            dates_1m = latest_date - timedelta(days=30)
            dates_3m = latest_date - timedelta(days=90)
            dates_1y = latest_date - timedelta(days=365)

            def calc_return(start_date):
                start_data = sector_data[sector_data["date"] >= start_date].head(1)
                if start_data.empty or current.empty:
                    return None
                start_price = start_data["close"].mean()
                current_price = current["close"].mean()
                return ((current_price - start_price) / start_price) * 100

            results.append(
                {
                    "sector": sector,
                    "current_avg_price": current["close"].mean(),
                    "return_1w": calc_return(dates_1w),
                    "return_1m": calc_return(dates_1m),
                    "return_3m": calc_return(dates_3m),
                    "return_1y": calc_return(dates_1y),
                    "avg_volume": current["volume"].mean(),
                }
            )

        return pd.DataFrame(results)

    def get_ticker_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Get data for a single ticker from cache.

        Args:
            ticker: Stock ticker (e.g., "VNQ", "PLD")

        Returns:
            DataFrame for that ticker or None
        """
        return self.storage.load_dataframe("reit_prices", filters={"ticker": ticker})

    def get_sector_data(self, sector: str) -> Optional[pd.DataFrame]:
        """
        Get data for a single sector from cache.

        Args:
            sector: Sector name (e.g., "office", "industrial")

        Returns:
            DataFrame for that sector or None
        """
        return self.storage.load_dataframe("reit_prices", filters={"sector": sector})
