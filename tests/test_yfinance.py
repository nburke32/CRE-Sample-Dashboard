"""
Tests for data/yfinance_fetcher.py — mock downloads, sentiment calculation, sector indices.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from data.yfinance_fetcher import YFinanceFetcher


class TestSentimentCalculation:
    """Test get_current_sentiment logic with pre-loaded data."""

    @pytest.fixture
    def fetcher_with_data(self):
        """Create fetcher with mock REIT price data in storage."""
        fetcher = YFinanceFetcher()

        # Build realistic price data for 2 sectors over 400 days
        dates = pd.date_range(end=datetime.now(), periods=400, freq="D")
        data = []

        for sector, tickers in [("office", ["BXP", "VNO"]), ("industrial", ["PLD"])]:
            for ticker in tickers:
                base_price = 100 + np.random.randn() * 10
                prices = base_price + np.cumsum(np.random.randn(len(dates)) * 0.5)
                for i, date in enumerate(dates):
                    data.append({
                        "date": date,
                        "ticker": ticker,
                        "name": f"{ticker} Inc",
                        "sector": sector,
                        "close": prices[i],
                        "volume": int(1e6 + np.random.randint(0, 5e5)),
                        "pct_change": np.random.randn() * 2,
                    })

        df = pd.DataFrame(data)
        fetcher.storage.save_dataframe(df, "reit_prices")
        yield fetcher
        fetcher.storage.delete_dataset("reit_prices")

    def test_sentiment_returns_dataframe(self, fetcher_with_data):
        """get_current_sentiment should return a DataFrame."""
        result = fetcher_with_data.get_current_sentiment()
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_sentiment_has_return_columns(self, fetcher_with_data):
        """Should have return columns for different time periods."""
        result = fetcher_with_data.get_current_sentiment()
        expected_cols = {"sector", "current_avg_price", "return_1w", "return_1m", "return_3m", "return_1y"}
        assert expected_cols.issubset(set(result.columns))

    def test_sentiment_has_both_sectors(self, fetcher_with_data):
        """Should have rows for each sector in the data."""
        result = fetcher_with_data.get_current_sentiment()
        assert set(result["sector"]) == {"office", "industrial"}

    def test_returns_are_numeric(self, fetcher_with_data):
        """Return values should be numeric (float)."""
        result = fetcher_with_data.get_current_sentiment()
        for col in ["return_1w", "return_1m", "return_3m"]:
            # Some may be None but those that exist should be numeric
            non_null = result[col].dropna()
            if not non_null.empty:
                assert pd.api.types.is_numeric_dtype(non_null)


class TestSectorIndices:
    """Test sector index normalization logic."""

    @pytest.fixture
    def fetcher_with_prices(self):
        """Create fetcher with minimal price data."""
        fetcher = YFinanceFetcher()

        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        data = []
        for date in dates:
            data.append({
                "date": date, "ticker": "BXP", "name": "Boston Properties",
                "sector": "office", "close": 100 + len(data) * 0.5,
                "volume": 1000000, "pct_change": 0.5,
            })
            data.append({
                "date": date, "ticker": "PLD", "name": "Prologis",
                "sector": "industrial", "close": 200 + len(data) * 0.3,
                "volume": 2000000, "pct_change": 0.3,
            })

        df = pd.DataFrame(data)
        fetcher.storage.save_dataframe(df, "reit_prices")
        yield fetcher
        fetcher.storage.delete_dataset("reit_prices")
        fetcher.storage.delete_dataset("reit_sector_indices")

    def test_sector_indices_returns_dataframe(self, fetcher_with_prices):
        """fetch_sector_indices should return a DataFrame."""
        result = fetcher_with_prices.fetch_sector_indices()
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_sector_index_starts_at_100(self, fetcher_with_prices):
        """Normalized sector index should start at 100."""
        result = fetcher_with_prices.fetch_sector_indices()
        for sector in result["sector"].unique():
            sector_data = result[result["sector"] == sector].sort_values("date")
            first_index = sector_data.iloc[0]["sector_index"]
            assert abs(first_index - 100.0) < 0.01

    def test_sector_indices_has_expected_columns(self, fetcher_with_prices):
        """Should have avg_close, total_volume, avg_pct_change, sector_index."""
        result = fetcher_with_prices.fetch_sector_indices()
        expected = {"date", "sector", "avg_close", "total_volume", "avg_pct_change", "sector_index"}
        assert expected.issubset(set(result.columns))


class TestTickerAndSectorFiltering:
    """Test get_ticker_data and get_sector_data convenience methods."""

    @pytest.fixture
    def fetcher_with_prices(self):
        fetcher = YFinanceFetcher()
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5, freq="D").tolist() * 2,
            "ticker": ["BXP"] * 5 + ["PLD"] * 5,
            "name": ["Boston Properties"] * 5 + ["Prologis"] * 5,
            "sector": ["office"] * 5 + ["industrial"] * 5,
            "close": [100.0] * 10,
            "volume": [1000000] * 10,
            "pct_change": [0.5] * 10,
        })
        fetcher.storage.save_dataframe(df, "reit_prices")
        yield fetcher
        fetcher.storage.delete_dataset("reit_prices")

    def test_get_ticker_data_filters_correctly(self, fetcher_with_prices):
        """get_ticker_data should return only rows for the specified ticker."""
        result = fetcher_with_prices.get_ticker_data("BXP")
        assert result is not None
        assert (result["ticker"] == "BXP").all()
        assert len(result) == 5

    def test_get_sector_data_filters_correctly(self, fetcher_with_prices):
        """get_sector_data should return only rows for the specified sector."""
        result = fetcher_with_prices.get_sector_data("industrial")
        assert result is not None
        assert (result["sector"] == "industrial").all()
        assert len(result) == 5

    def test_get_ticker_data_nonexistent(self, fetcher_with_prices):
        """get_ticker_data for a nonexistent ticker should return empty."""
        result = fetcher_with_prices.get_ticker_data("FAKE")
        assert result is not None
        assert len(result) == 0


class TestCacheBehavior:
    """Test cache hit/miss for REIT price fetching."""

    def test_cache_hit_avoids_download(self):
        """When cache is fresh (< 4 hours), should not call yf.download."""
        fetcher = YFinanceFetcher()

        # Save some data and set it as recent
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "ticker": ["VNQ"] * 5,
            "name": ["Vanguard RE ETF"] * 5,
            "sector": ["broad"] * 5,
            "close": [100.0] * 5,
            "volume": [1e6] * 5,
            "pct_change": [0.1] * 5,
        })
        fetcher.storage.save_dataframe(df, "reit_prices")

        # This should return cached data without downloading
        result = fetcher.fetch_reit_prices(force_refresh=False)
        assert len(result) == 5

        # Cleanup
        fetcher.storage.delete_dataset("reit_prices")

    def test_empty_data_returns_empty_sentiment(self):
        """get_current_sentiment with no data should return empty DataFrame."""
        from unittest.mock import patch

        fetcher = YFinanceFetcher()

        # Mock both the storage load (which checks cache + seed) and fetch
        with patch.object(fetcher.storage, "load_dataframe", return_value=None):
            with patch.object(fetcher, "fetch_reit_prices", return_value=pd.DataFrame()):
                result = fetcher.get_current_sentiment()
        assert result.empty
