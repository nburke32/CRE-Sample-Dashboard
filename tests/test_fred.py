"""
Tests for data/fred_fetcher.py — API mocking, data transformation, cache logic.
"""

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data.fred_fetcher import FREDFetcher


class TestFREDFetcherInit:
    """Test initialization and API key handling."""

    def test_init_with_explicit_key(self):
        """Should accept an explicit API key."""
        fetcher = FREDFetcher(api_key="test_key_123")
        assert fetcher.api_key == "test_key_123"

    def test_init_from_env_var(self):
        """Should read API key from environment if not passed."""
        with patch.dict(os.environ, {"FRED_API_KEY": "env_key_456"}):
            fetcher = FREDFetcher()
            assert fetcher.api_key == "env_key_456"

    def test_init_raises_without_key(self):
        """Should raise ValueError when no key is available."""
        with patch.dict(os.environ, {}, clear=True):
            # Also clear any existing FRED_API_KEY
            env = os.environ.copy()
            env.pop("FRED_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises(ValueError, match="FRED API key not found"):
                    FREDFetcher()


class TestFetchSeries:
    """Test _fetch_series with mocked HTTP responses."""

    @pytest.fixture
    def fetcher(self):
        return FREDFetcher(api_key="test_key")

    def test_successful_fetch(self, fetcher):
        """Should parse a valid FRED API response into a DataFrame."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "observations": [
                {"date": "2024-01-01", "value": "4.25"},
                {"date": "2024-02-01", "value": "4.30"},
                {"date": "2024-03-01", "value": "."},  # FRED uses "." for missing
            ]
        }
        mock_response.raise_for_status = MagicMock()

        # requests is imported locally inside _fetch_series, patch via the real module
        import requests
        with patch.object(requests, "get", return_value=mock_response):
            result = fetcher._fetch_series("DGS10", "2024-01-01")

        assert result is not None
        assert len(result) == 2  # "." should be dropped as NaN
        assert "date" in result.columns
        assert "value" in result.columns

    def test_fetch_no_observations_key(self, fetcher):
        """Should return None when response has no 'observations' key."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"error": "bad request"}
        mock_response.raise_for_status = MagicMock()

        mock_requests = MagicMock()
        mock_requests.get.return_value = mock_response

        with patch.dict("sys.modules", {"requests": mock_requests}):
            result = fetcher._fetch_series("INVALID", "2024-01-01")

        assert result is None

    def test_fetch_empty_observations(self, fetcher):
        """Should return None when observations list is empty."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"observations": []}
        mock_response.raise_for_status = MagicMock()

        mock_requests = MagicMock()
        mock_requests.get.return_value = mock_response

        with patch.dict("sys.modules", {"requests": mock_requests}):
            result = fetcher._fetch_series("DGS10", "2024-01-01")

        assert result is None

    def test_fetch_network_error(self, fetcher):
        """Should return None on network errors."""
        mock_requests = MagicMock()
        mock_requests.get.side_effect = Exception("Connection timeout")

        with patch.dict("sys.modules", {"requests": mock_requests}):
            result = fetcher._fetch_series("DGS10", "2024-01-01")

        assert result is None


class TestDataTransformation:
    """Test data pivot and column ordering in fetch methods."""

    @pytest.fixture
    def fetcher(self):
        return FREDFetcher(api_key="test_key")

    def test_national_data_returns_wide_format(self, fetcher):
        """fetch_national_data should return wide-format DataFrame with indicator columns."""
        # Mock _fetch_series to return controlled data
        def mock_fetch(series_id, start_date):
            return pd.DataFrame({
                "date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
                "value": [4.25, 4.30],
            })

        with patch.object(fetcher, "_fetch_series", side_effect=mock_fetch):
            with patch.object(fetcher.storage, "dataset_exists", return_value=False):
                with patch.object(fetcher.storage, "save_dataframe"):
                    result = fetcher.fetch_national_data(force_refresh=True)

        assert not result.empty
        assert "date" in result.columns
        # Should have indicator columns (pivoted from long to wide)
        assert len(result.columns) > 1

    def test_metro_data_has_metro_columns(self, fetcher):
        """fetch_metro_data should include metro_code and metro_name columns."""
        def mock_fetch(series_id, start_date):
            return pd.DataFrame({
                "date": pd.to_datetime(["2024-01-01"]),
                "value": [5.0],
            })

        with patch.object(fetcher, "_fetch_series", side_effect=mock_fetch):
            with patch.object(fetcher.storage, "dataset_exists", return_value=False):
                with patch.object(fetcher.storage, "save_dataframe"):
                    result = fetcher.fetch_metro_data(force_refresh=True)

        assert not result.empty
        assert "metro_code" in result.columns
        assert "metro_name" in result.columns
        assert "date" in result.columns
        # First 3 columns should be date, metro_code, metro_name
        assert list(result.columns[:3]) == ["date", "metro_code", "metro_name"]

    def test_national_empty_when_all_fetches_fail(self, fetcher):
        """Should return empty DataFrame when all series fail to fetch."""
        with patch.object(fetcher, "_fetch_series", return_value=None):
            with patch.object(fetcher.storage, "dataset_exists", return_value=False):
                result = fetcher.fetch_national_data(force_refresh=True)

        assert result.empty


class TestCacheBehavior:
    """Test cache hit/miss logic."""

    @pytest.fixture
    def fetcher(self):
        return FREDFetcher(api_key="test_key")

    def test_national_uses_cache_when_fresh(self, fetcher):
        """Should return cached data without fetching when cache is fresh."""
        from datetime import datetime

        cached_df = pd.DataFrame({"date": ["2024-01-01"], "treasury_10y": [4.25]})

        with patch.object(fetcher.storage, "dataset_exists", return_value=True):
            with patch.object(fetcher.storage, "get_last_updated", return_value=datetime.now()):
                with patch.object(fetcher.storage, "load_dataframe", return_value=cached_df):
                    result = fetcher.fetch_national_data(force_refresh=False)

        assert len(result) == 1
        assert result.iloc[0]["treasury_10y"] == 4.25

    def test_force_refresh_bypasses_cache(self, fetcher):
        """force_refresh=True should skip cache and fetch fresh data."""
        def mock_fetch(series_id, start_date):
            return pd.DataFrame({
                "date": pd.to_datetime(["2024-01-01"]),
                "value": [99.0],
            })

        with patch.object(fetcher, "_fetch_series", side_effect=mock_fetch):
            with patch.object(fetcher.storage, "save_dataframe"):
                result = fetcher.fetch_national_data(force_refresh=True)

        # Should have fetched fresh data, not used cache
        assert not result.empty
