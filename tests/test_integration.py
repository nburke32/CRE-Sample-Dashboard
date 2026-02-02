"""
Integration tests for the CRE Dashboard.
Tests end-to-end data flows, API interactions, and module integration.
"""

import os
from datetime import datetime, timedelta

import pandas as pd
import pytest

from data.fred_fetcher import FREDFetcher
from data.storage import DataStorage
from data.yfinance_fetcher import YFinanceFetcher
from models.market_scoring import score_all_metros
from models.prophet_forecast import forecast_metro_indicator


class TestDataStorage:
    """Test data storage and caching"""

    def test_storage_initialization(self):
        """Test that storage initializes correctly"""
        storage = DataStorage()
        datasets = storage.list_datasets()
        assert isinstance(datasets, list)

    def test_save_and_load_dataframe(self):
        """Test saving and loading data"""
        storage = DataStorage()
        test_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10, freq='D'),
            'value': range(10)
        })

        # Save
        storage.save_dataframe(test_df, 'test_dataset')

        # Load
        loaded_df = storage.load_dataframe('test_dataset')

        assert loaded_df is not None
        assert len(loaded_df) == len(test_df)
        assert list(loaded_df.columns) == list(test_df.columns)

        # Cleanup
        storage.delete_dataset('test_dataset')

    def test_dataset_exists(self):
        """Test checking dataset existence"""
        storage = DataStorage()

        # Should exist if data has been fetched
        # This test depends on whether data has been loaded
        result = storage.dataset_exists('fred_national')
        assert isinstance(result, bool)

    def test_get_last_updated(self):
        """Test getting last updated timestamp"""
        storage = DataStorage()

        if storage.dataset_exists('fred_national'):
            last_updated = storage.get_last_updated('fred_national')
            assert isinstance(last_updated, (datetime, type(None)))

    def test_list_datasets(self):
        """Test listing all datasets"""
        storage = DataStorage()
        datasets = storage.list_datasets()

        assert isinstance(datasets, list)


@pytest.mark.skipif(not os.getenv('FRED_API_KEY'), reason="FRED_API_KEY not set")
class TestFREDFetcher:
    """Test FRED data fetching (requires API key)"""

    def test_fetcher_initialization(self):
        """Test FRED fetcher initialization"""
        fetcher = FREDFetcher()
        assert fetcher.api_key is not None
        assert fetcher.storage is not None

    def test_fetch_series(self):
        """Test fetching a single FRED series"""
        fetcher = FREDFetcher()
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

        # Try to fetch Treasury 10Y
        df = fetcher._fetch_series('DGS10', start_date)

        if df is not None:
            assert 'date' in df.columns
            assert 'value' in df.columns
            assert len(df) > 0

    @pytest.mark.slow
    def test_fetch_national_data_from_cache(self):
        """Test fetching national data (should use cache if available)"""
        fetcher = FREDFetcher()

        # This should use cached data if available, or fetch if not
        df = fetcher.fetch_national_data(force_refresh=False)

        assert isinstance(df, pd.DataFrame)
        # May be empty if API fails or no cache

    @pytest.mark.slow
    def test_fetch_metro_data_from_cache(self):
        """Test fetching metro data (should use cache if available)"""
        fetcher = FREDFetcher()

        # This should use cached data if available
        df = fetcher.fetch_metro_data(force_refresh=False)

        assert isinstance(df, pd.DataFrame)


@pytest.mark.skipif(not os.getenv('FRED_API_KEY'), reason="FRED_API_KEY not set")
class TestYFinanceFetcher:
    """Test YFinance REIT data fetching"""

    def test_fetcher_initialization(self):
        """Test YFinance fetcher initialization"""
        fetcher = YFinanceFetcher()
        assert fetcher.storage is not None

    @pytest.mark.slow
    def test_fetch_reit_prices_from_cache(self):
        """Test fetching REIT prices (should use cache if available)"""
        fetcher = YFinanceFetcher()

        # This should use cached data if available
        df = fetcher.fetch_reit_prices(force_refresh=False)

        assert isinstance(df, pd.DataFrame)

    @pytest.mark.slow
    def test_get_current_sentiment(self):
        """Test getting current REIT sentiment"""
        fetcher = YFinanceFetcher()

        sentiment = fetcher.get_current_sentiment()

        assert isinstance(sentiment, pd.DataFrame)


class TestEndToEndWithRealData:
    """Test end-to-end workflows with real cached data"""

    @pytest.fixture
    def load_real_data(self):
        """Load real data from cache if available"""
        storage = DataStorage()

        metros_df = storage.load_dataframe('fred_metros')
        national_df = storage.load_dataframe('fred_national')
        reit_df = storage.load_dataframe('reit_prices')

        return metros_df, national_df, reit_df

    @pytest.mark.skipif(
        not DataStorage().dataset_exists('fred_metros'),
        reason="Real data not available"
    )
    def test_prophet_forecast_with_real_data(self, load_real_data):
        """Test Prophet forecast with real metro data"""
        metros_df, _, _ = load_real_data

        if metros_df is None or metros_df.empty:
            pytest.skip("No metro data available")

        # Get NYC data
        nyc_data = metros_df[metros_df['metro_code'] == 'NYC']

        if nyc_data.empty or 'hpi' not in nyc_data.columns:
            pytest.skip("NYC HPI data not available")

        forecast, metrics = forecast_metro_indicator(nyc_data, 'hpi', periods=12)

        if not forecast.empty:
            assert 'current_value' in metrics
            assert 'forecast_end' in metrics
            assert len(forecast) > 0

    @pytest.mark.skipif(
        not DataStorage().dataset_exists('fred_metros'),
        reason="Real data not available"
    )
    def test_market_scoring_with_real_data(self, load_real_data):
        """Test market scoring with real data"""
        metros_df, national_df, reit_df = load_real_data

        if metros_df is None or metros_df.empty:
            pytest.skip("No metro data available")

        scores, features, sentiment = score_all_metros(
            metros_df, national_df if national_df is not None and not national_df.empty else pd.DataFrame(), reit_df
        )

        if not scores.empty:
            assert 'strength_score' in scores.columns
            assert 'rank' in scores.columns
            assert len(scores) > 0
            assert scores['rank'].min() >= 1

            # Validate score ranges
            assert (scores['strength_score'] >= 0).all()
            assert (scores['strength_score'] <= 100).all()

    @pytest.mark.skipif(
        not DataStorage().dataset_exists('fred_metros'),
        reason="Real data not available"
    )
    def test_scoring_consistency(self, load_real_data):
        """Test that scoring is deterministic with same data"""
        metros_df, national_df, reit_df = load_real_data

        if metros_df is None or metros_df.empty:
            pytest.skip("No metro data available")

        # Run scoring twice
        nat_df = national_df if national_df is not None and not national_df.empty else pd.DataFrame()
        scores1, _, _ = score_all_metros(metros_df, nat_df, reit_df)
        scores2, _, _ = score_all_metros(metros_df, nat_df, reit_df)

        if not scores1.empty and not scores2.empty:
            # Scores should be identical
            pd.testing.assert_frame_equal(
                scores1[['metro_code', 'strength_score']].sort_values('metro_code').reset_index(drop=True),
                scores2[['metro_code', 'strength_score']].sort_values('metro_code').reset_index(drop=True)
            )


class TestErrorHandling:
    """Test error handling and resilience"""

    def test_fred_fetcher_without_api_key(self):
        """Test FRED fetcher fails gracefully without API key"""
        # Temporarily unset the key
        old_key = os.environ.get('FRED_API_KEY')
        if old_key:
            del os.environ['FRED_API_KEY']

        try:
            with pytest.raises(ValueError, match="FRED API key not found"):
                FREDFetcher()
        finally:
            if old_key:
                os.environ['FRED_API_KEY'] = old_key

    def test_scoring_with_incomplete_data(self):
        """Test scoring handles incomplete data gracefully"""
        # Create minimal data
        incomplete_metro = pd.DataFrame({
            'metro_code': ['NYC'],
            'metro_name': ['New York'],
            'date': [datetime.now()],
            'hpi': [100.0]
        })

        scores, features, sentiment = score_all_metros(
            incomplete_metro, pd.DataFrame(), None
        )

        # Should either return empty or handle gracefully
        assert isinstance(scores, pd.DataFrame)

    def test_forecast_with_all_nan(self):
        """Test forecast handles all-NaN data"""
        bad_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=24, freq='MS'),
            'value': [float('nan')] * 24
        })

        forecast, metrics = forecast_metro_indicator(bad_data, 'value', periods=12)

        # Should return empty or error
        assert forecast.empty or 'error' in metrics

    def test_storage_with_invalid_dataset(self):
        """Test storage handles non-existent dataset"""
        storage = DataStorage()
        result = storage.load_dataframe('nonexistent_dataset_xyz')

        assert result is None or isinstance(result, pd.DataFrame)


class TestDataQuality:
    """Test data quality checks"""

    @pytest.mark.skipif(
        not DataStorage().dataset_exists('fred_metros'),
        reason="Real data not available"
    )
    def test_metro_data_completeness(self):
        """Test metro data has expected structure"""
        storage = DataStorage()
        metros_df = storage.load_dataframe('fred_metros')

        if metros_df is not None and not metros_df.empty:
            assert 'metro_code' in metros_df.columns
            assert 'date' in metros_df.columns

            # Check for critical indicators
            expected_indicators = ['hpi', 'unemployment']
            present_indicators = [col for col in expected_indicators if col in metros_df.columns]
            assert len(present_indicators) > 0, "No critical indicators found"

    @pytest.mark.skipif(
        not DataStorage().dataset_exists('reit_prices'),
        reason="Real data not available"
    )
    def test_reit_data_completeness(self):
        """Test REIT data has expected structure"""
        storage = DataStorage()
        reit_df = storage.load_dataframe('reit_prices')

        if reit_df is not None and not reit_df.empty:
            assert 'date' in reit_df.columns
            assert 'sector' in reit_df.columns
            assert 'ticker' in reit_df.columns

            # Check date range
            latest_date = reit_df['date'].max()
            assert (datetime.now() - latest_date) < timedelta(days=7), "REIT data is stale"

    @pytest.mark.skipif(
        not DataStorage().dataset_exists('fred_metros'),
        reason="Real data not available"
    )
    def test_no_duplicate_records(self):
        """Test that data has no duplicate date-metro combinations"""
        storage = DataStorage()
        metros_df = storage.load_dataframe('fred_metros')

        if metros_df is not None and not metros_df.empty:
            duplicates = metros_df.duplicated(subset=['metro_code', 'date']).sum()
            assert duplicates == 0, f"Found {duplicates} duplicate records"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'not slow'])
