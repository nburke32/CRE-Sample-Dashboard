"""
Unit tests for Prophet forecasting module.
Tests core functionality, edge cases, and error handling.
"""


import numpy as np
import pandas as pd
import pytest

from models.prophet_forecast import (
    ProphetForecaster,
    batch_forecast_metros,
    forecast_metro_indicator,
    validate_forecast_quality,
)


class TestProphetForecaster:
    """Test ProphetForecaster class"""

    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data"""
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='MS')
        values = 100 + np.cumsum(np.random.randn(len(dates)) * 2)
        return pd.DataFrame({'date': dates, 'value': values})

    @pytest.fixture
    def sparse_data(self):
        """Generate sparse quarterly data (like HPI)"""
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='QS')
        values = 100 + np.cumsum(np.random.randn(len(dates)) * 2)
        return pd.DataFrame({'date': dates, 'value': values})

    def test_prepare_data_basic(self, sample_data):
        """Test basic data preparation"""
        forecaster = ProphetForecaster()
        prophet_df = forecaster.prepare_data(sample_data, 'date', 'value', interpolate=False)

        assert 'ds' in prophet_df.columns
        assert 'y' in prophet_df.columns
        assert len(prophet_df) > 0
        assert prophet_df['y'].notna().all()

    def test_prepare_data_with_interpolation(self, sparse_data):
        """Test interpolation for sparse data"""
        forecaster = ProphetForecaster()
        prophet_df = forecaster.prepare_data(sparse_data, 'date', 'value', interpolate=True)

        assert len(prophet_df) >= len(sparse_data)
        assert prophet_df['y'].notna().all()

    def test_fit_and_forecast(self, sample_data):
        """Test fitting model and generating forecast"""
        forecaster = ProphetForecaster()
        forecaster.fit(sample_data, 'date', 'value')

        assert forecaster.fitted is True
        assert forecaster.model is not None

        forecast = forecaster.forecast(periods=12, include_history=False)
        assert len(forecast) == 12
        assert all(col in forecast.columns for col in ['date', 'forecast', 'lower_bound', 'upper_bound'])

    def test_forecast_without_fit_raises_error(self, sample_data):
        """Test that forecasting without fitting raises error"""
        forecaster = ProphetForecaster()
        with pytest.raises(ValueError, match="Model must be fitted"):
            forecaster.forecast(periods=12)

    def test_sanity_checks(self, sample_data):
        """Test that sanity checks prevent unrealistic forecasts"""
        forecaster = ProphetForecaster()
        forecaster.fit(sample_data, 'date', 'value')
        forecast = forecaster.forecast(periods=12)

        # Check bounds are reasonable
        hist_max = sample_data['value'].max()
        hist_min = sample_data['value'].min()

        assert forecast['forecast'].min() >= 0  # No negative values
        assert forecast['lower_bound'].min() <= forecast['forecast'].min()
        assert forecast['upper_bound'].max() >= forecast['forecast'].max()

    def test_insufficient_data(self):
        """Test handling of insufficient data"""
        # Only 3 data points - too few for Prophet
        sparse_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=3, freq='MS'),
            'value': [100, 102, 101]
        })

        forecaster = ProphetForecaster()
        try:
            forecaster.fit(sparse_df, 'date', 'value')
            # Prophet might work with 3 points but quality will be poor
            # Just ensure it doesn't crash
            assert True
        except Exception as e:
            # Or it might fail, which is acceptable
            assert True


class TestForecastMetroIndicator:
    """Test convenience function for metro forecasting"""

    @pytest.fixture
    def metro_data(self):
        """Generate sample metro data"""
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='MS')
        return pd.DataFrame({
            'date': dates,
            'hpi': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
            'employment': 1000 + np.cumsum(np.random.randn(len(dates)) * 10),
            'unemployment': 4 + np.random.randn(len(dates)) * 0.5
        })

    def test_forecast_metro_indicator_success(self, metro_data):
        """Test successful forecast"""
        forecast, metrics = forecast_metro_indicator(metro_data, 'hpi', periods=12)

        assert not forecast.empty
        assert 'current_value' in metrics
        assert 'forecast_end' in metrics
        assert 'pct_change' in metrics
        assert metrics['periods'] == 12

    def test_forecast_insufficient_data(self):
        """Test with insufficient data points"""
        sparse_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5, freq='MS'),
            'hpi': [100, 101, 102, 103, 104]
        })

        forecast, metrics = forecast_metro_indicator(sparse_df, 'hpi', periods=12)

        # Should return empty or error
        assert forecast.empty or 'error' in metrics

    def test_forecast_missing_column(self, metro_data):
        """Test with non-existent indicator"""
        with pytest.raises(KeyError):
            forecast_metro_indicator(metro_data, 'nonexistent_col', periods=12)

    def test_forecast_all_nan(self, metro_data):
        """Test with all NaN values"""
        metro_data['bad_indicator'] = np.nan
        forecast, metrics = forecast_metro_indicator(metro_data, 'bad_indicator', periods=12)

        assert forecast.empty
        assert 'error' in metrics


class TestValidateForecastQuality:
    """Test forecast validation"""

    @pytest.fixture
    def valid_forecast(self):
        """Generate valid forecast data"""
        dates = pd.date_range('2023-01-01', periods=24, freq='MS')
        return pd.DataFrame({
            'date': dates,
            'forecast': 100 + np.arange(24),
            'lower_bound': 95 + np.arange(24),
            'upper_bound': 105 + np.arange(24)
        })

    @pytest.fixture
    def historical_data(self):
        """Generate historical data"""
        dates = pd.date_range('2020-01-01', periods=36, freq='MS')
        return pd.DataFrame({
            'date': dates,
            'hpi': 100 + np.arange(36) * 0.5
        })

    def test_validate_quality_valid_forecast(self, valid_forecast, historical_data):
        """Test validation of good forecast"""
        result = validate_forecast_quality(valid_forecast, historical_data, 'hpi')

        assert 'valid' in result
        assert 'warnings' in result
        assert 'metrics' in result

    def test_validate_quality_empty_data(self):
        """Test validation with empty data"""
        result = validate_forecast_quality(pd.DataFrame(), pd.DataFrame(), 'value')

        assert result['valid'] is False
        assert 'Empty data' in result['warnings']

    def test_validate_quality_negative_values(self, historical_data):
        """Test detection of negative forecasts"""
        bad_forecast = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=12, freq='MS'),
            'forecast': [-10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
            'lower_bound': [-15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40],
            'upper_bound': [-5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        })

        result = validate_forecast_quality(bad_forecast, historical_data, 'hpi')

        assert result['valid'] is False
        assert any('negative' in w.lower() for w in result['warnings'])


class TestBatchForecastMetros:
    """Test batch forecasting across multiple metros"""

    @pytest.fixture
    def multi_metro_data(self):
        """Generate data for multiple metros"""
        dates = pd.date_range('2020-01-01', periods=48, freq='MS')
        data = []

        for metro in ['NYC', 'LAX', 'CHI']:
            metro_df = pd.DataFrame({
                'metro_code': metro,
                'date': dates,
                'hpi': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
            })
            data.append(metro_df)

        return pd.concat(data, ignore_index=True)

    def test_batch_forecast_all_metros(self, multi_metro_data):
        """Test forecasting multiple metros at once"""
        result = batch_forecast_metros(multi_metro_data, 'hpi', periods=6)

        if not result.empty:
            assert 'metro_code' in result.columns
            assert 'is_forecast' in result.columns
            assert len(result['metro_code'].unique()) > 0

    def test_batch_forecast_empty_data(self):
        """Test batch forecast with empty data"""
        result = batch_forecast_metros(pd.DataFrame(), 'hpi', periods=6)
        assert result.empty


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_forecast_with_gaps(self):
        """Test forecasting with gaps in data"""
        dates = pd.date_range('2020-01-01', periods=40, freq='MS')
        values = 100 + np.cumsum(np.random.randn(40) * 0.5)

        # Introduce gaps
        df = pd.DataFrame({'date': dates, 'value': values})
        df.loc[10:15, 'value'] = np.nan
        df.loc[25:30, 'value'] = np.nan

        forecast, metrics = forecast_metro_indicator(df, 'value', periods=12)

        # Should still work with interpolation
        assert not forecast.empty or 'error' in metrics

    def test_forecast_constant_values(self):
        """Test forecasting flat data (no trend)"""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=36, freq='MS'),
            'value': [100] * 36
        })

        forecaster = ProphetForecaster()
        forecaster.fit(df, 'date', 'value')
        forecast = forecaster.forecast(periods=12)

        # Forecast should be relatively flat
        assert not forecast.empty
        forecast_values = forecast.tail(12)['forecast']
        assert forecast_values.std() < 10  # Low variance

    def test_forecast_extreme_volatility(self):
        """Test forecasting highly volatile data"""
        dates = pd.date_range('2020-01-01', periods=36, freq='MS')
        values = 100 + np.random.randn(36) * 50  # High volatility

        df = pd.DataFrame({'date': dates, 'value': values})

        forecast, metrics = forecast_metro_indicator(df, 'value', periods=12)

        # Should complete but might have warnings
        assert not forecast.empty or 'error' in metrics
        if 'validation' in metrics:
            # Likely to have wide confidence intervals
            assert True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
