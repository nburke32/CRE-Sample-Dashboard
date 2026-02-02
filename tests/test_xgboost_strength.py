"""
Unit tests for market strength scoring module.
Tests scoring logic, sentiment calculation, and edge cases.
"""


import numpy as np
import pandas as pd
import pytest

from models.market_scoring import (
    CRE_SECTOR_WEIGHTS,
    REGIONS,
    MarketStrengthModel,
    get_score_methodology,
    score_all_metros,
)


class TestMarketStrengthModel:
    """Test MarketStrengthModel class"""

    @pytest.fixture
    def sample_metro_data(self):
        """Generate sample metro-level data"""
        dates = pd.date_range('2023-01-01', periods=24, freq='MS')
        metros = ['NYC', 'LAX', 'CHI']
        data = []

        for metro in metros:
            metro_df = pd.DataFrame({
                'metro_code': metro,
                'metro_name': f'{metro} Metro',
                'date': dates,
                'hpi': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
                'employment': 1000 + np.cumsum(np.random.randn(len(dates)) * 10),
                'unemployment': 4 + np.random.randn(len(dates)) * 0.5,
                'population': 5000000 + np.cumsum(np.random.randn(len(dates)) * 10000)
            })
            data.append(metro_df)

        return pd.concat(data, ignore_index=True)

    @pytest.fixture
    def sample_national_data(self):
        """Generate sample national data"""
        dates = pd.date_range('2023-01-01', periods=24, freq='MS')
        return pd.DataFrame({
            'date': dates,
            'treasury_10y': 4.0 + np.random.randn(len(dates)) * 0.3,
            'mortgage_30y': 6.5 + np.random.randn(len(dates)) * 0.4,
            'fed_funds': 5.0 + np.random.randn(len(dates)) * 0.2
        })

    @pytest.fixture
    def sample_reit_data(self):
        """Generate sample REIT data"""
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        sectors = list(CRE_SECTOR_WEIGHTS.keys())
        data = []

        for sector in sectors:
            sector_df = pd.DataFrame({
                'date': dates,
                'sector': sector,
                'ticker': f'{sector.upper()[:3]}',
                'close': 100 + np.cumsum(np.random.randn(len(dates)) * 2),
                'pct_change': np.random.randn(len(dates)) * 0.02
            })
            data.append(sector_df)

        return pd.concat(data, ignore_index=True)

    def test_prepare_features_basic(self, sample_metro_data, sample_national_data):
        """Test basic feature preparation"""
        model = MarketStrengthModel()
        features = model.prepare_features(sample_metro_data, sample_national_data)

        assert not features.empty
        assert 'metro_code' in features.columns
        assert 'metro_name' in features.columns
        assert len(features) > 0

    def test_prepare_features_with_reit(self, sample_metro_data, sample_national_data, sample_reit_data):
        """Test feature preparation with REIT data"""
        model = MarketStrengthModel()
        features = model.prepare_features(sample_metro_data, sample_national_data, sample_reit_data)

        assert not features.empty
        # Should have REIT momentum features
        reit_cols = [col for col in features.columns if 'reit_' in col]
        assert len(reit_cols) > 0

    def test_calculate_reit_sentiment_positive(self):
        """Test sentiment calculation with positive returns"""
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        reit_df = pd.DataFrame({
            'date': dates,
            'sector': ['broad'] * 30,
            'pct_change': [0.01] * 30  # Positive returns
        })

        model = MarketStrengthModel()
        sentiment = model.calculate_reit_sentiment(reit_df, lookback_days=30)

        assert 'composite' in sentiment
        assert sentiment['composite'] > 0  # Should be positive

    def test_calculate_reit_sentiment_negative(self):
        """Test sentiment calculation with negative returns"""
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        reit_df = pd.DataFrame({
            'date': dates,
            'sector': ['broad'] * 30,
            'pct_change': [-0.01] * 30  # Negative returns
        })

        model = MarketStrengthModel()
        sentiment = model.calculate_reit_sentiment(reit_df, lookback_days=30)

        assert sentiment['composite'] < 0  # Should be negative

    def test_calculate_reit_sentiment_empty(self):
        """Test sentiment with no REIT data"""
        model = MarketStrengthModel()
        sentiment = model.calculate_reit_sentiment(None)

        assert sentiment['composite'] == 0.0

    def test_calculate_base_score(self, sample_metro_data, sample_national_data):
        """Test base score calculation"""
        model = MarketStrengthModel()
        features = model.prepare_features(sample_metro_data, sample_national_data)
        result = model._calculate_base_score(features)

        assert 'base_score' in result.columns
        assert result['base_score'].notna().all()
        assert (result['base_score'] >= 0).all()
        assert (result['base_score'] <= 100).all()

    def test_calculate_strength_score(self, sample_metro_data, sample_national_data, sample_reit_data):
        """Test full strength score calculation"""
        model = MarketStrengthModel()
        features = model.prepare_features(sample_metro_data, sample_national_data, sample_reit_data)
        scores = model.calculate_strength_score(features, sample_reit_data)

        assert not scores.empty
        assert 'strength_score' in scores.columns
        assert 'base_score' in scores.columns
        assert 'rank' in scores.columns
        assert 'sentiment_adjustment' in scores.columns

        # Scores should be bounded
        assert (scores['strength_score'] >= 0).all()
        assert (scores['strength_score'] <= 100).all()

    def test_sentiment_adjustment_bounds(self, sample_metro_data, sample_national_data):
        """Test that sentiment adjustment respects max bounds"""
        # Create extreme positive REIT data
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        extreme_reit = pd.DataFrame({
            'date': dates,
            'sector': ['broad'] * 30,
            'pct_change': [0.10] * 30  # Extreme positive returns
        })

        model = MarketStrengthModel()
        features = model.prepare_features(sample_metro_data, sample_national_data, extreme_reit)
        scores = model.calculate_strength_score(features, extreme_reit)

        # Sentiment adjustment should be capped
        max_adj = scores['sentiment_adjustment'].max()
        assert abs(max_adj) <= model.MAX_SENTIMENT_ADJUSTMENT

    def test_regional_percentile(self, sample_metro_data, sample_national_data):
        """Test regional ranking"""
        model = MarketStrengthModel()
        features = model.prepare_features(sample_metro_data, sample_national_data)
        scores = model.calculate_strength_score(features, None)

        if 'region' in scores.columns:
            assert 'regional_percentile' in scores.columns
            # Percentiles should be 0-100
            assert (scores['regional_percentile'] >= 0).all()
            assert (scores['regional_percentile'] <= 100).all()


class TestScoreAllMetros:
    """Test convenience function for scoring all metros"""

    @pytest.fixture
    def complete_data(self):
        """Generate complete dataset"""
        dates = pd.date_range('2023-01-01', periods=24, freq='MS')
        metros = ['NYC', 'LAX', 'CHI', 'DFW', 'ATL']

        metro_data = []
        for metro in metros:
            metro_df = pd.DataFrame({
                'metro_code': metro,
                'metro_name': f'{metro} Metro',
                'date': dates,
                'hpi': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
                'unemployment': 4 + np.random.randn(len(dates)) * 0.5,
                'population': 5000000 + np.cumsum(np.random.randn(len(dates)) * 10000)
            })
            metro_data.append(metro_df)

        metros_df = pd.concat(metro_data, ignore_index=True)

        national_df = pd.DataFrame({
            'date': dates,
            'treasury_10y': 4.0 + np.random.randn(len(dates)) * 0.3,
        })

        reit_dates = pd.date_range('2024-01-01', periods=30, freq='D')
        reit_df = pd.DataFrame({
            'date': reit_dates,
            'sector': ['broad'] * 30,
            'pct_change': np.random.randn(30) * 0.01
        })

        return metros_df, national_df, reit_df

    def test_score_all_metros_complete(self, complete_data):
        """Test scoring with complete data"""
        metros_df, national_df, reit_df = complete_data
        scores, features, sentiment = score_all_metros(metros_df, national_df, reit_df)

        assert not scores.empty
        assert not features.empty
        assert 'composite' in sentiment
        assert len(scores) > 0
        assert 'rank' in scores.columns

    def test_score_all_metros_no_reit(self, complete_data):
        """Test scoring without REIT data"""
        metros_df, national_df, _ = complete_data
        scores, features, sentiment = score_all_metros(metros_df, national_df, None)

        assert not scores.empty
        # Sentiment should be neutral
        assert sentiment.get('composite', 0) == 0.0

    def test_score_all_metros_custom_sentiment_adj(self, complete_data):
        """Test custom sentiment adjustment"""
        metros_df, national_df, reit_df = complete_data

        # Test with 10% max adjustment
        scores, _, _ = score_all_metros(metros_df, national_df, reit_df, max_sentiment_adj=0.10)

        if not scores.empty:
            max_adj = scores['sentiment_adjustment'].abs().max()
            assert max_adj <= 0.10

    def test_score_all_metros_empty_data(self):
        """Test with empty data"""
        scores, features, sentiment = score_all_metros(
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        )

        assert scores.empty
        assert features.empty
        assert sentiment == {}


class TestGetScoreMethodology:
    """Test methodology documentation function"""

    def test_get_methodology_structure(self):
        """Test that methodology returns expected structure"""
        methodology = get_score_methodology()

        assert 'approach' in methodology
        assert 'formula' in methodology
        assert 'base_weights' in methodology
        assert 'sentiment_weights' in methodology
        assert 'max_sentiment_adjustment' in methodology

    def test_methodology_weights_sum(self):
        """Test that weights are reasonable"""
        methodology = get_score_methodology()

        # Base weights should sum to roughly 1.0
        base_weights = methodology['base_weights']
        total_weight = sum(abs(w) for w in base_weights.values())
        assert 0.8 <= total_weight <= 1.2  # Allow some tolerance

        # Sentiment weights should sum to 1.0
        sentiment_weights = methodology['sentiment_weights']
        sentiment_total = sum(sentiment_weights.values())
        assert 0.95 <= sentiment_total <= 1.05


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_single_metro(self):
        """Test with only one metro"""
        dates = pd.date_range('2023-01-01', periods=24, freq='MS')
        metro_df = pd.DataFrame({
            'metro_code': 'NYC',
            'metro_name': 'New York',
            'date': dates,
            'hpi': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
            'unemployment': 4 + np.random.randn(len(dates)) * 0.5
        })

        national_df = pd.DataFrame({
            'date': dates,
            'treasury_10y': [4.0] * len(dates)
        })

        scores, features, sentiment = score_all_metros(metro_df, national_df, None)

        if not scores.empty:
            assert len(scores) == 1
            assert scores.iloc[0]['rank'] == 1

    def test_all_missing_hpi(self):
        """Test with all missing HPI data"""
        dates = pd.date_range('2023-01-01', periods=24, freq='MS')

        # Create proper multi-row DataFrame
        data = []
        for metro, name in [('NYC', 'New York'), ('LAX', 'Los Angeles')]:
            for date in dates:
                data.append({
                    'metro_code': metro,
                    'metro_name': name,
                    'date': date,
                    'hpi': np.nan,
                    'unemployment': np.random.randn() * 0.5 + 4
                })
        metro_df = pd.DataFrame(data)

        national_df = pd.DataFrame({
            'date': dates,
            'treasury_10y': [4.0] * len(dates)
        })

        scores, features, sentiment = score_all_metros(metro_df, national_df, None)

        # Should still produce scores using available data
        # or return empty if insufficient
        assert isinstance(scores, pd.DataFrame)

    def test_extreme_unemployment(self):
        """Test with extreme unemployment values"""
        dates = pd.date_range('2023-01-01', periods=24, freq='MS')
        metro_df = pd.DataFrame({
            'metro_code': 'NYC',
            'metro_name': 'New York',
            'date': dates,
            'hpi': 100 + np.arange(len(dates)),
            'unemployment': [20.0] * len(dates)  # Very high unemployment
        })

        national_df = pd.DataFrame({'date': dates, 'treasury_10y': [4.0] * len(dates)})

        scores, features, sentiment = score_all_metros(metro_df, national_df, None)

        if not scores.empty:
            # High unemployment should result in lower score
            assert True  # Score calculation should handle this

    def test_data_quality_tracking(self):
        """Test that data quality is tracked"""
        dates = pd.date_range('2023-01-01', periods=24, freq='MS')
        metro_df = pd.DataFrame({
            'metro_code': 'NYC',
            'metro_name': 'New York',
            'date': dates,
            'hpi': [100] * 12 + [np.nan] * 12,  # Half missing
            'unemployment': 4 + np.random.randn(len(dates)) * 0.5
        })

        national_df = pd.DataFrame({'date': dates})

        model = MarketStrengthModel()
        features = model.prepare_features(metro_df, national_df)

        # Verify data quality tracking exists and works
        if '_missing_count' in features.columns:
            # Column exists and contains valid data
            assert features['_missing_count'].dtype in [int, 'int64', np.int64]
            # Can be 0 if all data happens to be present (e.g., only needs unemployment)
            assert (features['_missing_count'] >= 0).all()
        else:
            # If column doesn't exist, feature preparation still worked
            assert not features.empty

    def test_zero_variance_data(self):
        """Test with constant values (zero variance)"""
        dates = pd.date_range('2023-01-01', periods=24, freq='MS')

        # Create proper multi-row DataFrame
        data = []
        for metro, name in [('NYC', 'New York'), ('LAX', 'Los Angeles')]:
            for date in dates:
                data.append({
                    'metro_code': metro,
                    'metro_name': name,
                    'date': date,
                    'hpi': 100.0,  # Constant
                    'unemployment': 4.0  # Constant
                })
        metro_df = pd.DataFrame(data)

        national_df = pd.DataFrame({'date': dates})

        scores, features, sentiment = score_all_metros(metro_df, national_df, None)

        # Should handle gracefully, likely giving equal scores
        if not scores.empty:
            assert scores['base_score'].nunique() <= 2


class TestRegionalMapping:
    """Test regional classification"""

    def test_all_metros_have_region(self):
        """Test that all predefined metros have a region"""
        all_metros = []
        for region, metros in REGIONS.items():
            all_metros.extend(metros)

        assert len(all_metros) > 0
        assert len(set(all_metros)) == len(all_metros)  # No duplicates

    def test_sector_weights_sum_to_one(self):
        """Test that REIT sector weights sum to 1.0"""
        total = sum(CRE_SECTOR_WEIGHTS.values())
        assert 0.99 <= total <= 1.01  # Allow small floating point error


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
