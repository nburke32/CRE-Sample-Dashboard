"""
XGBoost-based market strength scoring model.
Ranks metros by predicted growth potential using economic indicators.

Scoring methodology:
- Base Score: Backward-looking fundamentals (HPI, employment, unemployment)
- Sentiment Adjustment: Forward-looking REIT momentum (multiplicative)
- Final Score = Base Score × (1 + Sentiment Adjustment)
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict
from datetime import datetime, timedelta

# Regional groupings for percentile ranking
REGIONS = {
    "Northeast": ["NYC", "BOS", "PHI"],
    "Southeast": ["MIA", "ATL", "NSH", "CLT", "RDU"],
    "Midwest": ["CHI", "MSP"],
    "Southwest": ["DFW", "HOU", "PHX", "AUS", "DEN"],
    "West": ["LAX", "SFO", "SEA", "SAN"],
    "Mid-Atlantic": ["WAS"],
}

# Sector weights for CRE sentiment (which REIT sectors matter most)
CRE_SECTOR_WEIGHTS = {
    "broad": 0.25,           # VNQ, IYR - overall market sentiment
    "office": 0.15,          # BXP, VNO, SLG
    "industrial": 0.20,      # PLD, DRE - logistics/warehousing demand
    "retail_openair": 0.08,  # O, REG, KIM - grocery-anchored, net lease (healthier fundamentals)
    "retail_mall": 0.02,     # SPG - malls (challenged sector, lower weight)
    "multifamily": 0.20,     # EQR, AVB, MAA - residential demand
    "data_center": 0.05,     # EQIX, DLR
    "homebuilders": 0.05,    # XHB, DHI, LEN - leading indicator
}


class MarketStrengthModel:
    """
    XGBoost model for scoring metro market strength.
    Uses economic indicators to predict relative market performance.

    Scoring approach:
    1. Calculate base_score from backward-looking fundamentals (0-100)
    2. Calculate sentiment_adjustment from forward-looking REIT momentum (-0.20 to +0.20)
    3. Final: final_score = base_score × (1 + sentiment_adjustment)
    """

    # Maximum sentiment adjustment (caps at ±20%)
    MAX_SENTIMENT_ADJUSTMENT = 0.20

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.fitted = False
        self.scaler = None
        self.sentiment_score = None  # Cached REIT sentiment

    def prepare_features(
        self,
        metro_df: pd.DataFrame,
        national_df: pd.DataFrame,
        reit_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Prepare feature matrix from metro, national, and REIT data.

        Creates features like:
        - HPI growth (3m, 12m) - quarterly data
        - Unemployment rate & change - monthly data
        - Population growth - annual data
        - REIT sector momentum

        Data quality tracking:
        - Each metro gets a 'data_quality' dict tracking which features are real vs imputed
        - Handles mixed-frequency data (quarterly HPI, monthly unemployment, annual population)

        Args:
            metro_df: Metro-level economic data
            national_df: National economic data
            reit_df: Optional REIT price data

        Returns:
            Feature DataFrame ready for modeling
        """
        features_list = []

        for metro in metro_df["metro_code"].unique():
            metro_data = metro_df[metro_df["metro_code"] == metro].sort_values("date")

            if len(metro_data) < 13:  # Need at least 13 months for 12m changes
                continue

            features = {
                "metro_code": metro,
                "metro_name": metro_data.iloc[-1].get("metro_name", metro),
                "date": metro_data.iloc[-1]["date"],
            }

            # Track data quality - which features have real data
            data_quality = {"real": [], "missing": []}

            # HPI features - QUARTERLY data, find most recent non-null values
            if "hpi" in metro_data.columns:
                hpi_data = metro_data[["date", "hpi"]].dropna()
                if len(hpi_data) >= 5:  # Need at least 5 quarters for YoY
                    hpi_current = hpi_data.iloc[-1]["hpi"]
                    hpi_year = hpi_data.iloc[-5]["hpi"] if len(hpi_data) >= 5 else None  # 4 quarters back
                    hpi_quarter = hpi_data.iloc[-2]["hpi"] if len(hpi_data) >= 2 else None

                    if hpi_year and hpi_year > 0:
                        features["hpi_growth_12m"] = ((hpi_current - hpi_year) / hpi_year) * 100
                        data_quality["real"].append("hpi_growth_12m")
                    if hpi_quarter and hpi_quarter > 0:
                        features["hpi_growth_3m"] = ((hpi_current - hpi_quarter) / hpi_quarter) * 100
                        data_quality["real"].append("hpi_growth_3m")
                    features["hpi_current"] = hpi_current
                else:
                    data_quality["missing"].extend(["hpi_growth_12m", "hpi_growth_3m"])

            # Unemployment features - MONTHLY data, use most recent non-null
            if "unemployment" in metro_data.columns:
                unemp_data = metro_data[["date", "unemployment"]].dropna()
                if len(unemp_data) >= 13:  # Need 13 months for YoY
                    unemp_current = unemp_data.iloc[-1]["unemployment"]
                    unemp_year = unemp_data.iloc[-13]["unemployment"]

                    features["unemployment_rate"] = unemp_current
                    features["unemployment_change_12m"] = unemp_current - unemp_year
                    data_quality["real"].extend(["unemployment_rate", "unemployment_change_12m"])
                elif len(unemp_data) >= 1:
                    # Have some data but not enough for YoY
                    features["unemployment_rate"] = unemp_data.iloc[-1]["unemployment"]
                    data_quality["real"].append("unemployment_rate")
                    data_quality["missing"].append("unemployment_change_12m")
                else:
                    data_quality["missing"].extend(["unemployment_rate", "unemployment_change_12m"])

            # Population growth - ANNUAL data
            if "population" in metro_data.columns:
                pop_data = metro_data[["date", "population"]].dropna()
                if len(pop_data) >= 2:
                    pop_latest = pop_data.iloc[-1]["population"]
                    pop_prev = pop_data.iloc[-2]["population"]
                    if pop_prev > 0:
                        features["population_growth_1y"] = ((pop_latest - pop_prev) / pop_prev) * 100
                        features["population"] = pop_latest
                        data_quality["real"].append("population_growth_1y")
                else:
                    data_quality["missing"].append("population_growth_1y")

            # Store data quality info
            features["_data_quality"] = data_quality
            features["_missing_count"] = len(data_quality["missing"])

            features_list.append(features)

        if not features_list:
            return pd.DataFrame()

        features_df = pd.DataFrame(features_list)

        # Add national context
        if not national_df.empty:
            latest_national = national_df.sort_values("date").iloc[-1]

            # Interest rate environment
            if "treasury_10y" in latest_national:
                features_df["treasury_10y"] = latest_national["treasury_10y"]
            if "mortgage_30y" in latest_national:
                features_df["mortgage_30y"] = latest_national["mortgage_30y"]
            if "fed_funds" in latest_national:
                features_df["fed_funds"] = latest_national["fed_funds"]

        # Add REIT sentiment (if available)
        if reit_df is not None and not reit_df.empty:
            latest_date = reit_df["date"].max()
            recent_reit = reit_df[reit_df["date"] >= latest_date - timedelta(days=30)]

            for sector in recent_reit["sector"].unique():
                sector_data = recent_reit[recent_reit["sector"] == sector]
                if not sector_data.empty:
                    avg_return = sector_data["pct_change"].mean()
                    features_df[f"reit_{sector}_momentum"] = avg_return

        return features_df

    def calculate_reit_sentiment(
        self,
        reit_df: Optional[pd.DataFrame] = None,
        lookback_days: int = 30,
    ) -> Dict[str, float]:
        """
        Calculate forward-looking sentiment from REIT momentum.

        Uses weighted average of sector returns over recent period.
        Returns a sentiment score normalized to [-1, +1] range,
        which will be scaled to MAX_SENTIMENT_ADJUSTMENT for final use.

        Args:
            reit_df: REIT price data with columns: date, sector, pct_change
            lookback_days: Days to look back for momentum calculation

        Returns:
            Dict with 'composite' score and per-sector scores
        """
        if reit_df is None or reit_df.empty:
            return {"composite": 0.0}

        latest_date = reit_df["date"].max()
        recent_reit = reit_df[reit_df["date"] >= latest_date - timedelta(days=lookback_days)]

        if recent_reit.empty:
            return {"composite": 0.0}

        sector_returns = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for sector, weight in CRE_SECTOR_WEIGHTS.items():
            sector_data = recent_reit[recent_reit["sector"] == sector]
            if not sector_data.empty:
                # Calculate cumulative return over lookback period
                avg_daily_return = sector_data["pct_change"].mean()
                # Annualize for interpretability (approx trading days)
                cumulative_return = avg_daily_return * min(lookback_days, len(sector_data))
                sector_returns[sector] = cumulative_return

                weighted_sum += cumulative_return * weight
                total_weight += weight

        # Normalize to [-1, +1] based on typical REIT monthly returns
        # Typical monthly return range: -10% to +10%
        if total_weight > 0:
            composite = weighted_sum / total_weight
            # Normalize: ±10% monthly return maps to ±1.0
            normalized_composite = np.clip(composite / 10.0, -1.0, 1.0)
        else:
            normalized_composite = 0.0

        result = {"composite": normalized_composite}
        result.update({f"sector_{k}": v for k, v in sector_returns.items()})

        self.sentiment_score = result
        return result

    def _calculate_base_score(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate base strength score from backward-looking fundamentals.

        This is the foundation score before sentiment adjustment.
        Uses median imputation for missing values but tracks imputation.

        Args:
            features_df: Feature DataFrame from prepare_features()

        Returns:
            DataFrame with base_score column (0-100 scale) and data quality info
        """
        if features_df.empty:
            return pd.DataFrame()

        result = features_df[["metro_code", "metro_name", "date"]].copy()

        # Define scoring weights (positive = good for CRE)
        # Note: employment data not available from FRED metros, so using proxy indicators
        # Total weights should sum to 1.0 for available features
        weights = {
            "hpi_growth_12m": 0.30,           # Home price appreciation (major indicator)
            "hpi_growth_3m": 0.15,            # Recent momentum
            "unemployment_rate": -0.20,       # Lower is better (proxy for job market)
            "unemployment_change_12m": -0.15, # Improving is better
            "population_growth_1y": 0.20,     # Population growth (demand driver)
        }

        # Normalize each feature to 0-100 scale
        score_components = {}
        imputation_flags = {}

        for feature, weight in weights.items():
            if feature in features_df.columns:
                # Track which values are imputed
                is_missing = features_df[feature].isna()
                imputation_flags[f"{feature}_imputed"] = is_missing

                # Use median for missing values
                median_val = features_df[feature].median()
                if pd.isna(median_val):
                    median_val = 0  # Fallback if all values are NaN
                values = features_df[feature].fillna(median_val)

                # Normalize to 0-100
                if values.std() > 0:
                    normalized = (values - values.min()) / (values.max() - values.min()) * 100
                else:
                    normalized = pd.Series([50] * len(values), index=values.index)

                # Flip if negative weight
                if weight < 0:
                    normalized = 100 - normalized

                score_components[feature] = normalized * abs(weight)

        # Calculate total score
        if score_components:
            score_df = pd.DataFrame(score_components)
            result["base_score"] = score_df.sum(axis=1)

            # Normalize final score to 0-100
            if result["base_score"].std() > 0:
                result["base_score"] = (
                    (result["base_score"] - result["base_score"].min())
                    / (result["base_score"].max() - result["base_score"].min())
                    * 100
                )
        else:
            result["base_score"] = 50

        # Add individual component values for transparency
        for feature in weights.keys():
            if feature in features_df.columns:
                result[feature] = features_df[feature]

        # Add imputation count for each metro
        if imputation_flags:
            impute_df = pd.DataFrame(imputation_flags)
            result["imputed_count"] = impute_df.sum(axis=1)
        else:
            result["imputed_count"] = 0

        # Preserve data quality info if present
        if "_missing_count" in features_df.columns:
            result["missing_features"] = features_df["_missing_count"]

        return result

    def _add_regional_percentile(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add regional percentile ranking to scores.

        Args:
            scores_df: DataFrame with metro_code and strength_score

        Returns:
            DataFrame with additional region and regional_percentile columns
        """
        # Map metros to regions
        metro_to_region = {}
        for region, metros in REGIONS.items():
            for metro in metros:
                metro_to_region[metro] = region

        scores_df["region"] = scores_df["metro_code"].map(metro_to_region)

        # Calculate percentile within each region
        def calc_regional_percentile(group):
            group = group.copy()
            group["regional_percentile"] = group["strength_score"].rank(pct=True) * 100
            return group

        scores_df = scores_df.groupby("region", group_keys=False).apply(
            calc_regional_percentile, include_groups=False
        )
        # Re-add the region column since include_groups=False excludes it
        scores_df["region"] = scores_df.index.get_level_values(0) if scores_df.index.nlevels > 1 else scores_df["metro_code"].map(metro_to_region)

        return scores_df

    def calculate_strength_score(
        self,
        features_df: pd.DataFrame,
        reit_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Calculate market strength score using multiplicative sentiment approach.

        Methodology:
        1. Calculate base_score from backward-looking fundamentals (0-100)
        2. Calculate sentiment_adjustment from REIT momentum (-0.20 to +0.20)
        3. Apply multiplicative adjustment: final_score = base_score × (1 + sentiment_adjustment)

        Args:
            features_df: Feature DataFrame from prepare_features()
            reit_df: Optional REIT price data for sentiment calculation

        Returns:
            DataFrame with strength scores, rankings, and component breakdown
        """
        if features_df.empty:
            return pd.DataFrame()

        # Step 1: Calculate base score from backward-looking fundamentals
        result = self._calculate_base_score(features_df)

        # Step 2: Calculate forward-looking sentiment from REIT momentum
        sentiment = self.calculate_reit_sentiment(reit_df, lookback_days=30)
        composite_sentiment = sentiment.get("composite", 0.0)

        # Scale to ±MAX_SENTIMENT_ADJUSTMENT (±20%)
        sentiment_adjustment = composite_sentiment * self.MAX_SENTIMENT_ADJUSTMENT

        # Step 3: Normalize base scores to 0-100, apply multiplicative adjustment, clamp to bounds
        # Simple, transparent, standard practice in financial scoring
        result["sentiment_adjustment"] = sentiment_adjustment
        result["sentiment_pct"] = sentiment_adjustment * 100  # For display (e.g., +15%)

        # Normalize base_score to 0-100
        if result["base_score"].std() > 0:
            min_raw = result["base_score"].min()
            max_raw = result["base_score"].max()
            result["base_score"] = (
                (result["base_score"] - min_raw) / (max_raw - min_raw) * 100
            )

        # Apply multiplicative adjustment: Final = Base × (1 + Sentiment)
        raw_adjusted = result["base_score"] * (1 + sentiment_adjustment)

        # Calculate sentiment contribution for visualization
        result["sentiment_contribution"] = raw_adjusted - result["base_score"]

        # Clamp to 0-100 bounds (only affects extreme cases)
        result["strength_score"] = np.clip(raw_adjusted, 0, 100)

        # Step 4: Add rankings
        result["rank"] = result["strength_score"].rank(ascending=False).astype(int)

        # Step 5: Add regional context
        result = self._add_regional_percentile(result)

        # Add sector sentiment breakdown for transparency
        for key, value in sentiment.items():
            if key.startswith("sector_"):
                result[f"reit_{key.replace('sector_', '')}_return"] = value

        return result.sort_values("rank")

    def fit_xgboost(
        self,
        features_df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
    ) -> "MarketStrengthModel":
        """
        Fit XGBoost model to predict a target variable.
        Use this when you have historical target data to train on.

        Args:
            features_df: Feature DataFrame
            target_col: Column name of target variable
            feature_cols: List of feature columns (auto-detected if None)

        Returns:
            self (for method chaining)
        """
        from xgboost import XGBRegressor
        from sklearn.preprocessing import StandardScaler

        if feature_cols is None:
            # Auto-detect numeric feature columns
            exclude = ["metro_code", "metro_name", "date", target_col, "rank"]
            feature_cols = [
                c for c in features_df.columns
                if c not in exclude and features_df[c].dtype in ["float64", "int64"]
            ]

        self.feature_names = feature_cols

        X = features_df[feature_cols].fillna(0)
        y = features_df[target_col]

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Fit XGBoost
        self.model = XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
        )
        self.model.fit(X_scaled, y)
        self.fitted = True

        return self

    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions using fitted XGBoost model.

        Args:
            features_df: Feature DataFrame

        Returns:
            DataFrame with predictions
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first. Call fit_xgboost() or use calculate_strength_score().")

        X = features_df[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X)

        predictions = self.model.predict(X_scaled)

        result = features_df[["metro_code", "metro_name"]].copy()
        result["prediction"] = predictions
        result["rank"] = pd.Series(predictions).rank(ascending=False).astype(int)

        return result.sort_values("rank")

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance from fitted XGBoost model.

        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.fitted or self.model is None:
            return None

        importance = self.model.feature_importances_

        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False)


def score_all_metros(
    metro_df: pd.DataFrame,
    national_df: pd.DataFrame,
    reit_df: Optional[pd.DataFrame] = None,
    max_sentiment_adj: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Convenience function to score all metros with multiplicative sentiment.

    Args:
        metro_df: Metro-level data from FRED
        national_df: National data from FRED
        reit_df: Optional REIT data from YFinance
        max_sentiment_adj: Optional custom max sentiment adjustment (0.0 to 0.5).
                          If None, uses default of 0.20 (±20%)

    Returns:
        Tuple of (scores_df, features_df, sentiment_dict)
    """
    model = MarketStrengthModel()

    # Override max sentiment adjustment if provided
    if max_sentiment_adj is not None:
        model.MAX_SENTIMENT_ADJUSTMENT = max_sentiment_adj

    features = model.prepare_features(metro_df, national_df, reit_df)

    if features.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    # Pass reit_df to calculate_strength_score for sentiment calculation
    scores = model.calculate_strength_score(features, reit_df)

    # Return sentiment breakdown for UI display
    sentiment = model.sentiment_score or {}

    return scores, features, sentiment


def get_score_methodology() -> Dict[str, any]:
    """
    Return methodology details for UI display.

    Returns:
        Dict with methodology explanation and weights
    """
    return {
        "approach": "Multiplicative Adjustment with Clamp",
        "formula": "Final = clamp(Base × (1 + Sentiment), 0, 100)",
        "base_weights": {
            "HPI Growth (12m)": 0.30,
            "HPI Growth (3m)": 0.15,
            "Unemployment Rate": -0.20,
            "Unemployment Change (12m)": -0.15,
            "Population Growth (1y)": 0.20,
        },
        "sentiment_weights": CRE_SECTOR_WEIGHTS,
        "max_sentiment_adjustment": MarketStrengthModel.MAX_SENTIMENT_ADJUSTMENT,
        "regions": REGIONS,
        "data_notes": [
            "HPI data is quarterly (FHFA All-Transactions Index)",
            "Unemployment data is monthly (BLS metro estimates)",
            "Population data is annual (Census estimates)",
            "Missing values are imputed using median of available metros",
        ],
    }
