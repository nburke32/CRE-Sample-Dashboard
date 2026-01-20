"""
Prophet time-series forecasting for metro-level CRE indicators.
Provides forecasts with confidence intervals for housing/economic metrics.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from datetime import datetime


class ProphetForecaster:
    """
    Time-series forecasting using Facebook Prophet.
    Designed for metro-level CRE indicator forecasting.
    """

    def __init__(self):
        self.model = None
        self.fitted = False
        self.training_data = None

    def prepare_data(
        self,
        df: pd.DataFrame,
        date_col: str,
        value_col: str,
        interpolate: bool = True,
    ) -> pd.DataFrame:
        """
        Prepare data for Prophet (requires 'ds' and 'y' columns).
        Handles quarterly data by interpolating to monthly.

        Args:
            df: Input DataFrame
            date_col: Name of date column
            value_col: Name of value column to forecast
            interpolate: Whether to interpolate missing values (for quarterly data)

        Returns:
            DataFrame formatted for Prophet
        """
        prophet_df = df[[date_col, value_col]].copy()
        prophet_df = prophet_df.rename(columns={date_col: "ds", value_col: "y"})
        prophet_df = prophet_df.sort_values("ds")

        # Check data density - if sparse (like quarterly HPI), interpolate
        non_null_count = prophet_df["y"].notna().sum()
        total_count = len(prophet_df)
        density = non_null_count / total_count if total_count > 0 else 0

        if interpolate and density < 0.5:
            # Sparse data (quarterly) - interpolate to fill gaps
            prophet_df = prophet_df.set_index("ds")
            prophet_df["y"] = prophet_df["y"].interpolate(method="time")
            prophet_df = prophet_df.reset_index()

        # Drop any remaining NaN
        prophet_df = prophet_df.dropna()

        return prophet_df

    def fit(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        value_col: str = "value",
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = False,
        daily_seasonality: bool = False,
        changepoint_prior_scale: float = 0.05,
    ) -> "ProphetForecaster":
        """
        Fit Prophet model to historical data.

        Args:
            df: Historical data
            date_col: Name of date column
            value_col: Name of value column
            yearly_seasonality: Include yearly patterns
            weekly_seasonality: Include weekly patterns
            daily_seasonality: Include daily patterns
            changepoint_prior_scale: Flexibility of trend (lower = more stable)

        Returns:
            self (for method chaining)
        """
        from prophet import Prophet

        prophet_df = self.prepare_data(df, date_col, value_col)
        self.training_data = prophet_df.copy()

        # Use conservative settings for stable forecasts
        self.model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            interval_width=0.95,
            changepoint_prior_scale=changepoint_prior_scale,  # Lower = more stable trend
            seasonality_prior_scale=0.1,  # Dampen seasonality
        )

        self.model.fit(prophet_df)
        self.fitted = True

        return self

    def forecast(
        self,
        periods: int = 12,
        freq: str = "MS",
        include_history: bool = True,
    ) -> pd.DataFrame:
        """
        Generate forecast for future periods with sanity checks.

        Args:
            periods: Number of periods to forecast
            freq: Frequency ('MS' for monthly, 'QS' for quarterly)
            include_history: Include historical fitted values

        Returns:
            DataFrame with forecast, lower/upper bounds
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before forecasting. Call fit() first.")

        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        forecast = self.model.predict(future)

        result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        result = result.rename(
            columns={
                "ds": "date",
                "yhat": "forecast",
                "yhat_lower": "lower_bound",
                "yhat_upper": "upper_bound",
            }
        )

        # Apply sanity checks
        result = self._apply_sanity_checks(result)

        if not include_history:
            result = result.tail(periods)

        return result

    def _apply_sanity_checks(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply sanity checks to forecast output.
        Ensures forecasts are within reasonable bounds while preserving CI ranges.
        """
        if self.training_data is None or self.training_data.empty:
            return forecast_df

        # Get historical statistics
        hist_mean = self.training_data["y"].mean()
        hist_std = self.training_data["y"].std()
        hist_min = self.training_data["y"].min()
        hist_max = self.training_data["y"].max()
        hist_range = hist_max - hist_min

        # Define reasonable absolute bounds (prevent extreme outliers)
        # Allow wider range for confidence intervals
        abs_lower_bound = max(hist_min - hist_range, hist_mean - 5 * hist_std, 0)
        abs_upper_bound = hist_max + hist_range

        # Only clip extreme outliers, not normal CI ranges
        forecast_df["forecast"] = forecast_df["forecast"].clip(abs_lower_bound, abs_upper_bound)
        forecast_df["lower_bound"] = forecast_df["lower_bound"].clip(abs_lower_bound, abs_upper_bound)
        forecast_df["upper_bound"] = forecast_df["upper_bound"].clip(abs_lower_bound, abs_upper_bound)

        # Ensure proper CI ordering: lower <= forecast <= upper
        # But don't collapse them - just fix ordering if violated
        forecast_df["lower_bound"] = forecast_df[["lower_bound", "forecast"]].min(axis=1)
        forecast_df["upper_bound"] = forecast_df[["upper_bound", "forecast"]].max(axis=1)

        return forecast_df

    def get_components(self) -> Optional[pd.DataFrame]:
        """Get trend and seasonality components."""
        if not self.fitted:
            return None

        future = self.model.make_future_dataframe(periods=0)
        forecast = self.model.predict(future)

        components = ["ds", "trend"]
        if "yearly" in forecast.columns:
            components.append("yearly")
        if "weekly" in forecast.columns:
            components.append("weekly")

        return forecast[components].rename(columns={"ds": "date"})


def validate_forecast_quality(
    forecast_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    value_col: str,
) -> dict:
    """
    Validate forecast quality with multiple tests.

    Returns:
        Dict with validation results and warnings
    """
    results = {
        "valid": True,
        "warnings": [],
        "metrics": {},
    }

    if forecast_df.empty or historical_df.empty:
        results["valid"] = False
        results["warnings"].append("Empty data")
        return results

    # Get historical stats
    hist_values = historical_df[value_col].dropna()
    if len(hist_values) < 10:
        results["warnings"].append(f"Limited historical data ({len(hist_values)} points)")

    hist_mean = hist_values.mean()
    hist_std = hist_values.std()
    last_value = hist_values.iloc[-1]

    # Get forecast values (future only)
    last_hist_date = historical_df["date"].max()
    future_forecast = forecast_df[forecast_df["date"] > last_hist_date]

    if future_forecast.empty:
        results["valid"] = False
        results["warnings"].append("No future forecast generated")
        return results

    forecast_mean = future_forecast["forecast"].mean()
    forecast_end = future_forecast["forecast"].iloc[-1]

    # Test 1: Forecast should be positive (for HPI, unemployment)
    if (future_forecast["forecast"] < 0).any():
        results["warnings"].append("Forecast contains negative values")
        results["valid"] = False

    # Test 2: Forecast shouldn't deviate too far from recent history
    pct_change = abs((forecast_end - last_value) / last_value) * 100
    results["metrics"]["pct_change_from_last"] = pct_change
    if pct_change > 50:
        results["warnings"].append(f"Forecast deviates {pct_change:.1f}% from last value")

    # Test 3: Confidence interval should be reasonable (not too wide)
    ci_width = (future_forecast["upper_bound"] - future_forecast["lower_bound"]).mean()
    ci_pct = (ci_width / forecast_mean) * 100 if forecast_mean > 0 else 0
    results["metrics"]["avg_ci_width_pct"] = ci_pct
    if ci_pct > 100:
        results["warnings"].append(f"Very wide confidence interval ({ci_pct:.1f}%)")

    # Test 4: Trend should be continuous (no sudden jumps)
    forecast_values = forecast_df["forecast"].values
    max_jump = np.max(np.abs(np.diff(forecast_values)))
    max_jump_pct = (max_jump / hist_mean) * 100 if hist_mean > 0 else 0
    results["metrics"]["max_period_jump_pct"] = max_jump_pct
    if max_jump_pct > 20:
        results["warnings"].append(f"Large jump detected in forecast ({max_jump_pct:.1f}%)")

    return results


def forecast_metro_indicator(
    metro_df: pd.DataFrame,
    indicator_col: str,
    periods: int = 12,
    date_col: str = "date",
) -> Tuple[pd.DataFrame, dict]:
    """
    Convenience function to forecast a metro indicator with validation.

    Args:
        metro_df: DataFrame with metro data
        indicator_col: Column name to forecast
        periods: Months to forecast
        date_col: Date column name

    Returns:
        Tuple of (forecast_df, metrics_dict)
    """
    # Filter to valid data
    df = metro_df[[date_col, indicator_col]].copy()
    df = df.sort_values(date_col)

    # Check minimum data points (after potential interpolation)
    non_null = df[indicator_col].notna().sum()
    if non_null < 8:
        return pd.DataFrame(), {"error": f"Insufficient data (only {non_null} points)"}

    try:
        # Fit with conservative settings
        forecaster = ProphetForecaster()
        forecaster.fit(
            df,
            date_col=date_col,
            value_col=indicator_col,
            changepoint_prior_scale=0.01,  # Very stable trend
        )
        forecast = forecaster.forecast(periods=periods, include_history=True)

        # Validate forecast quality
        validation = validate_forecast_quality(forecast, df, indicator_col)

        # Get metrics
        valid_values = df[indicator_col].dropna()
        current_value = valid_values.iloc[-1]
        forecast_end = forecast["forecast"].iloc[-1]
        pct_change = ((forecast_end - current_value) / current_value) * 100

        metrics = {
            "current_value": current_value,
            "forecast_end": forecast_end,
            "pct_change": pct_change,
            "forecast_lower": forecast["lower_bound"].iloc[-1],
            "forecast_upper": forecast["upper_bound"].iloc[-1],
            "periods": periods,
            "validation": validation,
        }

        # If forecast is invalid, add warning to metrics
        if not validation["valid"]:
            metrics["warnings"] = validation["warnings"]

        return forecast, metrics

    except Exception as e:
        return pd.DataFrame(), {"error": str(e)}


def batch_forecast_metros(
    all_metros_df: pd.DataFrame,
    indicator_col: str,
    periods: int = 12,
    metro_col: str = "metro_code",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Forecast an indicator for all metros.
    """
    # Handle empty DataFrame
    if all_metros_df.empty:
        return pd.DataFrame()

    results = []

    for metro in all_metros_df[metro_col].unique():
        metro_data = all_metros_df[all_metros_df[metro_col] == metro]

        forecast, metrics = forecast_metro_indicator(
            metro_data, indicator_col, periods, date_col
        )

        if not forecast.empty:
            forecast[metro_col] = metro
            forecast["is_forecast"] = forecast["date"] > metro_data[date_col].max()
            results.append(forecast)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)
