"""
Tests for data/config.py — structural validation of configuration constants.
Catches typos, missing fields, and configuration drift.
"""

from data.config import (
    DATA_STORE_PATH,
    FRED_SERIES,
    HISTORICAL_YEARS,
    METROS,
    NYC_BBOX,
    REFRESH_INTERVAL_HOURS,
    REIT_TICKERS,
    SEED_DATA_PATH,
)


class TestMetroConfiguration:
    """Validate METROS dictionary structure."""

    REQUIRED_FIELDS = {"name", "unemployment", "hpi", "population"}

    def test_metro_count(self):
        """Should have exactly 20 metros defined."""
        assert len(METROS) == 20

    def test_no_duplicate_metro_codes(self):
        """Metro codes should be unique (dict keys enforce this, but be explicit)."""
        codes = list(METROS.keys())
        assert len(codes) == len(set(codes))

    def test_all_metros_have_required_fields(self):
        """Every metro must have name, unemployment, hpi, and population series IDs."""
        for code, info in METROS.items():
            missing = self.REQUIRED_FIELDS - set(info.keys())
            assert not missing, f"Metro {code} missing fields: {missing}"

    def test_metro_names_are_nonempty_strings(self):
        """Metro names should be meaningful strings."""
        for code, info in METROS.items():
            assert isinstance(info["name"], str), f"{code} name is not a string"
            assert len(info["name"]) > 3, f"{code} name is suspiciously short"

    def test_series_ids_are_nonempty_strings(self):
        """All FRED series IDs should be nonempty strings."""
        for code, info in METROS.items():
            for field in ["unemployment", "hpi", "population"]:
                val = info[field]
                assert isinstance(val, str) and len(val) > 0, (
                    f"Metro {code}.{field} is empty or not a string"
                )


class TestFREDSeriesConfiguration:
    """Validate FRED national series definitions."""

    def test_national_series_exist(self):
        """FRED_SERIES should have a 'national' key."""
        assert "national" in FRED_SERIES

    def test_national_series_count(self):
        """Should have 17 national series."""
        assert len(FRED_SERIES["national"]) == 17

    def test_series_ids_are_strings(self):
        """All series IDs should be nonempty strings."""
        for indicator, series_id in FRED_SERIES["national"].items():
            assert isinstance(series_id, str) and len(series_id) > 0, (
                f"National indicator '{indicator}' has invalid series ID"
            )

    def test_key_indicators_present(self):
        """Critical CRE indicators should be present."""
        expected = [
            "treasury_10y", "mortgage_30y", "fed_funds",
            "unemployment_national", "cre_delinquency"
        ]
        for indicator in expected:
            assert indicator in FRED_SERIES["national"], (
                f"Missing critical indicator: {indicator}"
            )


class TestREITTickerConfiguration:
    """Validate REIT ticker definitions."""

    def test_ticker_count(self):
        """Should have 20 REIT tickers."""
        assert len(REIT_TICKERS) == 20

    def test_all_tickers_have_name_and_sector(self):
        """Every ticker must have name and sector fields."""
        for ticker, info in REIT_TICKERS.items():
            assert "name" in info, f"Ticker {ticker} missing 'name'"
            assert "sector" in info, f"Ticker {ticker} missing 'sector'"

    def test_expected_sectors_present(self):
        """Should cover the main CRE sectors."""
        sectors = {info["sector"] for info in REIT_TICKERS.values()}
        expected = {"broad", "office", "industrial", "multifamily", "data_center"}
        assert expected.issubset(sectors), f"Missing sectors: {expected - sectors}"


class TestGeospatialConfig:
    """Validate NYC bounding box."""

    def test_bbox_is_4_tuple(self):
        assert len(NYC_BBOX) == 4

    def test_bbox_covers_nyc(self):
        """Bounding box should encompass all 5 boroughs."""
        west, south, east, north = NYC_BBOX
        # Rough checks — NYC spans roughly -74.25 to -73.69 W, 40.49 to 40.92 N
        assert west < -74.0
        assert east > -73.7
        assert south < 40.5
        assert north > 40.9

    def test_bbox_is_valid(self):
        """West < East and South < North."""
        west, south, east, north = NYC_BBOX
        assert west < east
        assert south < north


class TestConstants:
    """Validate configuration constants."""

    def test_refresh_interval_positive(self):
        assert REFRESH_INTERVAL_HOURS > 0

    def test_historical_years_positive(self):
        assert HISTORICAL_YEARS > 0

    def test_data_store_path_is_pathlib(self):
        from pathlib import Path
        assert isinstance(DATA_STORE_PATH, Path)

    def test_seed_data_path_is_pathlib(self):
        from pathlib import Path
        assert isinstance(SEED_DATA_PATH, Path)

    def test_seed_is_subdirectory_of_data_store(self):
        """Seed data should live inside the data store directory."""
        assert str(SEED_DATA_PATH).startswith(str(DATA_STORE_PATH))
