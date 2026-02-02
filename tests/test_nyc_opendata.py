"""
Tests for data/nyc_opendata_fetcher.py — data cleaning, property categorization, borough mapping.
Focuses on the transformation logic, not the API calls.
"""

import numpy as np
import pandas as pd
import pytest

from data.nyc_opendata_fetcher import NYCOpenDataFetcher


class TestPropertyTypeCategorization:
    """Test _categorize_property_type mapping logic."""

    @pytest.fixture
    def fetcher(self):
        """Create fetcher instance (no API calls needed for these tests)."""
        # Patch the __init__ to avoid Socrata client initialization
        fetcher = object.__new__(NYCOpenDataFetcher)
        return fetcher

    @pytest.mark.parametrize("input_val,expected", [
        ("OFFICE BUILDINGS", "Office"),
        ("Office Rentals", "Office"),
        ("RETAIL STORES", "Retail"),
        ("Department Store", "Retail"),
        ("WAREHOUSE", "Industrial"),
        ("Industrial Buildings", "Industrial"),
        ("FACTORY", "Industrial"),
        ("HOTELS", "Hotel"),
        ("Hotel Rooms", "Hotel"),
        ("ELEVATOR APARTMENTS", "Multifamily"),
        ("Walk-up Family Dwelling", "Multifamily"),
        ("GARAGES", "Parking"),
        ("Parking Lots", "Parking"),
        ("CONDOMINIUMS", "Mixed Use"),
        ("CHURCHES", "Mixed Use"),
    ])
    def test_categorization_mapping(self, fetcher, input_val, expected):
        """Test each property type category maps correctly."""
        assert fetcher._categorize_property_type(input_val) == expected

    def test_nan_returns_other(self, fetcher):
        """NaN building class should return 'Other'."""
        assert fetcher._categorize_property_type(np.nan) == "Other"
        assert fetcher._categorize_property_type(None) == "Other"

    def test_case_insensitivity(self, fetcher):
        """Categorization should be case-insensitive."""
        assert fetcher._categorize_property_type("office") == "Office"
        assert fetcher._categorize_property_type("OFFICE") == "Office"
        assert fetcher._categorize_property_type("Office") == "Office"


class TestDataCleaning:
    """Test _clean_property_data transformation logic."""

    @pytest.fixture
    def fetcher(self):
        """Create fetcher with mocked storage."""
        fetcher = object.__new__(NYCOpenDataFetcher)
        return fetcher

    @pytest.fixture
    def raw_data(self):
        """Simulate raw API response data."""
        return pd.DataFrame({
            "sale_date": ["2024-06-15T00:00:00", "2024-07-20T00:00:00", "2024-08-10T00:00:00"],
            "sale_price": ["1500000", "25000", "5000000"],
            "gross_square_feet": ["10000", "5000", "20000"],
            "land_square_feet": ["5000", "2500", "10000"],
            "residential_units": ["0", "0", "0"],
            "commercial_units": ["5", "2", "10"],
            "total_units": ["5", "2", "10"],
            "lot": ["1", "2", "3"],
            "borough": ["1", "2", "3"],
            "building_class_category": ["OFFICE BUILDINGS", "RETAIL STORES", "WAREHOUSE"],
        })

    def test_date_conversion(self, fetcher, raw_data):
        """sale_date should be converted to datetime."""
        result = fetcher._clean_property_data(raw_data)
        assert pd.api.types.is_datetime64_any_dtype(result["sale_date"])

    def test_numeric_conversion(self, fetcher, raw_data):
        """Numeric columns should be converted from strings to numeric types."""
        result = fetcher._clean_property_data(raw_data)
        assert pd.api.types.is_numeric_dtype(result["sale_price"])
        assert pd.api.types.is_numeric_dtype(result["gross_square_feet"])

    def test_price_per_sqft_calculated(self, fetcher, raw_data):
        """price_per_sqft should be calculated from sale_price / gross_square_feet."""
        result = fetcher._clean_property_data(raw_data)
        assert "price_per_sqft" in result.columns

        # $1,500,000 / 10,000 sqft = $150/sqft
        row = result[result["sale_price"] == 1_500_000]
        if not row.empty:
            assert abs(row.iloc[0]["price_per_sqft"] - 150.0) < 0.01

    def test_outlier_price_per_sqft_nulled(self, fetcher):
        """price_per_sqft > $5000 or < $10 should be set to None."""
        df = pd.DataFrame({
            "sale_date": ["2024-01-01T00:00:00", "2024-01-01T00:00:00", "2024-01-01T00:00:00"],
            "sale_price": ["10000000", "100000", "50000"],
            "gross_square_feet": ["100", "100000", "1000"],  # $100k/sqft, $1/sqft, $50/sqft
            "borough": ["1", "1", "1"],
        })
        result = fetcher._clean_property_data(df)

        # Only the $50/sqft row should survive with a valid price_per_sqft
        valid = result[result["price_per_sqft"].notna()]
        assert len(valid) >= 1
        assert (valid["price_per_sqft"] >= 10).all()
        assert (valid["price_per_sqft"] <= 5000).all()

    def test_borough_mapping(self, fetcher, raw_data):
        """Borough codes 1-5 should be mapped to names."""
        result = fetcher._clean_property_data(raw_data)
        expected_boroughs = {"Manhattan", "Bronx", "Brooklyn"}
        assert set(result["borough"].unique()).issubset(
            {"Manhattan", "Bronx", "Brooklyn", "Queens", "Staten Island"}
        )

    def test_all_borough_codes(self, fetcher):
        """All 5 borough codes should map correctly."""
        df = pd.DataFrame({
            "sale_date": ["2024-01-01T00:00:00"] * 5,
            "sale_price": ["500000"] * 5,
            "borough": ["1", "2", "3", "4", "5"],
        })
        result = fetcher._clean_property_data(df)
        assert set(result["borough"]) == {
            "Manhattan", "Bronx", "Brooklyn", "Queens", "Staten Island"
        }

    def test_low_price_filtered(self, fetcher):
        """Sales under $10,000 should be removed."""
        df = pd.DataFrame({
            "sale_date": ["2024-01-01T00:00:00", "2024-01-01T00:00:00"],
            "sale_price": ["5000", "500000"],
            "borough": ["1", "1"],
        })
        result = fetcher._clean_property_data(df)
        assert len(result) == 1
        assert result.iloc[0]["sale_price"] == 500_000

    def test_null_sale_price_dropped(self, fetcher):
        """Rows with null sale_price should be dropped."""
        df = pd.DataFrame({
            "sale_date": ["2024-01-01T00:00:00", "2024-01-01T00:00:00"],
            "sale_price": [None, "500000"],
            "borough": ["1", "1"],
        })
        result = fetcher._clean_property_data(df)
        assert len(result) == 1

    def test_empty_dataframe_returns_empty(self, fetcher):
        """Cleaning an empty DataFrame should return empty."""
        result = fetcher._clean_property_data(pd.DataFrame())
        assert result.empty

    def test_property_type_column_added(self, fetcher, raw_data):
        """property_type column should be added from building_class_category."""
        result = fetcher._clean_property_data(raw_data)
        assert "property_type" in result.columns
