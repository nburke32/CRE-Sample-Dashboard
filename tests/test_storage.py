"""
Tests for data/storage.py — filtering logic, seed fallback, error handling.
"""

from datetime import datetime

import pandas as pd
import pytest

from data.storage import DataStorage


class TestStorageFiltering:
    """Test the filter logic in _load_parquet."""

    @pytest.fixture
    def storage_with_data(self, tmp_path):
        """Create a storage instance with test data saved."""
        import os
        os.environ["STORAGE_BACKEND"] = "parquet"

        storage = DataStorage(backend="parquet")

        # Save a test dataset
        df = pd.DataFrame({
            "metro_code": ["NYC", "NYC", "LAX", "LAX", "CHI"],
            "sector": ["office", "retail", "office", "industrial", "office"],
            "value": [100, 200, 300, 400, 500],
        })
        storage.save_dataframe(df, "test_filter_data")
        yield storage
        # Cleanup
        storage.delete_dataset("test_filter_data")

    def test_filter_single_value(self, storage_with_data):
        """Filter by a single column value."""
        result = storage_with_data.load_dataframe(
            "test_filter_data", filters={"metro_code": "NYC"}
        )
        assert len(result) == 2
        assert (result["metro_code"] == "NYC").all()

    def test_filter_list_values(self, storage_with_data):
        """Filter by a list of values (isin)."""
        result = storage_with_data.load_dataframe(
            "test_filter_data", filters={"metro_code": ["NYC", "CHI"]}
        )
        assert len(result) == 3
        assert set(result["metro_code"].unique()) == {"NYC", "CHI"}

    def test_filter_multiple_columns(self, storage_with_data):
        """Filter by multiple columns simultaneously."""
        result = storage_with_data.load_dataframe(
            "test_filter_data", filters={"metro_code": "NYC", "sector": "office"}
        )
        assert len(result) == 1
        assert result.iloc[0]["value"] == 100

    def test_filter_nonexistent_column_ignored(self, storage_with_data):
        """Filtering on a column that doesn't exist should not crash."""
        result = storage_with_data.load_dataframe(
            "test_filter_data", filters={"nonexistent_col": "foo"}
        )
        # Should return all rows since the filter column doesn't exist
        assert len(result) == 5

    def test_filter_no_matches(self, storage_with_data):
        """Filter that matches nothing should return empty DataFrame."""
        result = storage_with_data.load_dataframe(
            "test_filter_data", filters={"metro_code": "DOES_NOT_EXIST"}
        )
        assert len(result) == 0

    def test_no_filter_returns_all(self, storage_with_data):
        """No filter should return the full dataset."""
        result = storage_with_data.load_dataframe("test_filter_data")
        assert len(result) == 5


class TestStorageSeedFallback:
    """Test seed data fallback behavior."""

    def test_load_nonexistent_returns_none(self):
        """Loading a dataset that doesn't exist anywhere returns None."""
        storage = DataStorage()
        result = storage.load_dataframe("completely_nonexistent_dataset_xyz_123")
        assert result is None

    def test_dataset_exists_false_for_missing(self):
        """dataset_exists returns False for nonexistent data."""
        storage = DataStorage()
        assert storage.dataset_exists("completely_nonexistent_dataset_xyz_123") is False


class TestStorageBackendValidation:
    """Test backend validation."""

    def test_invalid_backend_raises_on_save(self):
        """Saving with an unknown backend should raise ValueError."""
        storage = DataStorage(backend="parquet")
        storage.backend = "unknown_backend"  # Force invalid backend
        with pytest.raises(ValueError, match="Unknown storage backend"):
            storage.save_dataframe(pd.DataFrame({"a": [1]}), "test")

    def test_invalid_backend_raises_on_load(self):
        """Loading with an unknown backend should raise ValueError."""
        storage = DataStorage(backend="parquet")
        storage.backend = "unknown_backend"
        with pytest.raises(ValueError, match="Unknown storage backend"):
            storage.load_dataframe("test")


class TestStorageCRUD:
    """Test save/load/delete round-trip."""

    def test_save_load_roundtrip(self):
        """Save and load should preserve data."""
        storage = DataStorage()
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "value": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        storage.save_dataframe(df, "test_roundtrip")

        loaded = storage.load_dataframe("test_roundtrip")
        assert loaded is not None
        assert len(loaded) == 5
        assert list(loaded.columns) == list(df.columns)

        # Cleanup
        storage.delete_dataset("test_roundtrip")

    def test_delete_returns_true_when_exists(self):
        """delete_dataset returns True when file exists."""
        storage = DataStorage()
        storage.save_dataframe(pd.DataFrame({"a": [1]}), "test_delete_me")
        assert storage.delete_dataset("test_delete_me") is True

    def test_delete_returns_false_when_missing(self):
        """delete_dataset returns False when file doesn't exist."""
        storage = DataStorage()
        assert storage.delete_dataset("nonexistent_xyz") is False

    def test_get_last_updated_returns_datetime(self):
        """get_last_updated should return a datetime for existing datasets."""
        storage = DataStorage()
        storage.save_dataframe(pd.DataFrame({"a": [1]}), "test_mtime")

        mtime = storage.get_last_updated("test_mtime")
        assert isinstance(mtime, datetime)

        storage.delete_dataset("test_mtime")

    def test_get_last_updated_returns_none_for_missing(self):
        """get_last_updated should return None for nonexistent datasets."""
        storage = DataStorage()
        assert storage.get_last_updated("nonexistent_xyz") is None

    def test_list_datasets_returns_list(self):
        """list_datasets should return a list of strings."""
        storage = DataStorage()
        datasets = storage.list_datasets()
        assert isinstance(datasets, list)
        assert all(isinstance(d, str) for d in datasets)
