"""
Data storage layer with switchable backends.
Currently supports: Parquet (local)
Future: Snowflake (cloud)
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional

from .config import DATA_STORE_PATH, SEED_DATA_PATH, STORAGE_BACKEND


class DataStorage:
    """
    Unified interface for data storage.
    Abstracts away the backend so we can switch from Parquet to Snowflake.
    """

    def __init__(self, backend: Optional[str] = None):
        self.backend = backend or STORAGE_BACKEND
        self._ensure_storage_exists()

    def _ensure_storage_exists(self):
        """Create storage directory if using parquet backend."""
        if self.backend == "parquet":
            DATA_STORE_PATH.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # PUBLIC API - Use these methods in the app
    # =========================================================================

    def save_dataframe(self, df: pd.DataFrame, name: str, partition_by: Optional[str] = None) -> None:
        """
        Save a DataFrame to storage.

        Args:
            df: DataFrame to save
            name: Dataset name (e.g., "fred_national", "fred_metros", "reit_prices")
            partition_by: Optional column to partition by (e.g., "metro_code")
        """
        if self.backend == "parquet":
            self._save_parquet(df, name)
        else:
            raise ValueError(f"Unknown storage backend: {self.backend}")

    def load_dataframe(self, name: str, filters: Optional[dict] = None) -> Optional[pd.DataFrame]:
        """
        Load a DataFrame from storage.

        Args:
            name: Dataset name
            filters: Optional dict of column: value to filter by

        Returns:
            DataFrame or None if not found
        """
        if self.backend == "parquet":
            return self._load_parquet(name, filters)
        else:
            raise ValueError(f"Unknown storage backend: {self.backend}")

    def get_last_updated(self, name: str) -> Optional[datetime]:
        """Get the last update timestamp for a dataset."""
        if self.backend == "parquet":
            return self._get_parquet_modified_time(name)
        return None

    def dataset_exists(self, name: str) -> bool:
        """Check if a dataset exists (cache or seed)."""
        if self.backend == "parquet":
            path = DATA_STORE_PATH / f"{name}.parquet"
            if path.exists():
                return True
            seed = SEED_DATA_PATH / f"{name}.parquet"
            return seed.exists()
        return False

    def list_datasets(self) -> list[str]:
        """List all available datasets."""
        if self.backend == "parquet":
            return [p.stem for p in DATA_STORE_PATH.glob("*.parquet")]
        return []

    def delete_dataset(self, name: str) -> bool:
        """Delete a dataset. Returns True if successful."""
        if self.backend == "parquet":
            path = DATA_STORE_PATH / f"{name}.parquet"
            if path.exists():
                path.unlink()
                return True
        return False

    # =========================================================================
    # PARQUET BACKEND (LOCAL)
    # =========================================================================

    def _save_parquet(self, df: pd.DataFrame, name: str) -> None:
        """Save DataFrame as parquet file."""
        path = DATA_STORE_PATH / f"{name}.parquet"
        df.to_parquet(path, index=False, engine="pyarrow")

    def _load_parquet(self, name: str, filters: Optional[dict] = None) -> Optional[pd.DataFrame]:
        """Load DataFrame from parquet file, falling back to seed data."""
        path = DATA_STORE_PATH / f"{name}.parquet"
        if not path.exists():
            path = SEED_DATA_PATH / f"{name}.parquet"
            if not path.exists():
                return None

        df = pd.read_parquet(path, engine="pyarrow")

        if filters:
            for col, val in filters.items():
                if col in df.columns:
                    if isinstance(val, list):
                        df = df[df[col].isin(val)]
                    else:
                        df = df[df[col] == val]

        return df

    def _get_parquet_modified_time(self, name: str) -> Optional[datetime]:
        """Get parquet file modification time."""
        path = DATA_STORE_PATH / f"{name}.parquet"
        if path.exists():
            return datetime.fromtimestamp(path.stat().st_mtime)
        return None

    # =========================================================================
    # SNOWFLAKE BACKEND (CLOUD) - TODO
    # =========================================================================
    # Future: implement Snowflake backend using snowflake-connector-python.
    # Use parameterized queries (cursor.execute(query, params)) to avoid
    # SQL injection. Set STORAGE_BACKEND="snowflake" in .env when ready.
