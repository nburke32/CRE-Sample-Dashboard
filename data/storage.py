"""
Data storage layer with switchable backends.
Currently supports: Parquet (local)
Future: Snowflake (cloud)
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional
import os

from .config import DATA_STORE_PATH, STORAGE_BACKEND


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
        # elif self.backend == "snowflake":
        #     self._save_snowflake(df, name)
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
        # elif self.backend == "snowflake":
        #     return self._load_snowflake(name, filters)
        else:
            raise ValueError(f"Unknown storage backend: {self.backend}")

    def get_last_updated(self, name: str) -> Optional[datetime]:
        """Get the last update timestamp for a dataset."""
        if self.backend == "parquet":
            return self._get_parquet_modified_time(name)
        # elif self.backend == "snowflake":
        #     return self._get_snowflake_modified_time(name)
        return None

    def dataset_exists(self, name: str) -> bool:
        """Check if a dataset exists."""
        if self.backend == "parquet":
            path = DATA_STORE_PATH / f"{name}.parquet"
            return path.exists()
        # elif self.backend == "snowflake":
        #     return self._snowflake_table_exists(name)
        return False

    def list_datasets(self) -> list[str]:
        """List all available datasets."""
        if self.backend == "parquet":
            return [p.stem for p in DATA_STORE_PATH.glob("*.parquet")]
        # elif self.backend == "snowflake":
        #     return self._list_snowflake_tables()
        return []

    def delete_dataset(self, name: str) -> bool:
        """Delete a dataset. Returns True if successful."""
        if self.backend == "parquet":
            path = DATA_STORE_PATH / f"{name}.parquet"
            if path.exists():
                path.unlink()
                return True
        # elif self.backend == "snowflake":
        #     return self._drop_snowflake_table(name)
        return False

    # =========================================================================
    # PARQUET BACKEND (LOCAL)
    # =========================================================================

    def _save_parquet(self, df: pd.DataFrame, name: str) -> None:
        """Save DataFrame as parquet file."""
        path = DATA_STORE_PATH / f"{name}.parquet"
        df.to_parquet(path, index=False, engine="pyarrow")

    def _load_parquet(self, name: str, filters: Optional[dict] = None) -> Optional[pd.DataFrame]:
        """Load DataFrame from parquet file."""
        path = DATA_STORE_PATH / f"{name}.parquet"
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
    # SNOWFLAKE BACKEND (CLOUD) - READY FOR FUTURE USE
    # =========================================================================
    #
    # Uncomment and configure when ready to switch to Snowflake.
    # Make sure to:
    # 1. Set STORAGE_BACKEND="snowflake" in .env
    # 2. Fill in Snowflake credentials in .env
    # 3. Install snowflake-connector-python
    #
    # -------------------------------------------------------------------------

    # def _get_snowflake_connection(self):
    #     """Get Snowflake connection using credentials from environment."""
    #     import snowflake.connector
    #     from dotenv import load_dotenv
    #     load_dotenv()
    #
    #     return snowflake.connector.connect(
    #         account=os.getenv("SNOWFLAKE_ACCOUNT"),
    #         user=os.getenv("SNOWFLAKE_USER"),
    #         password=os.getenv("SNOWFLAKE_PASSWORD"),
    #         warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
    #         database=os.getenv("SNOWFLAKE_DATABASE"),
    #         schema=os.getenv("SNOWFLAKE_SCHEMA"),
    #     )

    # def _save_snowflake(self, df: pd.DataFrame, name: str) -> None:
    #     """Save DataFrame to Snowflake table."""
    #     from snowflake.connector.pandas_tools import write_pandas
    #
    #     conn = self._get_snowflake_connection()
    #     try:
    #         # Create table if not exists (schema inferred from DataFrame)
    #         write_pandas(
    #             conn,
    #             df,
    #             table_name=name.upper(),
    #             auto_create_table=True,
    #             overwrite=True,
    #         )
    #     finally:
    #         conn.close()

    # def _load_snowflake(self, name: str, filters: Optional[dict] = None) -> Optional[pd.DataFrame]:
    #     """Load DataFrame from Snowflake table."""
    #     conn = self._get_snowflake_connection()
    #     try:
    #         query = f"SELECT * FROM {name.upper()}"
    #
    #         if filters:
    #             where_clauses = []
    #             for col, val in filters.items():
    #                 if isinstance(val, list):
    #                     vals = ", ".join([f"'{v}'" for v in val])
    #                     where_clauses.append(f"{col} IN ({vals})")
    #                 else:
    #                     where_clauses.append(f"{col} = '{val}'")
    #             query += " WHERE " + " AND ".join(where_clauses)
    #
    #         cursor = conn.cursor()
    #         cursor.execute(query)
    #         df = cursor.fetch_pandas_all()
    #         return df
    #     except Exception:
    #         return None
    #     finally:
    #         conn.close()

    # def _get_snowflake_modified_time(self, name: str) -> Optional[datetime]:
    #     """Get last modified time from Snowflake table metadata."""
    #     conn = self._get_snowflake_connection()
    #     try:
    #         cursor = conn.cursor()
    #         cursor.execute(f"""
    #             SELECT LAST_ALTERED
    #             FROM INFORMATION_SCHEMA.TABLES
    #             WHERE TABLE_NAME = '{name.upper()}'
    #         """)
    #         result = cursor.fetchone()
    #         return result[0] if result else None
    #     except Exception:
    #         return None
    #     finally:
    #         conn.close()

    # def _snowflake_table_exists(self, name: str) -> bool:
    #     """Check if Snowflake table exists."""
    #     conn = self._get_snowflake_connection()
    #     try:
    #         cursor = conn.cursor()
    #         cursor.execute(f"""
    #             SELECT COUNT(*)
    #             FROM INFORMATION_SCHEMA.TABLES
    #             WHERE TABLE_NAME = '{name.upper()}'
    #         """)
    #         result = cursor.fetchone()
    #         return result[0] > 0
    #     except Exception:
    #         return False
    #     finally:
    #         conn.close()

    # def _list_snowflake_tables(self) -> list[str]:
    #     """List all tables in Snowflake schema."""
    #     conn = self._get_snowflake_connection()
    #     try:
    #         cursor = conn.cursor()
    #         cursor.execute("SHOW TABLES")
    #         return [row[1] for row in cursor.fetchall()]
    #     except Exception:
    #         return []
    #     finally:
    #         conn.close()

    # def _drop_snowflake_table(self, name: str) -> bool:
    #     """Drop a Snowflake table."""
    #     conn = self._get_snowflake_connection()
    #     try:
    #         cursor = conn.cursor()
    #         cursor.execute(f"DROP TABLE IF EXISTS {name.upper()}")
    #         return True
    #     except Exception:
    #         return False
    #     finally:
    #         conn.close()
