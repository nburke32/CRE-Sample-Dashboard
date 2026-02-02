"""
Overture Maps building data fetcher.
Downloads NYC building footprints with geometry and height data
from Overture Maps Foundation's open GeoParquet distribution.

Supports two modes:
  - Commercial-only (12K buildings for standalone view)
  - Transaction-proximity (35K buildings within 50m of a known transaction)
"""

from datetime import datetime

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from .config import DATA_STORE_PATH, NYC_BBOX, SEED_DATA_PATH

COMMERCIAL_SUBTYPES = {"commercial", "industrial", "service"}


class OvertureFetcher:
    """
    Fetches building footprint data from Overture Maps.
    Uses the overturemaps Python package to stream GeoParquet from S3.
    Caches locally in data_store/ as geoparquet.
    """

    COMMERCIAL_CACHE = "overture_nyc_buildings"
    MATCHED_CACHE = "overture_nyc_matched"

    def __init__(self):
        self.commercial_path = DATA_STORE_PATH / f"{self.COMMERCIAL_CACHE}.parquet"
        self.matched_path = DATA_STORE_PATH / f"{self.MATCHED_CACHE}.parquet"

    def _download_all(self, bbox):
        """Stream all buildings from Overture for the bounding box."""
        import overturemaps

        print("Downloading NYC buildings from Overture Maps (this may take 30-60s)...")
        reader = overturemaps.record_batch_reader("building", bbox=bbox)
        if reader is None:
            return gpd.GeoDataFrame()

        table = reader.read_all()
        gdf = gpd.GeoDataFrame.from_arrow(table)
        gdf = gdf.set_crs("EPSG:4326")
        print(f"Downloaded {len(gdf):,} total buildings")

        # Keep only needed columns
        keep_cols = ["id", "geometry", "subtype", "class", "height", "num_floors", "names"]
        gdf = gdf[[c for c in keep_cols if c in gdf.columns]].copy()

        # Extract primary name from nested struct
        if "names" in gdf.columns:
            gdf["name"] = gdf["names"].apply(
                lambda x: x.get("primary", "") if isinstance(x, dict) else ""
            )
            gdf = gdf.drop(columns=["names"])

        # Clean height
        if "height" in gdf.columns:
            gdf["height"] = pd.to_numeric(gdf["height"], errors="coerce").fillna(10.0)

        return gdf

    def fetch_commercial(
        self,
        bbox: tuple[float, float, float, float] = NYC_BBOX,
        force_refresh: bool = False,
    ) -> gpd.GeoDataFrame:
        """
        Download commercial-only building footprints (~12K).
        Used for the standalone Overture view.
        """
        if not force_refresh and self.commercial_path.exists():
            return gpd.read_parquet(self.commercial_path)

        gdf = self._download_all(bbox)
        if gdf.empty:
            return gdf

        gdf = gdf[gdf["subtype"].isin(COMMERCIAL_SUBTYPES)].copy()
        print(f"Filtered to {len(gdf):,} commercial/industrial/service buildings")

        DATA_STORE_PATH.mkdir(parents=True, exist_ok=True)
        gdf.to_parquet(self.commercial_path)
        print(f"Cached to {self.commercial_path}")
        return gdf

    def fetch_matched(
        self,
        transactions_df: pd.DataFrame,
        bbox: tuple[float, float, float, float] = NYC_BBOX,
        buffer_m: float = 50.0,
        force_refresh: bool = False,
    ) -> gpd.GeoDataFrame:
        """
        Download all buildings within buffer_m meters of a geocoded transaction.
        Returns ~35K buildings at 50m buffer — 89% strict-within match rate.
        Used for the Overture + Transactions case study tab.

        Args:
            transactions_df: DataFrame with latitude/longitude columns
            bbox: Bounding box for Overture download
            buffer_m: Buffer distance in meters around each transaction
            force_refresh: Force re-download
        """
        if not force_refresh and self.matched_path.exists():
            return gpd.read_parquet(self.matched_path)

        gdf = self._download_all(bbox)
        if gdf.empty:
            return gdf

        # Build transaction points
        txn_geo = transactions_df.dropna(subset=["latitude", "longitude"])
        points = gpd.GeoDataFrame(
            txn_geo,
            geometry=[Point(xy) for xy in zip(txn_geo["longitude"], txn_geo["latitude"])],
            crs="EPSG:4326",
        )

        # Project to meters, buffer, filter
        gdf_m = gdf.to_crs(epsg=32618)
        points_m = points.to_crs(epsg=32618)
        txn_hull = points_m.geometry.union_all().buffer(buffer_m)
        nearby = gdf_m[gdf_m.geometry.intersects(txn_hull)]
        nearby = nearby.to_crs("EPSG:4326")

        print(f"Filtered to {len(nearby):,} buildings within {buffer_m}m of a transaction")

        # Spatial join with 5m buffer on buildings to account for geocoding offset
        nearby_m = nearby.to_crs(epsg=32618)
        nearby_buffered = nearby_m.copy()
        nearby_buffered["geometry"] = nearby_m.geometry.buffer(5)
        nearby_buffered = nearby_buffered.to_crs("EPSG:4326")
        joined = gpd.sjoin(nearby_buffered, points[["geometry"]], how="left", predicate="contains")
        nearby["has_transaction"] = nearby.index.isin(joined.dropna(subset=["index_right"]).index)

        # Count matched transactions (from the transaction side)
        txn_joined = gpd.sjoin(points[["geometry"]], nearby_buffered[["geometry"]], how="left", predicate="within")
        matched_txn = txn_joined["index_right"].notna().sum()
        unique_matched = len(txn_joined.dropna(subset=["index_right"]).index.unique())
        print(f"Transactions matched: {unique_matched:,} / {len(points):,} ({unique_matched/len(points)*100:.1f}%)")

        DATA_STORE_PATH.mkdir(parents=True, exist_ok=True)
        nearby.to_parquet(self.matched_path)
        print(f"Cached to {self.matched_path}")
        return nearby

    def get_commercial(self) -> gpd.GeoDataFrame | None:
        """Load cached commercial buildings, falling back to seed data."""
        if self.commercial_path.exists():
            return gpd.read_parquet(self.commercial_path)
        seed = SEED_DATA_PATH / f"{self.COMMERCIAL_CACHE}.parquet"
        if seed.exists():
            return gpd.read_parquet(seed)
        return None

    def get_matched(self) -> gpd.GeoDataFrame | None:
        """Load cached transaction-proximity buildings, falling back to seed data."""
        if self.matched_path.exists():
            return gpd.read_parquet(self.matched_path)
        seed = SEED_DATA_PATH / f"{self.MATCHED_CACHE}.parquet"
        if seed.exists():
            return gpd.read_parquet(seed)
        return None

    def get_last_updated(self, which="commercial") -> datetime | None:
        """Get cache timestamp."""
        path = self.commercial_path if which == "commercial" else self.matched_path
        if path.exists():
            return datetime.fromtimestamp(path.stat().st_mtime)
        return None
