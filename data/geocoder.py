"""
NYC address geocoder using NYC GeoSearch (Pelias).
Adds lat/lng coordinates to property transaction records.
Free API, no key required.
"""

import time

import pandas as pd
import requests

from .config import DATA_STORE_PATH, NYC_GEOSEARCH_URL, SEED_DATA_PATH


class NYCGeocoder:
    """
    Geocodes NYC addresses using the NYC Planning Labs GeoSearch API.
    Pelias-based, free, no API key required.
    Caches results as parquet in data_store/.
    """

    CACHE_NAME = "nyc_transactions_geocoded"
    REQUEST_DELAY = 0.1  # 100ms between requests to be polite

    def __init__(self):
        self.cache_path = DATA_STORE_PATH / f"{self.CACHE_NAME}.parquet"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "CRE-Dashboard/1.0"})

    def geocode_address(
        self, address: str, borough: str = "", zip_code: str = ""
    ) -> tuple[float, float] | None:
        """
        Geocode a single NYC address.

        Args:
            address: Street address (e.g. "501 LEXINGTON AVENUE")
            borough: Borough name (e.g. "Manhattan")
            zip_code: ZIP code (e.g. "10017")

        Returns:
            (latitude, longitude) tuple, or None if not found
        """
        # Build search text — include borough/zip for accuracy
        parts = [address]
        if borough:
            parts.append(borough)
        if zip_code:
            parts.append(f"NY {zip_code}")
        search_text = ", ".join(parts)

        try:
            resp = self.session.get(
                NYC_GEOSEARCH_URL,
                params={"text": search_text, "size": 1},
                timeout=5,
            )
            resp.raise_for_status()
            data = resp.json()

            features = data.get("features", [])
            if features:
                coords = features[0]["geometry"]["coordinates"]
                # GeoJSON is [lng, lat] — return as (lat, lng)
                return (coords[1], coords[0])

        except Exception:
            pass

        return None

    def geocode_transactions(
        self,
        df: pd.DataFrame,
        force_refresh: bool = False,
        progress_callback=None,
    ) -> pd.DataFrame:
        """
        Batch geocode NYC property transactions.

        Args:
            df: DataFrame with 'address', 'borough', 'zip_code' columns
            force_refresh: Re-geocode even if cached
            progress_callback: Optional callable(current, total) for progress updates

        Returns:
            DataFrame with 'latitude' and 'longitude' columns added
        """
        # Return cache if available
        if not force_refresh and self.cache_path.exists():
            return pd.read_parquet(self.cache_path)

        result = df.copy()
        result["latitude"] = None
        result["longitude"] = None

        total = len(result)
        geocoded = 0
        failed = 0

        for idx, row in result.iterrows():
            address = str(row.get("address", ""))
            if not address or address == "nan":
                failed += 1
                continue

            borough = str(row.get("borough", ""))
            zip_code = str(row.get("zip_code", ""))

            coords = self.geocode_address(address, borough, zip_code)

            if coords:
                result.at[idx, "latitude"] = coords[0]
                result.at[idx, "longitude"] = coords[1]
                geocoded += 1
            else:
                failed += 1

            # Progress updates
            current = geocoded + failed
            if progress_callback and current % 50 == 0:
                progress_callback(current, total)

            # Rate limiting
            time.sleep(self.REQUEST_DELAY)

        # Store results
        DATA_STORE_PATH.mkdir(parents=True, exist_ok=True)
        result.to_parquet(self.cache_path)

        print(f"Geocoded {geocoded}/{total} addresses ({failed} failed)")
        return result

    def get_geocoded_transactions(self) -> pd.DataFrame | None:
        """Load cached geocoded transactions, falling back to seed data."""
        if self.cache_path.exists():
            return pd.read_parquet(self.cache_path)
        seed = SEED_DATA_PATH / f"{self.CACHE_NAME}.parquet"
        if seed.exists():
            return pd.read_parquet(seed)
        return None

    def clear_cache(self):
        """Remove cached geocoded data."""
        if self.cache_path.exists():
            self.cache_path.unlink()
