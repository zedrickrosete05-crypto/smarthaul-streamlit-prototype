# utils/geocode.py
from functools import lru_cache
import time
import typing as t
import requests

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

@lru_cache(maxsize=4096)
def geocode_place(place: str) -> t.Optional[tuple[float, float]]:
    """Return (lat, lon) for a place string, or None if not found."""
    if not place or not str(place).strip():
        return None
    try:
        resp = requests.get(
            NOMINATIM_URL,
            params={"q": place, "format": "json", "limit": 1},
            headers={"User-Agent": "SmartHaul/1.0 (Streamlit demo)"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return None
        lat = float(data[0]["lat"])
        lon = float(data[0]["lon"])
        return (lat, lon)
    except Exception:
        return None
    finally:
        # Free service → be polite: ~1 req/sec
        time.sleep(1.0)
