# smarthaul_core/distance.py
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# A small local cache for distance matrices (persists across reruns on the same host)
DATA_DIR = Path(".cache")
DATA_DIR.mkdir(exist_ok=True)

def _hash_key(obj) -> str:
    """Stable short hash for caching."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]

def haversine_matrix(points: list[tuple[float, float]]) -> np.ndarray:
    """
    Fast, vectorized great-circle distance matrix (meters).
    Replace later with OSRM/ORS durations; keep order of points stable.
    """
    R = 6371000.0  # Earth radius (m)
    pts = np.radians(np.array(points, dtype=float))
    lat = pts[:, 0][:, None]
    lon = pts[:, 1][:, None]
    dlat = lat - lat.T
    dlon = lon - lon.T
    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.minimum(1, np.sqrt(a)))

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_distance_matrix(points: list[tuple[float, float]], profile: str = "driving") -> np.ndarray:
    """
    Returns an NxN matrix in meters. Cached via Streamlit AND written to .cache as Parquet.
    Keyed by the exact set/order of points and the profile.
    """
    key = _hash_key({"pts": points, "profile": profile})
    fpath = DATA_DIR / f"dist_{key}.parquet"

    # Disk cache hit?
    if fpath.exists():
        return pd.read_parquet(fpath).values

    # Compute (fallback is Haversine for now; later swap with OSRM/ORS durations)
    mat = haversine_matrix(points)

    # Persist to disk so repeated runs don’t recompute
    pd.DataFrame(mat).to_parquet(fpath, index=False)
    return mat
