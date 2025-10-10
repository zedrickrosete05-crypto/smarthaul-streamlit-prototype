# smarthaul_core/solver.py
import hashlib, json
from typing import List, Dict, Any
import numpy as np
import streamlit as st

def _digest(arr: np.ndarray) -> str:
    """Short stable digest for caching numeric inputs."""
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]

@st.cache_data(ttl=180, show_spinner=False)
def solve_greedy(dist: np.ndarray, demand: np.ndarray, vehicle_caps: List[int]) -> Dict[str, Any]:
    """
    Very simple placeholder 'solver' so we can cache heavy inputs and wire the UI.
    - dist: NxN matrix (meters)
    - demand: length-N vector
    - vehicle_caps: list of capacities

    Returns:
      {
        "routes": [ [0, 5, 9, 0], [0, 2, 7, 0], ... ],   # depot assumed index 0 for now
        "km": 123.4
      }
    """
    # NOTE: Replace this with your actual greedy/OR-Tools later.
    # For now we just return empty routes and total km as a sanity KPI.

    # Cache key is built automatically by Streamlit from args,
    # but we also compute hashes to keep it deterministic if you refactor.
    _ = json.dumps({
        "dist": _digest(dist.astype(np.float64)),
        "dem": _digest(demand.astype(np.float64)),
        "caps": vehicle_caps
    }, sort_keys=True)

    total_km = float(dist.sum()) / 1000.0
    return {"routes": [], "km": total_km}
