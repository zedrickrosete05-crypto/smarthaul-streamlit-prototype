# utils/travel_time.py
from __future__ import annotations
import math, time, os
from typing import List, Tuple, Optional
import requests
import pandas as pd

# -------- helpers --------
def _chunk(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

def _to_latlng_str(pt: Tuple[float, float]) -> str:
    return f"{pt[0]:.6f},{pt[1]:.6f}"

# -------- Google Distance Matrix (live traffic) --------
# Limits: up to 100 elements/request (origins * destinations),
# practical batch: 10x10 = 100 elements.
def google_distance_matrix_minutes(
    points: List[Tuple[float,float]],
    api_key: str,
    departure_epoch: int,
    traffic_model: str = "best_guess",
    units: str = "metric",
    timeout: int = 10,
) -> pd.DataFrame:
    n = len(points)
    mins = [[0.0]*n for _ in range(n)]
    base = "https://maps.googleapis.com/maps/api/distancematrix/json"

    # batch 10x10 to stay within 100 elements
    batch_size = 10
    for oi, orig_block in enumerate(_chunk(points, batch_size)):
        for di, dest_block in enumerate(_chunk(points, batch_size)):
            origins = "|".join(_to_latlng_str(p) for p in orig_block)
            dests   = "|".join(_to_latlng_str(p) for p in dest_block)
            params = {
                "origins": origins,
                "destinations": dests,
                "departure_time": departure_epoch,      # 'now' for traffic
                "traffic_model": traffic_model,
                "units": units,
                "key": api_key,
            }
            r = requests.get(base, params=params, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            if data.get("status") != "OK":
                raise RuntimeError(f"Google DM status={data.get('status')}: {data}")

            rows = data["rows"]
            for i, row in enumerate(rows):
                elements = row["elements"]
                for j, el in enumerate(elements):
                    # Prefer duration_in_traffic if available, else duration
                    if el.get("status") != "OK":
                        val_min = float("inf")
                    else:
                        sec = el.get("duration_in_traffic", el.get("duration", {})).get("value", 0)
                        val_min = sec / 60.0
                    mins[oi*batch_size + i][di*batch_size + j] = float(val_min)
            time.sleep(0.05)  # be nice to the API

    return pd.DataFrame(mins)

# -------- HERE Routing v8 matrix (traffic) [optional] --------
# You can swap to HERE by implementing your app_code / apiKey here.
def here_matrix_minutes(
    points: List[Tuple[float,float]],
    api_key: str,
    departure_epoch: int,
    timeout: int = 10
) -> pd.DataFrame:
    # Minimal illustrative example using synchronous pairwise calls
    # For production, use the official HERE Matrix API batching.
    n = len(points)
    mins = [[0.0]*n for _ in range(n)]
    base = "https://router.hereapi.com/v8/routes"
    for i, o in enumerate(points):
        for j, d in enumerate(points):
            if i == j:
                mins[i][j] = 0.0
                continue
            params = {
                "origin": _to_latlng_str(o),
                "destination": _to_latlng_str(d),
                "transportMode": "car",
                "departureTime": departure_epoch,
                "return": "travelSummary",
                "apikey": api_key,
            }
            r = requests.get(base, params=params, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            try:
                sec = data["routes"][0]["sections"][0]["summary"]["duration"]
                mins[i][j] = sec / 60.0
            except Exception:
                mins[i][j] = float("inf")
            time.sleep(0.02)
    return pd.DataFrame(mins)

# -------- Open-Meteo (no key) to compute slowdown factors --------
# Simple rule-of-thumb penalties; tune as you gather data.
def weather_slowdown_factor(lat: float, lon: float, epoch: int) -> float:
    """
    Return multiplicative slowdown factor due to weather at given place & time.
    1.00 = no change, 1.10 = +10% time.
    """
    # Query hourly forecast around the departure hour
    # Open-Meteo: free, no key
    from datetime import datetime, timezone
    dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
    date_str = dt.strftime("%Y-%m-%d")
    hour = dt.hour

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat:.4f}&longitude={lon:.4f}"
        "&hourly=precipitation,wind_speed_10m"
        f"&start_date={date_str}&end_date={date_str}"
        "&timezone=UTC"
    )
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        js = r.json()
        hours = js["hourly"]["time"]
        precip = js["hourly"]["precipitation"]
        wind = js["hourly"]["wind_speed_10m"]
        # find this hour
        for t, p, w in zip(hours, precip, wind):
            # t like '2025-10-14T09:00'
            if int(t.split("T")[1].split(":")[0]) == hour:
                factor = 1.0
                if p >= 1.0:    # ≥1mm/h rain
                    factor += 0.10
                if p >= 5.0:    # heavy rain
                    factor += 0.10
                if w >= 35:     # strong wind km/h
                    factor += 0.05
                return min(1.35, factor)
    except Exception:
        pass
    return 1.0

# -------- Main entry: duration matrix with optional weather factor --------
def duration_matrix_minutes(
    points: List[Tuple[float,float]],
    provider: str,
    api_key: Optional[str],
    departure_epoch: int,
    use_weather: bool = True,
) -> pd.DataFrame:
    """
    Returns NxN minutes matrix using the chosen provider (google|here).
    Applies a single weather factor based on the depot point.
    """
    provider = (provider or "google").lower()
    if provider == "here":
        if not api_key:
            raise ValueError("HERE API key required")
        M = here_matrix_minutes(points, api_key, departure_epoch)
    else:
        if not api_key:
            raise ValueError("Google Maps API key required")
        M = google_distance_matrix_minutes(points, api_key, departure_epoch)

    if use_weather and len(points) > 0:
        f = weather_slowdown_factor(points[0][0], points[0][1], departure_epoch)
        M = M * float(f)

    # Safety: replace inf/NaN with big numbers
    M = M.replace([float("inf")], pd.NA).fillna(9e6)
    return M
