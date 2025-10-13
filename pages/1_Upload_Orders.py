# pages/1_Upload_Orders.py
from __future__ import annotations

import io
import re
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

# ---------------------------------------------------
# Page config
# ---------------------------------------------------
st.set_page_config(page_title="SmartHaul – Upload Orders", layout="wide")
st.title("📥 Upload Orders")

# ---------------------------------------------------
# Columns & schema
# ---------------------------------------------------
# Required logical fields. Users may supply HH:MM strings; we normalize.
REQUIRED = ["order_id", "tw_start", "tw_end", "service_min"]
OPTIONAL = ["lat", "lon", "place", "demand", "priority", "hub", "notes"]

CANONICAL_ORDER = [
    "order_id", "place", "lat", "lon",
    "tw_start", "tw_end",
    "service_min", "demand", "priority", "hub", "notes",
    # normalized/minute cols (produced here)
    "tw_start_min", "tw_end_min", "service_time_min"
]

LAT_BOUNDS = (-90.0, 90.0)
LON_BOUNDS = (-180.0, 180.0)

# A tiny dict of "common aliases" -> canonical
ALIASES = {
    "latitude": "lat", "long": "lon", "lng": "lon", "longitude": "lon",
    "time_window_start": "tw_start", "tw_start_min": "tw_start",  # accept *_min but we’ll recompute
    "time_window_end": "tw_end", "tw_end_min": "tw_end",
    "service_time": "service_min", "service_minutes": "service_min",
    "svc_min": "service_min",
}

# ---------------------------------------------------
# Utilities
# ---------------------------------------------------
def _to_minutes_hhmm(x: Any) -> Optional[int]:
    """Parse 'HH:MM' or pandas Timestamp -> minutes from midnight."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    if not s:
        return None
    # already looks like "HH:MM"
    if re.fullmatch(r"\d{1,2}:\d{2}", s):
        h, m = [int(t) for t in s.split(":")]
        if not (0 <= h < 24 and 0 <= m < 60):
            return None
        return h * 60 + m
    # try datetime-ish
    try:
        ts = pd.to_datetime(s, errors="raise")
        return ts.hour * 60 + ts.minute
    except Exception:
        return None

def _within(val: Any, lo: float, hi: float) -> bool:
    try:
        fv = float(val)
    except Exception:
        return False
    return lo <= fv <= hi

def _norm_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Lower/strip headers, map aliases, keep original if unique."""
    mapping = {}
    seen = set()
    for c in df.columns:
        low = c.strip().lower()
        target = ALIASES.get(low, low)
        # Avoid collisions: if two columns collapse to same name, keep original for the latter
        if target in seen:
            target = c  # keep original unique
        mapping[c] = target
        seen.add(target)
    df = df.rename(columns=mapping)
    return df

# ---------------------------------------------------
# Geocoding with cache
# ---------------------------------------------------
@st.cache_data(show_spinner=False)
def _geocode_osm(place: str) -> Optional[Tuple[float, float]]:
    """Free OSM Nominatim (polite). Swap with your production geocoder as needed."""
    import urllib.parse, urllib.request, json
    base = "https://nominatim.openstreetmap.org/search"
    url = f"{base}?q={urllib.parse.quote(place)}&format=json&limit=1"
    req = urllib.request.Request(url, headers={"User-Agent": "SmartHaul/1.0"})
    with urllib.request.urlopen(req, timeout=8) as r:
        data = json.loads(r.read().decode("utf-8"))
    if data:
        return float(data[0]["lat"]), float(data[0]["lon"])
    return None

def geocode_unique_places(df: pd.DataFrame, *, suffix_country: str = "", delay_s: float = 1.0) -> tuple[pd.DataFrame, list[dict]]:
    """
    Geocode only unique places missing coordinates.
    Uses st.session_state['geocode_cache'] to persist results across runs.
    """
    cache: Dict[str, Tuple[float, float]] = st.session_state.setdefault("geocode_cache", {})

    # rows needing geocode
    need_mask = (df["place"].fillna("").str.strip() != "") & (df["lat"].isna() | df["lon"].isna())
    if not need_mask.any():
        return df, []

    places = df.loc[need_mask, "place"].astype(str).str.strip()
    unique_places = sorted(set(places.tolist()))

    notes = []
    prog = st.progress(0.0, text="Geocoding places…")
    for i, p in enumerate(unique_places, start=1):
        key = p + (f", {suffix_country}" if suffix_country else "")
        if p in cache:
            latlon = cache[p]
        else:
            try:
                latlon = _geocode_osm(key)
            except Exception as e:
                notes.append({"place": p, "level": "warning", "message": f"geocode error: {e}"})
                latlon = None
            # be nice to OSM
            time.sleep(max(0.2, delay_s))
            if latlon:
                cache[p] = latlon

        if not latlon:
            notes.append({"place": p, "level": "warning", "message": "no geocode result"})

        prog.progress(i / len(unique_places), text=f"Geocoding places… ({i}/{len(unique_places)})")

    # fill back
    for idx in df.index[need_mask]:
        p = str(df.at[idx, "place"]).strip()
        if p in cache:
            lat, lon = cache[p]
            if _within(lat, *LAT_BOUNDS) and _within(lon, *LON_BOUNDS):
                df.at[idx, "lat"] = float(lat)
                df.at[idx, "lon"] = float(lon)
            else:
                notes.append({"place": p, "level": "warning", "message": "out-of-bounds coordinates"})
    prog.empty()
    return df, notes

# ---------------------------------------------------
# CSV template
# ---------------------------------------------------
def _template_csv() -> bytes:
    tmpl = pd.DataFrame({
        "order_id": ["O-1001", "O-1002", "O-1003"],
        "place": ["JY Square, Cebu City", "Cebu IT Park", "SM City Cebu"],
        "tw_start": ["08:30", "09:00", "10:00"],
        "tw_end":   ["11:00", "12:00", "13:00"],
        "service_min": [7, 5, 10],
        "lat": [np.nan, np.nan, np.nan],
        "lon": [np.nan, np.nan, np.nan],
        "demand": [10, 5, 8],
        "priority": [2, 1, 3],
        "hub": ["Main", "Main", "Main"],
        "notes": ["fragile", "", ""],
    })
    buf = io.StringIO()
    tmpl.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

with st.expander("Download CSV template"):
    st.download_button("Download template.csv", data=_template_csv(),
                       file_name="orders_template_places.csv", mime="text/csv")
    st.caption("Required: order_id, tw_start (HH:MM), tw_end (HH:MM), service_min (minutes). "
               "Provide either lat+lon **or** place. Optional: demand, priority, hub, notes.")

# ---------------------------------------------------
# File uploader & options
# ---------------------------------------------------
left, right = st.columns([2, 1])
with left:
    file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"], key="orders_csv", accept_multiple_files=False)
with right:
    st.write("**Options**")
    use_geocoding = st.toggle("Geocode missing coords", value=True)
    bias_ph = st.toggle("Bias to Philippines", value=True, help="Append ', Philippines' to place queries")
    throttle = st.slider("Geocode delay (sec)", 0.2, 2.0, 1.0, 0.1)

if file is None:
    st.caption("Tip: you can upload multiple times; the latest valid upload replaces the current orders.")
    st.stop()

# ---------------------------------------------------
# Read file
# ---------------------------------------------------
try:
    if file.name.lower().endswith(".csv"):
        df_raw = pd.read_csv(file)
    else:
        df_raw = pd.read_excel(file)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

st.subheader("Preview (raw upload)")
st.dataframe(df_raw.head(50), use_container_width=True)

# ---------------------------------------------------
# Normalize headers & ensure columns
# ---------------------------------------------------
df = _norm_headers(df_raw.copy())

for col in REQUIRED + OPTIONAL:
    if col not in df.columns:
        df[col] = pd.NA

# Keep only known columns (but don’t drop unknowns yet—show to user)
unknown_cols = [c for c in df.columns if c not in set(REQUIRED + OPTIONAL)]
if unknown_cols:
    st.info(f"Ignoring unrecognized columns: {', '.join(unknown_cols)}")

# Strip strings
for c in ["order_id", "place", "tw_start", "tw_end", "hub", "notes"]:
    if c in df.columns:
        df[c] = df[c].astype("string").str.strip()

# Convert numeric-ish
for c in ["lat", "lon", "service_min", "demand", "priority"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# ---------------------------------------------------
# Validate
# ---------------------------------------------------
issues = []

# Required columns presence
for c in REQUIRED:
    if c not in df.columns:
        issues.append({"row": None, "field": c, "level": "error", "message": f"Missing required column '{c}'"})

# Row-level checks
for idx, row in df.iterrows():
    if not str(row["order_id"] or "").strip():
        issues.append({"row": idx, "field": "order_id", "level": "error", "message": "order_id is required"})

    # service_min
    if pd.isna(row["service_min"]) or float(row["service_min"]) < 0:
        issues.append({"row": idx, "field": "service_min", "level": "error", "message": "service_min must be ≥ 0"})

    # time windows
    ws = _to_minutes_hhmm(row["tw_start"])
    we = _to_minutes_hhmm(row["tw_end"])
    if row["tw_start"] and ws is None:
        issues.append({"row": idx, "field": "tw_start", "level": "error", "message": f"Unrecognized time '{row['tw_start']}'"})
    if row["tw_end"] and we is None:
        issues.append({"row": idx, "field": "tw_end", "level": "error", "message": f"Unrecognized time '{row['tw_end']}'"})
    if (ws is not None) and (we is not None) and (we < ws):
        issues.append({"row": idx, "field": "tw_end", "level": "error", "message": "tw_end earlier than tw_start"})

# Duplicates
dups = df["order_id"].duplicated(keep=False)
if dups.any():
    for i in df.index[dups]:
        issues.append({"row": i, "field": "order_id", "level": "error", "message": "duplicate order_id"})

# If both coords are provided, sanity-check bounds
for idx, row in df.iterrows():
    if not (pd.isna(row["lat"]) or pd.isna(row["lon"])):
        if not (_within(row["lat"], *LAT_BOUNDS) and _within(row["lon"], *LON_BOUNDS)):
            issues.append({"row": idx, "field": "lat/lon", "level": "error", "message": "Coordinates out of bounds"})

issues_df = pd.DataFrame(issues)
if not issues_df.empty:
    if (issues_df["level"] == "error").any():
        st.error("Validation found errors. Please fix these and re-upload.")
    else:
        st.warning("There are warnings you may want to review.")
    st.dataframe(issues_df.sort_values(["level", "row", "field"], na_position="first"),
                 use_container_width=True)

# Early stop on blocking errors
if not issues_df.empty and (issues_df["level"] == "error").any():
    st.stop()

# ---------------------------------------------------
# Geocoding (only for rows with missing coords & non-empty place)
# ---------------------------------------------------
if use_geocoding:
    with st.spinner("Geocoding rows with missing coordinates from 'place'…"):
        df, notes = geocode_unique_places(
            df,
            suffix_country="Philippines" if bias_ph else "",
            delay_s=float(throttle),
        )
    if notes:
        st.info("Geocoding notes")
        st.dataframe(pd.DataFrame(notes), use_container_width=True)

# Final check: we require either lat+lon OR place geocoded
missing_coords = df["lat"].isna() | df["lon"].isna()
missing_with_no_place = missing_coords & (df["place"].fillna("").str.strip() == "")
if missing_with_no_place.any():
    st.error(f"{int(missing_with_no_place.sum())} row(s) missing both coordinates and place. Provide at least one.")
    st.dataframe(df.loc[missing_with_no_place, ["order_id", "place", "lat", "lon"]], use_container_width=True)
    st.stop()

# ---------------------------------------------------
# Normalize minute columns for downstream pages
# ---------------------------------------------------
df["tw_start_min"] = df["tw_start"].map(_to_minutes_hhmm)
df["tw_end_min"] = df["tw_end"].map(_to_minutes_hhmm)
df["service_time_min"] = pd.to_numeric(df["service_min"], errors="coerce").fillna(0).astype(int)

# Round lat/lon neatly for display (keeps full precision for computation)
df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

# Keep canonical ordering for clarity
for c in CANONICAL_ORDER:
    if c not in df.columns:
        df[c] = pd.NA
df = df[CANONICAL_ORDER].copy()

# ---------------------------------------------------
# Summary + preview
# ---------------------------------------------------
ok_rows = len(df.dropna(subset=["lat", "lon"]))
st.success(f"Loaded {ok_rows} order(s) ✅")

stats_cols = st.columns(3)
with stats_cols[0]:
    st.metric("Total orders", len(df))
with stats_cols[1]:
    st.metric("With coordinates", ok_rows)
with stats_cols[2]:
    gaps = int((df["tw_start_min"].isna() | df["tw_end_min"].isna()).sum())
    st.metric("Missing time windows", gaps)

st.dataframe(df, use_container_width=True)

# Map preview (if we have coordinates)
if ok_rows > 0:
    st.subheader("Map preview")
    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(
            latitude=float(df["lat"].dropna().mean()),
            longitude=float(df["lon"].dropna().mean()),
            zoom=11
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=df.dropna(subset=["lat", "lon"]),
                get_position="[lon, lat]",
                get_radius=60,
                pickable=True,
            )
        ],
        tooltip={"text": "{order_id}\n{place}"}
    ))

# ---------------------------------------------------
# Persist & export
# ---------------------------------------------------
st.session_state["orders_df"] = df.copy()

st.download_button(
    "⬇️ Download cleaned orders",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="orders_cleaned.csv",
    mime="text/csv",
)

st.info("Next: open **Optimize Routes** to generate assignments and ETAs.")
