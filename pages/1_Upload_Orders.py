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

st.set_page_config(page_title="SmartHaul – Upload Orders", layout="wide")
st.title("📥 Upload Orders")

# ---------------------------------------------------
# Column schema
# ---------------------------------------------------
REQUIRED = ["order_id", "tw_start", "tw_end", "service_min"]
OPTIONAL = ["lat", "lon", "place", "demand", "priority", "hub", "notes"]

CANONICAL_ORDER = [
    "order_id", "place", "lat", "lon",
    "tw_start", "tw_end", "service_min",
    "demand", "priority", "hub", "notes",
    # normalized/min columns (produced here)
    "tw_start_min", "tw_end_min", "service_time_min"
]

ALIASES = {
    "latitude": "lat", "long": "lon", "lng": "lon", "longitude": "lon",
    "time_window_start": "tw_start", "tw_start_min": "tw_start",
    "time_window_end": "tw_end", "tw_end_min": "tw_end",
    "service_time": "service_min", "service_minutes": "service_min",
    "svc_min": "service_min",
}

LAT_BOUNDS = (-90.0, 90.0)
LON_BOUNDS = (-180.0, 180.0)

# ---------------------------------------------------
# Helpers
# ---------------------------------------------------
_AMPM_RE = re.compile(r"^\s*(1[0-2]|0?[1-9]):([0-5][0-9])\s*([AaPp][Mm])\s*$")
_24H_RE  = re.compile(r"^\s*([01]?\d|2[0-3]):([0-5]\d)\s*$")

def to_minutes_any(x: Any) -> Optional[int]:
    """
    Accepts:
      - 'HH:MM AM/PM'  (e.g., '1:05 pm', '01:05 PM')
      - 'HH:MM' 24h    (e.g., '13:05', '8:05')
      - pandas/ISO datetimes
    Returns minutes after midnight, or None.
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    if not s:
        return None

    m = _AMPM_RE.match(s)
    if m:
        h = int(m.group(1))
        mm = int(m.group(2))
        ampm = m.group(3).upper()
        if ampm == "PM" and h != 12:
            h += 12
        if ampm == "AM" and h == 12:
            h = 0
        return h * 60 + mm

    m = _24H_RE.match(s)
    if m:
        h = int(m.group(1)); mm = int(m.group(2))
        return h * 60 + mm

    # try datetime-like
    try:
        ts = pd.to_datetime(s, errors="raise")
        return ts.hour * 60 + ts.minute
    except Exception:
        return None

def within(val: Any, lo: float, hi: float) -> bool:
    try:
        fv = float(val)
    except Exception:
        return False
    return lo <= fv <= hi

def norm_headers(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    seen = set()
    for c in df.columns:
        key = c.strip().lower()
        target = ALIASES.get(key, key)
        if target in seen:  # avoid collisions
            target = c
        mapping[c] = target
        seen.add(target)
    return df.rename(columns=mapping)

# ---------------------------------------------------
# Geocoding (unique places; cached)
# ---------------------------------------------------
@st.cache_data(show_spinner=False)
def geocode_osm(place: str) -> Optional[Tuple[float, float]]:
    import urllib.parse, urllib.request, json
    base = "https://nominatim.openstreetmap.org/search"
    url = f"{base}?q={urllib.parse.quote(place)}&format=json&limit=1"
    req = urllib.request.Request(url, headers={"User-Agent": "SmartHaul/1.0"})
    with urllib.request.urlopen(req, timeout=8) as r:
        data = json.loads(r.read().decode("utf-8"))
    if data:
        return float(data[0]["lat"]), float(data[0]["lon"])
    return None

def geocode_unique(df: pd.DataFrame, *, suffix_country: str = "", delay_s: float = 1.0):
    cache: Dict[str, Tuple[float, float]] = st.session_state.setdefault("geocode_cache", {})
    need_mask = (df["place"].fillna("").str.strip() != "") & (df["lat"].isna() | df["lon"].isna())
    if not need_mask.any():
        return df, []

    uniq = sorted(set(df.loc[need_mask, "place"].astype(str).str.strip().tolist()))
    notes = []
    prog = st.progress(0.0, text="Geocoding places…")
    for i, p in enumerate(uniq, start=1):
        key = p + (f", {suffix_country}" if suffix_country else "")
        if p not in cache:
            try:
                latlon = geocode_osm(key)
                if latlon:
                    cache[p] = (float(latlon[0]), float(latlon[1]))
                else:
                    notes.append({"place": p, "level": "warning", "message": "no geocode result"})
            except Exception as e:
                notes.append({"place": p, "level": "warning", "message": f"geocode error: {e}"})
            time.sleep(max(0.2, delay_s))
        prog.progress(i / len(uniq), text=f"Geocoding places… ({i}/{len(uniq)})")

    for idx in df.index[need_mask]:
        p = str(df.at[idx, "place"]).strip()
        if p in cache:
            lat, lon = cache[p]
            if within(lat, *LAT_BOUNDS) and within(lon, *LON_BOUNDS):
                df.at[idx, "lat"] = float(lat)
                df.at[idx, "lon"] = float(lon)
            else:
                notes.append({"place": p, "level": "warning", "message": "out-of-bounds coordinates"})
    prog.empty()
    return df, notes

# ---------------------------------------------------
# Template (AM/PM)
# ---------------------------------------------------
def template_csv() -> bytes:
    tmpl = pd.DataFrame({
        "order_id": ["O-1001", "O-1002", "O-1003"],
        "place": ["JY Square, Cebu City", "Cebu IT Park", "SM City Cebu"],
        "tw_start": ["08:30 AM", "09:00 AM", "10:00 AM"],
        "tw_end":   ["11:00 AM", "12:00 PM", "01:00 PM"],
        "service_min": [7, 5, 10],
        "lat": [np.nan, np.nan, np.nan],
        "lon": [np.nan, np.nan, np.nan],
    })
    buf = io.StringIO()
    tmpl.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

with st.expander("Download CSV template"):
    st.download_button("Download template.csv", data=template_csv(),
                       file_name="orders_template_places.csv", mime="text/csv")
    st.caption("Required: order_id, tw_start, tw_end, service_min. "
               "Times can be **AM/PM** (e.g. '1:05 PM') or **24h** ('13:05'). "
               "Provide either **lat+lon** or **place** (will geocode).")

# ---------------------------------------------------
# Upload + options
# ---------------------------------------------------
left, right = st.columns([2, 1])
with left:
    file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"], key="orders_csv", accept_multiple_files=False)
with right:
    st.write("**Options**")
    use_geocoding = st.toggle("Geocode missing coords", value=True)
    bias_ph = st.toggle("Bias to Philippines", value=True,
                        help="Append ', Philippines' to place queries")
    throttle = st.slider("Geocode delay (sec)", 0.2, 2.0, 1.0, 0.1)

if not file:
    st.caption("Tip: You can upload multiple times; the latest valid upload replaces the current orders.")
    st.stop()

# Read
try:
    df_raw = pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

st.subheader("Preview (raw upload)")
st.dataframe(df_raw.head(50), use_container_width=True)

# ---------------------------------------------------
# Normalize headers & coerce types
# ---------------------------------------------------
df = norm_headers(df_raw.copy())

for col in REQUIRED + OPTIONAL:
    if col not in df.columns:
        df[col] = pd.NA

unknown_cols = [c for c in df.columns if c not in set(REQUIRED + OPTIONAL)]
if unknown_cols:
    st.info(f"Ignoring unrecognized columns: {', '.join(unknown_cols)}")

for c in ["order_id", "place", "tw_start", "tw_end", "hub", "notes"]:
    if c in df.columns:
        df[c] = df[c].astype("string").str.strip()

for c in ["lat", "lon", "service_min", "demand", "priority"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# ---------------------------------------------------
# Validate & fix
# ---------------------------------------------------
issues = []

for c in REQUIRED:
    if c not in df.columns:
        issues.append({"row": None, "field": c, "level": "error", "message": f"Missing required column '{c}'"})

for idx, row in df.iterrows():
    if not str(row["order_id"] or "").strip():
        issues.append({"row": idx, "field": "order_id", "level": "error", "message": "order_id is required"})

    if pd.isna(row["service_min"]) or float(row["service_min"]) < 0:
        issues.append({"row": idx, "field": "service_min", "level": "error", "message": "service_min must be ≥ 0"})

    ws = to_minutes_any(row["tw_start"])
    we = to_minutes_any(row["tw_end"])
    if row["tw_start"] and ws is None:
        issues.append({"row": idx, "field": "tw_start", "level": "error", "message": f"Unrecognized time '{row['tw_start']}'"})
    if row["tw_end"] and we is None:
        issues.append({"row": idx, "field": "tw_end", "level": "error", "message": f"Unrecognized time '{row['tw_end']}'"})
    if (ws is not None) and (we is not None) and (we < ws):
        issues.append({"row": idx, "field": "tw_end", "level": "error", "message": "tw_end earlier than tw_start"})

dups = df["order_id"].duplicated(keep=False)
if dups.any():
    for i in df.index[dups]:
        issues.append({"row": i, "field": "order_id", "level": "error", "message": "duplicate order_id"})

for idx, row in df.iterrows():
    if not (pd.isna(row["lat"]) or pd.isna(row["lon"])):
        if not (within(row["lat"], *LAT_BOUNDS) and within(row["lon"], *LON_BOUNDS)):
            issues.append({"row": idx, "field": "lat/lon", "level": "error", "message": "Coordinates out of bounds"})

issues_df = pd.DataFrame(issues)
if not issues_df.empty:
    if (issues_df["level"] == "error").any():
        st.error("Validation found errors. Please fix these and re-upload.")
    else:
        st.warning("There are warnings you may want to review.")
    st.dataframe(issues_df.sort_values(["level", "row", "field"], na_position="first"),
                 use_container_width=True)

if not issues_df.empty and (issues_df["level"] == "error").any():
    st.stop()

# ---------------------------------------------------
# Geocode missing coordinates from 'place'
# ---------------------------------------------------
if use_geocoding:
    with st.spinner("Geocoding rows with missing coordinates from 'place'…"):
        df, notes = geocode_unique(
            df,
            suffix_country="Philippines" if bias_ph else "",
            delay_s=float(throttle),
        )
    if notes:
        st.info("Geocoding notes")
        st.dataframe(pd.DataFrame(notes), use_container_width=True)

# Final location requirement
missing_coords = df["lat"].isna() | df["lon"].isna()
missing_with_no_place = missing_coords & (df["place"].fillna("").str.strip() == "")
if missing_with_no_place.any():
    st.error(f"{int(missing_with_no_place.sum())} row(s) missing both coordinates and place. Provide at least one.")
    st.dataframe(df.loc[missing_with_no_place, ["order_id", "place", "lat", "lon"]], use_container_width=True)
    st.stop()

# ---------------------------------------------------
# Normalize to minutes for downstream pages
# ---------------------------------------------------
df["tw_start_min"] = df["tw_start"].map(to_minutes_any)
df["tw_end_min"] = df["tw_end"].map(to_minutes_any)
df["service_time_min"] = pd.to_numeric(df["service_min"], errors="coerce").fillna(0).astype(int)

df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

for c in CANONICAL_ORDER:
    if c not in df.columns:
        df[c] = pd.NA
df = df[CANONICAL_ORDER].copy()

# ---------------------------------------------------
# Summary + preview
# ---------------------------------------------------
ok_rows = len(df.dropna(subset=["lat", "lon"]))
st.success(f"Loaded {ok_rows} order(s) ✅")

m1, m2, m3 = st.columns(3)
with m1: st.metric("Total orders", len(df))
with m2: st.metric("With coordinates", ok_rows)
with m3: st.metric("Missing time windows", int((df['tw_start_min'].isna() | df['tw_end_min'].isna()).sum()))

st.dataframe(df, use_container_width=True)

# Map preview
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
    "⬇️ Download cleaned orders (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="orders_cleaned.csv",
    mime="text/csv",
)

st.info("Next: open **Optimize Routes** to generate assignments and ETAs (AM/PM).")
