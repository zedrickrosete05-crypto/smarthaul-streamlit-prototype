# pages/1_Upload_Orders.py
from __future__ import annotations

import io
import math
import re
import time
from typing import Any, Optional, Tuple

import pandas as pd
import streamlit as st
import pydeck as pdk

# -------------------- Page setup --------------------
st.set_page_config(page_title="SmartHaul – Upload Orders", layout="wide")
st.title("Upload Orders")

# -------------------- Schema --------------------
# Required columns (strings as provided by your existing app)
REQUIRED_COLS = ["order_id", "tw_start", "tw_end", "service_min"]
# Optional columns. At least (lat & lon) OR place must be present per row.
OPTIONAL_COLS = ["lat", "lon", "place", "demand", "priority", "hub", "notes"]

ALL_COLS = REQUIRED_COLS + OPTIONAL_COLS

LAT_MIN, LAT_MAX = -90.0, 90.0
LON_MIN, LON_MAX = -180.0, 180.0


# -------------------- Helpers --------------------
def _coerce_numeric(series: pd.Series, dtype: str = "float64") -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(dtype)


def _within(v: float, lo: float, hi: float) -> bool:
    return v is not None and not pd.isna(v) and lo <= float(v) <= hi


def _parse_time(s: Any) -> Optional[pd.Timestamp]:
    """Accept 'HH:MM'/'H:MM' or full datetime; return pandas Timestamp or None."""
    if pd.isna(s):
        return None
    s = str(s).strip()
    if not s:
        return None
    try:
        if re.fullmatch(r"\d{1,2}:\d{2}", s):
            return pd.to_datetime(f"1970-01-01 {s}", utc=False, errors="raise")
        return pd.to_datetime(s, utc=False, errors="raise")
    except Exception:
        return None


def _fmt_hhmm(ts: Optional[pd.Timestamp]) -> str:
    return "" if ts is None else ts.strftime("%H:%M")


# ---- polite, no-key OSM geocoder (swap with Mapbox/Google in prod) ----
@st.cache_data(show_spinner=False)
def geocode_place_nominatim(q: str) -> Optional[Tuple[float, float]]:
    import urllib.parse, urllib.request, json

    base = "https://nominatim.openstreetmap.org/search"
    url = f"{base}?q={urllib.parse.quote(q)}&format=json&limit=1"
    req = urllib.request.Request(url, headers={"User-Agent": "SmartHaul/1.0"})
    with urllib.request.urlopen(req, timeout=8) as r:
        data = json.loads(r.read().decode("utf-8"))
    time.sleep(1.1)  # be nice to the free service
    if data:
        return float(data[0]["lat"]), float(data[0]["lon"])
    return None


# -------------------- CSV Template --------------------
template = pd.DataFrame(
    [
        {
            "order_id": "O-1001",
            "place": "JY Square, Cebu City",
            "tw_start": "08:30",
            "tw_end": "11:00",
            "service_min": 7,
            "lat": pd.NA,
            "lon": pd.NA,
            "demand": 10,
            "priority": 2,
            "hub": "Main",
            "notes": "fragile",
        },
        {
            "order_id": "O-1002",
            "place": "Cebu IT Park",
            "tw_start": "09:00",
            "tw_end": "12:00",
            "service_min": 5,
            "lat": pd.NA,
            "lon": pd.NA,
            "demand": 5,
            "priority": 1,
            "hub": "Main",
            "notes": "",
        },
        {
            "order_id": "O-1003",
            "place": "SM City Cebu",
            "tw_start": "10:00",
            "tw_end": "13:00",
            "service_min": 10,
            "lat": pd.NA,
            "lon": pd.NA,
            "demand": 8,
            "priority": 3,
            "hub": "Main",
            "notes": "",
        },
    ]
)

st.download_button(
    "Download CSV template (with place)",
    template.to_csv(index=False).encode(),
    file_name="orders_template_places.csv",
    mime="text/csv",
)

st.caption(
    "Required columns: **order_id, tw_start (HH:MM), tw_end (HH:MM), service_min (minutes)**. "
    "Provide **either** `lat & lon` **or** `place` (address/name) for each row. "
    "Optional: demand, priority, hub, notes."
)

# -------------------- File uploader --------------------
file = st.file_uploader("Upload Orders CSV", type=["csv"], key="orders_csv")

use_geocoding = st.toggle(
    "Geocode rows missing coordinates from `place`",
    value=True,
    help="If latitude/longitude are blank, try to fetch from address using OpenStreetMap (Nominatim).",
)


# -------------------- Validation & Cleaning --------------------
def clean_and_validate(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (clean_df, issues_df).
    issues_df columns: row, field, level ('error'|'warning'), message.
    """
    issues = []

    # Normalize header names (lower, strip) → canonical names
    lookup = {c.lower().strip(): c for c in df_raw.columns}
    def col(name: str) -> Optional[str]:
        return lookup.get(name.lower())

    # Create missing optional columns, check required ones
    df = df_raw.copy()
    for c in ALL_COLS:
        src = col(c)
        if src is None:
            if c in REQUIRED_COLS:
                issues.append(
                    {"row": None, "field": c, "level": "error", "message": f"Missing required column '{c}'"}
                )
            else:
                df[c] = pd.Series(dtype="object")
        else:
            if src != c:
                df.rename(columns={src: c}, inplace=True)

    if any(i["level"] == "error" for i in issues):
        return pd.DataFrame(), pd.DataFrame(issues)

    # Types / coercions
    df["order_id"] = df["order_id"].astype("string").str.strip()
    df["place"] = df["place"].astype("string").str.strip() if "place" in df.columns else ""
    df["service_min"] = _coerce_numeric(df["service_min"], "Float64")

    # optional numerics
    for coln in ["lat", "lon", "demand", "priority"]:
        if coln in df.columns:
            df[coln] = pd.to_numeric(df[coln], errors="coerce")

    # Row-level checks
    for idx, row in df.iterrows():
        # order_id
        if not row["order_id"]:
            issues.append({"row": idx, "field": "order_id", "level": "error", "message": "order_id is required"})

        # service_min
        if pd.isna(row["service_min"]) or float(row["service_min"]) < 0:
            issues.append({"row": idx, "field": "service_min", "level": "error", "message": "service_min must be ≥ 0"})

        # time windows
        ws = _parse_time(row.get("tw_start"))
        we = _parse_time(row.get("tw_end"))
        if row.get("tw_start") and ws is None:
            issues.append({"row": idx, "field": "tw_start", "level": "error", "message": f"Unrecognized time '{row.get('tw_start')}'"})
        if row.get("tw_end") and we is None:
            issues.append({"row": idx, "field": "tw_end", "level": "error", "message": f"Unrecognized time '{row.get('tw_end')}'"})
        if ws and we and we < ws:
            issues.append({"row": idx, "field": "tw_end", "level": "error", "message": "tw_end earlier than tw_start"})

        # location presence (either lat+lon OR place)
        has_coords = _within(row.get("lat"), LAT_MIN, LAT_MAX) and _within(row.get("lon"), LON_MIN, LON_MAX)
        has_place = bool(str(row.get("place") or "").strip())
        if not has_coords and not has_place:
            issues.append(
                {"row": idx, "field": "location", "level": "error", "message": "Provide lat+lon or place"}
            )

        # coords sanity if provided
        if pd.notna(row.get("lat")) or pd.notna(row.get("lon")):
            if not (_within(row.get("lat"), LAT_MIN, LAT_MAX) and _within(row.get("lon"), LON_MIN, LON_MAX)):
                issues.append(
                    {"row": idx, "field": "lat/lon", "level": "error", "message": "Coordinates out of bounds"}
                )

    # duplicate order_id
    dups = df["order_id"].duplicated(keep=False)
    if dups.any():
        for i in df.index[dups]:
            issues.append({"row": i, "field": "order_id", "level": "error", "message": "duplicate order_id"})

    # Early return on blocking errors
    issues_df = pd.DataFrame(issues)
    if not issues_df.empty and (issues_df["level"] == "error").any():
        return pd.DataFrame(), issues_df.sort_values(["level", "row", "field"], na_position="first")

    # Normalize times to HH:MM strings
    df["tw_start"] = [_fmt_hhmm(_parse_time(v)) for v in df["tw_start"]]
    df["tw_end"] = [_fmt_hhmm(_parse_time(v)) for v in df["tw_end"]]

    return df.reset_index(drop=True), issues_df.sort_values(["level", "row", "field"], na_position="first")


def try_geocode_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Fill missing lat/lon using 'place'. Returns updated df and warnings list."""
    notes = []
    if "place" not in df.columns:
        return df, notes

    for idx, row in df.iterrows():
        lat, lon, place = row.get("lat"), row.get("lon"), row.get("place")
        has_coords = _within(lat, LAT_MIN, LAT_MAX) and _within(lon, LON_MIN, LON_MAX)
        if (not has_coords) and str(place or "").strip():
            try:
                res = geocode_place_nominatim(place)
                if res:
                    glat, glon = res
                    if _within(glat, LAT_MIN, LAT_MAX) and _within(glon, LON_MIN, LON_MAX):
                        df.at[idx, "lat"] = float(glat)
                        df.at[idx, "lon"] = float(glon)
                    else:
                        notes.append({"row": idx, "field": "geocode", "level": "warning", "message": "geocoder returned out-of-bounds coords"})
                else:
                    notes.append({"row": idx, "field": "geocode", "level": "warning", "message": "no geocode result"})
            except Exception as e:
                notes.append({"row": idx, "field": "geocode", "level": "warning", "message": f"geocode error: {e}"})
    return df, notes


# -------------------- Main UI --------------------
if file is None:
    st.caption("Tip: You can upload multiple times; the latest valid upload replaces the current orders.")
    st.stop()

# Read CSV
try:
    df_raw = pd.read_csv(file)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

st.subheader("Preview")
st.dataframe(df_raw.head(50), use_container_width=True)

with st.spinner("Validating…"):
    base_df, issues_df = clean_and_validate(df_raw)

if not issues_df.empty:
    err = issues_df[issues_df["level"] == "error"]
    if not err.empty:
        st.error("Found validation errors. Fix these and re-upload.")
    else:
        st.warning("No blocking errors, but there are warnings you may want to review.")
    st.dataframe(issues_df, use_container_width=True)

if base_df.empty:
    st.stop()

# Optional geocoding
if use_geocoding:
    with st.spinner("Geocoding rows with missing coordinates from 'place'…"):
        geo_df, notes = try_geocode_missing(base_df.copy())
    if notes:
        st.info("Geocoding notes:")
        st.dataframe(pd.DataFrame(notes), use_container_width=True)
    work_df = geo_df
else:
    work_df = base_df

# Final check: still missing coords?
missing = work_df["lat"].isna() | work_df["lon"].isna()
if missing.any():
    st.error(
        f"{missing.sum()} row(s) still missing coordinates. Provide lat/lon or enable geocoding."
    )
    st.dataframe(work_df[missing][["order_id", "place", "lat", "lon"]], use_container_width=True)
    st.stop()

# Success: save and show
st.success(f"Validated {len(work_df)} orders ✅")
st.dataframe(work_df, use_container_width=True)

# Save for downstream pages
st.session_state["orders_df"] = work_df

# Map preview
try:
    st.subheader("Map preview")
    st.pydeck_chart(
        pdk.Deck(
            initial_view_state=pdk.ViewState(
                latitude=float(work_df["lat"].mean()),
                longitude=float(work_df["lon"].mean()),
                zoom=11,
            ),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=work_df,
                    get_position="[lon, lat]",
                    get_radius=50,
                    pickable=True,
                )
            ],
            tooltip={"text": "{order_id}\n{place}"},
        )
    )
except Exception:
    st.caption("Map preview unavailable (pydeck needs numeric lat/lon).")

# Download cleaned file
st.download_button(
    "⬇️ Download cleaned orders",
    data=work_df.to_csv(index=False).encode("utf-8"),
    file_name="orders_cleaned.csv",
    mime="text/csv",
)

st.info("Next: go to **Optimize Routes** to generate assignments and ETAs.")
