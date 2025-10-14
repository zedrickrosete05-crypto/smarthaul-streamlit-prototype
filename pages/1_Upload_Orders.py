# pages/1_Upload_Orders.py
from __future__ import annotations

# — clear any stale caches (safe on reloads) —
import streamlit as st
st.cache_data.clear()

import io, re, time, json, os, urllib.parse, urllib.request
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
import pydeck as pdk

# ────────────────────────────────────────────────────────────────────────────────
# Page header
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="SmartHaul – Upload Orders", layout="wide")
BUILD = "uploader-am/pm-proof-v4 (with persistence)"
st.title("📦 Upload Orders")
st.caption(f"Build: {BUILD}")

# ────────────────────────────────────────────────────────────────────────────────
# Simple persistence helpers (CSV on local disk)
# ────────────────────────────────────────────────────────────────────────────────
DATA_DIR = "data"
ORDERS_PATH = os.path.join(DATA_DIR, "orders_df.csv")
os.makedirs(DATA_DIR, exist_ok=True)

def save_orders(df: pd.DataFrame) -> None:
    try:
        df.to_csv(ORDERS_PATH, index=False)
    except Exception:
        pass

def load_orders() -> Optional[pd.DataFrame]:
    try:
        if os.path.exists(ORDERS_PATH):
            return pd.read_csv(ORDERS_PATH)
    except Exception:
        return None
    return None

with st.sidebar:
    st.subheader("Session")
    if st.button("Load last session"):
        prev = load_orders()
        if prev is not None and not prev.empty:
            st.session_state["orders_df"] = prev.copy()
            st.success(f"Restored {len(prev)} orders from last session.")
            st.stop()  # stop here so other pages see restored data immediately
        else:
            st.info("No previous orders found.")

# ────────────────────────────────────────────────────────────────────────────────
# Columns
# ────────────────────────────────────────────────────────────────────────────────
REQUIRED = ["order_id", "tw_start", "tw_end", "service_min"]
OPTIONAL = ["place", "lat", "lon", "demand", "priority", "hub", "notes"]
CANONICAL = [
    "order_id", "place", "lat", "lon",
    "tw_start", "tw_end", "service_min",
    "tw_start_min", "tw_end_min", "service_time_min",
    "demand", "priority", "hub", "notes",
]

# allow a couple of header aliases (people often vary these)
ALIASES = {
    "service time (min)": "service_min",
    "service_time": "service_min",
    "service_minutes": "service_min",
    "start": "tw_start",
    "end": "tw_end",
}

# ────────────────────────────────────────────────────────────────────────────────
# Time parsing (AM/PM + 24h + Excel + datetime)
# ────────────────────────────────────────────────────────────────────────────────
_AMPM = re.compile(
    r"""^\s*
        (?P<h>\d{1,2})
        (?:
            [:\u2236](?P<m>\d{2})   # 8:30 AM
            |
            (?P<m2>\d{2})?         # 830PM -> m2=30
        )
        \s*(?P<ampm>a\.?m\.?|p\.?m\.?)\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)
_H24 = re.compile(r"^\s*(?P<h>[01]?\d|2[0-3])[:\u2236](?P<m>[0-5]\d)\s*$")

def to_minutes(val: Any) -> Optional[int]:
    """Return minutes after midnight or None; tolerant to many formats."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None

    # datetime / time-like
    if hasattr(val, "hour") and hasattr(val, "minute"):
        return int(val.hour) * 60 + int(val.minute)

    # Excel time fraction [0,1]
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        x = float(val)
        if 0.0 <= x <= 1.0:
            return int(round(x * 24 * 60))

    s = str(val).strip().replace("\u00A0", " ").replace(".", "")  # normalize NBSP + a.m./p.m.

    m = _AMPM.match(s)
    if m:
        h = int(m.group("h"))
        mm = m.group("m") or m.group("m2") or "0"
        minute = int(mm)
        ampm = m.group("ampm").lower()
        if "p" in ampm and h != 12:
            h += 12
        if "a" in ampm and h == 12:
            h = 0
        if 0 <= h < 24 and 0 <= minute < 60:
            return h * 60 + minute

    m = _H24.match(s)
    if m:
        return int(m.group("h")) * 60 + int(m.group("m"))

    # last resort: pandas parser
    try:
        ts = pd.to_datetime(s, errors="raise")
        return int(ts.hour) * 60 + int(ts.minute)
    except Exception:
        return None

# ────────────────────────────────────────────────────────────────────────────────
# Geocoding (cached, unique places)
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def geocode_place(place: str) -> Optional[Tuple[float, float]]:
    url = (
        "https://nominatim.openstreetmap.org/search?"
        + "q=" + urllib.parse.quote(place)
        + "&format=json&limit=1"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "SmartHaul/1.0"})
    with urllib.request.urlopen(req, timeout=8) as r:
        data = json.loads(r.read().decode("utf-8"))
    if data:
        return float(data[0]["lat"]), float(data[0]["lon"])
    return None

def geocode_missing(df: pd.DataFrame, suffix=" Philippines", delay_s: float = 1.0) -> None:
    cache = st.session_state.setdefault("geo_cache", {})
    need = (df["place"].fillna("").str.strip() != "") & (df["lat"].isna() | df["lon"].isna())
    if not need.any():
        return
    places = sorted(set(df.loc[need, "place"].astype(str)))
    prog = st.progress(0.0, text="Geocoding places…")
    for i, p in enumerate(places, 1):
        if p not in cache:
            try:
                coords = geocode_place(p + suffix)
                if coords:
                    cache[p] = coords
            except Exception:
                pass
            time.sleep(max(0.2, delay_s))
        prog.progress(i / len(places), text=f"Geocoding {i}/{len(places)}")
    prog.empty()
    for idx in df.index[need]:
        p = str(df.at[idx, "place"]).strip()
        if p in cache:
            df.at[idx, "lat"], df.at[idx, "lon"] = cache[p]

# ────────────────────────────────────────────────────────────────────────────────
# Template download
# ────────────────────────────────────────────────────────────────────────────────
def template_bytes() -> bytes:
    demo = pd.DataFrame({
        "order_id": ["O-1001", "O-1002", "O-1003", "O-1004", "O-1005"],
        "place": ["JY Square, Cebu City", "Cebu IT Park", "SM City Cebu",
                  "Robinsons Galleria Cebu", "Ayala Center Cebu"],
        "tw_start": ["08:30 AM", "09:00 AM", "10:00 AM", "11:30 AM", "01:00 PM"],
        "tw_end":   ["11:00 AM", "12:00 PM", "01:00 PM", "02:30 PM", "04:00 PM"],
        "service_min": [7, 5, 10, 6, 8],
        "demand": [1, 2, 1, 3, 1],
    })
    buf = io.StringIO(); demo.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

with st.expander("📁 Download CSV template"):
    st.download_button("Download orders_template_places.csv", template_bytes(),
                       "orders_template_places.csv", "text/csv")
    st.caption("Accepted time examples: '08:30 AM', '8:30am', '0830PM', '14:00', Excel time cells.")

# ────────────────────────────────────────────────────────────────────────────────
# Upload + read
# ────────────────────────────────────────────────────────────────────────────────
up = st.file_uploader("Upload Orders (CSV/XLSX)", type=["csv", "xlsx"])
if not up:
    st.stop()

try:
    if up.name.lower().endswith(".csv"):
        df = pd.read_csv(up)
    else:
        try:
            df = pd.read_excel(up, engine="openpyxl")
        except Exception:
            df = pd.read_excel(up)  # fallback
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

st.subheader("Preview (raw upload)")
st.dataframe(df.head(50), use_container_width=True)

# normalize headers (aliases)
ren = {}
for c in df.columns:
    key = c.strip().lower()
    ren[c] = ALIASES.get(key, c)
df = df.rename(columns=ren)

# Ensure columns exist
for c in REQUIRED + OPTIONAL:
    if c not in df.columns:
        df[c] = pd.NA

# Normalize types
for c in ["order_id", "place", "tw_start", "tw_end", "hub", "notes"]:
    df[c] = df[c].astype("string").str.strip()
for c in ["lat", "lon", "service_min", "demand", "priority"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Default demand = 1
df["demand"] = df["demand"].fillna(1).clip(lower=0).astype(int)

# Drop duplicate order_ids (keep first, show info)
dups = df["order_id"].duplicated(keep="first")
if dups.any():
    st.warning(f"Dropped {int(dups.sum())} duplicate order_id row(s) (keeping first occurrence).")
    df = df.loc[~dups].copy()

# Check hard-required
missing_id = df["order_id"].isna() | (df["order_id"].str.strip() == "")
if missing_id.any():
    st.error("Some rows are missing order_id. Please fix and re-upload.")
    st.dataframe(df.loc[missing_id, ["order_id", "place", "tw_start", "tw_end"]], use_container_width=True)
    st.stop()

# Parse times (AM/PM aware)
df["tw_start_min"] = df["tw_start"].map(to_minutes)
df["tw_end_min"]   = df["tw_end"].map(to_minutes)

bad_mask = df["tw_start_min"].isna() | df["tw_end_min"].isna()
if bad_mask.any():
    st.error(f"{int(bad_mask.sum())} row(s) have invalid time format in tw_start/tw_end.")
    st.caption("Accepted: '08:30 AM', '8:30am', '0830PM', '14:00', Excel time cells, or datetime.")
    st.dataframe(df.loc[bad_mask, ["order_id", "tw_start", "tw_end"]], use_container_width=True)
    st.stop()

# Same-day window sanity (simple rule; adjust if you need overnight windows)
bad_window = (df["tw_start_min"].notna() & df["tw_end_min"].notna()
              & (df["tw_end_min"] < df["tw_start_min"]))
if bad_window.any():
    st.error("Some rows have tw_end earlier than tw_start (same-day windows). Please fix:")
    st.dataframe(df.loc[bad_window, ["order_id", "tw_start", "tw_end"]], use_container_width=True)
    st.stop()

# Service time (minutes)
df["service_time_min"] = pd.to_numeric(df["service_min"], errors="coerce").fillna(0).clip(lower=0).astype(int)

# Geocode if missing
if "lat" not in df.columns or "lon" not in df.columns:
    df["lat"], df["lon"] = np.nan, np.nan
need_geo = (df["lat"].isna() | df["lon"].isna()) & (df["place"].fillna("").str.strip() != "")
if need_geo.any():
    st.info("Geocoding rows with missing coordinates from 'place'…")
    geocode_missing(df, suffix=" Philippines", delay_s=1.0)

# Warn if still missing coordinates after geocoding
still_missing = df["lat"].isna() | df["lon"].isna()
if still_missing.any():
    st.warning(f"{int(still_missing.sum())} row(s) still lack coordinates after geocoding.")

# Canonical order + persist to session and disk
for c in CANONICAL:
    if c not in df.columns:
        df[c] = pd.NA
df = df[CANONICAL].copy()
st.session_state["orders_df"] = df.copy()
save_orders(df)  # <-- persistence

# Summary + preview
ok_rows = int((~(df["lat"].isna() | df["lon"].isna())).sum())
st.success(f"Loaded {len(df)} order(s). With coordinates: {ok_rows}")
st.dataframe(df, use_container_width=True)

# Optional map
if ok_rows:
    st.subheader("Map preview")
    st.pydeck_chart(
        pdk.Deck(
            initial_view_state=pdk.ViewState(
                latitude=float(df["lat"].dropna().mean()),
                longitude=float(df["lon"].dropna().mean()),
                zoom=11,
            ),
            layers=[pdk.Layer(
                "ScatterplotLayer",
                data=df.dropna(subset=["lat","lon"]),
                get_position="[lon, lat]",
                get_radius=60,
                pickable=True,
            )],
            tooltip={"text": "{order_id}\n{place}"},
        )
    )

# Download cleaned file
st.download_button(
    "⬇️ Download cleaned orders (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="orders_cleaned.csv",
    mime="text/csv",
)

st.info("Next: open **Optimize Routes** to generate assignments and ETAs (AM/PM).")
