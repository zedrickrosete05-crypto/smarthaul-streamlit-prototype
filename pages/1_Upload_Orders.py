# pages/1_Upload_Orders.py
from __future__ import annotations

import io, re, time, json, urllib.parse, urllib.request
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

# ---------- Page ----------
st.set_page_config(page_title="SmartHaul – Upload Orders", layout="wide")
BUILD = "upload-am-pm-final"
st.title("📦 Upload Orders")
st.caption(f"Build: {BUILD}")

# ---------- Columns ----------
REQUIRED = ["order_id", "tw_start", "tw_end", "service_min"]
OPTIONAL = ["place", "lat", "lon", "demand", "priority", "hub", "notes"]
CANONICAL = [
    "order_id", "place", "lat", "lon",
    "tw_start", "tw_end", "service_min",
    "tw_start_min", "tw_end_min", "service_time_min",
    "demand", "priority", "hub", "notes",
]

# ---------- Robust time parsing (AM/PM + 24h + Excel + datetime) ----------
_AMPM = re.compile(
    r"""^\s*(?P<h>\d{1,2})(?:
            [:\u2236](?P<m>\d{2}) | (?P<m2>\d{2})?
        )\s*(?P<ampm>a\.?m\.?|p\.?m\.?)\s*$""",
    re.IGNORECASE | re.VERBOSE,
)
_H24 = re.compile(r"^\s*(?P<h>[01]?\d|2[0-3])[:\u2236](?P<m>[0-5]\d)\s*$")

def to_minutes(val: Any) -> Optional[int]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if hasattr(val, "hour") and hasattr(val, "minute"):
        return int(val.hour) * 60 + int(val.minute)
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        x = float(val)
        if 0.0 <= x <= 1.0:  # Excel time fraction of a day
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
    try:
        ts = pd.to_datetime(s, errors="raise")
        return int(ts.hour) * 60 + int(ts.minute)
    except Exception:
        return None

# ---------- Geocoding ----------
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

def geocode_unique(df: pd.DataFrame, suffix=" Philippines", delay_s: float = 1.0) -> None:
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
            time.sleep(max(0.3, delay_s))
        prog.progress(i / len(places), text=f"Geocoding {i}/{len(places)}")
    prog.empty()
    for idx in df.index[need]:
        p = str(df.at[idx, "place"]).strip()
        if p in cache:
            df.at[idx, "lat"], df.at[idx, "lon"] = cache[p]

# ---------- Template download ----------
def template_bytes() -> bytes:
    demo = pd.DataFrame({
        "order_id": ["O-1001", "O-1002", "O-1003", "O-1004", "O-1005"],
        "place": ["JY Square, Cebu City", "Cebu IT Park", "SM City Cebu",
                  "Robinsons Galleria Cebu", "Ayala Center Cebu"],
        "tw_start": ["08:30 AM", "09:00 AM", "10:00 AM", "11:30 AM", "01:00 PM"],
        "tw_end":   ["11:00 AM", "12:00 PM", "01:00 PM", "02:30 PM", "04:00 PM"],
        "service_min": [7, 5, 10, 6, 8],
    })
    buf = io.StringIO(); demo.to_csv(buf, index=False)
    return buf.getvalue().encode()

with st.expander("📁 Download CSV template"):
    st.download_button("Download orders_template_places.csv", template_bytes(),
                       "orders_template_places.csv", "text/csv")
    st.caption("Times may be **AM/PM** (e.g. '08:30 AM', '1:15pm', '0830PM') or **24-hour** ('14:00').")

# ---------- Upload ----------
uploaded = st.file_uploader("Upload Orders CSV/XLSX", type=["csv", "xlsx"])
if not uploaded:
    st.stop()
try:
    df = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
except Exception as e:
    st.error(f"Could not read file: {e}"); st.stop()

st.subheader("Preview (raw upload)")
st.dataframe(df.head(50), use_container_width=True)

# Ensure columns exist
for c in REQUIRED + OPTIONAL:
    if c not in df.columns:
        df[c] = pd.NA

# Normalize types
for c in ["order_id", "place", "tw_start", "tw_end", "hub", "notes"]:
    df[c] = df[c].astype("string").str.strip()
for c in ["lat", "lon", "service_min", "demand", "priority"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Parse times (AM/PM aware)
df["tw_start_min"] = df["tw_start"].map(to_minutes)
df["tw_end_min"]   = df["tw_end"].map(to_minutes)
bad = df[df["tw_start_min"].isna() | df["tw_end_min"].isna()]
if not bad.empty:
    st.error(f"{len(bad)} row(s) have invalid time format in tw_start/tw_end.")
    st.caption("Accepted: '08:30 AM', '8:30am', '0830PM', '14:00', Excel time cells, datetime/time.")
    st.dataframe(bad[["order_id","tw_start","tw_end"]], use_container_width=True)
    st.stop()

# Service time
df["service_time_min"] = pd.to_numeric(df["service_min"], errors="coerce").fillna(0).clip(lower=0).astype(int)

# Coordinates via geocoding if needed
if "lat" not in df.columns or "lon" not in df.columns:
    df["lat"], df["lon"] = np.nan, np.nan
need_geo = (df["lat"].isna() | df["lon"].isna()) & (df["place"].fillna("").str.strip() != "")
if need_geo.any():
    st.info("Geocoding missing coordinates…")
    geocode_unique(df, " Philippines", 1.0)

# Canonical order + persist
for c in CANONICAL:
    if c not in df.columns:
        df[c] = pd.NA
df = df[CANONICAL].copy()
st.session_state["orders_df"] = df.copy()

# Summary
ok_rows = int((~(df["lat"].isna() | df["lon"].isna())).sum())
st.success(f"Loaded {len(df)} order(s). With coordinates: {ok_rows}")
st.dataframe(df, use_container_width=True)

# Map
if ok_rows:
    st.subheader("Map preview")
    st.pydeck_chart(
        pdk.Deck(
            initial_view_state=pdk.ViewState(
                latitude=float(df["lat"].dropna().mean()),
                longitude=float(df["lon"].dropna().mean()),
                zoom=11,
            ),
            layers=[pdk.Layer("ScatterplotLayer",
                              data=df.dropna(subset=["lat","lon"]),
                              get_position="[lon, lat]", get_radius=60, pickable=True)],
            tooltip={"text": "{order_id}\n{place}"},
        )
    )

# Download cleaned
st.download_button(
    "⬇️ Download cleaned orders (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="orders_cleaned.csv",
    mime="text/csv",
)

st.info("Next: open **Optimize Routes** to generate assignments and ETAs (AM/PM).")
