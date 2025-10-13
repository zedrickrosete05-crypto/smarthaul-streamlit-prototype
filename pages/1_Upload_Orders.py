# pages/1_Upload_Orders.py
from __future__ import annotations

import io, re, time, math
from typing import Optional, Tuple, Any
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

st.set_page_config(page_title="SmartHaul – Upload Orders", layout="wide")
st.title("📦 Upload Orders")

# ---------- Required/Optional columns ----------
REQUIRED = ["order_id", "tw_start", "tw_end", "service_min"]
OPTIONAL = ["place", "lat", "lon", "demand", "priority", "hub", "notes"]

# ---------- Time parsing (very tolerant) ----------
_AMPM = re.compile(
    r"""^\s*
        (?P<h>\d{1,2})
        [:\u2236]?(?P<m>\d{2})?         # 8:30 or 830 also ok
        \s*(?P<ampm>a\.?m\.?|p\.?m\.?)  # AM/PM variants with/without dots
        \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)
_HHMM = re.compile(r"^\s*(?P<h>[01]?\d|2[0-3])[:\u2236](?P<m>[0-5]\d)\s*$")  # 24h HH:MM

def _to_minutes(value: Any) -> Optional[int]:
    """Return minutes after midnight from many formats:
       '08:30 AM', '8:30am', '0830 pm', '14:05', pandas.Timestamp, datetime/time, Excel serial."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    # Pandas/py datetime/time
    if hasattr(value, "hour") and hasattr(value, "minute"):
        return int(value.hour) * 60 + int(value.minute)

    # Excel serial (date/time) or pure float hours
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        x = float(value)
        if 0 <= x <= 1:  # Excel time fraction of a day
            return int(round(x * 24 * 60))
        if x > 59 and x < 24 * 60:  # already minutes (rare)
            return int(x)

    s = str(value).strip().replace("\u00A0", " ")  # normalize NBSP
    s = s.replace(".", "")  # remove dots in a.m./p.m.

    # AM/PM (with or without colon/space)
    m = _AMPM.match(s)
    if m:
        h = int(m.group("h"))
        mnt = int(m.group("m") or 0)
        ampm = m.group("ampm").lower()
        if "p" in ampm and h != 12:
            h += 12
        if "a" in ampm and h == 12:
            h = 0
        if 0 <= h < 24 and 0 <= mnt < 60:
            return h * 60 + mnt

    # 24h HH:MM
    m = _HHMM.match(s)
    if m:
        h = int(m.group("h"))
        mnt = int(m.group("m"))
        return h * 60 + mnt

    # Last resort: let pandas try
    try:
        ts = pd.to_datetime(s, errors="raise")
        return int(ts.hour) * 60 + int(ts.minute)
    except Exception:
        return None

# ---------- Geocoding unique places (cached) ----------
@st.cache_data(show_spinner=False)
def geocode_place(place: str) -> Optional[Tuple[float, float]]:
    import urllib.parse, urllib.request, json
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

def geocode_unique(df: pd.DataFrame, country_suffix: str = " Philippines", delay_s: float = 1.0):
    cache = st.session_state.setdefault("geocode_cache", {})
    need = (df["place"].fillna("").str.strip() != "") & (df["lat"].isna() | df["lon"].isna())
    if not need.any():
        return

    unique_places = sorted(set(df.loc[need, "place"].astype(str).str.strip()))
    prog = st.progress(0.0, text="Geocoding places…")
    for i, p in enumerate(unique_places, start=1):
        if p not in cache:
            try:
                coords = geocode_place(p + (country_suffix if country_suffix else ""))
                if coords:
                    cache[p] = coords
            except Exception:
                pass
            time.sleep(max(0.2, delay_s))
        prog.progress(i / len(unique_places), text=f"Geocoding places… ({i}/{len(unique_places)})")
    prog.empty()

    for idx in df.index[need]:
        p = str(df.at[idx, "place"]).strip()
        if p in cache:
            df.at[idx, "lat"], df.at[idx, "lon"] = map(float, cache[p])

# ---------- Template (AM/PM) ----------
def template_csv() -> bytes:
    demo = pd.DataFrame({
        "order_id": ["O-1001", "O-1002", "O-1003"],
        "place": ["JY Square, Cebu City", "Cebu IT Park", "SM City Cebu"],
        "tw_start": ["08:30 AM", "09:00 AM", "10:00 AM"],
        "tw_end":   ["11:00 AM", "12:00 PM", "01:00 PM"],
        "service_min": [7, 5, 10],
    })
    buf = io.StringIO()
    demo.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

with st.expander("📁 Download CSV template"):
    st.download_button("Download template", data=template_csv(),
                       file_name="orders_template_places.csv", mime="text/csv")
    st.caption("Times may be **AM/PM** (e.g. '1:05 PM', '0830PM') or **24-hour** ('13:05'). "
               "Provide **place** (will geocode) or **lat+lon**.")

# ---------- Upload ----------
file = st.file_uploader("Upload orders (CSV or Excel)", type=["csv", "xlsx"])
if not file:
    st.stop()

try:
    df_raw = pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

st.subheader("Preview (raw upload)")
st.dataframe(df_raw.head(50), use_container_width=True)

# ---------- Ensure columns ----------
df = df_raw.copy()
for col in REQUIRED + OPTIONAL:
    if col not in df.columns:
        df[col] = pd.NA

# Normalize types
for c in ["order_id", "place", "tw_start", "tw_end", "hub", "notes"]:
    df[c] = df[c].astype("string").str.strip()
for c in ["lat", "lon", "service_min", "demand", "priority"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ---------- Validate times (fixed) ----------
df["tw_start_min"] = df["tw_start"].map(_to_minutes)
df["tw_end_min"]   = df["tw_end"].map(_to_minutes)

bad_mask = df["tw_start_min"].isna() | df["tw_end_min"].isna()
if bad_mask.any():
    st.error(f"{int(bad_mask.sum())} row(s) have invalid time in tw_start/tw_end.")
    st.dataframe(df.loc[bad_mask, ["order_id", "tw_start", "tw_end"]], use_container_width=True)
    st.stop()

# Ensure order_id and service_min
missing_id = df["order_id"].isna() | (df["order_id"].str.strip() == "")
if missing_id.any():
    st.error("Some rows are missing order_id.")
    st.dataframe(df.loc[missing_id, ["order_id", "place"]], use_container_width=True)
    st.stop()

df["service_time_min"] = pd.to_numeric(df["service_min"], errors="coerce").fillna(0).clip(lower=0).astype(int)

# ---------- Geocode if needed ----------
if "lat" not in df.columns or "lon" not in df.columns:
    df["lat"], df["lon"] = np.nan, np.nan

if (df["lat"].isna() | df["lon"].isna()).any():
    st.info("Geocoding rows with missing coordinates from 'place'…")
    geocode_unique(df, country_suffix=" Philippines", delay_s=1.0)

# Final missing coords check
missing_coords = df["lat"].isna() | df["lon"].isna()
if missing_coords.any():
    st.warning(f"{int(missing_coords.sum())} row(s) still missing coordinates.")
    st.dataframe(df.loc[missing_coords, ["order_id", "place"]], use_container_width=True)

# ---------- Canonical column order ----------
cols = [
    "order_id", "place", "lat", "lon",
    "tw_start", "tw_end", "service_min",
    "tw_start_min", "tw_end_min", "service_time_min",
    "demand", "priority", "hub", "notes",
]
for c in cols:
    if c not in df.columns:
        df[c] = pd.NA
df = df[cols].copy()

# ---------- Persist + UI ----------
ok_rows = int((~(df["lat"].isna() | df["lon"].isna())).sum())
st.success(f"Loaded {len(df)} order(s). With coordinates: {ok_rows}")

st.dataframe(df, use_container_width=True)

if ok_rows:
    st.subheader("Map preview")
    st.pydeck_chart(
        pdk.Deck(
            initial_view_state=pdk.ViewState(
                latitude=float(df["lat"].dropna().mean()),
                longitude=float(df["lon"].dropna().mean()),
                zoom=11,
            ),
            layers=[pdk.Layer("ScatterplotLayer", data=df.dropna(subset=["lat","lon"]),
                              get_position="[lon, lat]", get_radius=60, pickable=True)],
            tooltip={"text": "{order_id}\n{place}"},
        )
    )

st.session_state["orders_df"] = df.copy()

st.download_button(
    "⬇️ Download cleaned orders (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="orders_cleaned.csv",
    mime="text/csv",
)

st.info("Next: open **Optimize Routes** to generate assignments and ETAs (AM/PM).")
