# pages/1_Upload_Orders.py
from __future__ import annotations
import io, re, json, time, urllib.parse, urllib.request
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

# ---- Force-clear caches so old validators disappear
st.cache_data.clear()

st.set_page_config(page_title="SmartHaul – Upload Orders", layout="wide")
st.title("📦 Upload Orders")
st.caption("Build: uploader-am/pm-proof")

REQUIRED = ["order_id", "tw_start", "tw_end", "service_min"]
OPTIONAL = ["place", "lat", "lon"]

# ------- AM/PM + 24h + Excel-time tolerant parser -------
_AMPM = re.compile(
    r"""^\s*(?P<h>\d{1,2})(?:
            [:\u2236](?P<m>\d{2}) | (?P<m2>\d{2})?
        )\s*(?P<ampm>a\.?m\.?|p\.?m\.?)\s*$""",
    re.IGNORECASE | re.VERBOSE,
)
_H24  = re.compile(r"^\s*(?P<h>[01]?\d|2[0-3])[:\u2236](?P<m>[0-5]\d)\s*$")

def to_minutes(x: Any) -> Optional[int]:
    if x is None or (isinstance(x, float) and pd.isna(x)): return None
    if hasattr(x, "hour") and hasattr(x, "minute"): return int(x.hour)*60 + int(x.minute)
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        v = float(x)
        if 0.0 <= v <= 1.0:  # Excel time fraction
            return int(round(v*24*60))

    s = str(x).strip().replace("\u00A0"," ").replace(".","")
    m = _AMPM.match(s)
    if m:
        h = int(m.group("h"))
        mm = int((m.group("m") or m.group("m2") or "0"))
        ampm = m.group("ampm").lower()
        if "p" in ampm and h != 12: h += 12
        if "a" in ampm and h == 12: h = 0
        return h*60 + mm
    m = _H24.match(s)
    if m: return int(m.group("h"))*60 + int(m.group("m"))
    try:
        ts = pd.to_datetime(s, errors="raise")
        return int(ts.hour)*60 + int(ts.minute)
    except Exception:
        return None

# ------- Geocode unique places (cached) -------
@st.cache_data(show_spinner=False)
def geocode_place(place: str) -> Optional[Tuple[float,float]]:
    url = ("https://nominatim.openstreetmap.org/search?"
           f"q={urllib.parse.quote(place)}&format=json&limit=1")
    req = urllib.request.Request(url, headers={"User-Agent":"SmartHaul/1.0"})
    with urllib.request.urlopen(req, timeout=8) as r:
        data = json.loads(r.read().decode("utf-8"))
    if data: return float(data[0]["lat"]), float(data[0]["lon"])
    return None

def geocode_missing(df: pd.DataFrame, suffix=" Philippines", delay_s=1.0):
    cache = st.session_state.setdefault("geo_cache", {})
    need = (df["place"].fillna("").str.strip()!="") & (df["lat"].isna() | df["lon"].isna())
    if not need.any(): return
    places = sorted(set(df.loc[need,"place"].astype(str)))
    prog = st.progress(0.0, text="Geocoding…")
    for i,p in enumerate(places,1):
        if p not in cache:
            try:
                c = geocode_place(p + suffix)
                if c: cache[p]=c
            except Exception: pass
            time.sleep(max(0.2, delay_s))
        prog.progress(i/len(places), text=f"Geocoding {i}/{len(places)}")
    prog.empty()
    for idx in df.index[need]:
        p = str(df.at[idx,"place"]).strip()
        if p in cache:
            df.at[idx,"lat"], df.at[idx,"lon"] = cache[p]

# ------- Template -------
def template_csv() -> bytes:
    demo = pd.DataFrame({
        "order_id": ["O-1001","O-1002","O-1003","O-1004","O-1005"],
        "place": ["JY Square, Cebu City","Cebu IT Park","SM City Cebu",
                  "Robinsons Galleria Cebu","Ayala Center Cebu"],
        "tw_start": ["08:30 AM","09:00 AM","10:00 AM","11:30 AM","01:00 PM"],
        "tw_end":   ["11:00 AM","12:00 PM","01:00 PM","02:30 PM","04:00 PM"],
        "service_min": [7,5,10,6,8],
    })
    buf = io.StringIO(); demo.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

with st.expander("📁 Download CSV template"):
    st.download_button("Download orders_template_places.csv", template_csv(),
                       "orders_template_places.csv", "text/csv")
    st.caption("Accepted: '08:30 AM', '8:30am', '0830PM', '14:00', Excel time cells.")

# ------- Upload -------
up = st.file_uploader("Upload Orders (CSV/XLSX)", type=["csv","xlsx"])
if not up: st.stop()
try:
    df = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
except Exception as e:
    st.error(f"Could not read file: {e}"); st.stop()

st.subheader("Preview (raw upload)")
st.dataframe(df.head(50), use_container_width=True)

# ensure columns
for c in REQUIRED + OPTIONAL:
    if c not in df.columns: df[c]=pd.NA
for c in ["order_id","place","tw_start","tw_end"]:
    df[c]=df[c].astype("string").str.strip()
if "lat" not in df.columns or "lon" not in df.columns:
    df["lat"], df["lon"] = np.nan, np.nan

# parse times (AM/PM aware)
df["tw_start_min"]=df["tw_start"].map(to_minutes)
df["tw_end_min"]=df["tw_end"].map(to_minutes)

bad = df[df["tw_start_min"].isna() | df["tw_end_min"].isna()]
if not bad.empty:
    st.error(f"{len(bad)} row(s) have invalid time format in tw_start/tw_end.")
    st.dataframe(bad[["order_id","tw_start","tw_end"]], use_container_width=True)
    st.stop()

# coerce service minutes
df["service_time_min"]=pd.to_numeric(df["service_min"], errors="coerce").fillna(0).astype(int)

# geocode if missing
need_geo = (df["lat"].isna() | df["lon"].isna()) & (df["place"].fillna("").str.strip()!="")
if need_geo.any():
    st.info("Geocoding rows with missing coordinates…")
    geocode_missing(df)

# persist canonical order
cols = ["order_id","place","lat","lon","tw_start","tw_end","service_min",
        "tw_start_min","tw_end_min","service_time_min"]
for c in cols:
    if c not in df.columns: df[c]=pd.NA
df = df[cols].copy()
st.session_state["orders_df"]=df.copy()

st.success(f"Loaded {len(df)} order(s).")
st.dataframe(df, use_container_width=True)

st.download_button("⬇️ Download cleaned (CSV)",
                   df.to_csv(index=False).encode("utf-8"),
                   "orders_cleaned.csv","text/csv")
