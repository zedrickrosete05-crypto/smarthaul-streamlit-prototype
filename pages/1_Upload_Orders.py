# pages/1_Upload_Orders.py
import io
import re
import time
from typing import Optional, Tuple
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk

st.set_page_config(page_title="SmartHaul – Upload Orders", layout="wide")
st.title("📦 Upload Orders")

# -----------------------------
# Helpers
# -----------------------------
def parse_time(value: str) -> Optional[int]:
    """Convert HH:MM AM/PM or 24H time to minutes."""
    if not value or pd.isna(value):
        return None
    value = str(value).strip().upper()

    # 12-hour AM/PM pattern
    match_12 = re.match(r"^(\d{1,2}):(\d{2})\s*(AM|PM)$", value)
    if match_12:
        h, m = int(match_12.group(1)), int(match_12.group(2))
        ampm = match_12.group(3)
        if ampm == "PM" and h != 12:
            h += 12
        if ampm == "AM" and h == 12:
            h = 0
        return h * 60 + m

    # 24-hour pattern
    match_24 = re.match(r"^(\d{1,2}):(\d{2})$", value)
    if match_24:
        h, m = int(match_24.group(1)), int(match_24.group(2))
        return h * 60 + m

    return None


@st.cache_data(show_spinner=False)
def geocode_place(place: str) -> Optional[Tuple[float, float]]:
    """Simple Nominatim geocoder."""
    import urllib.parse, urllib.request, json
    try:
        url = (
            f"https://nominatim.openstreetmap.org/search?"
            f"q={urllib.parse.quote(place + ', Philippines')}&format=json&limit=1"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "SmartHaulApp"})
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read().decode())
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        return None
    return None


# -----------------------------
# Template
# -----------------------------
def example_csv() -> bytes:
    df = pd.DataFrame({
        "order_id": ["O-1001", "O-1002", "O-1003"],
        "place": ["JY Square, Cebu City", "Cebu IT Park", "SM City Cebu"],
        "tw_start": ["08:30 AM", "09:00 AM", "10:00 AM"],
        "tw_end": ["11:00 AM", "12:00 PM", "01:00 PM"],
        "service_min": [7, 5, 10],
    })
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


with st.expander("📁 Download Template"):
    st.download_button(
        "Download orders_template_places.csv",
        data=example_csv(),
        file_name="orders_template_places.csv",
        mime="text/csv",
    )
    st.caption("Supports both 24-hour and AM/PM times like '08:30 AM' or '14:00'.")

# -----------------------------
# File Upload
# -----------------------------
uploaded = st.file_uploader("Upload your orders file", type=["csv", "xlsx"])

if not uploaded:
    st.stop()

try:
    df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

st.subheader("Preview (raw upload)")
st.dataframe(df.head(), use_container_width=True)

required_cols = ["order_id", "place", "tw_start", "tw_end", "service_min"]
for c in required_cols:
    if c not in df.columns:
        st.error(f"Missing required column: {c}")
        st.stop()

# -----------------------------
# Validate and clean
# -----------------------------
df["tw_start_min"] = df["tw_start"].map(parse_time)
df["tw_end_min"] = df["tw_end"].map(parse_time)

invalid_times = df[df["tw_start_min"].isna() | df["tw_end_min"].isna()]
if not invalid_times.empty:
    st.error(f"{len(invalid_times)} row(s) have invalid time formats.")
    st.dataframe(invalid_times, use_container_width=True)
    st.stop()

# Geocode
if "lat" not in df.columns or "lon" not in df.columns:
    df["lat"], df["lon"] = np.nan, np.nan

missing_coords = df["lat"].isna() | df["lon"].isna()
if missing_coords.any():
    st.info("Geocoding rows with missing coordinates...")
    for idx in df[missing_coords].index:
        place = df.at[idx, "place"]
        coords = geocode_place(place)
        if coords:
            df.at[idx, "lat"], df.at[idx, "lon"] = coords
        time.sleep(1)

st.success(f"Loaded {len(df)} orders ✅")

df["service_time_min"] = df["service_min"].astype(int)
st.session_state["orders_df"] = df.copy()

st.dataframe(df, use_container_width=True)

# Map
if st.checkbox("Show map preview", value=True):
    st.pydeck_chart(
        pdk.Deck(
            initial_view_state=pdk.ViewState(
                latitude=df["lat"].mean(), longitude=df["lon"].mean(), zoom=11
            ),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=df,
                    get_position="[lon, lat]",
                    get_radius=60,
                    get_fill_color=[0, 128, 255],
                )
            ],
        )
    )

st.download_button(
    "Download cleaned file",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="orders_cleaned.csv",
)
