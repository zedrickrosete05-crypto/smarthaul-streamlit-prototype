import io, pandas as pd, streamlit as st
from datetime import datetime
from functools import lru_cache

st.title("📤 Upload Orders")

# Users can give either place or lat/lon
ACCEPTED_COLS = {"order_id", "place", "lat", "lon", "tw_start", "tw_end", "service_min"}

st.caption("Tip: You can upload either **place names** (we’ll geocode) or **lat/lon**. Time format HH:MM.")

file = st.file_uploader("CSV file", type="csv")

# ---------- Geocoding helpers ----------
def _setup_geocoder():
    try:
        from geopy.geocoders import Nominatim
        from geopy.extra.rate_limiter import RateLimiter
        geolocator = Nominatim(user_agent="smarthaul-demo")
        reverse = None
        forward = RateLimiter(geolocator.geocode, min_delay_seconds=1)  # be nice to OSM!
        return forward
    except Exception:
        import requests
        def forward(q: str):
            try:
                url = "https://nominatim.openstreetmap.org/search"
                params = {"q": q, "format": "json", "limit": 1, "addressdetails": 0}
                headers = {"User-Agent": "smarthaul-demo/1.0 (contact: example@example.com)"}
                r = requests.get(url, params=params, headers=headers, timeout=10)
                r.raise_for_status()
                js = r.json()
                if js:
                    return {"lat": float(js[0]["lat"]), "lon": float(js[0]["lon"])}
            except Exception:
                return None
        return forward

_forward = _setup_geocoder()

@st.cache_data(show_spinner=False)
def geocode_cached(q: str):
    if not q or not isinstance(q, str): 
        return None
    res = _forward(q)
    if res is None:
        return None
    # geopy returns Location; http returns dict
    if hasattr(res, "latitude"):
        return {"lat": float(res.latitude), "lon": float(res.longitude)}
    return {"lat": float(res["lat"]), "lon": float(res["lon"])}

# ---------- Template download ----------
tmpl = pd.DataFrame([
    {"order_id":"O-1001","place":"SM Mall of Asia, Pasay","tw_start":"08:30","tw_end":"11:00","service_min":7},
    {"order_id":"O-1002","place":"Ayala Triangle Gardens, Makati","tw_start":"09:00","tw_end":"12:00","service_min":5},
    {"order_id":"O-1003","lat":14.6091,"lon":121.0223,"tw_start":"10:00","tw_end":"13:00","service_min":10},
])
st.download_button(
    "Download CSV template (place or lat/lon)",
    data=tmpl.to_csv(index=False).encode(),
    file_name="orders_template_friendly.csv",
    mime="text/csv",
)

if not file:
    st.stop()

# ---------- Read & validate ----------
df = pd.read_csv(file).rename(columns=str.lower)
unknown = [c for c in df.columns if c not in ACCEPTED_COLS]
if unknown:
    st.info(f"Ignoring unknown columns: {unknown}")

# Normalize types
for c in ["tw_start","tw_end"]:
    if c in df:
        df[c] = df[c].astype(str).str.extract(r"^(\d{1,2}:\d{2})", expand=False).fillna("")

if "service_min" in df:
    df["service_min"] = pd.to_numeric(df["service_min"], errors="coerce").fillna(0).clip(lower=0)
else:
    df["service_min"] = 0

# ---------- Geocode missing lat/lon using 'place' ----------
needs_geo = df[ (df.get("lat").isna() | ~df.get("lat").apply(lambda x: str(x).strip().replace('.','',1).replace('-','',1).isdigit() if pd.notna(x) else False)) |
                (df.get("lon").isna() | ~df.get("lon").apply(lambda x: str(x).strip().replace('.','',1).replace('-','',1).isdigit() if pd.notna(x) else False)) ]

if "place" in df.columns and len(needs_geo):
    with st.spinner("Geocoding places…"):
        lat_list, lon_list = [], []
        for _, r in df.iterrows():
            lat, lon = r.get("lat"), r.get("lon")
            if pd.isna(lat) or pd.isna(lon):
                res = geocode_cached(str(r.get("place","")).strip())
                if res:
                    lat, lon = res["lat"], res["lon"]
            lat_list.append(lat)
            lon_list.append(lon)
        df["lat"], df["lon"] = lat_list, lon_list

df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

bad = df[df["lat"].isna() | df["lon"].isna()]
if len(bad):
    st.warning(f"{len(bad)} row(s) couldn’t be geocoded. Fix the place text or add lat/lon. Showing them below.")
    st.dataframe(bad, use_container_width=True)

good = df.dropna(subset=["lat","lon"]).copy()
if good.empty:
    st.error("No valid rows with coordinates. Please correct your file.")
    st.stop()

# Minimal required for optimizer
required_missing = [c for c in ["order_id","tw_start","tw_end"] if c not in good.columns]
for c in required_missing:
    good[c] = ""  # allow empty windows if user didn’t provide

st.session_state["orders_df"] = good[["order_id","lat","lon","tw_start","tw_end","service_min"]]
st.success(f"Loaded {len(good)} orders. {len(bad)} need attention.")
st.dataframe(st.session_state["orders_df"].head(20), use_container_width=True)
