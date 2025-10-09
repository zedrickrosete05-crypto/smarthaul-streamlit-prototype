import io
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SmartHaul – Upload Orders", page_icon="📤")
st.title("📤 Upload Orders")

ACCEPTED_COLS = {"order_id", "place", "lat", "lon", "tw_start", "tw_end", "service_min"}

st.caption(
    "You can upload **place names** (we’ll geocode) or **lat/lon**. "
    "Optional: tw_start, tw_end in HH:MM and service_min in minutes."
)

# -------- Template download (friendly) --------
tmpl = pd.DataFrame([
    {"order_id":"MNL-2001","place":"SM Mall of Asia, Pasay","tw_start":"08:30","tw_end":"11:00","service_min":7},
    {"order_id":"MNL-2002","place":"Ayala Triangle Gardens, Makati","tw_start":"09:00","tw_end":"12:00","service_min":5},
    {"order_id":"MNL-2003","lat":14.6091,"lon":121.0223,"tw_start":"10:00","tw_end":"13:00","service_min":10},
])
st.download_button(
    "Download CSV template (place or lat/lon)",
    data=tmpl.to_csv(index=False).encode(),
    file_name="orders_template_friendly.csv",
    mime="text/csv",
)

file = st.file_uploader("Upload CSV", type="csv")
if not file:
    st.stop()

# -------- Read & basic normalize --------
df = pd.read_csv(file)
df.columns = [c.strip().lower() for c in df.columns]
unknown = [c for c in df.columns if c not in ACCEPTED_COLS]
if unknown:
    st.info(f"Ignoring unknown columns: {unknown}")

for c in ("tw_start", "tw_end"):
    if c in df.columns:
        df[c] = df[c].astype(str).str.extract(r"^(\d{1,2}:\d{2})", expand=False).fillna("")

if "service_min" in df.columns:
    df["service_min"] = pd.to_numeric(df["service_min"], errors="coerce").fillna(0).clip(lower=0)
else:
    df["service_min"] = 0

# -------- Geocoding helpers (forward) --------
def _setup_geocoder():
    try:
        from geopy.geocoders import Nominatim
        from geopy.extra.rate_limiter import RateLimiter
        geolocator = Nominatim(user_agent="smarthaul-demo")
        forward = RateLimiter(geolocator.geocode, min_delay_seconds=1)
        def f(q: str):
            if not q:
                return None
            try:
                loc = forward(q)
                if loc:
                    return {"lat": float(loc.latitude), "lon": float(loc.longitude)}
            except Exception:
                return None
            return None
        return f
    except Exception:
        import requests
        def f(q: str):
            if not q:
                return None
            try:
                url = "https://nominatim.openstreetmap.org/search"
                params = {"q": q, "format": "json", "limit": 1}
                headers = {"User-Agent": "smarthaul-demo/1.0 (contact: example@example.com)"}
                r = requests.get(url, params=params, headers=headers, timeout=10)
                r.raise_for_status()
                js = r.json()
                if js:
                    return {"lat": float(js[0]["lat"]), "lon": float(js[0]["lon"])}
            except Exception:
                return None
            return None
        return f

_forward = _setup_geocoder()

@st.cache_data(show_spinner=False)
def geocode_cached(q: str):
    return _forward(q) if q else None

# -------- Geocode rows missing lat/lon but having place --------
df["lat"] = pd.to_numeric(df.get("lat"), errors="coerce")
df["lon"] = pd.to_numeric(df.get("lon"), errors="coerce")

if "place" in df.columns:
    need_geo = df[df["lat"].isna() | df["lon"].isna()]
    if len(need_geo):
        with st.spinner("Geocoding place names…"):
            for i, r in need_geo.iterrows():
                res = geocode_cached(str(r.get("place", "")).strip())
                if res:
                    df.at[i, "lat"] = res["lat"]
                    df.at[i, "lon"] = res["lon"]

# -------- Final validation --------
bad = df[df["lat"].isna() | df["lon"].isna()]
if len(bad):
    st.warning(f"{len(bad)} row(s) couldn’t be geocoded. Fix the place text or add lat/lon. Showing them below.")
    st.dataframe(bad, use_container_width=True)

good = df.dropna(subset=["lat","lon"]).copy()
if good.empty:
    st.error("No valid rows with coordinates. Please correct your file.")
    st.stop()

if "order_id" not in good.columns:
    st.error("Missing required column: order_id")
    st.stop()

if "place" not in good.columns:
    good["place"] = ""

# Only keep what downstream needs (+ place for display)
out_cols = ["order_id","lat","lon","tw_start","tw_end","service_min","place"]
st.session_state["orders_df"] = good[out_cols].copy()

st.success(f"Loaded {len(good)} orders. {len(bad)} need attention.")
st.dataframe(st.session_state["orders_df"].head(20), use_container_width=True)
