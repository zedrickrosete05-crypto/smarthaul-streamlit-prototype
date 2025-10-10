from branding import setup_branding, section
setup_branding("SmartHaul – Upload Orders")

import pandas as pd
import streamlit as st

REQUIRED = ["order_id","lat","lon","tw_start","tw_end","service_min"]

section("Upload Orders")

# Template to download
tmpl = pd.DataFrame([
    {"order_id":"O-1001","lat":10.3157,"lon":123.8854,"tw_start":"08:30","tw_end":"11:00","service_min":7},
    {"order_id":"O-1002","lat":10.3099,"lon":123.9180,"tw_start":"09:00","tw_end":"12:00","service_min":5},
    {"order_id":"O-1003","lat":10.3070,"lon":123.9011,"tw_start":"10:00","tw_end":"13:00","service_min":10},
])
st.download_button("Download CSV template (lat/lon)",
                   tmpl.to_csv(index=False).encode(),
                   file_name="orders_template.csv", mime="text/csv")

st.caption("Upload CSV with columns: order_id, lat, lon, tw_start, tw_end, service_min")
file = st.file_uploader(" ", type="csv")
if not file:
    st.stop()

df = pd.read_csv(file)
df.columns = [c.strip() for c in df.columns]

missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

# type coercions & light cleaning
df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
df["service_min"] = pd.to_numeric(df["service_min"], errors="coerce").fillna(0).clip(lower=0)
for c in ("tw_start","tw_end"):
    df[c] = df[c].astype(str).str.extract(r"^(\d{1,2}:\d{2})", expand=False)

good = df.dropna(subset=["lat","lon"]).copy()
if good.empty:
    st.error("All rows have invalid lat/lon."); st.stop()

st.session_state["orders_df"] = good[REQUIRED].copy()
st.success(f"Loaded {len(good)} orders.")
st.dataframe(st.session_state["orders_df"].head(20), use_container_width=True)
@@ -23,25 +23,80 @@ file = st.file_uploader(" ", type="csv")
if not file:
    st.stop()

df = pd.read_csv(file)
df.columns = [c.strip() for c in df.columns]

missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

# type coercions & light cleaning
df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
df["service_min"] = pd.to_numeric(df["service_min"], errors="coerce").fillna(0).clip(lower=0)
for c in ("tw_start","tw_end"):
    df[c] = df[c].astype(str).str.extract(r"^(\d{1,2}:\d{2})", expand=False)

good = df.dropna(subset=["lat","lon"]).copy()
if good.empty:
    st.error("All rows have invalid lat/lon."); st.stop()

st.session_state["orders_df"] = good[REQUIRED].copy()
st.success(f"Loaded {len(good)} orders.")
st.dataframe(st.session_state["orders_df"].head(20), use_container_width=True)

section("Order overview")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Orders loaded", len(good))
with c2:
    avg_service = pd.to_numeric(good["service_min"], errors="coerce").mean()
    if pd.notna(avg_service):
        st.metric("Avg. service time", f"{avg_service:.1f} min")
    else:
        st.metric("Avg. service time", "—")
with c3:
    time_windows = good[["tw_start", "tw_end"]].dropna().astype(str)
    if not time_windows.empty:
        st.metric("Time window span", f"{time_windows['tw_start'].min()} – {time_windows['tw_end'].max()}")
    else:
        st.metric("Time window span", "—")

if {"lat", "lon"}.issubset(good.columns):
    latlon = good.dropna(subset=["lat", "lon"])
    if not latlon.empty:
        centroid_lat = float(latlon["lat"].mean())
        centroid_lon = float(latlon["lon"].mean())
        st.session_state.setdefault("depot_lat", round(centroid_lat, 4))
        st.session_state.setdefault("depot_lon", round(centroid_lon, 4))

        if st.checkbox("Show map preview", value=True):
            try:
                import pydeck as pdk

                points = latlon.copy()
                points["tooltip"] = points.apply(
                    lambda r: (
                        f"{r['order_id']}\nTW {r['tw_start']}–{r['tw_end']}"
                        if pd.notna(r["tw_start"]) and pd.notna(r["tw_end"])
                        else str(r["order_id"])
                    ),
                    axis=1,
                )
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=points,
                    get_position='[lon, lat]',
                    get_radius=70,
                    get_fill_color=[11, 60, 93, 200],
                    get_line_color=[255, 255, 255],
                    pickable=True,
                )
                view = pdk.ViewState(latitude=centroid_lat, longitude=centroid_lon, zoom=11)
                st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view,
                                         tooltip={"text": "{tooltip}"}))
            except Exception as exc:
                st.info(f"Map preview unavailable: {exc}")

st.session_state.setdefault("start_time", None)
