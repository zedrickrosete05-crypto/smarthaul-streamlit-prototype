# pages/2_Optimize_Routes.py
from __future__ import annotations

import math
import datetime as dt
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ────────────────────────────────────────────────────────────────────────────────
# Page
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="SmartHaul – Optimize Routes", layout="wide")
BUILD = "optimize-am/pm-v3"
st.title("Optimize Routes")
st.caption(f"Build: {BUILD}")

# ────────────────────────────────────────────────────────────────────────────────
# Guards
# ────────────────────────────────────────────────────────────────────────────────
if "orders_df" not in st.session_state or st.session_state["orders_df"] is None:
    st.warning("No orders loaded. Go to **Upload Orders** first.")
    st.page_link("pages/1_Upload_Orders.py", label="← Open Upload Orders", icon="⬅️")
    st.stop()

orders = st.session_state["orders_df"].reset_index(drop=True)
# Expected columns: order_id, place, lat, lon, tw_start_min, tw_end_min, service_time_min

# Basic safety for coords
if orders[["lat", "lon"]].isna().any().any():
    st.warning("Some orders have missing coordinates. They will be skipped for planning.")
orders = orders.dropna(subset=["lat", "lon"]).reset_index(drop=True)
if orders.empty:
    st.error("No orders with coordinates available to plan.")
    st.stop()

# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────
def min_to_ampm(m: int | float | None) -> str:
    if m is None or pd.isna(m): return "—"
    m = int(m) % (24 * 60)
    h, mm = divmod(m, 60)
    ampm = "AM" if h < 12 else "PM"
    h12 = (h % 12) or 12
    return f"{h12}:{mm:02d} {ampm}"

def haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    R = 6371.0088
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    x = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(min(1.0, math.sqrt(x)))

@st.cache_data(show_spinner=False)
def distance_matrix(points: List[Tuple[float, float]]) -> np.ndarray:
    n = len(points)
    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_km(points[i], points[j])
            M[i, j] = M[j, i] = d
    return M

# ────────────────────────────────────────────────────────────────────────────────
# Planner settings (sidebar)
# ────────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Planner Settings")
    speed_kph = st.slider("Avg speed (km/h)", 15, 90, 30, 5)
    max_stops_per_vehicle = st.slider("Max stops per route", 3, 50, 15, 1)
    num_vehicles = st.slider("Vehicles", 1, 50, 5, 1)

    st.subheader("Shift start")
    use_now = st.toggle("Use current time", value=True)
    shift_time = (
        dt.datetime.now().time().replace(second=0, microsecond=0)
        if use_now
        else st.time_input("Pick time", dt.time(8, 0), step=dt.timedelta(minutes=5))
    )

    # Optional manual depot
    st.subheader("Depot")
    use_manual_depot = st.checkbox("Use manual depot (place)", value=False)
    depot_coords = None
    if use_manual_depot:
        depot_place = st.text_input("Depot place", "Cebu IT Park")
        try:
            # Reuse your uploader geocoder if present
            from utils.geocode import geocode_place  # type: ignore
            depot_coords = geocode_place(depot_place)
        except Exception:
            depot_coords = None
        if depot_coords is None:
            st.info("Using fallback depot = centroid (could not geocode place).")

# Fallback depot = centroid of orders
if depot_coords is None:
    depot = (float(orders["lat"].mean()), float(orders["lon"].mean()))
else:
    depot = (float(depot_coords[0]), float(depot_coords[1]))

# ────────────────────────────────────────────────────────────────────────────────
# Distance matrix (include depot as index 0)
# ────────────────────────────────────────────────────────────────────────────────
points = [depot] + list(zip(orders["lat"].tolist(), orders["lon"].tolist()))
D_km = distance_matrix(points)

# ────────────────────────────────────────────────────────────────────────────────
# Clustering + routing (sweep -> NN -> 2-opt)
# ────────────────────────────────────────────────────────────────────────────────
def sweep_clusters(v_count: int, cap: int) -> list[list[int]]:
    """Split orders (1..N) into v_count clusters by polar angle around depot."""
    if len(points) <= 1:
        return [[] for _ in range(v_count)]
    def angle(idx: int) -> float:
        lat, lon = points[idx]
        dy = lat - depot[0]
        dx = (lon - depot[1]) * math.cos(math.radians(depot[0]))
        return (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0

    nodes = list(range(1, len(points)))
    nodes.sort(key=angle)

    clusters = [[] for _ in range(v_count)]
    i = 0
    for n in nodes:
        clusters[i].append(n)
        if len(clusters[i]) >= cap:
            i = (i + 1) % v_count
    return clusters

def nearest_neighbor_route(cluster: list[int]) -> list[int]:
    if not cluster:
        return [0, 0]
    un = set(cluster)
    route = [0]
    cur = 0
    while un:
        nxt = min(un, key=lambda j: D_km[cur, j])
        un.remove(nxt)
        route.append(nxt)
        cur = nxt
    route.append(0)
    return route

def route_len(route: list[int]) -> float:
    return float(sum(D_km[route[i], route[i+1]] for i in range(len(route)-1)))

def two_opt(route: list[int]) -> list[int]:
    best = route[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best)-2):
            for k in range(i+1, len(best)-1):
                if k - i == 1:
                    continue
                new = best[:i] + best[i:k][::-1] + best[k:]
                if route_len(new) + 1e-9 < route_len(best):
                    best = new
                    improved = True
    return best

clusters = sweep_clusters(int(num_vehicles), int(max_stops_per_vehicle))
routes_idx = [two_opt(nearest_neighbor_route(c)) for c in clusters]

# ────────────────────────────────────────────────────────────────────────────────
# Build ETAs and plan table
# ────────────────────────────────────────────────────────────────────────────────
v_speed_km_min = max(float(speed_kph), 5.0) / 60.0
shift_start_min = int(shift_time.hour) * 60 + int(shift_time.minute)

rows = []
for v, route in enumerate(routes_idx, start=1):
    tmin = shift_start_min
    for i in range(len(route) - 1):
        a, b = route[i], route[i + 1]
        leg_km = float(D_km[a, b])
        drive_min = int(round(leg_km / v_speed_km_min)) if v_speed_km_min > 0 else 0
        tmin += drive_min

        if b != 0:
            r = orders.iloc[b - 1]
            tw_s = int(r["tw_start_min"]) if pd.notna(r["tw_start_min"]) else None
            tw_e = int(r["tw_end_min"]) if pd.notna(r["tw_end_min"]) else None
            svc = int(r["service_time_min"]) if pd.notna(r["service_time_min"]) else 0

            # wait if arriving earlier than window start
            if tw_s is not None and tmin < tw_s:
                tmin = tw_s

            within = True
            if tw_e is not None and tmin > tw_e:
                within = False

            rows.append(dict(
                vehicle_id=f"V{v}",
                order_id=str(r["order_id"]),
                lat=float(r["lat"]),
                lon=float(r["lon"]),
                eta=min_to_ampm(tmin),
                tw_start=min_to_ampm(tw_s),
                tw_end=min_to_ampm(tw_e),
                within_window=within,
                leg_km=round(leg_km, 2),
            ))

            # add service duration before next leg
            tmin += svc

df_plan = pd.DataFrame(rows)
if df_plan.empty:
    st.warning("No routes could be constructed. Check data and settings.")
    st.stop()

df_plan["alert"] = df_plan.apply(
    lambda r: "" if r["within_window"] in (True, "—") else "Late risk", axis=1
)
df_plan["status"] = "Planned"

# Persist for other pages
st.session_state["routes_df"] = df_plan.copy()
st.session_state["settings"] = dict(
    speed_kph=float(speed_kph),
    max_stops_per_vehicle=int(max_stops_per_vehicle),
    num_vehicles=int(num_vehicles),
    shift_start=min_to_ampm(shift_start_min),
    depot_lat=depot[0],
    depot_lon=depot[1],
)

# ────────────────────────────────────────────────────────────────────────────────
# Results table (AM/PM)
# ────────────────────────────────────────────────────────────────────────────────
st.success(f"Planned {df_plan['vehicle_id'].nunique()} route(s) for {len(df_plan)} stop(s).")
st.markdown("### Planned routes")

df_show = df_plan.copy()
df_show["Stop #"] = df_show.groupby("vehicle_id").cumcount() + 1
df_show["Time window"] = df_show["tw_start"].astype(str) + " – " + df_show["tw_end"].astype(str)
df_show["Lat"] = pd.to_numeric(df_show["lat"], errors="coerce").round(4)
df_show["Lon"] = pd.to_numeric(df_show["lon"], errors="coerce").round(4)

hide_coords = st.checkbox("Hide coordinates", value=True)

display_df = df_show.rename(columns={"vehicle_id": "Vehicle", "order_id": "Order", "eta": "ETA"})
display_cols = ["Vehicle", "Stop #", "Order", "ETA", "Time window", "alert"]
if not hide_coords:
    display_cols += ["Lat", "Lon"]

st.dataframe(display_df[display_cols], use_container_width=True, hide_index=True)

# ────────────────────────────────────────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("### Route summary")
summary = (
    display_df.sort_values(["Vehicle", "Stop #"], kind="mergesort")
              .groupby("Vehicle", sort=False)
              .agg(Stops=("Order", "count"),
                   **{"First ETA": ("ETA", "first")},
                   **{"Last ETA": ("ETA", "last")},
                   Alerts=("alert", lambda s: int((s != "").sum())))
              .reset_index()
)
st.dataframe(summary, hide_index=True, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────────
# Optional map (pydeck)
# ────────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("### Map")
if st.checkbox("Show map (pydeck)", value=False):
    try:
        import pydeck as pdk

        df_map = display_df.copy()
        df_map["stop_idx"] = df_map.groupby("Vehicle").cumcount() + 1
        df_map = df_map.sort_values(["Vehicle", "stop_idx"], kind="mergesort").reset_index(drop=True)

        palette = np.array([
            [0,122,255],[255,45,85],[88,86,214],[255,149,0],
            [52,199,89],[175,82,222],[255,59,48],[90,200,250]
        ])
        veh_ids = df_map["Vehicle"].unique()
        cmap = {v: palette[i % len(palette)].tolist() for i, v in enumerate(veh_ids)}
        df_map["color"] = df_map["Vehicle"].map(cmap)

        rows_path = []
        for v, g in df_map.groupby("Vehicle", sort=False):
            g = g.sort_values("stop_idx")
            pts = [{"lon": float(r.Lon), "lat": float(r.Lat)} for r in g.itertuples(index=False)]
            rows_path.append({"Vehicle": v, "path": pts, "color": cmap[v]})
        paths = pd.DataFrame(rows_path)

        layers = [
            pdk.Layer("PathLayer", data=paths, get_path="path", get_width=4,
                      get_color="color", width_min_pixels=2, pickable=False),
            pdk.Layer("ScatterplotLayer", data=df_map, get_position='[Lon, Lat]',
                      get_radius=60, get_fill_color="color", get_line_color=[255,255,255],
                      line_width_min_pixels=1, pickable=True),
            pdk.Layer("TextLayer", data=df_map, get_position='[Lon, Lat]',
                      get_text="stop_idx", get_size=12, get_color=[230,230,230],
                      get_alignment_baseline="'center'")
        ]
        view = pdk.ViewState(latitude=float(df_map.Lat.mean()),
                             longitude=float(df_map.Lon.mean()), zoom=11)
        tooltip = {"text": "Stop {stop_idx}\n{Vehicle}\nETA {ETA}"}
        st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, tooltip=tooltip))
    except Exception as e:
        st.info(f"Map skipped: {e}")

# ────────────────────────────────────────────────────────────────────────────────
# Download
# ────────────────────────────────────────────────────────────────────────────────
st.download_button(
    "Download planned routes (CSV)",
    data=df_plan.to_csv(index=False).encode("utf-8"),
    file_name="routes_plan.csv",
    mime="text/csv",
)
