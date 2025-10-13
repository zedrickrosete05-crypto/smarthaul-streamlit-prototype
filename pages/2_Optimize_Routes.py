# pages/2_Optimize_Routes.py
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ---------------- Page setup ----------------
st.set_page_config(page_title="SmartHaul – Optimize Routes", layout="wide")
st.title("Optimize Routes")

# ---------------- Guards ----------------
if "orders_df" not in st.session_state or st.session_state["orders_df"] is None:
    st.warning("No orders loaded. Go to **Upload Orders** first.")
    st.page_link("pages/1_Upload_Orders.py", label="← Open Upload Orders", icon="⬅️")
    st.stop()

orders = st.session_state["orders_df"].reset_index(drop=True)
# Expected (best-case) columns from Upload page:
#   order_id, lat, lon, service_time_min, tw_start_min, tw_end_min
# But we also support: service_min, tw_start, tw_end (HH:MM) and convert here.

# ---------------- Helpers ----------------
def min_to_hhmm(m: int | float | None) -> str:
    if m is None or pd.isna(m):
        return "—"
    m = int(m) % (24 * 60)
    return f"{m // 60:02d}:{m % 60:02d}"

def hhmm_to_min(s: str | int | float | None) -> int | None:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    s = str(s).strip()
    if not s or ":" not in s:
        return None
    try:
        h, m = [int(x) for x in s.split(":")]
        return h * 60 + m
    except Exception:
        return None

def haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    R = 6371.0088
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    x = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(min(1, math.sqrt(x)))

@st.cache_data(show_spinner=False)
def distance_matrix(points: List[Tuple[float, float]]) -> np.ndarray:
    n = len(points)
    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            km = haversine_km(points[i], points[j])
            M[i, j] = M[j, i] = km
    return M

# ---------------- Normalize uploads ----------------
# Accept both *_min and HH:MM string columns
if "service_time_min" not in orders.columns and "service_min" in orders.columns:
    orders["service_time_min"] = pd.to_numeric(orders["service_min"], errors="coerce").fillna(0).astype(int)

if "tw_start_min" not in orders.columns and "tw_start" in orders.columns:
    orders["tw_start_min"] = orders["tw_start"].map(hhmm_to_min)

if "tw_end_min" not in orders.columns and "tw_end" in orders.columns:
    orders["tw_end_min"] = orders["tw_end"].map(hhmm_to_min)

need_cols = ["order_id", "lat", "lon", "service_time_min", "tw_start_min", "tw_end_min"]
missing = [c for c in need_cols if c not in orders.columns]
if missing:
    st.error(f"Upload page didn't produce required columns: {missing}")
    st.stop()

# Coerce numerics to be safe
orders["lat"] = pd.to_numeric(orders["lat"], errors="coerce")
orders["lon"] = pd.to_numeric(orders["lon"], errors="coerce")
orders["service_time_min"] = pd.to_numeric(orders["service_time_min"], errors="coerce").fillna(0).astype(int)

# ---------------- Sidebar parameters ----------------
with st.sidebar:
    st.header("Planner Settings")
    speed_kph = st.slider("Avg speed (km/h)", 15, 90, 30, 5)
    max_stops_per_vehicle = st.slider("Max stops per route", 3, 50, 15, 1)
    num_vehicles = st.slider("Vehicles", 1, 50, 5, 1)
    shift_start = st.text_input("Shift start (HH:MM)", "08:00")

    st.header("Depot")
    use_manual_depot = st.checkbox("Use manual depot (place)", value=False)
    depot = None
    if use_manual_depot:
        depot_place = st.text_input("Depot place", "Cebu IT Park")
        depot_coords = None
        try:
            # If you have utils/geocode.py with geocode_place() you can use it here.
            from utils.geocode import geocode_place  # type: ignore
            depot_coords = geocode_place(depot_place)
        except Exception:
            depot_coords = None
        if depot_coords is None:
            st.warning("Could not geocode depot place. Falling back to centroid.")
        else:
            depot = (float(depot_coords[0]), float(depot_coords[1]))

# Fallback depot = centroid of orders
if depot is None:
    depot = (float(orders["lat"].mean()), float(orders["lon"].mean()))

# ---------------- Build distance matrix ----------------
points = [depot] + list(zip(orders["lat"].tolist(), orders["lon"].tolist()))
D_km = distance_matrix(points)  # includes depot at index 0

# ---------------- Greedy planner ----------------
def plan_routes(D: np.ndarray, num_veh: int, max_stops: int) -> List[List[int]]:
    """
    Return routes as sequences of node indices (0 = depot).
    Orders are nodes 1..N.
    """
    n = D.shape[0] - 1
    unassigned = set(range(1, n + 1))
    routes: List[List[int]] = []

    # Build up to num_veh initial routes
    for _ in range(num_veh):
        if not unassigned:
            routes.append([0, 0])
            continue
        route = [0]
        current = 0
        while unassigned and (len(route) - 1) < max_stops:
            nxt = min(unassigned, key=lambda j: D[current, j])
            unassigned.remove(nxt)
            route.append(nxt)
            current = nxt
        route.append(0)
        routes.append(route)

    # Round-robin add remaining stops
    r = 0
    while unassigned:
        nxt = min(unassigned, key=lambda j: D[routes[r][-2], j])
        unassigned.remove(nxt)
        routes[r].insert(-1, nxt)
        r = (r + 1) % len(routes)

    return routes

routes_idx = plan_routes(D_km, int(num_vehicles), int(max_stops_per_vehicle))

# ---------------- Build plan with ETAs & window checks ----------------
v_speed_km_min = max(float(speed_kph), 5.0) / 60.0  # km/min (avoid zero)
time_now = hhmm_to_min(shift_start) or 8 * 60

rows = []
for v, route in enumerate(routes_idx, start=1):
    tmin = time_now
    for i in range(len(route) - 1):
        a, b = route[i], route[i + 1]
        leg_km = float(D_km[a, b])
        drive_min = int(round(leg_km / v_speed_km_min)) if v_speed_km_min > 0 else 0
        tmin += drive_min

        # If arriving at an order node
        if b != 0:
            ord_row = orders.iloc[b - 1]
            tw_s = int(ord_row["tw_start_min"]) if pd.notna(ord_row["tw_start_min"]) else None
            tw_e = int(ord_row["tw_end_min"]) if pd.notna(ord_row["tw_end_min"]) else None
            svc = int(ord_row["service_time_min"]) if pd.notna(ord_row["service_time_min"]) else 0

            # Wait if early
            if tw_s is not None and tmin < tw_s:
                tmin = tw_s

            eta_hhmm = min_to_hhmm(tmin)
            within_window = True
            if tw_e is not None and tmin > tw_e:
                within_window = False

            rows.append(
                dict(
                    vehicle_id=f"V{v}",
                    order_id=str(ord_row["order_id"]),
                    lat=float(ord_row["lat"]),
                    lon=float(ord_row["lon"]),
                    eta=eta_hhmm,
                    tw_start=min_to_hhmm(tw_s),
                    tw_end=min_to_hhmm(tw_e),
                    within_window=within_window,
                    leg_km=round(leg_km, 2),
                )
            )

            # Add service time before next leg
            tmin += svc

df_plan = pd.DataFrame(rows)
if df_plan.empty:
    st.warning("No routes could be constructed. Check your data (coordinates/windows).")
    st.stop()

# Alerts + status
df_plan["alert"] = df_plan.apply(lambda r: "" if (r["within_window"] in (True, "—")) else "Late risk", axis=1)
df_plan["status"] = "Planned"

# Persist for other pages
st.session_state["routes_df"] = df_plan.copy()
st.session_state["settings"] = dict(
    speed_kph=float(speed_kph),
    max_stops_per_vehicle=int(max_stops_per_vehicle),
    num_vehicles=int(num_vehicles),
    shift_start=shift_start,
    depot_lat=depot[0],
    depot_lon=depot[1],
)

# ---------------- Results table ----------------
st.success(f"Planned {df_plan['vehicle_id'].nunique()} route(s) for {len(df_plan)} stops.")
st.markdown("### Planned routes")

df_show = df_plan.copy()
df_show["Stop #"] = df_show.groupby("vehicle_id").cumcount() + 1
df_show["Time window"] = df_show["tw_start"].astype(str) + " – " + df_show["tw_end"].astype(str)
df_show["Lat"] = pd.to_numeric(df_show["lat"], errors="coerce").round(4)
df_show["Lon"] = pd.to_numeric(df_show["lon"], errors="coerce").round(4)

hide_coords = st.checkbox("Hide coordinates", value=True)

# Rename FIRST, then select using the NEW names (prevents KeyError)
display_df = df_show.rename(columns={"vehicle_id": "Vehicle", "order_id": "Order", "eta": "ETA"})
display_cols = ["Vehicle", "Stop #", "Order", "ETA", "Time window", "alert"]
if not hide_coords:
    display_cols += ["Lat", "Lon"]

st.dataframe(display_df[display_cols], use_container_width=True, hide_index=True)

# ---------------- Summary ----------------
st.divider()
st.markdown("### Route summary")

def first_eta(series: pd.Series) -> str:
    for v in series:
        if v not in ("—", "N/A"):
            return v
    return "—"

def last_eta(series: pd.Series) -> str:
    ser = list(series)
    for v in reversed(ser):
        if v not in ("—", "N/A"):
            return v
    return "—"

# distance per route
def route_distance_km(route_idx: list[int]) -> float:
    return float(sum(D_km[route_idx[i], route_idx[i + 1]] for i in range(len(route_idx) - 1)))

veh_routes = {f"V{i+1}": r for i, r in enumerate(routes_idx)}

summary = (
    display_df.sort_values(["Vehicle", "Stop #"], kind="mergesort")
    .groupby("Vehicle", sort=False)
    .agg(
        Stops=("Order", "count"),
        **{"First ETA": ("ETA", first_eta)},
        **{"Last ETA": ("ETA", last_eta)},
        Alerts=("alert", lambda s: int((s != "").sum())),
    )
    .reset_index()
)
summary["Distance (km)"] = summary["Vehicle"].map(lambda v: round(route_distance_km(veh_routes.get(v, [0, 0])), 2))

st.dataframe(summary, hide_index=True, use_container_width=True)

# ---------------- Optional map (pydeck) ----------------
st.divider()
st.markdown("### Map")
if st.checkbox("Show map (pydeck)", value=False):
    try:
        import pydeck as pdk

        df_map = display_df.copy()
        df_map["stop_idx"] = df_map.groupby("Vehicle").cumcount() + 1
        df_map = df_map.sort_values(["Vehicle", "stop_idx"], kind="mergesort").reset_index(drop=True)

        # color per vehicle
        palette = np.array(
            [
                [0, 122, 255],
                [255, 45, 85],
                [88, 86, 214],
                [255, 149, 0],
                [52, 199, 89],
                [175, 82, 222],
                [255, 59, 48],
                [90, 200, 250],
            ]
        )
        veh_ids = df_map["Vehicle"].unique()
        cmap = {v: palette[i % len(palette)].tolist() for i, v in enumerate(veh_ids)}
        df_map["color"] = df_map["Vehicle"].map(cmap)

        # path rows
        rows_path = []
        for v, g in df_map.groupby("Vehicle", sort=False):
            g = g.sort_values("stop_idx")
            pts = [{"lon": float(r.Lon), "lat": float(r.Lat)} for r in g.itertuples(index=False)]
            rows_path.append({"Vehicle": v, "path": pts, "color": cmap[v]})
        paths = pd.DataFrame(rows_path)

        # depot point
        depot_df = pd.DataFrame(
            [
                {
                    "Vehicle": "Depot",
                    "Lat": st.session_state.get("settings", {}).get("depot_lat", depot[0]),
                    "Lon": st.session_state.get("settings", {}).get("depot_lon", depot[1]),
                    "stop_idx": 0,
                    "ETA": "—",
                }
            ]
        )

        layers = [
            pdk.Layer("PathLayer", data=paths, get_path="path", get_width=4, get_color="color",
                      width_min_pixels=2, pickable=False),
            pdk.Layer("ScatterplotLayer", data=df_map, get_position='[Lon, Lat]', get_radius=60,
                      get_fill_color="color", get_line_color=[255, 255, 255], line_width_min_pixels=1, pickable=True),
            pdk.Layer("TextLayer", data=df_map, get_position='[Lon, Lat]', get_text="stop_idx",
                      get_size=12, get_color=[230, 230, 230], get_alignment_baseline="'center'"),
            pdk.Layer("ScatterplotLayer", data=depot_df, get_position='[Lon, Lat]', get_radius=80,
                      get_fill_color=[255, 255, 255], get_line_color=[0, 0, 0], line_width_min_pixels=1, pickable=False),
        ]
        view = pdk.ViewState(latitude=float(df_map.Lat.mean()), longitude=float(df_map.Lon.mean()), zoom=11)
        tooltip = {"text": "Stop {stop_idx}\n{Vehicle}\nETA {ETA}"}
        st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, tooltip=tooltip))
    except Exception as e:
        st.info(f"Map skipped: {e}")

# ---------------- Download ----------------
st.download_button(
    "Download planned routes (CSV)",
    data=df_plan.to_csv(index=False).encode("utf-8"),
    file_name="routes_plan.csv",
    mime="text/csv",
)
