# pages/2_Optimize_Routes.py
from __future__ import annotations

import math, os
import datetime as dt
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ────────────────────────────────────────────────────────────────────────────────
# Page
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="SmartHaul – Optimize Routes", layout="wide")
BUILD = "optimize-capacity-duration-v1 + persist"
st.title("Optimize Routes")
st.caption(f"Build: {BUILD}")

# ────────────────────────────────────────────────────────────────────────────────
# Simple persistence (routes saved to ./data/routes_df.csv)
# ────────────────────────────────────────────────────────────────────────────────
DATA_DIR = "data"; os.makedirs(DATA_DIR, exist_ok=True)
ROUTES_PATH = os.path.join(DATA_DIR, "routes_df.csv")

def save_routes(df: pd.DataFrame) -> None:
    try:
        df.to_csv(ROUTES_PATH, index=False)
    except Exception:
        pass

def load_routes() -> pd.DataFrame | None:
    try:
        if os.path.exists(ROUTES_PATH):
            return pd.read_csv(ROUTES_PATH)
    except Exception:
        return None
    return None

with st.sidebar:
    if st.button("Load last plan"):
        prev = load_routes()
        if prev is not None and not prev.empty:
            st.session_state["routes_df"] = prev.copy()
            st.success(f"Restored plan with {len(prev)} stop(s).")
            st.stop()
        else:
            st.info("No previous plan found.")

# ────────────────────────────────────────────────────────────────────────────────
# Guards
# ────────────────────────────────────────────────────────────────────────────────
if "orders_df" not in st.session_state or st.session_state["orders_df"] is None:
    st.warning("No orders loaded. Go to **Upload Orders** first.")
    try:
        st.page_link("pages/1_Upload_Orders.py", label="← Open Upload Orders")
    except Exception:
        st.markdown("[← Open Upload Orders](pages/1_Upload_Orders.py)")
    st.stop()

orders_raw = st.session_state["orders_df"].copy()

# Coerce required numeric fields
for col in ["lat", "lon", "service_time_min", "tw_start_min", "tw_end_min", "demand"]:
    if col not in orders_raw.columns:
        orders_raw[col] = np.nan
orders_raw["lat"] = pd.to_numeric(orders_raw["lat"], errors="coerce")
orders_raw["lon"] = pd.to_numeric(orders_raw["lon"], errors="coerce")
orders_raw["service_time_min"] = pd.to_numeric(orders_raw["service_time_min"], errors="coerce").fillna(0).clip(lower=0).astype(int)
orders_raw["tw_start_min"] = pd.to_numeric(orders_raw["tw_start_min"], errors="coerce")
orders_raw["tw_end_min"] = pd.to_numeric(orders_raw["tw_end_min"], errors="coerce")
orders_raw["demand"] = pd.to_numeric(orders_raw.get("demand", 1), errors="coerce").fillna(1).clip(lower=0).astype(int)

# Drop rows without coordinates
orders = orders_raw.dropna(subset=["lat", "lon"]).reset_index(drop=True)
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
    max_stops_per_vehicle = st.slider("Max stops per route", 3, 100, 30, 1)

    st.subheader("Vehicles")
    num_vehicles = st.slider("Number of vehicles", 1, 100, 5, 1)
    vehicle_capacity = st.number_input("Vehicle capacity (sum of order 'demand')", min_value=0, value=30, step=1)
    max_route_minutes = st.number_input("Max route duration (minutes)", min_value=30, value=8*60, step=15)

    st.subheader("Shift start")
    use_now = st.toggle("Use current time", value=True)
    shift_time = (
        dt.datetime.now().time().replace(second=0, microsecond=0)
        if use_now else st.time_input("Pick time", dt.time(8, 0), step=dt.timedelta(minutes=5))
    )

    st.subheader("Depot")
    use_manual_depot = st.checkbox("Use manual depot (place)", value=False)
    depot_coords = None
    if use_manual_depot:
        depot_place = st.text_input("Depot place", "Cebu IT Park")
        try:
            from utils.geocode import geocode_place  # optional; ignore if missing
            depot_coords = geocode_place(depot_place)
        except Exception:
            depot_coords = None
        if depot_coords is None:
            st.info("Using centroid as depot (could not geocode).")

# Fallback depot = centroid
if depot_coords is None:
    depot = (float(orders["lat"].mean()), float(orders["lon"].mean()))
else:
    depot = (float(depot_coords[0]), float(depot_coords[1]))

# ────────────────────────────────────────────────────────────────────────────────
# Distance matrix (include depot as index 0)
# ────────────────────────────────────────────────────────────────────────────────
points = [depot] + list(zip(orders["lat"].tolist(), orders["lon"].tolist()))
D_km = distance_matrix(points)

# Speed in minutes per km
v_speed_km_min = max(float(speed_kph), 5.0) / 60.0
shift_start_min = int(shift_time.hour) * 60 + int(shift_time.minute)

# ────────────────────────────────────────────────────────────────────────────────
# Capacity-aware sweep clustering (angle ordering) with stop cap
# ────────────────────────────────────────────────────────────────────────────────
def angle(idx: int) -> float:
    lat, lon = points[idx]
    dy = lat - depot[0]
    dx = (lon - depot[1]) * math.cos(math.radians(depot[0]))
    return (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0

nodes = list(range(1, len(points)))
nodes.sort(key=angle)

clusters: list[list[int]] = [[] for _ in range(int(num_vehicles))]
cluster_loads = [0 for _ in range(int(num_vehicles))]
i = 0
for n in nodes:
    dem = int(orders.iloc[n-1]["demand"])
    # rotate until we find a vehicle with space
    tries = 0
    while tries < num_vehicles and (cluster_loads[i] + dem > vehicle_capacity or len(clusters[i]) >= max_stops_per_vehicle):
        i = (i + 1) % int(num_vehicles)
        tries += 1
    # assign if any fits; otherwise leave unassigned for later
    if cluster_loads[i] + dem <= vehicle_capacity and len(clusters[i]) < max_stops_per_vehicle:
        clusters[i].append(n)
        cluster_loads[i] += dem
        i = (i + 1) % int(num_vehicles)

# Any nodes still not assigned because of capacity/stop-cap?
assigned_set = set([n for cl in clusters for n in cl])
unassigned_nodes = [n for n in nodes if n not in assigned_set]

# ────────────────────────────────────────────────────────────────────────────────
# Route construction: nearest neighbor → 2-opt, with duration check
# ────────────────────────────────────────────────────────────────────────────────
def route_len_km(route: list[int]) -> float:
    return float(sum(D_km[route[i], route[i+1]] for i in range(len(route)-1)))

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

def two_opt(route: list[int]) -> list[int]:
    best = route[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for k in range(i + 1, len(best) - 1):
                if k - i == 1:
                    continue
                new = best[:i] + best[i:k][::-1] + best[k:]
                if route_len_km(new) + 1e-9 < route_len_km(best):
                    best = new
                    improved = True
    return best

def simulate_route_time_minutes(route: list[int]) -> int:
    """
    Returns total minutes from shift start until returning to depot,
    including driving, waiting for windows, and service times.
    """
    tmin = shift_start_min
    for i in range(len(route) - 1):
        a, b = route[i], route[i+1]
        leg_km = float(D_km[a, b])
        drive = int(round(leg_km / v_speed_km_min)) if v_speed_km_min > 0 else 0
        tmin += drive
        if b != 0:
            row = orders.iloc[b - 1]
            tw_s = int(row["tw_start_min"]) if pd.notna(row["tw_start_min"]) else None
            svc = int(row["service_time_min"]) if pd.notna(row["service_time_min"]) else 0
            if tw_s is not None and tmin < tw_s:
                tmin = tw_s
            tmin += svc
    return tmin - shift_start_min

# Build routes per cluster and trim if duration exceeds limit
routes_idx: list[list[int]] = []
overflow_nodes: list[int] = []

for cl in clusters:
    base = two_opt(nearest_neighbor_route(cl))
    if base == [0, 0]:
        routes_idx.append(base)
        continue

    # If duration fits, take it. Otherwise, trim from the tail into overflow.
    if simulate_route_time_minutes(base) <= max_route_minutes:
        routes_idx.append(base)
        continue

    # Greedy trimming: pop interior nodes (before depot) until within limit.
    route = base[:]
    removed = []
    while len(route) > 3 and simulate_route_time_minutes(route) > max_route_minutes:
        # Remove the node whose removal reduces time the most (simple heuristic)
        best_gain = -1
        best_pos = None
        for pos in range(1, len(route) - 1):  # don't remove depots at 0 or -1
            trial = route[:pos] + route[pos+1:]
            gain = simulate_route_time_minutes(route) - simulate_route_time_minutes(trial)
            if gain > best_gain:
                best_gain, best_pos = gain, pos
        if best_pos is None:
            break
        removed.append(route[best_pos])
        route = route[:best_pos] + route[best_pos+1:]

    routes_idx.append(route if route else [0, 0])
    overflow_nodes.extend(removed)

# Try to place overflow + initially unassigned onto any route with spare capacity & duration
def route_demand(route: list[int]) -> int:
    if not route: return 0
    return int(sum(int(orders.iloc[n-1]["demand"]) for n in route if n != 0))

for n in overflow_nodes + unassigned_nodes:
    placed = False
    dem = int(orders.iloc[n-1]["demand"])
    for r_i, r in enumerate(routes_idx):
        # Capacity check
        if route_demand(r) + dem > vehicle_capacity:
            continue
        # Try cheap insertions at best spot
        best_route = None
        best_len = float("inf")
        for pos in range(1, len(r)):  # insert before depot at end too
            cand = r[:pos] + [n] + r[pos:]
            if simulate_route_time_minutes(cand) <= max_route_minutes:
                L = route_len_km(cand)
                if L < best_len:
                    best_len = L; best_route = cand
        if best_route is not None:
            routes_idx[r_i] = best_route
            placed = True
            break
    if not placed:
        # Could not fit anywhere; keep unassigned
        unassigned_nodes.append(n)

# Deduplicate unassigned list
unassigned_nodes = sorted(set(unassigned_nodes) - set([0]))

# ────────────────────────────────────────────────────────────────────────────────
# Build route plan with ETAs (AM/PM) + window checks
# ────────────────────────────────────────────────────────────────────────────────
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
                demand=int(r["demand"]),
            ))
            tmin += svc

df_plan = pd.DataFrame(rows)

# ────────────────────────────────────────────────────────────────────────────────
# Persist + UI
# ────────────────────────────────────────────────────────────────────────────────
if df_plan.empty and unassigned_nodes:
    st.warning("All orders violated capacity/duration constraints and were left unassigned.")
    st.stop()
elif df_plan.empty:
    st.warning("No routes could be constructed. Check your data and settings.")
    st.stop()

df_plan["alert"] = df_plan.apply(
    lambda r: "" if r["within_window"] in (True, "—") else "Late risk",
    axis=1,
)
df_plan["status"] = "Planned"

# Save for other pages (and to disk)
st.session_state["routes_df"] = df_plan.copy()
st.session_state["settings"] = dict(
    speed_kph=float(speed_kph),
    max_stops_per_vehicle=int(max_stops_per_vehicle),
    num_vehicles=int(num_vehicles),
    vehicle_capacity=int(vehicle_capacity),
    max_route_minutes=int(max_route_minutes),
    shift_start=min_to_ampm(shift_start_min),
    depot_lat=depot[0],
    depot_lon=depot[1],
)
save_routes(df_plan)  # ← persist the plan

# ────────────────────────────────────────────────────────────────────────────────
# Results (AM/PM)
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
display_cols = ["Vehicle", "Stop #", "Order", "ETA", "Time window", "demand", "alert"]
if not hide_coords:
    display_cols += ["Lat", "Lon"]

st.dataframe(display_df[display_cols], use_container_width=True, hide_index=True)

# ────────────────────────────────────────────────────────────────────────────────
# Unassigned orders (if any)
# ────────────────────────────────────────────────────────────────────────────────
if unassigned_nodes:
    ua = orders.iloc[[n-1 for n in sorted(set(unassigned_nodes))]][
        ["order_id", "place", "demand", "tw_start_min", "tw_end_min"]
    ].copy()
    ua["Window"] = ua["tw_start_min"].map(min_to_ampm) + " – " + ua["tw_end_min"].map(min_to_ampm)
    ua = ua.rename(columns={"order_id": "Order"})
    st.warning(f"{len(ua)} order(s) could not be assigned due to capacity/duration/stop-cap.")
    st.dataframe(ua[["Order", "place", "demand", "Window"]], use_container_width=True, hide_index=True)
    st.session_state["unassigned_df"] = ua.copy()
else:
    st.session_state["unassigned_df"] = pd.DataFrame(columns=["Order"])

# ────────────────────────────────────────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("### Route summary")

summary = (
    display_df.sort_values(["Vehicle", "Stop #"], kind="mergesort")
              .groupby("Vehicle", sort=False)
              .agg(Stops=("Order", "count"),
                   Load=("demand", "sum"),
                   First_ETA=("ETA", "first"),
                   Last_ETA=("ETA", "last"),
                   Alerts=("alert", lambda s: int((s != "").sum())))
              .reset_index()
              .rename(columns={"First_ETA":"First ETA","Last_ETA":"Last ETA"})
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
