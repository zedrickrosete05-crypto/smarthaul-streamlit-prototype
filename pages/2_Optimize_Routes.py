# pages/2_Optimize_Routes.py
from __future__ import annotations

import math, time, datetime as dt
from zoneinfo import ZoneInfo
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ---------------- Page setup ----------------
st.set_page_config(page_title="SmartHaul – Optimize Routes", layout="wide")
st.title("Optimize Routes")
PH_TZ = ZoneInfo("Asia/Manila")

# ---------------- Guards ----------------
if "orders_df" not in st.session_state or st.session_state["orders_df"] is None:
    st.warning("No orders loaded. Go to **Upload Orders** first.")
    st.page_link("pages/1_Upload_Orders.py", label="← Open Upload Orders", icon="⬅️")
    st.stop()

orders = st.session_state["orders_df"].reset_index(drop=True)

# ---------------- Helpers ----------------
def to_ampm(minutes: int | float | None) -> str:
    """Minutes after midnight -> 'h:MM AM/PM'. Returns '—' if None/NaN."""
    if minutes is None or pd.isna(minutes):
        return "—"
    m = int(minutes) % (24 * 60)
    h24, mm = divmod(m, 60)
    suffix = "AM" if h24 < 12 else "PM"
    h12 = h24 % 12
    if h12 == 0:
        h12 = 12
    return f"{h12}:{mm:02d} {suffix}"

def hhmm_to_min(s: str | int | float | None) -> int | None:
    if s is None or (isinstance(s, float) and pd.isna(s)): return None
    s = str(s).strip()
    if ":" not in s: return None
    try:
        h, m = [int(x) for x in s.split(":")]
        return h * 60 + m
    except Exception:
        return None

def time_to_minutes(t: dt.time | None) -> int | None:
    return None if t is None else t.hour * 60 + t.minute

def haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    R = 6371.0088
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    x = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
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

orders["lat"] = pd.to_numeric(orders["lat"], errors="coerce")
orders["lon"] = pd.to_numeric(orders["lon"], errors="coerce")
orders["service_time_min"] = pd.to_numeric(orders["service_time_min"], errors="coerce").fillna(0).astype(int)

# ---------------- Sidebar: Planner Settings ----------------
with st.sidebar:
    st.header("Planner Settings")
    speed_kph = st.slider("Avg speed (km/h)", 15, 90, 30, 5)
    max_stops_per_vehicle = st.slider("Max stops per route", 3, 50, 15, 1)
    num_vehicles = st.slider("Vehicles", 1, 50, 5, 1)

    st.subheader("Shift start (Asia/Manila)")
    use_now = st.toggle("Use current local time", value=True)
    now_local = dt.datetime.now(PH_TZ).time().replace(second=0, microsecond=0)
    shift_time = now_local if use_now else st.time_input("Pick time", value=dt.time(8, 0), step=dt.timedelta(minutes=5))
    shift_start_min = time_to_minutes(shift_time) or 8 * 60

    st.subheader("Realtime")
    auto_refresh = st.toggle("Auto-refresh", value=True, help="Reruns the page at an interval so ETAs move in real time.")
    refresh_sec = st.slider("Refresh every (seconds)", 10, 120, 30, 5, disabled=not auto_refresh)
    # True auto-refresh without extra packages: change a query param every N seconds
    if auto_refresh:
        st.experimental_set_query_params(t=int(time.time() // refresh_sec))

    st.header("Depot")
    use_manual_depot = st.checkbox("Use manual depot (place)", value=False)
    depot = None
    if use_manual_depot:
        depot_place = st.text_input("Depot place", "Cebu IT Park")
        if st.button("Geocode depot"):
            try:
                import urllib.parse, urllib.request, json
                url = f"https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(depot_place+', Philippines')}&format=json&limit=1"
                req = urllib.request.Request(url, headers={"User-Agent": "SmartHaul/1.0"})
                with urllib.request.urlopen(req, timeout=8) as r:
                    data = json.loads(r.read().decode("utf-8"))
                if data:
                    depot = (float(data[0]["lat"]), float(data[0]["lon"]))
                    st.success(f"Depot set to: {depot[0]:.5f}, {depot[1]:.5f}")
                else:
                    st.error("No result for depot place.")
            except Exception as e:
                st.error(f"Depot geocode failed: {e}")

# Fallback depot = centroid of orders
if depot is None:
    depot = (float(orders["lat"].mean()), float(orders["lon"].mean()))

# ---------------- Distance matrix ----------------
points = [depot] + list(zip(orders["lat"].tolist(), orders["lon"].tolist()))
D_km = distance_matrix(points)  # depot at index 0

# ---------------- Sweep clustering + 2-opt ----------------
def polar_angle(p: Tuple[float, float], origin: Tuple[float, float]) -> float:
    y = p[0] - origin[0]
    x = (p[1] - origin[1]) * math.cos(math.radians(origin[0]))
    return math.degrees(math.atan2(y, x)) % 360.0

def sweep_assign(max_routes: int, cap_stops: int) -> List[List[int]]:
    n = len(points) - 1
    if n <= 0:
        return [[] for _ in range(max_routes)]
    nodes = list(range(1, n + 1))
    angs = [(i, polar_angle(points[i], depot)) for i in nodes]
    angs.sort(key=lambda t: t[1])
    clusters: List[List[int]] = [[] for _ in range(max_routes)]
    ridx = 0
    for i, _a in angs:
        start = ridx
        while len(clusters[ridx]) >= cap_stops:
            ridx = (ridx + 1) % max_routes
            if ridx == start:
                break
        clusters[ridx].append(i)
        if len(clusters[ridx]) >= cap_stops:
            ridx = (ridx + 1) % max_routes
    return clusters

def nearest_neighbor_route(cluster: List[int], D: np.ndarray) -> List[int]:
    if not cluster: return [0, 0]
    un = set(cluster)
    route, cur = [0], 0
    while un:
        nxt = min(un, key=lambda j: D[cur, j])
        un.remove(nxt); route.append(nxt); cur = nxt
    route.append(0)
    return route

def route_len(route: List[int], D: np.ndarray) -> float:
    return float(sum(D[route[i], route[i + 1]] for i in range(len(route) - 1)))

def two_opt(route: List[int], D: np.ndarray) -> List[int]:
    best = route[:]; best_len = route_len(best, D); improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for k in range(i + 1, len(best) - 1):
                if k - i == 1: continue
                new = best[:i] + best[i:k][::-1] + best[k:]
                new_len = route_len(new, D)
                if new_len + 1e-9 < best_len:
                    best, best_len, improved = new, new_len, True
    return best

# Build routes using current planner settings
N = len(orders)
V = max(1, min(int(num_vehicles), max(1, N)))
cap = int(max_stops_per_vehicle)

clusters = sweep_assign(V, cap)
routes_idx: List[List[int]] = []
for cl in clusters:
    r = nearest_neighbor_route(cl, D_km)
    if len(r) > 4:
        r = two_opt(r, D_km)
    routes_idx.append(r)

# Any leftovers (should be none if sweep respected cap, but keep it safe)
assigned = set(j for cl in clusters for j in cl)
if len(assigned) < N:
    leftovers = [j for j in range(1, N + 1) if j not in assigned]
    for j in leftovers:
        best_r, best_pos, best_cost = None, None, float("inf")
        for ri, r in enumerate(routes_idx):
            for pos in range(1, len(r)):
                cand = r[:pos] + [j] + r[pos:]
                cost = route_len(cand, D_km)
                if cost < best_cost:
                    best_r, best_pos, best_cost = ri, pos, cost
        routes_idx[best_r].insert(best_pos, j)
        if len(routes_idx[best_r]) > 4:
            routes_idx[best_r] = two_opt(routes_idx[best_r], D_km)

# ---------------- Build plan (AM/PM + realtime) ----------------
v_speed_km_min = max(float(speed_kph), 5.0) / 60.0  # km/min
time_now_min = shift_start_min

rows = []
for v, route in enumerate(routes_idx, start=1):
    tmin = time_now_min
    for i in range(len(route) - 1):
        a, b = route[i], route[i + 1]
        leg_km = float(D_km[a, b])
        drive_min = int(round(leg_km / v_speed_km_min)) if v_speed_km_min > 0 else 0
        tmin += drive_min

        if b != 0:
            ord_row = orders.iloc[b - 1]
            tw_s = int(ord_row["tw_start_min"]) if pd.notna(ord_row["tw_start_min"]) else None
            tw_e = int(ord_row["tw_end_min"]) if pd.notna(ord_row["tw_end_min"]) else None
            svc = int(ord_row["service_time_min"]) if pd.notna(ord_row["service_time_min"]) else 0

            if tw_s is not None and tmin < tw_s:
                tmin = tw_s

            rows.append(
                dict(
                    vehicle_id=f"V{v}",
                    order_id=str(ord_row["order_id"]),
                    lat=float(ord_row["lat"]),
                    lon=float(ord_row["lon"]),
                    eta=to_ampm(tmin),
                    tw_start=to_ampm(tw_s),
                    tw_end=to_ampm(tw_e),
                    within_window=(True if (tw_e is None or tmin <= tw_e) else False),
                    leg_km=round(leg_km, 2),
                )
            )

            tmin += svc

df_plan = pd.DataFrame(rows)
if df_plan.empty:
    st.warning("No routes could be constructed. Check your data (coordinates/windows).")
    st.stop()

df_plan["alert"] = df_plan.apply(lambda r: "" if (r["within_window"] in (True, "—")) else "Late risk", axis=1)
df_plan["status"] = "Planned"

# Persist
st.session_state["routes_df"] = df_plan.copy()
st.session_state["settings"] = dict(
    speed_kph=float(speed_kph),
    max_stops_per_vehicle=int(max_stops_per_vehicle),
    num_vehicles=int(num_vehicles),
    depot_lat=depot[0],
    depot_lon=depot[1],
    shift_start_min=time_now_min,
)

# ---------------- Results ----------------
st.success(f"Planned {df_plan['vehicle_id'].nunique()} route(s) for {len(df_plan)} stops.")
st.caption(f"Shift start: **{to_ampm(time_now_min)}** (Asia/Manila)")
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
st.metric("Total distance (km)", round(summary["Distance (km)"].sum(), 2))

# ---------------- Optional map ----------------
st.divider()
st.markdown("### Map")
if st.checkbox("Show map (pydeck)", value=False):
    try:
        import pydeck as pdk
        df_map = display_df.copy()
        df_map["stop_idx"] = df_map.groupby("Vehicle").cumcount() + 1
        df_map = df_map.sort_values(["Vehicle", "stop_idx"], kind="mergesort").reset_index(drop=True)
        palette = np.array([[0,122,255],[255,45,85],[88,86,214],[255,149,0],[52,199,89],[175,82,222],[255,59,48],[90,200,250]])
        veh_ids = df_map["Vehicle"].unique()
        cmap = {v: palette[i % len(palette)].tolist() for i, v in enumerate(veh_ids)}
        df_map["color"] = df_map["Vehicle"].map(cmap)
        rows_path = []
        for v, g in df_map.groupby("Vehicle", sort=False):
            g = g.sort_values("stop_idx")
            pts = [{"lon": float(r.Lon), "lat": float(r.Lat)} for r in g.itertuples(index=False)]
            rows_path.append({"Vehicle": v, "path": pts, "color": cmap[v]})
        paths = pd.DataFrame(rows_path)
        depot_df = pd.DataFrame([{"Vehicle": "Depot", "Lat": depot[0], "Lon": depot[1], "stop_idx": 0, "ETA": "—"}])
        layers = [
            pdk.Layer("PathLayer", data=paths, get_path="path", get_width=4, get_color="color", width_min_pixels=2, pickable=False),
            pdk.Layer("ScatterplotLayer", data=df_map, get_position='[Lon, Lat]', get_radius=60, get_fill_color="color", get_line_color=[255,255,255], line_width_min_pixels=1, pickable=True),
            pdk.Layer("TextLayer", data=df_map, get_position='[Lon, Lat]', get_text="stop_idx", get_size=12, get_color=[230,230,230], get_alignment_baseline="'center'"),
            pdk.Layer("ScatterplotLayer", data=depot_df, get_position='[Lon, Lat]', get_radius=80, get_fill_color=[255,255,255], get_line_color=[0,0,0], line_width_min_pixels=1, pickable=False),
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
