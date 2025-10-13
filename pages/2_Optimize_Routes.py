# pages/2_Optimize_Routes.py
from __future__ import annotations

import math, time, datetime as dt
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SmartHaul – Optimize Routes", layout="wide")
BUILD = "optimize-am-pm-v2"
st.title("Optimize Routes")
st.caption(f"Build: {BUILD}")

# ---------- Guards ----------
if "orders_df" not in st.session_state or st.session_state["orders_df"] is None:
    st.warning("No orders loaded. Go to **Upload Orders** first.")
    st.page_link("pages/1_Upload_Orders.py", label="← Open Upload Orders", icon="⬅️")
    st.stop()

orders = st.session_state["orders_df"].reset_index(drop=True)

# ---------- Helpers ----------
def min_to_ampm(m: int | float | None) -> str:
    if m is None or pd.isna(m): return "—"
    m = int(m) % (24 * 60)
    h, mm = divmod(m, 60)
    ampm = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    return f"{h12}:{mm:02d} {ampm}"

def hhmm_to_min(s: str) -> int | None:
    try:
        h, m = [int(x) for x in str(s).split(":")]
        return h * 60 + m
    except Exception:
        return None

def haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    R = 6371.0088
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    x = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(min(1, math.sqrt(x)))

@st.cache_data(show_spinner=False)
def distance_matrix(points: List[Tuple[float,float]]) -> np.ndarray:
    n = len(points)
    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            km = haversine_km(points[i], points[j])
            M[i, j] = M[j, i] = km
    return M

# ---------- Sidebar parameters ----------
with st.sidebar:
    st.header("Planner Settings")
    speed_kph = st.slider("Avg speed (km/h)", 15, 90, 30, 5)
    max_stops_per_vehicle = st.slider("Max stops per route", 3, 50, 15, 1)
    num_vehicles = st.slider("Vehicles", 1, 50, 5, 1)

    st.subheader("Shift start (local)")
    use_now = st.toggle("Use current time", value=True)
    shift_time = dt.datetime.now().time().replace(second=0, microsecond=0) if use_now \
                 else st.time_input("Pick time", dt.time(8,0), step=dt.timedelta(minutes=5))

# ---------- Depot ----------
depot = (float(orders["lat"].mean()), float(orders["lon"].mean()))

# ---------- Build matrix ----------
points = [depot] + list(zip(orders["lat"].tolist(), orders["lon"].tolist()))
D_km = distance_matrix(points)  # includes depot at index 0

# ---------- Greedy planner + 2-opt ----------
def nearest_neighbor(cluster: list[int]) -> list[int]:
    if not cluster: return [0,0]
    un = set(cluster); route = [0]; cur = 0
    while un:
        nxt = min(un, key=lambda j: D_km[cur, j])
        un.remove(nxt); route.append(nxt); cur = nxt
    route.append(0); return route

def route_len(r: list[int]) -> float:
    return float(sum(D_km[r[i], r[i+1]] for i in range(len(r)-1)))

def two_opt(route: list[int]) -> list[int]:
    best = route[:]; improved=True
    while improved:
        improved=False
        for i in range(1,len(best)-2):
            for k in range(i+1,len(best)-1):
                if k-i==1: continue
                new = best[:i]+best[i:k][::-1]+best[k:]
                if route_len(new)+1e-9 < route_len(best):
                    best=new; improved=True
    return best

# simple sweep by angle to split among vehicles
def sweep_assign(v: int, cap: int) -> list[list[int]]:
    if len(points)<=1: return [[] for _ in range(v)]
    def ang(p, o):
        y = p[0]-o[0]; x = (p[1]-o[1])*math.cos(math.radians(o[0]))
        return (math.degrees(math.atan2(y,x))%360)
    nodes = list(range(1,len(points)))
    nodes.sort(key=lambda i: ang(points[i], depot))
    clusters=[[] for _ in range(v)]
    idx=0
    for n in nodes:
        while len(clusters[idx])>=cap: idx=(idx+1)%v
        clusters[idx].append(n)
        if len(clusters[idx])>=cap: idx=(idx+1)%v
    return clusters

clusters = sweep_assign(int(num_vehicles), int(max_stops_per_vehicle))
routes_idx = [two_opt(nearest_neighbor(cl)) for cl in clusters]

# ---------- Build route plan with ETAs (AM/PM) ----------
v_speed_km_min = max(float(speed_kph), 5.0) / 60.0
shift_start_min = shift_time.hour*60 + shift_time.minute

rows = []
for v, route in enumerate(routes_idx, start=1):
    tmin = shift_start_min
    for i in range(len(route)-1):
        a,b = route[i], route[i+1]
        leg_km = float(D_km[a,b])
        drive_min = int(round(leg_km / v_speed_km_min)) if v_speed_km_min>0 else 0
        tmin += drive_min

        if b != 0:
            row = orders.iloc[b-1]
            tw_s = int(row["tw_start_min"]) if pd.notna(row["tw_start_min"]) else None
            tw_e = int(row["tw_end_min"]) if pd.notna(row["tw_end_min"]) else None
            svc  = int(row["service_time_min"]) if pd.notna(row["service_time_min"]) else 0

            if tw_s is not None and tmin < tw_s:  # wait until window opens
                tmin = tw_s

            rows.append(dict(
                vehicle_id=f"V{v}",
                order_id=str(row["order_id"]),
                lat=float(row["lat"]),
                lon=float(row["lon"]),
                eta=min_to_ampm(tmin),
                tw_start=min_to_ampm(tw_s),
                tw_end=min_to_ampm(tw_e),
                within_window=(True if (tw_e is None or tmin <= tw_e) else False),
                leg_km=round(leg_km,2),
            ))
            tmin += svc

df_plan = pd.DataFrame(rows)
if df_plan.empty:
    st.warning("No routes could be constructed. Check your data.")
    st.stop()

df_plan["alert"] = df_plan.apply(lambda r: "" if r["within_window"] in (True,"—") else "Late risk", axis=1)
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

# ---------- Results (AM/PM displayed) ----------
st.success(f"Planned {df_plan['vehicle_id'].nunique()} route(s) for {len(df_plan)} stops.")
st.markdown("### Planned routes")

df_show = df_plan.copy()
df_show["Stop #"] = df_show.groupby("vehicle_id").cumcount() + 1
df_show["Time window"] = df_show["tw_start"].astype(str) + " – " + df_show["tw_end"].astype(str)
df_show["Lat"] = pd.to_numeric(df_show["lat"], errors="coerce").round(4)
df_show["Lon"] = pd.to_numeric(df_show["lon"], errors="coerce").round(4)

hide_coords = st.checkbox("Hide coordinates", value=True)
display_df = df_show.rename(columns={"vehicle_id":"Vehicle","order_id":"Order","eta":"ETA"})
cols = ["Vehicle","Stop #","Order","ETA","Time window","alert"]
if not hide_coords:
    cols += ["Lat","Lon"]
st.dataframe(display_df[cols], use_container_width=True, hide_index=True)

# ---------- Summary ----------
st.divider(); st.markdown("### Route summary")
summary = (
    display_df.sort_values(["Vehicle","Stop #"], kind="mergesort")
              .groupby("Vehicle", sort=False)
              .agg(Stops=("Order","count"),
                   **{"First ETA":("ETA","first")},
                   **{"Last ETA":("ETA","last")},
                   Alerts=("alert", lambda s: int((s!="").sum())))
              .reset_index()
)
st.dataframe(summary, hide_index=True, use_container_width=True)

# ---------- Download ----------
st.download_button(
    "Download planned routes (CSV)",
    data=df_plan.to_csv(index=False).encode("utf-8"),
    file_name="routes_plan.csv",
    mime="text/csv",
)
