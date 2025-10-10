from branding import setup_branding, section
setup_branding("SmartHaul – Optimize Routes")

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import math
from datetime import datetime, timedelta, time

import pandas as pd
import streamlit as st

section("Optimize Routes")

# ---------- helpers ----------
def haversine_km(a: float, b: float, c: float, d: float) -> float:
    R = 6371.0088
    p1, p2 = math.radians(a), math.radians(c)
    dphi = math.radians(c - a)
    dlmb = math.radians(d - b)
    x = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(x))


def travel_min(a, b, c, d, speed: float = 25.0) -> float:
    return (haversine_km(a, b, c, d) / max(speed, 5)) * 60.0

def t(s: str) -> datetime:
    return datetime.strptime(str(s), "%H:%M")

def parse_time(value) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    if ":" in text:
        parts = text.split(":")
        if len(parts) >= 2:
            text = f"{parts[0]:0>2}:{parts[1]:0>2}"
    elif text.isdigit() and len(text) == 4:
        text = f"{text[:2]}:{text[2:]}"
    else:
        return None
    try:
        return datetime.strptime(text, "%H:%M")
    except ValueError:
        return None


def format_time(dt: datetime | None) -> str:
    if isinstance(dt, datetime):
        return dt.strftime("%H:%M")
    return "N/A"


# ---------- greedy planner ----------
def greedy(df: pd.DataFrame, depot=(14.5995, 120.9842), speed: float = 25.0, max_stops: int = 10) -> pd.DataFrame:
def greedy(
    df: pd.DataFrame,
    depot=(14.5995, 120.9842),
    speed: float = 25.0,
    max_stops: int = 10,
    start_time: str = "08:00",
) -> pd.DataFrame:
    orders = df.copy().reset_index(drop=True)
    orders["lat"] = pd.to_numeric(orders["lat"], errors="coerce")
    orders["lon"] = pd.to_numeric(orders["lon"], errors="coerce")
    orders["service_min"] = pd.to_numeric(orders.get("service_min", 0), errors="coerce").fillna(0)
    orders["tw_start"] = orders["tw_start"].astype(str)
    orders["tw_end"]   = orders["tw_end"].astype(str)
    orders["tw_end"] = orders["tw_end"].astype(str)
    orders["done"] = False

    routes, vid = [], 1
    day_start = parse_time(start_time) or datetime.strptime("08:00", "%H:%M")
    while not orders.done.all():
        lat, lon = depot
        now = t("08:00")
        now = day_start
        route = []

        while True:
            cand = []
            cand: list[tuple[int, float, datetime, float]] = []
            for i, row in orders[~orders.done].iterrows():
                if pd.isna(row.lat) or pd.isna(row.lon):
                    continue
                drive = travel_min(lat, lon, row.lat, row.lon, speed)
                distance_km = haversine_km(lat, lon, row.lat, row.lon)
                arr = now + timedelta(minutes=drive)
                start = max(arr, t(row.tw_start))
                if start <= t(row.tw_end):
                    cand.append((i, drive, start))
                tw_start = parse_time(row.tw_start)
                tw_end = parse_time(row.tw_end)
                start = max(arr, tw_start) if tw_start else arr
                if tw_end is None or start <= tw_end:
                    cand.append((i, drive, start, distance_km))

            if not cand or len(route) >= max_stops:
                break

            i, drive, start = sorted(cand, key=lambda x: x[1])[0]
            i, drive, start, distance_km = sorted(cand, key=lambda x: x[1])[0]
            r = orders.loc[i]
            svc_min = float(r.service_min) if pd.notna(r.service_min) else 0.0
            leave = start + timedelta(minutes=svc_min)

            route.append(dict(
                vehicle_id=f"V{vid}",
                order_id=r.order_id,
                lat=r.lat, lon=r.lon,
                eta=start.strftime("%H:%M"),
                tw_start=r.tw_start, tw_end=r.tw_end
            ))
            route.append(
                dict(
                    vehicle_id=f"V{vid}",
                    order_id=r.order_id,
                    lat=r.lat,
                    lon=r.lon,
                    eta=format_time(start),
                    depart=format_time(leave),
                    tw_start=r.tw_start,
                    tw_end=r.tw_end,
                    drive_min=float(drive),
                    drive_km=float(distance_km),
                    service_min=svc_min,
                    arrival_dt=start,
                    depart_dt=leave,
                )
            )
            orders.at[i, "done"] = True
            lat, lon, now = r.lat, r.lon, leave

        if route:
            routes.append(pd.DataFrame(route)); vid += 1
            routes.append(pd.DataFrame(route))
            vid += 1
        else:
            i = orders[~orders.done].index[0]; r = orders.loc[i]
            routes.append(pd.DataFrame([dict(
                vehicle_id=f"V{vid}", order_id=r.order_id, lat=r.lat, lon=r.lon,
                eta="N/A", tw_start=r.tw_start, tw_end=r.tw_end
            )]))
            orders.at[i, "done"] = True; vid += 1
            i = orders[~orders.done].index[0]
            r = orders.loc[i]
            routes.append(
                pd.DataFrame(
                    [
                        dict(
                            vehicle_id=f"V{vid}",
                            order_id=r.order_id,
                            lat=r.lat,
                            lon=r.lon,
                            eta="N/A",
                            depart="N/A",
                            tw_start=r.tw_start,
                            tw_end=r.tw_end,
                            drive_min=float("nan"),
                            drive_km=float("nan"),
                            service_min=float(r.service_min)
                            if pd.notna(r.service_min)
                            else 0.0,
                            arrival_dt=pd.NaT,
                            depart_dt=pd.NaT,
                        )
                    ]
                )
            )
            orders.at[i, "done"] = True
            vid += 1

    return pd.concat(routes, ignore_index=True)


# ---------- require data ----------
if "orders_df" not in st.session_state:
    st.warning("Upload orders first on the Upload page."); st.stop()
    st.warning("Upload orders first on the Upload page.")
    st.stop()

orders_df = st.session_state["orders_df"]

mean_lat = pd.to_numeric(orders_df.get("lat"), errors="coerce").mean()
mean_lon = pd.to_numeric(orders_df.get("lon"), errors="coerce").mean()
default_lat = st.session_state.get("depot_lat", float(mean_lat) if not math.isnan(mean_lat) else 14.5995)
default_lon = st.session_state.get("depot_lon", float(mean_lon) if not math.isnan(mean_lon) else 120.9842)

section("Planner settings")
col1, col2, col3 = st.columns(3)
with col1:
    speed = st.slider("Average speed (km/h)", 15, 45, 25, 1)
with col2:
    maxst = st.slider("Max stops per route", 5, 20, 10, 1)
with col3:
    default_start = st.session_state.get("start_time") or time(8, 0)
    start_time_input = st.time_input("Driver start time", value=default_start)
    st.session_state["start_time"] = start_time_input

dc1, dc2 = st.columns(2)
with dc1:
    depot_lat = st.number_input("Depot latitude", value=float(default_lat), format="%.4f")
with dc2:
    depot_lon = st.number_input("Depot longitude", value=float(default_lon), format="%.4f")

speed = st.slider("Average speed (km/h)", 15, 45, 25, 1)
maxst = st.slider("Max stops per route", 5, 20, 10, 1)
st.session_state["depot_lat"] = depot_lat
st.session_state["depot_lon"] = depot_lon

# ---------- compute ----------
if st.button("Compute routes"):
    df_plan = greedy(st.session_state["orders_df"], speed=speed, max_stops=maxst)
    df_plan = greedy(
        st.session_state["orders_df"],
        speed=speed,
        max_stops=maxst,
        depot=(depot_lat, depot_lon),
        start_time=start_time_input.strftime("%H:%M"),
    )

    df_plan["alert"] = df_plan.apply(
        lambda r: "Late risk" if r["eta"] != "N/A" and r["eta"] > r["tw_end"] else "",
        lambda r: (
            "Late risk"
            if isinstance(r.get("arrival_dt"), datetime)
            and (end := parse_time(r.get("tw_end"))) is not None
            and r["arrival_dt"] > end
            else ""
        ),
        axis=1,
    )
    df_plan["status"] = "Planned"
    st.session_state["routes_df"] = df_plan
    st.session_state["dispatch_df"] = df_plan.copy()
    st.success(f"Computed {df_plan['vehicle_id'].nunique()} route(s) for {len(df_plan)} stops.")
    st.success(
        f"Computed {df_plan['vehicle_id'].nunique()} route(s) for {len(df_plan)} stops."
    )

# ---------- results ----------
if "routes_df" in st.session_state:
    df = st.session_state["routes_df"].copy()

    df["drive_min"] = pd.to_numeric(df.get("drive_min"), errors="coerce")
    df["drive_km"] = pd.to_numeric(df.get("drive_km"), errors="coerce")
    df["service_min"] = pd.to_numeric(df.get("service_min"), errors="coerce")
    df["arrival_dt"] = pd.to_datetime(df.get("arrival_dt"), errors="coerce")
    df["depart_dt"] = pd.to_datetime(df.get("depart_dt"), errors="coerce")

    total_distance = float(df["drive_km"].sum(skipna=True))
    total_drive = float(df["drive_min"].sum(skipna=True))
    total_service = float(df["service_min"].sum(skipna=True))

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Total distance", f"{total_distance:.1f} km")
    mc2.metric("Drive time", f"{total_drive:.0f} min")
    mc3.metric("Service time", f"{total_service:.0f} min")

    section("Planned routes")
    df["Stop #"] = df.groupby("vehicle_id").cumcount() + 1
    df["Time window"] = df["tw_start"].astype(str) + " – " + df["tw_end"].astype(str)
    df["Lat"] = pd.to_numeric(df["lat"], errors="coerce").round(4)
    df["Lon"] = pd.to_numeric(df["lon"], errors="coerce").round(4)
    df["Depart"] = df["depart_dt"].dt.strftime("%H:%M").fillna("—")
    df["Drive (min)"] = df["drive_min"].round(1)
    df["Distance (km)"] = df["drive_km"].round(1)
    df["Service (min)"] = df["service_min"].round(1)

    display_df = df.rename(columns={
        "vehicle_id": "Vehicle", "order_id": "Order", "eta": "ETA", "alert": "Alert"
    })
    display_df = df.rename(
        columns={
            "vehicle_id": "Vehicle",
            "order_id": "Order",
            "eta": "ETA",
            "alert": "Alert",
        }
    )

    hide_coords = st.checkbox("Hide coordinates", value=True)
    show_travel = st.checkbox("Show travel metrics", value=False)

    cols = ["Vehicle", "Stop #", "Order", "ETA", "Time window", "Alert"]
    cols = ["Vehicle", "Stop #", "Order", "ETA", "Depart", "Time window", "Alert"]
    if not hide_coords:
        cols += ["Lat", "Lon"]
    if show_travel:
        cols += ["Drive (min)", "Distance (km)", "Service (min)"]

    st.dataframe(display_df[[c for c in cols if c in display_df.columns]],
                 use_container_width=True, hide_index=True)
    st.dataframe(
        display_df[[c for c in cols if c in display_df.columns]],
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    section("Route summary")
    def first_valid_eta(s):
        for v in s:
            if v != "N/A": return v

    def first_valid_eta(series: pd.Series) -> str:
        for value in series:
            if value != "N/A":
                return value
        return "—"
    def last_valid_eta(s):
        lst = list(s)
        for v in reversed(lst):
            if v != "N/A": return v

    def last_valid_eta(series: pd.Series) -> str:
        lst = list(series)
        for value in reversed(lst):
            if value != "N/A":
                return value
        return "—"

    def route_duration_minutes(group: pd.DataFrame) -> float:
        starts = group["arrival_dt"].dropna()
        ends = group["depart_dt"].dropna()
        if starts.empty or ends.empty:
            return float("nan")
        return (ends.max() - starts.min()).total_seconds() / 60.0

    summary = (
        df.sort_values(["vehicle_id","eta"], kind="mergesort")
          .groupby("vehicle_id", sort=False)
          .agg(Stops=("order_id","count"),
               **{"First ETA":("eta", first_valid_eta)},
               **{"Last ETA":("eta", last_valid_eta)},
               Alerts=("alert", lambda s: int((s!="").sum())))
          .reset_index().rename(columns={"vehicle_id":"Vehicle"})
        df.sort_values(["vehicle_id", "eta"], kind="mergesort")
        .groupby("vehicle_id", sort=False)
        .agg(
            Stops=("order_id", "count"),
            **{"First ETA": ("eta", first_valid_eta)},
            **{"Last ETA": ("eta", last_valid_eta)},
            Alerts=("alert", lambda s: int((s != "").sum())),
            Distance_km=("drive_km", "sum"),
            Drive_min=("drive_min", "sum"),
            Service_min=("service_min", "sum"),
        )
        .reset_index()
        .rename(columns={"vehicle_id": "Vehicle"})
    )

    durations = []
    for vid, group in df.groupby("vehicle_id", sort=False):
        durations.append(route_duration_minutes(group))
    summary["Distance (km)"] = summary.pop("Distance_km").fillna(0.0).round(1)
    summary["Drive (min)"] = summary.pop("Drive_min").fillna(0.0).round(1)
    summary["Service (min)"] = summary.pop("Service_min").fillna(0.0).round(1)
    summary["Route duration (min)"] = pd.Series(durations).fillna(0.0).round(1)

    st.dataframe(summary, hide_index=True, use_container_width=True)

    st.divider()
    section("Map")
    if st.checkbox("Show map", value=True):
        try:
            import pydeck as pdk, numpy as np
            import numpy as np
            import pydeck as pdk

            df_map = df.copy()
            df_map["stop_idx"] = df_map.groupby("vehicle_id").cumcount() + 1
            df_map = df_map.sort_values(["vehicle_id","stop_idx"], kind="mergesort").reset_index(drop=True)
            df_map = (
                df_map.sort_values(["vehicle_id", "stop_idx"], kind="mergesort")
                .reset_index(drop=True)
            )
            df_map["drive_label"] = df_map["Drive (min)"].fillna(0).round(1).astype(str)

            palette = np.array([[0,122,255],[255,45,85],[88,86,214],[255,149,0],
                                [52,199,89],[175,82,222],[255,59,48],[90,200,250]])
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
            veh_ids = df_map["vehicle_id"].unique()
            cmap = {v: palette[i % len(palette)].tolist() for i, v in enumerate(veh_ids)}
            df_map["color"] = df_map["vehicle_id"].map(cmap)

            rows = []
            for v, g in df_map.groupby("vehicle_id", sort=False):
                g = g.sort_values("stop_idx")
                pts = [{"lon": float(r.lon), "lat": float(r.lat)} for r in g.itertuples(index=False)]
                pts = [
                    {"lon": float(r.lon), "lat": float(r.lat)}
                    for r in g.itertuples(index=False)
                ]
                rows.append({"vehicle_id": v, "path": pts, "color": cmap[v]})
            paths = pd.DataFrame(rows)

            layers = [
                pdk.Layer("PathLayer", data=paths, get_path="path", get_width=4,
                          get_color="color", width_min_pixels=2, pickable=False),
                pdk.Layer("ScatterplotLayer", data=df_map, get_position='[lon, lat]',
                          get_radius=60, get_fill_color="color", get_line_color=[255,255,255],
                          line_width_min_pixels=1, pickable=True),
                pdk.Layer("TextLayer", data=df_map, get_position='[lon, lat]',
                          get_text="stop_idx", get_size=12, get_color=[230,230,230],
                          get_alignment_baseline="'center'")
                pdk.Layer(
                    "PathLayer",
                    data=paths,
                    get_path="path",
                    get_width=4,
                    get_color="color",
                    width_min_pixels=2,
                    pickable=False,
                ),
                pdk.Layer(
                    "ScatterplotLayer",
                    data=df_map,
                    get_position="[lon, lat]",
                    get_radius=60,
                    get_fill_color="color",
                    get_line_color=[255, 255, 255],
                    line_width_min_pixels=1,
                    pickable=True,
                ),
                pdk.Layer(
                    "TextLayer",
                    data=df_map,
                    get_position="[lon, lat]",
                    get_text="stop_idx",
                    get_size=12,
                    get_color=[230, 230, 230],
                    get_alignment_baseline="'center'",
                ),
            ]
            view = pdk.ViewState(latitude=float(df_map.lat.mean()),
                                 longitude=float(df_map.lon.mean()), zoom=11)
            tooltip = {"text": "Stop {stop_idx}\nVehicle {vehicle_id}\nETA {eta}"}
            st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, tooltip=tooltip))
        except Exception as e:
            st.info(f"Map rendering skipped: {e}")

    st.download_button("Download planned routes (CSV)",
                       data=df.to_csv(index=False).encode("utf-8"),
                       file_name="routes_plan.csv", mime="text/csv")
            view = pdk.ViewState(
                latitude=float(df_map.lat.mean()),
                longitude=float(df_map.lon.mean()),
                zoom=11,
            )
            tooltip = {
                "text": "Stop {stop_idx}\nVehicle {vehicle_id}\nETA {eta}\nDrive {drive_label} min"
            }
            st.pydeck_chart(
                pdk.Deck(layers=layers, initial_view_state=view, tooltip=tooltip)
            )
        except Exception as exc:
            st.info(f"Map rendering skipped: {exc}")

    st.download_button(
        "Download planned routes (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="routes_plan.csv",
        mime="text/csv",
    )
