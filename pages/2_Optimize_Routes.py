import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, time as dtime
import math

st.set_page_config(page_title="SmartHaul – Optimize Routes", page_icon="🧭")
st.title("🧭 Optimize Routes")

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
    s = str(s)
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return datetime.strptime("08:00", "%H:%M")

# ---------- greedy planner ----------
def greedy(df: pd.DataFrame,
           depot=(14.5995, 120.9842),
           speed: float = 25.0,
           max_stops: int = 10,
           start_time_str: str = "08:00") -> pd.DataFrame:

    orders = df.copy().reset_index(drop=True)
    orders["lat"] = pd.to_numeric(orders["lat"], errors="coerce")
    orders["lon"] = pd.to_numeric(orders["lon"], errors="coerce")
    orders["service_min"] = pd.to_numeric(orders.get("service_min", 0), errors="coerce").fillna(0)
    orders["tw_start"] = orders["tw_start"].astype(str)
    orders["tw_end"] = orders["tw_end"].astype(str)
    orders["done"] = False

    routes, vid = [], 1
    while not orders.done.all():
        lat, lon = depot
        now = t(start_time_str)
        route = []

        while True:
            cand = []
            for i, row in orders[~orders.done].iterrows():
                if pd.isna(row.lat) or pd.isna(row.lon):
                    continue
                drive = travel_min(lat, lon, row.lat, row.lon, speed)
                arr = now + timedelta(minutes=drive)
                start = max(arr, t(row.tw_start))
                if start <= t(row.tw_end):
                    cand.append((i, drive, start))

            if not cand or len(route) >= max_stops:
                break

            i, drive, start = sorted(cand, key=lambda x: x[1])[0]
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
            orders.at[i, "done"] = True
            lat, lon, now = r.lat, r.lon, leave

        if route:
            routes.append(pd.DataFrame(route)); vid += 1
        else:
            i = orders[~orders.done].index[0]; r = orders.loc[i]
            routes.append(pd.DataFrame([dict(
                vehicle_id=f"V{vid}", order_id=r.order_id, lat=r.lat, lon=r.lon,
                eta="N/A", tw_start=r.tw_start, tw_end=r.tw_end
            )]))
            orders.at[i, "done"] = True; vid += 1

    return pd.concat(routes, ignore_index=True)

# ---------- require data ----------
if "orders_df" not in st.session_state:
    st.warning("Upload orders first on the Upload page.")
    st.stop()

orders_df = st.session_state["orders_df"]

# Controls
speed = st.slider("Average speed (km/h)", 15, 60, 30, 1)
maxst = st.slider("Max stops per route", 5, 30, 10, 1)
start_time = st.time_input("Start time", value=dtime(8, 0), help="Driver leaves depot at this time")

depot_mode = st.radio(
    "Depot location",
    ["Use centroid of uploaded orders (recommended)", "Enter manually"],
    index=0,
    help="Make sure depot is near your orders so ETAs are feasible."
)

if depot_mode == "Use centroid of uploaded orders (recommended)":
    depot_lat = float(pd.to_numeric(orders_df["lat"], errors="coerce").mean())
    depot_lon = float(pd.to_numeric(orders_df["lon"], errors="coerce").mean())
else:
    c1, c2 = st.columns(2)
    depot_lat = c1.number_input("Depot latitude", value=float(pd.to_numeric(orders_df["lat"], errors="coerce").mean()))
    depot_lon = c2.number_input("Depot longitude", value=float(pd.to_numeric(orders_df["lon"], errors="coerce").mean()))

depot_tuple = (depot_lat, depot_lon)
st.caption(f"Depot: {depot_lat:.5f}, {depot_lon:.5f}")

# ---------- compute ----------
if st.button("Compute routes"):
    df_plan = greedy(
        orders_df[["order_id","lat","lon","tw_start","tw_end","service_min"]],
        depot=depot_tuple,
        speed=speed,
        max_stops=maxst,
        start_time_str=start_time.strftime("%H:%M")
    )
    df_plan["alert"] = df_plan.apply(
        lambda r: "Late risk" if r["eta"] != "N/A" and r["eta"] > r["tw_end"] else "",
        axis=1,
    )
    df_plan["status"] = "Planned"
    st.session_state["routes_df"] = df_plan
    st.session_state["dispatch_df"] = df_plan.copy()
    st.success(f"Computed {df_plan['vehicle_id'].nunique()} route(s) for {len(df_plan)} stops.")

# ---------- results ----------
if "routes_df" in st.session_state:
    df = st.session_state["routes_df"].copy()

    # Prefer uploaded place names (no geocoding needed)
    if "place" in orders_df.columns:
        df = df.merge(
            orders_df[["order_id","lat","lon","place"]],
            on=["order_id","lat","lon"],
            how="left"
        )

    # Optional reverse-geocode (OFF by default)
    use_places = st.checkbox(
        "Fill missing place names via reverse-geocoding (OpenStreetMap)",
        value=False
    )
    place_available = False
    if use_places:
        try:
            from geopy.geocoders import Nominatim
            from geopy.extra.rate_limiter import RateLimiter
            geolocator = Nominatim(user_agent="smarthaul-demo")
            reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1)

            @st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
            def reverse_geocode(lat: float, lon: float) -> str:
                try:
                    loc = reverse((lat, lon), language="en", zoom=15)
                    if loc:
                        parts = [p.strip() for p in str(loc.address).split(",")]
                        return ", ".join(parts[:3])
                except Exception:
                    pass
                return ""
            with st.spinner("Resolving names…"):
                missing_mask = df["place"].isna() | (df["place"] == "")
                df.loc[missing_mask, "place"] = df[missing_mask].apply(
                    lambda r: reverse_geocode(float(r["lat"]), float(r["lon"])), axis=1
                )
            place_available = True
        except Exception:
            place_available = False

    # --------- Friendly table ----------
    st.subheader("Planned routes")
    df["Stop #"] = df.groupby("vehicle_id").cumcount() + 1
    df["Time window"] = df["tw_start"].astype(str) + " – " + df["tw_end"].astype(str)
    df["Lat"] = pd.to_numeric(df["lat"], errors="coerce").round(4)
    df["Lon"] = pd.to_numeric(df["lon"], errors="coerce").round(4)

    display_df = df.rename(columns={
        "vehicle_id":"Vehicle", "order_id":"Order", "eta":"ETA", "alert":"Alert", "place":"Place"
    })

    hide_coords = st.checkbox("Hide coordinates", value=True)

    wanted_cols = ["Vehicle", "Stop #", "Order"]
    if "Place" in display_df.columns:
        wanted_cols.append("Place")
    wanted_cols += ["ETA", "Time window", "Alert"]
    if not hide_coords:
        wanted_cols += ["Lat", "Lon"]

    cols = [c for c in wanted_cols if c in display_df.columns]
    missing = sorted(set(wanted_cols) - set(cols))
    if missing:
        st.info(f"Skipping missing columns {missing}")

    st.dataframe(
        display_df[cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Vehicle": st.column_config.TextColumn("Vehicle"),
            "Stop #": st.column_config.NumberColumn("Stop #"),
            "Order": st.column_config.TextColumn("Order ID"),
            "Place": st.column_config.TextColumn("Place"),
            "ETA": st.column_config.TextColumn("ETA"),
            "Time window": st.column_config.TextColumn("Time window"),
            "Alert": st.column_config.TextColumn("Alert"),
            "Lat": st.column_config.NumberColumn("Lat", format="%.4f"),
            "Lon": st.column_config.NumberColumn("Lon", format="%.4f"),
        },
    )

    st.divider()
    st.subheader("Route summary")

    def first_valid_eta(s):
        for v in s:
            if v != "N/A":
                return v
        return "—"

    def last_valid_eta(s):
        lst = list(s)
        for v in reversed(lst):
            if v != "N/A":
                return v
        return "—"

    summary = (
        df.sort_values(["vehicle_id", "eta"], kind="mergesort")
          .groupby("vehicle_id", sort=False)
          .agg(
              Stops=("order_id", "count"),
              **{"First ETA": ("eta", first_valid_eta)},
              **{"Last ETA": ("eta", last_valid_eta)},
              Alerts=("alert", lambda s: int((s != "").sum())),
          )
          .reset_index()
          .rename(columns={"vehicle_id": "Vehicle"})
    )
    st.dataframe(summary, hide_index=True, use_container_width=True)

    # ---------- map ----------
    show_map = st.checkbox("Show map", value=True)
    if show_map:
        try:
            import pydeck as pdk
            import numpy as np

            df_map = df.copy()
            df_map["stop_idx"] = df_map.groupby("vehicle_id").cumcount() + 1
            df_map = df_map.sort_values(["vehicle_id", "stop_idx"], kind="mergesort").reset_index(drop=True)

            palette = np.array([
                [  0,122,255], [255, 45, 85], [ 88, 86,214], [255,149,  0],
                [ 52,199, 89], [175, 82,222], [255, 59, 48], [ 90,200,250]
            ])
            veh_ids = df_map["vehicle_id"].unique()
            color_map = {v: palette[i % len(palette)].tolist() for i, v in enumerate(veh_ids)}
            df_map["color"] = df_map["vehicle_id"].map(color_map)

            # build paths (include depot point at start per-vehicle)
            rows = []
            for v in veh_ids:
                g = df_map[df_map["vehicle_id"] == v].sort_values("stop_idx")
                pts = [{"lon": float(r.lon), "lat": float(r.lat)} for r in g.itertuples(index=False)]
                rows.append({"vehicle_id": v, "path": pts, "color": color_map[v]})
            paths = pd.DataFrame(rows)

            path_layer = pdk.Layer("PathLayer", data=paths, get_path="path", get_width=4,
                                   get_color="color", width_min_pixels=2, pickable=False)
            point_layer = pdk.Layer("ScatterplotLayer", data=df_map, get_position='[lon, lat]',
                                    get_radius=60, get_fill_color="color",
                                    get_line_color=[255, 255, 255], line_width_min_pixels=1, pickable=True)

            # Labels: prefer Place; fallback to stop #
            if "place" in df_map.columns and df_map["place"].fillna("").ne("").any():
                df_map["label"] = df_map.apply(
                    lambda r: f"{int(r['stop_idx'])}. {r.get('place','') or ''}".strip(), axis=1
                )
            else:
                df_map["label"] = df_map.apply(lambda r: f"{int(r['stop_idx'])}", axis=1)

            text_layer = pdk.Layer("TextLayer", data=df_map, get_position='[lon, lat]',
                                   get_text="label", get_size=12, get_color=[230,230,230],
                                   get_angle=0, get_alignment_baseline="'center'")

            view = pdk.ViewState(
                latitude=float(df_map.lat.mean()),
                longitude=float(df_map.lon.mean()),
                zoom=11,
            )
            tooltip = {"text": "{label}\nVehicle {vehicle_id}\nETA {eta}"}

            st.pydeck_chart(pdk.Deck(
                layers=[path_layer, point_layer, text_layer],
                initial_view_state=view,
                tooltip=tooltip,
            ))
        except Exception as e:
            st.info(f"Map rendering skipped: {e}")

    # ---------- download ----------
    st.download_button(
        "Download planned routes (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="routes_plan.csv",
        mime="text/csv",
    )
