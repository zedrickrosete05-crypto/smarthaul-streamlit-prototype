# pages/2_Optimize_Routes.py
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import math

st.title("🧭 Optimize Routes")

# ---------- helpers ----------
def haversine_km(a: float, b: float, c: float, d: float) -> float:
    """Great-circle distance in km between (a,b) and (c,d)."""
    R = 6371.0088
    p1, p2 = math.radians(a), math.radians(c)
    dphi = math.radians(c - a)
    dlmb = math.radians(d - b)
    x = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(x))

def travel_min(a, b, c, d, speed: float = 25.0) -> float:
    """Travel time in minutes using straight-line distance and average speed."""
    return (haversine_km(a, b, c, d) / max(speed, 5)) * 60.0

def t(s: str) -> datetime:
    """'HH:MM' -> datetime on an arbitrary date."""
    return datetime.strptime(str(s), "%H:%M")

# ---------- greedy planner (robust typing) ----------
def greedy(
    df: pd.DataFrame,
    depot=(14.5995, 120.9842),
    speed: float = 25.0,
    max_stops: int = 10,
) -> pd.DataFrame:
    orders = df.copy().reset_index(drop=True)

    # Coerce critical columns to numeric, guard against bad rows
    orders["lat"] = pd.to_numeric(orders["lat"], errors="coerce")
    orders["lon"] = pd.to_numeric(orders["lon"], errors="coerce")
    orders["service_min"] = pd.to_numeric(orders.get("service_min", 0), errors="coerce").fillna(0)

    # Ensure time window strings exist
    orders["tw_start"] = orders["tw_start"].astype(str)
    orders["tw_end"] = orders["tw_end"].astype(str)

    orders["done"] = False
    routes, vid = [], 1

    while not orders.done.all():
        lat, lon = depot
        now = t("08:00")
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

            route.append(
                dict(
                    vehicle_id=f"V{vid}",
                    order_id=r.order_id,
                    lat=r.lat,
                    lon=r.lon,
                    eta=start.strftime("%H:%M"),
                    tw_start=r.tw_start,
                    tw_end=r.tw_end,
                )
            )

            orders.at[i, "done"] = True
            lat, lon, now = r.lat, r.lon, leave

        if route:
            routes.append(pd.DataFrame(route))
            vid += 1
        else:
            # If nothing feasible, assign one to progress and continue
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
                            tw_start=r.tw_start,
                            tw_end=r.tw_end,
                        )
                    ]
                )
            )
            orders.at[i, "done"] = True
            vid += 1

    return pd.concat(routes, ignore_index=True)

# ---------- require data ----------
if "orders_df" not in st.session_state:
    st.warning("Upload orders first on the Upload page.")
    st.stop()

speed = st.slider("Average speed (km/h)", 15, 45, 25, 1)
maxst = st.slider("Max stops per route", 5, 20, 10, 1)

if st.button("Compute routes"):
    df_plan = greedy(st.session_state["orders_df"], speed=speed, max_stops=maxst)
    df_plan["alert"] = df_plan.apply(
        lambda r: "Late risk" if r["eta"] != "N/A" and r["eta"] > r["tw_end"] else "",
        axis=1,
    )
    st.session_state["routes_df"] = df_plan
    st.success(f"Computed {df_plan['vehicle_id'].nunique()} route(s) for {len(df_plan)} stops.")

# ---------- results ----------
if "routes_df" in st.session_state:
    df = st.session_state["routes_df"].copy()

    # Friendly table
    st.subheader("Planned routes")
    df["Stop #"] = df.groupby("vehicle_id").cumcount() + 1
    df["Time window"] = df["tw_start"].astype(str) + " – " + df["tw_end"].astype(str)
    df["Lat"] = pd.to_numeric(df["lat"], errors="coerce").round(4)
    df["Lon"] = pd.to_numeric(df["lon"], errors="coerce").round(4)

    display_df = df.rename(
        columns={"vehicle_id": "Vehicle", "order_id": "Order", "eta": "ETA", "alert": "Alert"}
    )[["Vehicle", "Stop #", "Order", "ETA", "Time window", "Alert", "Lat", "Lon"]]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Vehicle": st.column_config.TextColumn("Vehicle", help="Assigned vehicle/route"),
            "Stop #": st.column_config.NumberColumn("Stop #", help="Sequence within the vehicle's route"),
            "Order": st.column_config.TextColumn("Order ID", help="Customer/job identifier"),
            "ETA": st.column_config.TextColumn("ETA", help="Estimated arrival time"),
            "Time window": st.column_config.TextColumn("Time window", help="Promised delivery window"),
            "Alert": st.column_config.TextColumn("Alert", help="Risk or note"),
            "Lat": st.column_config.NumberColumn("Lat", format="%.4f"),
            "Lon": st.column_config.NumberColumn("Lon", format="%.4f"),
        },
    )

    st.divider()
    st.subheader("Route summary")
    for v, g in df.groupby("vehicle_id", sort=False):
        stops = len(g)
        first_eta = g["eta"].iloc[0]
        last_eta = g["eta"].iloc[-1] if (g["eta"].iloc[-1] != "N/A") else "—"
        alerts = int((g["alert"] != "").sum())
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        with c1:
            st.metric(f"{v}", f"{stops} stops")
        with c2:
            st.metric("First ETA", first_eta)
        with c3:
            st.metric("Last ETA", last_eta)
        with c4:
            st.metric("Alerts", alerts)

    # ---------- map (colored routes + labels + depot) ----------
    show_map = st.checkbox("Show map", value=True)
    if show_map:
        try:
            import pydeck as pdk
            import numpy as np

            # stop index per vehicle for labels
            df["stop_idx"] = df.groupby("vehicle_id").cumcount() + 1

            # color per vehicle (RGB)
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
            veh_ids = df["vehicle_id"].unique()
            color_map = {v: palette[i % len(palette)].tolist() for i, v in enumerate(veh_ids)}
            df["color"] = df["vehicle_id"].map(color_map)

            # path per vehicle
            def path_from_group(g: pd.DataFrame):
                return [{"lon": float(r.lon), "lat": float(r.lat)} for r in g.itertuples(index=False)]

            paths = (
                df.groupby("vehicle_id", sort=False)
                .apply(path_from_group)
                .reset_index(name="path")
            )
            paths["color"] = paths["vehicle_id"].map(color_map)

            path_layer = pdk.Layer(
                "PathLayer",
                data=paths,
                get_path="path",
                get_width=3,
                get_color="color",
                width_min_pixels=2,
                pickable=False,
            )
            point_layer = pdk.Layer(
                "ScatterplotLayer",
                data=df,
                get_position="[lon, lat]",
                get_radius=55,
                get_fill_color="color",
                get_line_color=[255, 255, 255],
                line_width_min_pixels=1,
                pickable=True,
            )
            text_layer = pdk.Layer(
                "TextLayer",
                data=df,
                get_position="[lon, lat]",
                get_text="stop_idx",
                get_size=12,
                get_color=[230, 230, 230],
                get_angle=0,
                get_alignment_baseline="'center'",
            )

            # depot (use mean as fallback)
            if len(df):
                depot_lon = float(df.iloc[0].Lon) if "Lon" in df else float(df.iloc[0].lon)
                depot_lat = float(df.iloc[0].Lat) if "Lat" in df else float(df.iloc[0].lat)
            else:
                depot_lon = float(df.lon.mean())
                depot_lat = float(df.lat.mean())
            depot = pd.DataFrame([{"lon": depot_lon, "lat": depot_lat}])
            depot_layer = pdk.Layer(
                "ScatterplotLayer",
                data=depot,
                get_position="[lon, lat]",
                get_radius=100,
                get_fill_color=[255, 255, 0],
                get_line_color=[0, 0, 0],
                line_width_min_pixels=2,
            )

            view = pdk.ViewState(
                latitude=float(df.lat.mean()),
                longitude=float(df.lon.mean()),
                zoom=11,
            )
            tooltip = {"text": "Vehicle {vehicle_id}\nStop {stop_idx}\nETA {eta}"}

            st.pydeck_chart(
                pdk.Deck(
                    layers=[path_layer, depot_layer, point_layer, text_layer],
                    initial_view_state=view,
                    tooltip=tooltip,
                )
            )
        except Exception as e:
            st.info(f"Map rendering skipped: {e}")

    # ---------- download ----------
    st.download_button(
        "Download planned routes (CSV)",
        data=st.session_state["routes_df"].to_csv(index=False).encode("utf-8"),
        file_name="routes_plan.csv",
        mime="text/csv",
    )
