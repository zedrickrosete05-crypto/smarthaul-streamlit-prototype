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

    # ---------- map (connected path + labels like delivery apps) ----------
    show_map = st.checkbox("Show map", value=True)
    if show_map:
        try:
            import pydeck as pdk
            import numpy as np

            # Work on a fresh copy and ensure order by stop
            df_map = st.session_state["routes_df"].copy()
            df_map["stop_idx"] = df_map.groupby("vehicle_id").cumcount() + 1
            df_map = df_map.sort_values(["vehicle_id", "stop_idx"], kind="mergesort").reset_index(drop=True)

            # Color per vehicle (RGB palette)
            palette = np.array([
                [  0,122,255], [255, 45, 85], [ 88, 86,214], [255,149,  0],
                [ 52,199, 89], [175, 82,222], [255, 59, 48], [ 90,200,250]
            ])
            veh_ids = df_map["vehicle_id"].unique()
            color_map = {v: palette[i % len(palette)].tolist() for i, v in enumerate(veh_ids)}
            df_map["color"] = df_map["vehicle_id"].map(color_map)

            # Include depot as starting point (uses first stop per vehicle; swap with fixed coords if needed)
            include_depot = st.checkbox("Include depot as starting point", value=True)
            if include_depot and len(df_map):
                depots = (
                    df_map.groupby("vehicle_id", sort=False)
                          .first()[["lon", "lat"]]
                          .reset_index()
                          .rename(columns={"lon": "d_lon", "lat": "d_lat"})
                )
                df_map = df_map.merge(depots, on="vehicle_id", how="left")
            else:
                df_map["d_lon"] = np.nan
                df_map["d_lat"] = np.nan

            # Build a sorted path per vehicle (depot first if enabled)
            paths_rows = []
            for v, g in df_map.groupby("vehicle_id", sort=False):
                g = g.sort_values("stop_idx")
                pts = [{"lon": float(r.lon), "lat": float(r.lat)} for r in g.itertuples(index=False)]
                if include_depot and np.isfinite(g["d_lon"].iloc[0]) and np.isfinite(g["d_lat"].iloc[0]):
                    pts = [{"lon": float(g["d_lon"].iloc[0]), "lat": float(g["d_lat"].iloc[0])}] + pts
                paths_rows.append({"vehicle_id": v, "path": pts, "color": color_map[v]})
            paths = pd.DataFrame(paths_rows)

            path_layer = pdk.Layer(
                "PathLayer",
                data=paths,
                get_path="path",
                get_width=4,
                get_color="color",
                width_min_pixels=2,
                pickable=False,
            )

            point_layer = pdk.Layer(
                "ScatterplotLayer",
                data=df_map,
                get_position='[lon, lat]',
                get_radius=60,
                get_fill_color="color",
                get_line_color=[255, 255, 255],
                line_width_min_pixels=1,
                pickable=True,
            )

            # Text labels like V1-1, V1-2, ...
            df_map["label"] = df_map.apply(lambda r: f"{r['vehicle_id']}-{int(r['stop_idx'])}", axis=1)
            text_layer = pdk.Layer(
                "TextLayer",
                data=df_map,
                get_position='[lon, lat]',
                get_text="label",
                get_size=12,
                get_color=[230, 230, 230],
                get_angle=0,
                get_alignment_baseline="'center'",
            )

            # Start & End markers (bigger dots + labels)
            starts = df_map[df_map["stop_idx"] == 1]
            ends = df_map.loc[df_map.groupby("vehicle_id")["stop_idx"].idxmax()]

            start_layer = pdk.Layer(
                "ScatterplotLayer",
                data=starts,
                get_position='[lon, lat]',
                get_radius=90,
                get_fill_color=[255, 255, 0],  # yellow
                get_line_color=[0, 0, 0],
                line_width_min_pixels=2,
            )
            start_text = pdk.Layer(
                "TextLayer",
                data=starts.assign(lbl="START"),
                get_position='[lon, lat]',
                get_text="lbl",
                get_size=14,
                get_color=[255, 255, 0],
                get_alignment_baseline="'top'",
            )

            end_layer = pdk.Layer(
                "ScatterplotLayer",
                data=ends,
                get_position='[lon, lat]',
                get_radius=90,
                get_fill_color=[255, 59, 48],  # red
                get_line_color=[0, 0, 0],
                line_width_min_pixels=2,
            )
            end_text = pdk.Layer(
                "TextLayer",
                data=ends.assign(lbl="END"),
                get_position='[lon, lat]',
                get_text="lbl",
                get_size=14,
                get_color=[255, 59, 48],
                get_alignment_baseline="'bottom'",
            )

            view = pdk.ViewState(
                latitude=float(df_map.lat.mean()),
                longitude=float(df_map.lon.mean()),
                zoom=11,
            )

            tooltip = {"text": "Vehicle {vehicle_id}\nStop {stop_idx}\nETA {eta}"}

            st.pydeck_chart(
                pdk.Deck(
                    layers=[path_layer, point_layer, text_layer, start_layer, start_text, end_layer, end_text],
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
