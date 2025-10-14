# pages/4_KPIs_and_Reports.py
from __future__ import annotations

import os, math
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SmartHaul – KPIs & Reports", layout="wide")
BUILD = "kpis-v4 (auto-load + safe columns)"
st.title("KPIs & Reports")
st.caption(f"Build: {BUILD}")

# ───────────────── persistence: auto-load last plan if missing ─────────────────
DATA_DIR = "data"
ROUTES_PATH = os.path.join(DATA_DIR, "routes_df.csv")
os.makedirs(DATA_DIR, exist_ok=True)

if "routes_df" not in st.session_state or st.session_state["routes_df"] is None:
    if os.path.exists(ROUTES_PATH):
        try:
            st.session_state["routes_df"] = pd.read_csv(ROUTES_PATH)
            st.toast("Loaded last saved plan from disk.", icon="✅")
        except Exception:
            pass

# ───────────────── guards ─────────────────
def safe_page_link(page: str, label: str) -> None:
    try:
        st.page_link(page, label=label)
    except TypeError:
        st.markdown(f"[{label}]({page})")

if "routes_df" not in st.session_state or st.session_state["routes_df"] is None:
    st.warning("No plan available. Go to **Optimize Routes** first.")
    safe_page_link("pages/2_Optimize_Routes.py", "← Optimize Routes")
    st.stop()

df = st.session_state["routes_df"].copy()

# ensure expected columns exist (prevents KeyErrors if older plans)
for c in ["vehicle_id","order_id","eta","tw_start","tw_end","status","alert","leg_km","lat","lon","within_window"]:
    if c not in df.columns:
        df[c] = pd.NA

# Default settings (if optimizer didn't store them)
settings = st.session_state.get("settings", {})
speed_kph = float(settings.get("speed_kph", 30.0))

# ───────────────── helpers ─────────────────
def first_nonblank(series: pd.Series) -> str:
    for v in series:
        if pd.notna(v) and str(v).strip() not in ("—", "", "N/A"):
            return str(v)
    return "—"

def last_nonblank(series: pd.Series) -> str:
    ser = list(series)
    for v in reversed(ser):
        if pd.notna(v) and str(v).strip() not in ("—", "", "N/A"):
            return str(v)
    return "—"

# ───────────────── quick filters ─────────────────
with st.sidebar:
    st.header("Filters")
    vehicles = sorted(df["vehicle_id"].astype(str).unique().tolist())
    veh_sel = st.multiselect("Vehicle(s)", vehicles, default=vehicles)
    status_vals = sorted(df["status"].astype(str).fillna("Planned").unique().tolist())
    status_sel = st.multiselect("Status", status_vals, default=status_vals)
    show_only_late = st.checkbox("Show only late/at-risk", value=False)

f = df.copy()
f["vehicle_id"] = f["vehicle_id"].astype(str)
f["status"] = f["status"].astype(str).fillna("Planned")
f = f[f["vehicle_id"].isin(veh_sel) & f["status"].isin(status_sel)]
if show_only_late:
    f = f[(f["within_window"] == False) | (f["alert"].fillna("") != "")]

if f.empty:
    st.info("No rows match the current filters.")
    st.stop()

# ───────────────── headline KPIs ─────────────────
total_stops = len(f)
veh_count = f["vehicle_id"].nunique()
late = int((f["within_window"] == False).sum())
on_time = total_stops - late
on_time_rate = (on_time / total_stops * 100) if total_stops else 0.0

total_km = float(pd.to_numeric(f["leg_km"], errors="coerce").fillna(0).sum())
est_drive_hours = total_km / max(speed_kph, 1.0)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Vehicles", veh_count)
c2.metric("Total stops", total_stops)
c3.metric("On-time stops", on_time)
c4.metric("On-time rate", f"{on_time_rate:.1f}%")
c5.metric("Total distance", f"{total_km:.1f} km")
c6.metric("Est. drive time", f"{est_drive_hours:.1f} h")

# ───────────────── per-vehicle summary ─────────────────
st.divider()
st.subheader("Per-vehicle summary")

tmp = f.copy()
tmp["Stop #"] = tmp.groupby("vehicle_id").cumcount() + 1
tmp = tmp.sort_values(["vehicle_id", "Stop #"], kind="mergesort")

veh_summary = (
    tmp.groupby("vehicle_id", sort=False)
      .agg(
          Stops=("order_id", "count"),
          **{"First ETA": ("eta", first_nonblank)},
          **{"Last ETA": ("eta", last_nonblank)},
          Distance_km=("leg_km", lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0).sum())),
          OnTime=("within_window", lambda s: int((s == True).sum())),
          Late=("within_window", lambda s: int((s == False).sum())),
          Alerts=("alert", lambda s: int((s.fillna("") != "").sum())),
      )
      .reset_index()
      .rename(columns={"vehicle_id": "Vehicle"})
)

veh_summary["On-time %"] = veh_summary.apply(
    lambda r: (r["OnTime"] / r["Stops"] * 100.0) if r["Stops"] else 0.0, axis=1
)
veh_summary["Distance_km"] = veh_summary["Distance_km"].map(lambda x: round(float(x), 2))

st.dataframe(
    veh_summary[["Vehicle", "Stops", "First ETA", "Last ETA", "OnTime", "Late", "On-time %", "Alerts", "Distance_km"]],
    use_container_width=True,
    hide_index=True
)

# ───────────────── detailed plan (AM/PM) ─────────────────
st.divider()
st.subheader("Detailed plan (AM/PM)")

show = f.rename(columns={"vehicle_id": "Vehicle", "order_id": "Order", "eta": "ETA"})
show["Time window"] = show["tw_start"].astype(str) + " – " + show["tw_end"].astype(str)
show["Lat"] = pd.to_numeric(show["lat"], errors="coerce").round(4)
show["Lon"] = pd.to_numeric(show["lon"], errors="coerce").round(4)

cols = ["Vehicle", "Order", "ETA", "Time window", "status", "alert", "leg_km", "Lat", "Lon"]
st.dataframe(show[cols], use_container_width=True, hide_index=True)

# ───────────────── exports ─────────────────
st.download_button(
    "⬇️ Export detailed plan (CSV)",
    data=show[cols].to_csv(index=False).encode("utf-8"),
    file_name="route_plan_report.csv",
    mime="text/csv",
)

st.download_button(
    "⬇️ Export per-vehicle summary (CSV)",
    data=veh_summary.to_csv(index=False).encode("utf-8"),
    file_name="route_vehicle_summary.csv",
    mime="text/csv",
)

# ───────────────── late / alerts spotlight ─────────────────
st.divider()
st.subheader("Late & Alerts spotlight")

late_df = f[(f["within_window"] == False) | (f["alert"].fillna("") != "")]
if late_df.empty:
    st.success("No late or at-risk stops 🎉")
else:
    spotlight = late_df.rename(columns={"vehicle_id": "Vehicle", "order_id": "Order", "eta": "ETA"})
    spotlight["Time window"] = spotlight["tw_start"].astype(str) + " – " + spotlight["tw_end"].astype(str)
    st.dataframe(
        spotlight[["Vehicle", "Order", "ETA", "Time window", "alert", "status", "leg_km", "lat", "lon"]],
        use_container_width=True,
        hide_index=True
    )
    st.download_button(
        "⬇️ Export late/alerts (CSV)",
        data=spotlight.to_csv(index=False).encode("utf-8"),
        file_name="late_and_alerts.csv",
        mime="text/csv",
    )
