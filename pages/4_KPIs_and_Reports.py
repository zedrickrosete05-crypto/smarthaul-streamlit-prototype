# pages/4_KPIs_and_Reports.py
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SmartHaul – KPIs & Reports", layout="wide")
st.title("KPIs & Reports")

# --- Data guards --------------------------------------------------------------
if "dispatch_df" not in st.session_state or st.session_state["dispatch_df"] is None:
    st.warning("No plan yet. Go to **Optimize Routes** first.")
    st.page_link("pages/2_Optimize_Routes.py", label="← Optimize Routes", icon="⬅️")
    st.stop()

df = st.session_state["dispatch_df"].copy()

# Ensure basic columns exist
for c in ["status", "vehicle_id", "order_id"]:
    if c not in df.columns:
        df[c] = "" if c != "status" else "Planned"

# If alert/within_window weren’t set upstream, add safe defaults
if "alert" not in df.columns:
    df["alert"] = ""
if "within_window" not in df.columns:
    df["within_window"] = np.nan  # unknown

# --- High-level KPIs ----------------------------------------------------------
total_stops = int(len(df))
delivered = int((df["status"] == "Delivered").sum())
enroute   = int((df["status"] == "En route").sum())
issues    = int((df["status"].isin(["Issue", "Skipped"])).sum())

# Planned on-time: percentage of rows with no “late risk” alert (if available)
if total_stops and "alert" in df.columns:
    ontime_plan = float((df["alert"] == "").mean())
else:
    ontime_plan = np.nan

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total stops", total_stops)
c2.metric("Delivered", delivered)
c3.metric("En route", enroute)
c4.metric("Issues/Skipped", issues)

st.metric("On-time (planned)", f"{ontime_plan*100:.1f}%" if not np.isnan(ontime_plan) else "—")

# --- Filters (optional) -------------------------------------------------------
st.divider()
st.subheader("Filters")
cols = st.columns(3)
with cols[0]:
    veh_filter = st.selectbox(
        "Vehicle",
        options=["(All)"] + sorted([v for v in df["vehicle_id"].astype(str).unique() if v]),
        index=0,
    )
with cols[1]:
    status_filter = st.selectbox(
        "Status",
        options=["(All)"] + ["Planned", "En route", "Delivered", "Skipped", "Issue"],
        index=0,
    )
with cols[2]:
    only_alerts = st.checkbox("Show only rows with alerts", value=False)

view = df.copy()
if veh_filter != "(All)":
    view = view[view["vehicle_id"].astype(str) == veh_filter]
if status_filter != "(All)":
    view = view[view["status"] == status_filter]
if only_alerts:
    view = view[view["alert"] != ""]

# --- Per-vehicle detail -------------------------------------------------------
st.divider()
st.subheader("Per-vehicle detail")

def _alerts_count(s: pd.Series) -> int:
    if "alert" in df.columns:
        return int((s != "").sum())
    return 0

detail = (
    df.groupby("vehicle_id", sort=False)
      .agg(
          Stops=("order_id", "count"),
          Delivered=("status", lambda s: int((s == "Delivered").sum())),
          EnRoute=("status", lambda s: int((s == "En route").sum())),
          Issues=("status", lambda s: int((s.isin(['Issue', 'Skipped'])).sum())),
          Alerts=("alert", _alerts_count),
      )
      .reset_index()
      .rename(columns={"vehicle_id": "Vehicle"})
      .sort_values("Vehicle", kind="stable")
)

st.dataframe(detail, hide_index=True, use_container_width=True)

# --- Charts -------------------------------------------------------------------
st.divider()
st.subheader("Stops per vehicle")

if not detail.empty:
    bar_df = detail[["Vehicle", "Stops"]].set_index("Vehicle")
    st.bar_chart(bar_df)
else:
    st.info("No data to chart yet.")

# --- Detail table (post-filter) ----------------------------------------------
st.divider()
st.subheader("Dispatch rows (filtered view)")
show_cols = [c for c in ["vehicle_id", "order_id", "status", "eta", "tw_start", "tw_end", "alert", "lat", "lon"] if c in view.columns]
if show_cols:
    st.dataframe(view[show_cols].sort_values(["vehicle_id", "order_id"], kind="stable"),
                 use_container_width=True, hide_index=True)
else:
    st.info("No columns available to display a detailed view.")

# --- Export -------------------------------------------------------------------
st.download_button(
    "Download report CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="kpi_report.csv",
    mime="text/csv",
)
