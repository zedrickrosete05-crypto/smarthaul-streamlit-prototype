# pages/3_Dispatch_and_Monitor.py
from __future__ import annotations

import datetime as dt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SmartHaul – Dispatch and Monitor", layout="wide")
BUILD = "dispatch-v3"
st.title("Dispatch and Monitor")
st.caption(f"Build: {BUILD}")

# ---------- helpers ----------
def now_ampm() -> str:
    t = dt.datetime.now().time().replace(second=0, microsecond=0)
    h = t.hour; m = t.minute
    ampm = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    return f"{h12}:{m:02d} {ampm}"

def update_route_status(vehicle: str, order: str, status: str) -> None:
    """Update status in routes_df for the selected vehicle+order."""
    if "routes_df" not in st.session_state:  # safety
        return
    df = st.session_state["routes_df"]
    mask = (df["vehicle_id"].astype(str) == vehicle) & (df["order_id"].astype(str) == order)
    if mask.any():
        df.loc[mask, "status"] = status
        st.session_state["routes_df"] = df  # persist

# ---------- guards ----------
if "routes_df" not in st.session_state or st.session_state["routes_df"] is None:
    st.warning("No routes available. Plan routes first.")
    st.page_link("pages/2_Optimize_Routes.py", "← Optimize Routes", icon="⬅️")
    st.stop()

routes = st.session_state["routes_df"].copy()
routes["Stop #"] = routes.groupby("vehicle_id").cumcount() + 1
routes = routes.rename(columns={"vehicle_id": "Vehicle", "order_id": "Order", "eta": "ETA"})
routes["Time window"] = routes["tw_start"].astype(str) + " – " + routes["tw_end"].astype(str)

# init dispatch log
if "dispatch_log" not in st.session_state:
    st.session_state["dispatch_log"] = pd.DataFrame(
        columns=["Timestamp", "Vehicle", "Order", "Action", "Note"]
    )

# ---------- board + actions ----------
left, right = st.columns([1.25, 0.75])

with left:
    st.subheader("Live board")
    board = routes[["Vehicle", "Stop #", "Order", "ETA", "Time window", "status", "alert"]].copy()
    st.dataframe(board, use_container_width=True, hide_index=True)

with right:
    st.subheader("Actions")
    v_sel = st.selectbox("Vehicle", sorted(routes["Vehicle"].unique()))
    orders_for_v = routes.loc[routes["Vehicle"] == v_sel, "Order"].tolist()
    o_sel = st.selectbox("Order", orders_for_v)
    note = st.text_input("Note (optional)", "")

    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Start", use_container_width=True):
        st.session_state["dispatch_log"].loc[len(st.session_state["dispatch_log"])] = \
            [now_ampm(), v_sel, o_sel, "Started", note]
        update_route_status(v_sel, o_sel, "In progress")
        st.success("Marked as Started")

    if c2.button("Arrived", use_container_width=True):
        st.session_state["dispatch_log"].loc[len(st.session_state["dispatch_log"])] = \
            [now_ampm(), v_sel, o_sel, "Arrived", note]
        update_route_status(v_sel, o_sel, "Arrived")
        st.success("Marked as Arrived")

    if c3.button("Complete", use_container_width=True):
        st.session_state["dispatch_log"].loc[len(st.session_state["dispatch_log"])] = \
            [now_ampm(), v_sel, o_sel, "Completed", note]
        update_route_status(v_sel, o_sel, "Completed")
        st.success("Marked as Completed")

    if c4.button("Delay", use_container_width=True):
        st.session_state["dispatch_log"].loc[len(st.session_state["dispatch_log"])] = \
            [now_ampm(), v_sel, o_sel, "Delayed", note]
        update_route_status(v_sel, o_sel, "Delayed")
        st.warning("Marked as Delayed")

st.divider()
st.subheader("Event log (latest first)")

log = st.session_state["dispatch_log"].copy()
if log.empty:
    st.info("No events yet.")
else:
    st.dataframe(log.iloc[::-1].reset_index(drop=True), use_container_width=True, hide_index=True)

    cdl, clr = st.columns([0.5, 0.5])
    cdl.download_button(
        "⬇️ Export dispatch log (CSV)",
        data=log.to_csv(index=False).encode("utf-8"),
        file_name="dispatch_log.csv",
        mime="text/csv",
        use_container_width=True,
    )
    if clr.button("🗑️ Clear log", use_container_width=True):
        st.session_state["dispatch_log"] = st.session_state["dispatch_log"].iloc[0:0].copy()
        st.success("Dispatch log cleared.")
