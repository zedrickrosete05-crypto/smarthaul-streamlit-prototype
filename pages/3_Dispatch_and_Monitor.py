# pages/3_Dispatch_and_Monitor.py
from __future__ import annotations

import time, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SmartHaul – Dispatch and Monitor", layout="wide")
BUILD = "dispatch-v1"
st.title("Dispatch and Monitor")
st.caption(f"Build: {BUILD}")

def now_ampm() -> str:
    t = dt.datetime.now().time().replace(second=0, microsecond=0)
    h = t.hour; m = t.minute
    ampm = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    return f"{h12}:{m:02d} {ampm}"

if "routes_df" not in st.session_state:
    st.warning("No routes available. Plan routes first.")
    st.page_link("pages/2_Optimize_Routes.py", "← Optimize Routes", icon="⬅️")
    st.stop()

routes = st.session_state["routes_df"].copy()
routes["Stop #"] = routes.groupby("vehicle_id").cumcount() + 1
routes = routes.rename(columns={"vehicle_id":"Vehicle", "order_id":"Order", "eta":"ETA"})

# Init dispatch state
if "dispatch_log" not in st.session_state:
    st.session_state["dispatch_log"] = pd.DataFrame(columns=["Vehicle","Order","Status","Timestamp"])

left, right = st.columns([1,1])
with left:
    st.subheader("Live board")
    df_live = routes[["Vehicle","Stop #","Order","ETA","tw_start","tw_end","status","alert"]].copy()
    df_live["Time window"] = df_live["tw_start"].astype(str) + " – " + df_live["tw_end"].astype(str)
    st.dataframe(df_live.drop(columns=["tw_start","tw_end"]), use_container_width=True, hide_index=True)

with right:
    st.subheader("Actions")
    v_sel = st.selectbox("Vehicle", sorted(routes["Vehicle"].unique()))
    orders_for_v = routes[routes["Vehicle"]==v_sel]["Order"].tolist()
    o_sel = st.selectbox("Order", orders_for_v)
    c1, c2, c3 = st.columns(3)
    if c1.button("Start", use_container_width=True):
        st.session_state["dispatch_log"] = pd.concat([
            st.session_state["dispatch_log"],
            pd.DataFrame([{"Vehicle":v_sel,"Order":o_sel,"Status":"Started","Timestamp":now_ampm()}])
        ], ignore_index=True)
    if c2.button("Complete", use_container_width=True):
        st.session_state["dispatch_log"] = pd.concat([
            st.session_state["dispatch_log"],
            pd.DataFrame([{"Vehicle":v_sel,"Order":o_sel,"Status":"Completed","Timestamp":now_ampm()}])
        ], ignore_index=True)
    if c3.button("Delay", use_container_width=True):
        st.session_state["dispatch_log"] = pd.concat([
            st.session_state["dispatch_log"],
            pd.DataFrame([{"Vehicle":v_sel,"Order":o_sel,"Status":"Delayed","Timestamp":now_ampm()}])
        ], ignore_index=True)

st.divider()
st.subheader("Event log")
st.dataframe(st.session_state["dispatch_log"].sort_index(ascending=False), use_container_width=True, hide_index=True)
