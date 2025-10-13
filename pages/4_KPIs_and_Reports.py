# pages/4_KPIs_and_Reports.py
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SmartHaul – KPIs and Reports", layout="wide")
BUILD = "kpis-v1"
st.title("KPIs and Reports")
st.caption(f"Build: {BUILD}")

if "routes_df" not in st.session_state:
    st.warning("No routes available. Plan routes first.")
    st.page_link("pages/2_Optimize_Routes.py", "← Optimize Routes", icon="⬅️")
    st.stop()

df = st.session_state["routes_df"].copy()

# KPIs
total_stops = len(df)
vehicles = df["vehicle_id"].nunique()
late = int((df["within_window"] == False).sum())
on_time = total_stops - late
on_time_rate = (on_time / total_stops * 100) if total_stops else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Vehicles", vehicles)
c2.metric("Total stops", total_stops)
c3.metric("On-time stops", on_time)
c4.metric("On-time rate", f"{on_time_rate:.1f}%")

st.divider()
st.subheader("Detailed plan (AM/PM)")
show = df.rename(columns={"vehicle_id":"Vehicle","order_id":"Order","eta":"ETA"})
show["Time window"] = show["tw_start"].astype(str) + " – " + show["tw_end"].astype(str)
show = show[["Vehicle","Order","ETA","Time window","alert","status","lat","lon"]]
st.dataframe(show, use_container_width=True, hide_index=True)

st.download_button(
    "⬇️ Export plan (CSV)",
    data=show.to_csv(index=False).encode("utf-8"),
    file_name="route_plan_report.csv",
    mime="text/csv",
)
