from branding import setup_branding, smarthaul_header
setup_branding("SmartHaul – KPIs & Reports")
smarthaul_header("KPIs & Reports")


# pages/4_KPIs_and_Reports.py
import streamlit as st
import pandas as pd

st.set_page_config(page_title="SmartHaul – KPIs & Reports", page_icon="📊")
st.title("📊 KPIs and Reports")

if "dispatch_df" not in st.session_state:
    st.warning("No plan yet. Go to ‘2 Optimize Routes’.")
    st.stop()

df = st.session_state["dispatch_df"].copy()

# basic KPIs
total_stops = len(df)
delivered = int((df["status"] == "Delivered").sum())
enroute = int((df["status"] == "En route").sum())
issues = int((df["status"].isin(["Issue","Skipped"])).sum())
ontime_plan = float((df["alert"] == "").mean()) if "alert" in df.columns and total_stops else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total stops", total_stops)
c2.metric("Delivered", delivered)
c3.metric("En route", enroute)
c4.metric("Issues/Skipped", issues)

st.metric("On-time (planned)", f"{ontime_plan*100:.1f}%")

st.divider()
st.subheader("Per-vehicle detail")
detail = (
    df.groupby("vehicle_id", sort=False)
      .agg(
          Stops=("order_id","count"),
          Delivered=("status", lambda s: int((s=="Delivered").sum())),
          Alerts=("alert", lambda s: int((s!="").sum()) if "alert" in df.columns else 0),
      )
      .reset_index()
      .rename(columns={"vehicle_id":"Vehicle"})
)
st.dataframe(detail, hide_index=True, use_container_width=True)

st.download_button(
    "Download report CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="kpi_report.csv",
    mime="text/csv",
)
