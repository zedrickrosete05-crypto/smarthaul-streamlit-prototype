import streamlit as st
st.title("🛰️ Dispatch & Monitor")
if "routes_df" not in st.session_state:
    st.warning("Optimize routes first."); st.stop()

alerts = st.session_state["routes_df"].query("alert != ''")
if alerts.empty:
    st.success("No risks detected.")
else:
    st.error(f"{len(alerts)} stop(s) with risk")
    st.dataframe(alerts[["vehicle_id","order_id","eta","tw_end","alert"]], use_container_width=True)

st.caption("Telemetry simulation can be added later.")
