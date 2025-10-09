import streamlit as st, pandas as pd

st.title("📊 KPIs and Reports")
if "dispatch_df" not in st.session_state:
    st.warning("Nothing to report yet.")
    st.stop()

df = st.session_state["dispatch_df"].copy()
on_time = (df["alert"]=="").mean() if len(df) else 0
delivered = (df["status"]=="Delivered").mean() if len(df) else 0

c1,c2 = st.columns(2)
c1.metric("On-time (planned)", f"{on_time*100:.1f}%")
c2.metric("Delivered (actual)", f"{delivered*100:.1f}%")

st.download_button("Download dispatch log (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="dispatch_log.csv", mime="text/csv")
