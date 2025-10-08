import streamlit as st, pandas as pd, io

st.title("📥 Upload Orders")
st.caption("CSV columns: order_id, lat, lon, tw_start, tw_end, service_min, demand, priority")

sample = """order_id,lat,lon,tw_start,tw_end,service_min,demand,priority
A1,14.5995,120.9842,08:00,10:00,10,1,2
A2,14.5534,121.0190,09:00,11:00,10,1,1
A3,14.6091,121.0223,08:30,10:30,10,1,2
A4,14.6517,121.0493,10:00,12:00,10,1,2
A5,14.5700,121.0330,08:00,09:00,10,1,3
"""
c1,c2 = st.columns(2)
with c1: upl = st.file_uploader("Upload CSV", type=["csv"])
with c2:
    if st.button("Use sample data"):
        upl = io.BytesIO(sample.encode())

if upl:
    df = pd.read_csv(upl)
    st.dataframe(df, use_container_width=True)
    st.session_state["orders_df"] = df
    st.success(f"Loaded {len(df)} orders.")
else:
    st.info("Upload a CSV or click **Use sample data**.")
