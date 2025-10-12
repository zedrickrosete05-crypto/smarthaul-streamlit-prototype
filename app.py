import io
import pandas as pd
import streamlit as st

REQUIRED_COLS = ["order_id","lat","lon","demand","window_start_min","window_end_min"]

# Downloadable sample
sample = """order_id,lat,lon,demand,window_start_min,window_end_min
A1,10.3157,123.8854,10,480,1020
A2,10.3190,123.9030,5,540,1080
A3,10.3072,123.8950,8,600,1140
"""
st.download_button("Download sample CSV", sample, file_name="orders_sample.csv", mime="text/csv")

file = st.file_uploader("Upload CSV", type=["csv"])
if file:
    try:
        df = pd.read_csv(file)
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            st.success(f"Loaded {len(df)} rows ✅")
            st.dataframe(df.head(20))
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
