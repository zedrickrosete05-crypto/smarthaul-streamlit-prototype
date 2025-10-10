from branding import setup_branding, section
setup_branding("SmartHaul – Upload Orders")

import pandas as pd
import streamlit as st

REQUIRED = ["order_id","lat","lon","tw_start","tw_end","service_min"]

section("Upload Orders")

# Template to download
tmpl = pd.DataFrame([
    {"order_id":"O-1001","lat":10.3157,"lon":123.8854,"tw_start":"08:30","tw_end":"11:00","service_min":7},
    {"order_id":"O-1002","lat":10.3099,"lon":123.9180,"tw_start":"09:00","tw_end":"12:00","service_min":5},
    {"order_id":"O-1003","lat":10.3070,"lon":123.9011,"tw_start":"10:00","tw_end":"13:00","service_min":10},
])
st.download_button("Download CSV template (lat/lon)",
                   tmpl.to_csv(index=False).encode(),
                   file_name="orders_template.csv", mime="text/csv")

st.caption("Upload CSV with columns: order_id, lat, lon, tw_start, tw_end, service_min")
file = st.file_uploader(" ", type="csv")
