import pandas as pd
import streamlit as st

st.set_page_config(page_title="SmartHaul – Upload Orders", page_icon="📤")
st.title("📤 Upload Orders")

REQUIRED = ["order_id","lat","lon","tw_start","tw_end","service_min"]

# Template to download
tmpl = pd.DataFrame([
    {"order_id":"O-1001","lat":10.3157,"lon":123.8854,"tw_start":"08:30","tw_end":"11:00","service_min":7},
    {"order_id":"O-1002","lat":10.3099,"lon":123.9180,"tw_start":"09:00","tw_end":"12:00","service_min":5},
    {"order_id":"O-1003","lat":10.3070,"lon":123.9011,"tw_start":"10:00","tw_end":"13:00","service_min":10},
])
st.download_button(
    "Download CSV template (lat/lon)",
    data=tmpl.to_csv(index=False).encode(),
    file_name="orders_template.csv",
    mime="text/csv",
)

file = st.file_uploader("Upload CSV with columns: " + ", ".join(REQUIRED), type="csv")
if not file:
    st.stop()

df = pd.read_csv(file)
df.columns = [c.strip() for c in df.columns]

missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

# type coercions & light cleaning
df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
df["service_min"] = pd.to_numeric(df["service_min"], errors="coerce").fillna(0).clip(lower=0)

for c in ("tw_start","tw_end"):
    df[c] = df[c].astype(str).str.extract(r"^(\d{1,2}:\d{2})", expand=False)

good = df.dropna(subset=["lat","lon"]).copy()
if good.empty:
    st.error("All rows have invalid lat/lon.")
    st.stop()

st.session_state["orders_df"] = good[REQUIRED].copy()
st.success(f"Loaded {len(good)} orders.")
st.dataframe(st.session_state["orders_df"].head(20), use_container_width=True)
