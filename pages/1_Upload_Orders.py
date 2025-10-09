import io, pandas as pd, streamlit as st

REQUIRED_COLS = ["order_id","lat","lon","tw_start","tw_end","service_min"]

st.subheader("Upload orders")
file = st.file_uploader("CSV with orders", type="csv")

# Downloadable template
tmpl = pd.DataFrame([
    {"order_id":"O-1001","lat":10.3157,"lon":123.8854,"tw_start":"08:30","tw_end":"11:00","service_min":5},
    {"order_id":"O-1002","lat":10.3099,"lon":123.9180,"tw_start":"09:00","tw_end":"12:00","service_min":7},
])
st.download_button("Download CSV template", data=tmpl.to_csv(index=False).encode(),
                   file_name="orders_template.csv", mime="text/csv")

if file:
    df = pd.read_csv(file)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()
    # light coercion
    df["tw_start"] = df["tw_start"].astype(str).str[:5]
    df["tw_end"]   = df["tw_end"].astype(str).str[:5]
    st.session_state["orders_df"] = df
    st.success(f"Loaded {len(df)} orders.")
