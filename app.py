import streamlit as st

st.set_page_config(page_title="SmartHaul", layout="wide")

st.title("SmartHaul – Streamlit Prototype")
st.write("Use the sidebar to navigate:")
st.markdown("""
1. **Upload Orders** — load CSV or use the sample.
2. **Optimize Routes** — compute routes, ETAs, risks.
3. **Dispatch & Monitor** — review alerts.
4. **KPIs & Reports** — check on-time %.
""")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Upload Orders", "Optimize Routes", "Dispatch & Monitor", "KPIs & Reports"])

if page == "Upload Orders":
    st.subheader("Upload Orders")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        import pandas as pd
        try:
            df = pd.read_csv(file)
            st.success(f"Loaded {len(df)} rows.")
            st.dataframe(df.head(20))
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

elif page == "Optimize Routes":
    st.subheader("Optimize Routes")
    st.info("Route optimization logic goes here.")

elif page == "Dispatch & Monitor":
    st.subheader("Dispatch & Monitor")
    st.info("Real-time alerts and status go here.")

else:
    st.subheader("KPIs & Reports")
    st.info("Summary metrics and reports go here.")
