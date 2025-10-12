import streamlit as st
st.set_page_config(page_title="SmartHaul", layout="wide")
st.title("SmartHaul – Streamlit Prototype")
st.write("Upload a CSV to test the pipeline.")

file = st.file_uploader("Upload CSV", type=["csv"])
if file:
    import pandas as pd
    try:
        df = pd.read_csv(file)
        st.success(f"Loaded {len(df)} rows.")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
