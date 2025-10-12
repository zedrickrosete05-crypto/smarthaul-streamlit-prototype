import streamlit as st

st.set_page_config(page_title="SmartHaul", layout="wide")
st.title("SmartHaul – Streamlit Prototype")

st.write("""
Welcome to **SmartHaul**.  
Use the sidebar to open a module:
1. Upload Orders  
2. Optimize Routes  
3. Dispatch and Monitor  
4. KPIs and Reports
""")

st.info("Start with **Upload Orders** in the sidebar ➡️")
