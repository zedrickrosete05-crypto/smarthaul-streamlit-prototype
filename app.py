import streamlit as st
st.set_page_config(page_title="SmartHaul", layout="wide")
st.title("SmartHaul – Streamlit Prototype")
st.write("Use the sidebar to navigate:")
st.markdown("""
1. **Upload Orders** – load CSV or use the sample.  
2. **Optimize Routes** – compute routes, ETAs, risks.  
3. **Dispatch & Monitor** – review alerts.  
4. **KPIs & Reports** – check on-time %.  
""")
