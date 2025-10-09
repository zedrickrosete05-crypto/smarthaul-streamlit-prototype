# branding.py
from pathlib import Path
import streamlit as st

def _logo_path() -> str | None:
    # Prefer root file; fallback to pages/
    for p in (Path("smarthaul-logo.png"), Path("pages/smarthaul-logo.png")):
        if p.exists():
            return str(p)
    return None

def setup_page(title: str, subtitle: str):
    logo = _logo_path() or "🚚"
    st.set_page_config(page_title=title, page_icon=logo, layout="wide")
    if isinstance(logo, str) and logo.endswith((".png", ".jpg", ".jpeg")):
        st.sidebar.image(logo, use_column_width=True)

    # compact header
    col1, col2 = st.columns([1, 8])
    with col1:
        if isinstance(logo, str) and logo.endswith((".png", ".jpg", ".jpeg")):
            st.image(logo, width=64)
        else:
            st.write("🚚")
    with col2:
        st.markdown(f"<h1 style='margin:0'>SmartHaul</h1>"
                    f"<p style='margin:.2rem 0 0;opacity:.8'>{subtitle}</p>",
                    unsafe_allow_html=True)
    st.divider()
