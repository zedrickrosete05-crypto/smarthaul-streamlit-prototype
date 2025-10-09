# branding.py
from pathlib import Path
import streamlit as st

LOGO_PATH = Path("assets/smarthaul-logo.png")

def setup_page(title: str):
    """
    Sets the page title+favicon and shows the logo in the sidebar.
    Call this once at the very top of every page.
    """
    # Use image as favicon if available; fallback to an emoji
    page_icon = str(LOGO_PATH) if LOGO_PATH.exists() else "🚚"
    st.set_page_config(page_title=title, page_icon=page_icon, layout="wide")

    # Sidebar logo (safe if file missing)
    if LOGO_PATH.exists():
        st.sidebar.image(str(LOGO_PATH), use_column_width=True)

    # Small spacing / polish
    st.markdown("""
    <style>
      .block-container { padding-top: 1.0rem; }
      [data-testid="stSidebar"] img { margin: .25rem 0 .75rem 0; }
    </style>
    """, unsafe_allow_html=True)

def header(subtitle: str):
    """
    Compact banner with logo + subtitle at the top of the page.
    """
    left, right = st.columns([1, 8])
    with left:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=64)
        else:
            st.write("🚚")
    with right:
        st.markdown(
            f"<h1 style='margin:0'>SmartHaul</h1>"
            f"<p style='margin:.2rem 0 0; opacity:.8'>{subtitle}</p>",
            unsafe_allow_html=True
        )
    st.divider()
