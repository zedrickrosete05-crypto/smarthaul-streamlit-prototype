# branding.py
from pathlib import Path
import streamlit as st

# Tweak these two numbers to change sizes app-wide
SIDEBAR_LOGO_WIDTH = 200   # was 120
HEADER_LOGO_WIDTH  = 80    # was 36

def _find_logo() -> str | None:
    here = Path(__file__).resolve().parent
    for p in (
        Path("smarthaul-logo.png"),            # repo root (recommended)
        here / "smarthaul-logo.png",           # same folder as branding.py
        here / "pages" / "smarthaul-logo.png", # inside /pages
    ):
        if p.exists():
            return str(p)
    return None

LOGO = _find_logo()

def setup_branding(page_title: str):
    """Set page title/icon and show a big logo in the sidebar."""
    st.set_page_config(page_title=page_title, page_icon=(LOGO or "🚚"), layout="wide")
    if LOGO:
        st.sidebar.image(LOGO, width=SIDEBAR_LOGO_WIDTH)

def smarthaul_header(subtitle: str):
    """Header with bigger logo + subtitle."""
    col1, col2 = st.columns([0.1, 0.9])
    with col1:
        if LOGO:
            st.image(LOGO, width=HEADER_LOGO_WIDTH)
        else:
            st.write("🚚")
    with col2:
        st.markdown(
            f"<h1 style='margin:0'>SmartHaul</h1>"
            f"<p style='margin:.35rem 0 0; opacity:.8'>{subtitle}</p>",
            unsafe_allow_html=True
        )
    st.divider()
