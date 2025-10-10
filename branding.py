# branding.py
from pathlib import Path
import streamlit as st

# === tweak sizes here ===
SIDEBAR_FILL = True           # fill sidebar width
SIDEBAR_LOGO_WIDTH = 260      # used if SIDEBAR_FILL is False
HEADER_LOGO_WIDTH  = 140      # big header logo (px)
# =========================

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
    """Set page title/icon and show a larger logo in the sidebar."""
    st.set_page_config(page_title=page_title, page_icon=(LOGO or "🚚"), layout="wide")
    if LOGO:
        if SIDEBAR_FILL:
            # fill the sidebar with the logo (new Streamlit param)
            st.sidebar.image(LOGO, use_container_width=True)
        else:
            st.sidebar.image(LOGO, width=SIDEBAR_LOGO_WIDTH)

def smarthaul_header(subtitle: str):
    """
    Header with a larger logo + subtitle.
    Increase the left column ratio to give the logo more room.
    """
    col1, col2 = st.columns([0.22, 0.78])   # more space for the logo
    with col1:
        if LOGO:
            st.image(LOGO, width=HEADER_LOGO_WIDTH)
        else:
            st.write("🚚")
    with col2:
        st.markdown(
            f"<h1 style='margin:0'>SmartHaul</h1>"
            f"<p style='margin:.35rem 0 0;opacity:.8'>{subtitle}</p>",
            unsafe_allow_html=True
        )
    st.divider()
