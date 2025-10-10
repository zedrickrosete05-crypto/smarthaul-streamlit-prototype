# branding.py
from pathlib import Path
import streamlit as st

def _find_logo() -> str | None:
    here = Path(__file__).resolve().parent
    for p in (
        Path("smarthaul-logo.png"),       # repo root (recommended)
        here / "smarthaul-logo.png",      # same folder as branding.py
        here / "pages" / "smarthaul-logo.png",   # inside /pages
    ):
        if p.exists():
            return str(p)
    return None

LOGO = _find_logo()

def setup_branding(page_title: str):
    """Set page title/icon and show a small logo in the sidebar."""
    st.set_page_config(page_title=page_title, page_icon=(LOGO or "🚚"), layout="wide")
    if LOGO:
        st.sidebar.image(LOGO, width=120)  # small sidebar logo

def smarthaul_header(subtitle: str):
    """Tiny header (logo + subtitle) used on every page."""
    col1, col2 = st.columns([0.08, 0.92])
    with col1:
        if LOGO:
            st.image(LOGO, width=36)       # tiny header logo
        else:
            st.write("🚚")
    with col2:
        st.markdown(
            f"<h1 style='margin:0'>SmartHaul</h1>"
            f"<p style='margin:.25rem 0 0;opacity:.8'>{subtitle}</p>",
            unsafe_allow_html=True
        )
    st.divider()
