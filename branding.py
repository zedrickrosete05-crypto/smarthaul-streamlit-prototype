# branding.py
from pathlib import Path
import streamlit as st

# === tweak sizes/colors here ===
SIDEBAR_FILL = True              # fill sidebar width with the logo
SIDEBAR_LOGO_WIDTH = 260         # used only if SIDEBAR_FILL is False
HEADER_LOGO_WIDTH  = 140         # header logo size (px)
ACCENT = "#0B3C5D"               # brand blue (match config.toml)
TEXT_MUTED = "rgba(230,238,246,.75)"
# ===============================

def _find_logo() -> str | None:
    here = Path(__file__).resolve().parent
    for p in (
        Path("smarthaul-logo.png"),            # repo root (recommended)
        here / "smarthaul-logo.png",           # alongside branding.py
        here / "pages" / "smarthaul-logo.png", # inside /pages
    ):
        if p.exists():
            return str(p)
    return None

LOGO = _find_logo()

def setup_branding(page_title: str):
    """
    Set page config, paint sidebar logo, and render a large header with NO subtitle.
    Call this first in every page.
    """
    st.set_page_config(page_title=page_title, page_icon=(LOGO or "🚚"), layout="wide")

    # Sidebar branding
    if LOGO:
        if SIDEBAR_FILL:
            st.sidebar.image(LOGO, use_container_width=True)
        else:
            st.sidebar.image(LOGO, width=SIDEBAR_LOGO_WIDTH)

    # Global CSS (spacing, headings, buttons)
    st.markdown(f"""
    <style>
      .block-container {{ padding-top: 1.0rem; }}

      .sh-title {{ display:flex; gap:1rem; align-items:center; }}
      .sh-subtitle {{ margin:.25rem 0 0; color:{TEXT_MUTED}; }}

      .stButton>button {{
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,.06);
        background: linear-gradient(180deg, {ACCENT} 0%, #08283F 100%);
      }}

      hr.sh-divider {{
        border: 0; height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,.12), transparent);
        margin: .9rem 0 1.1rem 0;
      }}
    </style>
    """, unsafe_allow_html=True)

    # Header (no subtitle)
    col1, col2 = st.columns([0.18, 0.82])
    with col1:
        if LOGO:
            st.image(LOGO, width=HEADER_LOGO_WIDTH)
        else:
            st.write("🚚")
    with col2:
        st.markdown('<h1 style="margin:0">SmartHaul</h1>', unsafe_allow_html=True)

    st.markdown('<hr class="sh-divider"/>', unsafe_allow_html=True)

def section(title: str):
    """Nice section header for within-page sections."""
    st.markdown(f'<h2 style="margin:.25rem 0 .35rem 0;">{title}</h2>',
                unsafe_allow_html=True)
