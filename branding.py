# branding.py
from pathlib import Path
import streamlit as st
from PIL import Image

def _find_logo() -> Path | None:
    """
    Search common locations so it works whether we're in root or pages/.
    """
    here = Path(__file__).resolve().parent
    candidates = [
        Path("assets/smarthaul-logo.png"),                    # cwd (repo root)
        here / "assets" / "smarthaul-logo.png",               # alongside branding.py
        here.parent / "assets" / "smarthaul-logo.png",        # when called from pages/
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def setup_page(title: str):
    """
    Sets the page title+favicon and shows the logo in the sidebar.
    Call this at the very top of every page.
    """
    logo_path = _find_logo()
    page_icon = Image.open(logo_path) if logo_path else "🚚"
    st.set_page_config(page_title=title, page_icon=page_icon, layout="wide")

    # Sidebar logo
    if logo_path:
        st.sidebar.image(str(logo_path), use_column_width=True)
    else:
        st.sidebar.write("🚚")  # fallback

    # subtle spacing
    st.markdown("""
    <style>
      .block-container { padding-top: 1.0rem; }
      [data-testid="stSidebar"] img { margin: .25rem 0 .75rem 0; }
    </style>
    """, unsafe_allow_html=True)

    # debug hint (click to reveal path status)
    with st.expander("Branding debug", expanded=False):
        st.write("Working directory:", str(Path().resolve()))
        st.write("branding.py:", str(Path(__file__).resolve()))
        found = _find_logo()
        st.write("Logo found:", bool(found), str(found) if found else "—")

def header(subtitle: str):
    """Compact banner with logo + subtitle."""
    logo_path = _find_logo()
    col1, col2 = st.columns([1, 8])
    with col1:
        if logo_path:
            st.image(str(logo_path), width=64)
        else:
            st.write("🚚")
    with col2:
        st.markdown(
            f"<h1 style='margin:0'>SmartHaul</h1>"
            f"<p style='margin:.2rem 0 0; opacity:.8'>{subtitle}</p>",
            unsafe_allow_html=True
        )
    st.divider()
