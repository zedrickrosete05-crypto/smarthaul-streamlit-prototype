# --- SmartHaul branding (robust) ---
import os, streamlit as st
from pathlib import Path

def _find_logo() -> str | None:
    """
    Look for 'smarthaul-logo.png' in common spots and, if needed,
    search the repo for any *smart*haul*.png file.
    """
    here = Path(__file__).resolve().parent
    candidates = [
        Path("smarthaul-logo.png"),
        here / "smarthaul-logo.png",
        here.parent / "smarthaul-logo.png",
        Path("pages/smarthaul-logo.png"),
        here / "pages" / "smarthaul-logo.png",
        here.parent / "pages" / "smarthaul-logo.png",
    ]
    for p in candidates:
        if p.exists():
            return str(p)

    # last resort: search recursively for something that looks like the logo
    for root in {Path("."), here, here.parent}:
        matches = list(root.rglob("*smart*haul*.png"))
        if matches:
            return str(matches[0])
    return None

LOGO = _find_logo()
st.set_page_config(page_title="SmartHaul", page_icon=(LOGO or "🚚"), layout="wide")
if LOGO:
    st.sidebar.image(LOGO, use_column_width=True)
# (optional) show quick diagnostics; collapse when not needed
with st.expander("Branding debug", expanded=False):
    st.write("Working dir:", os.getcwd())
    st.write("This file:", str(Path(__file__).resolve()))
    st.write("Logo found:", bool(LOGO), LOGO or "—")
# -----------------------------------
