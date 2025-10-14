# pages/3_Dispatch_and_Monitor.py
from __future__ import annotations

import os, datetime as dt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SmartHaul – Dispatch and Monitor", layout="wide")
BUILD = "dispatch-v4 (persisted plan + log)"
st.title("Dispatch and Monitor")
st.caption(f"Build: {BUILD}")

# ────────────────────────────────────────────────────────────────────────────────
# Persistence helpers
# ────────────────────────────────────────────────────────────────────────────────
DATA_DIR = "data"
ROUTES_PATH = os.path.join(DATA_DIR, "routes_df.csv")
LOG_PATH = os.path.join(DATA_DIR, "dispatch_log.csv")
os.makedirs(DATA_DIR, exist_ok=True)

def load_routes() -> pd.DataFrame | None:
    return pd.read_csv(ROUTES_PATH) if os.path.exists(ROUTES_PATH) else None

def save_routes(df: pd.DataFrame) -> None:
    try:
        df.to_csv(ROUTES_PATH, index=False)
    except Exception:
        pass

def load_log() -> pd.DataFrame:
    if os.path.exists(LOG_PATH):
        try:
            df = pd.read_csv(LOG_PATH)
            # Ensure expected columns exist
            need = ["Timestamp","Vehicle","Order","Action","Note"]
            for c in need:
                if c not in df.columns:
                    df[c] = "" if c != "Note" else ""
            return df[need]
        except Exception:
            pass
    return pd.DataFrame(columns=["Timestamp","Vehicle","Order","Action","Note"])

def save_log(df: pd.DataFrame) -> None:
    try:
        df.to_csv(LOG_PATH, index=False)
    except Exception:
        pass

def safe_page_link(page: str, label: str) -> None:
    try:
        st.page_link(page, label=label)
    except TypeError:
        st.markdown(f"[{label}]({page})")

# ────────────────────────────────────────────────────────────────────────────────
# Init session: try auto-load plan/log if missing
# ────────────────────────────────────────────────────────────────────────────────
if "routes_df" not in st.session_state or st.session_state["routes_df"] is None:
    prev = load_routes()
    if prev is not None and not prev.empty:
        st.session_state["routes_df"] = prev.copy()

if "dispatch_log" not in st.session_state:
    st.session_state["dispatch_log"] = load_log()

# ────────────────────────────────────────────────────────────────────────────────
# Guards
# ────────────────────────────────────────────────────────────────────────────────
if "routes_df" not in st.session_state or st.session_state["routes_df"] is None:
    st.warning("No routes available. Plan routes first.")
    safe_page_link("pages/2_Optimize_Routes.py", "← Optimize Routes")
    st.stop()

# Ensure required columns exist in routes_df
routes_raw = st.session_state["routes_df"].copy()
for c in ["vehicle_id","order_id","eta","tw_start","tw_end","status","alert","lat","lon"]:
    if c not in routes_raw.columns:
        routes_raw[c] = pd.NA

# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────
def now_ampm() -> str:
    t = dt.datetime.now().time().replace(second=0, microsecond=0)
    h, m = t.hour, t.minute
    ampm = "AM" if h < 12 else "PM"
    h12 = (h % 12) or 12
    return f"{h12}:{m:02d} {ampm}"

def update_route_status(vehicle: str, order: str, status: str) -> None:
    """Update status in session + persist to disk."""
    if "routes_df" not in st.session_state: return
    df = st.session_state["routes_df"].copy()
    mask = (df["vehicle_id"].astype(str) == vehicle) & (df["order_id"].astype(str) == order)
    if mask.any():
        df.loc[mask, "status"] = status
        st.session_state["routes_df"] = df
        save_routes(df)  # persist status change

# ────────────────────────────────────────────────────────────────────────────────
# Sidebar: session controls
# ────────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Session")
    if st.button("Load last plan & log"):
        r_prev = load_routes()
        l_prev = load_log()
        if r_prev is not None and not r_prev.empty:
            st.session_state["routes_df"] = r_prev.copy()
            st.success(f"Loaded plan with {len(r_prev)} stop(s).")
        if not l_prev.empty:
            st.session_state["dispatch_log"] = l_prev.copy()
            st.success(f"Loaded log with {len(l_prev)} event(s).")
        if (r_prev is None or r_prev.empty) and l_prev.empty:
            st.info("Nothing to load yet.")

# ────────────────────────────────────────────────────────────────────────────────
# Prepare board
# ────────────────────────────────────────────────────────────────────────────────
routes = st.session_state["routes_df"].copy()
routes["Stop #"] = routes.groupby("vehicle_id").cumcount() + 1
routes = routes.rename(columns={"vehicle_id":"Vehicle","order_id":"Order","eta":"ETA"})
routes["Time window"] = routes["tw_start"].astype(str) + " – " + routes["tw_end"].astype(str)

# ────────────────────────────────────────────────────────────────────────────────
# Board + actions
# ────────────────────────────────────────────────────────────────────────────────
left, right = st.columns([1.25, 0.75])

with left:
    st.subheader("Live board")
    board = routes[["Vehicle","Stop #","Order","ETA","Time window","status","alert"]].copy()
    st.dataframe(board, use_container_width=True, hide_index=True)

with right:
    st.subheader("Actions")
    v_sel = st.selectbox("Vehicle", sorted(routes["Vehicle"].unique()))
    orders_for_v = routes.loc[routes["Vehicle"] == v_sel, "Order"].tolist()
    o_sel = st.selectbox("Order", orders_for_v)
    note = st.text_input("Note (optional)", "")

    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Start", use_container_width=True):
        st.session_state["dispatch_log"].loc[len(st.session_state["dispatch_log"])] = \
            [now_ampm(), v_sel, o_sel, "Started", note]
        save_log(st.session_state["dispatch_log"])
        update_route_status(v_sel, o_sel, "In progress")
        st.success("Marked as Started")

    if c2.button("Arrived", use_container_width=True):
        st.session_state["dispatch_log"].loc[len(st.session_state["dispatch_log"])] = \
            [now_ampm(), v_sel, o_sel, "Arrived", note]
        save_log(st.session_state["dispatch_log"])
        update_route_status(v_sel, o_sel, "Arrived")
        st.success("Marked as Arrived")

    if c3.button("Complete", use_container_width=True):
        st.session_state["dispatch_log"].loc[len(st.session_state["dispatch_log"])] = \
            [now_ampm(), v_sel, o_sel, "Completed", note]
        save_log(st.session_state["dispatch_log"])
        update_route_status(v_sel, o_sel, "Completed")
        st.success("Marked as Completed")

    if c4.button("Delay", use_container_width=True):
        st.session_state["dispatch_log"].loc[len(st.session_state["dispatch_log"])] = \
            [now_ampm(), v_sel, o_sel, "Delayed", note]
        save_log(st.session_state["dispatch_log"])
        update_route_status(v_sel, o_sel, "Delayed")
        st.warning("Marked as Delayed")

# ────────────────────────────────────────────────────────────────────────────────
# Event log
# ────────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Event log (latest first)")

log = st.session_state["dispatch_log"].copy()
if log.empty:
    st.info("No events yet.")
else:
    st.dataframe(log.iloc[::-1].reset_index(drop=True),
                 use_container_width=True, hide_index=True)
    cdl, clr = st.columns([0.5, 0.5])
    cdl.download_button(
        "⬇️ Export dispatch log (CSV)",
        data=log.to_csv(index=False).encode("utf-8"),
        file_name="dispatch_log.csv",
        mime="text/csv",
        use_container_width=True,
    )
    if clr.button("🗑️ Clear log", use_container_width=True):
        st.session_state["dispatch_log"] = log.iloc[0:0].copy()
        save_log(st.session_state["dispatch_log"])
        st.success("Dispatch log cleared.")
