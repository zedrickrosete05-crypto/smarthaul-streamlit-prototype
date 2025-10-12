# pages/3_Dispatch_and_Monitor.py
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SmartHaul – Dispatch & Monitor", layout="wide")
st.title("Dispatch & Monitor")

STATUS_OPTIONS = ["Planned", "En route", "Delivered", "Skipped", "Issue"]

REQUIRED_COLS = [
    "vehicle_id", "order_id", "eta", "lat", "lon", "tw_start", "tw_end", "status"
]

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the table has all required columns and sane defaults."""
    df = df.copy()
    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = "" if c != "status" else "Planned"
    # Normalize types a bit
    if "status" in df:
        df["status"] = df["status"].replace({None: "Planned"}).astype(str)
    return df

# ------------------------------------------------------------------------------
# Bootstrap dispatch state from plan if needed
# ------------------------------------------------------------------------------
if "dispatch_df" not in st.session_state:
    if "routes_df" in st.session_state and st.session_state["routes_df"] is not None:
        plan = st.session_state["routes_df"].copy()
        # routes_df expected: vehicle_id, order_id, eta, tw_start, tw_end, lat, lon, ...
        base_cols = {c for c in plan.columns}
        needed = {"vehicle_id","order_id","eta","tw_start","tw_end","lat","lon"}
        if needed.issubset(base_cols):
            boot = plan[list(needed)].copy()
            boot["status"] = "Planned"
            st.session_state["dispatch_df"] = ensure_columns(boot)
        else:
            st.warning("No dispatch data yet. Run **Optimize Routes** first.")
            st.page_link("pages/2_Optimize_Routes.py", label="← Optimize Routes", icon="⬅️")
            st.stop()
    else:
        st.warning("No dispatch data yet. Run **Optimize Routes** first.")
        st.page_link("pages/2_Optimize_Routes.py", label="← Optimize Routes", icon="⬅️")
        st.stop()

# Work copy
df = ensure_columns(st.session_state["dispatch_df"])

# ------------------------------------------------------------------------------
# Optional: load/save CSV
# ------------------------------------------------------------------------------
lc, rc = st.columns([2, 1], vertical_alignment="bottom")
with lc:
    up = st.file_uploader("Load dispatch CSV (optional)", type="csv", key="dispatch_csv")
    if up is not None:
        try:
            loaded = pd.read_csv(up)
            st.session_state["dispatch_df"] = ensure_columns(loaded)
            df = st.session_state["dispatch_df"]
            st.success("Loaded dispatch from CSV.")
        except Exception as e:
            st.error(f"Failed to load CSV: {e}")

with rc:
    st.download_button(
        "Download dispatch CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="dispatch_log.csv",
        mime="text/csv",
        use_container_width=True
    )

# ------------------------------------------------------------------------------
# Controls
# ------------------------------------------------------------------------------
vehicles = sorted([v for v in df["vehicle_id"].astype(str).unique() if v])
if not vehicles:
    st.warning("No vehicles found in dispatch data.")
    st.stop()

left, right = st.columns([2, 1], vertical_alignment="bottom")
with left:
    veh = st.selectbox("Vehicle", vehicles, index=0, key="veh_select")

with right:
    act = st.selectbox(
        "Quick action",
        ["—", "Mark all En route", "Mark all Delivered", "Reset to Planned", "Mark next stop Delivered"],
        index=0,
        key="veh_action"
    )

mask = df["vehicle_id"].astype(str) == veh
g = df[mask].copy().reset_index(drop=True)

# Apply quick actions
if act == "Mark all En route":
    df.loc[mask, "status"] = "En route"
elif act == "Mark all Delivered":
    df.loc[mask, "status"] = "Delivered"
elif act == "Reset to Planned":
    df.loc[mask, "status"] = "Planned"
elif act == "Mark next stop Delivered":
    # Find first non-delivered row in order of ETA if possible
    try:
        tmp = g.copy()
        # Sort by ETA if parseable (HH:MM), else leave as-is
        def _eta_key(x):
            try:
                h, m = str(x).split(":")
                return int(h) * 60 + int(m)
            except Exception:
                return 10**9  # push non-parseable to end
        tmp["__eta_key"] = tmp["eta"].map(_eta_key)
        nxt = tmp[tmp["status"] != "Delivered"].sort_values("__eta_key").head(1).index
        if len(nxt):
            df.loc[mask].iloc[nxt, df.columns.get_loc("status")] = "Delivered"
    except Exception:
        pass

# ------------------------------------------------------------------------------
# Editable table for the selected vehicle
# ------------------------------------------------------------------------------
st.markdown(f"### Stops for {veh}")

# Show a lighter view with editable Status only (fast and simple),
# but keep context columns visible.
view_cols = ["order_id", "eta", "tw_start", "tw_end", "status", "lat", "lon"]
view_cols = [c for c in view_cols if c in df.columns]
veh_view = df.loc[mask, view_cols].reset_index(drop=True)

# Use data_editor for better UX (Streamlit 1.37+)
try:
    edited_view = st.data_editor(
        veh_view,
        use_container_width=True,
        hide_index=True,
        column_config={
            "status": st.column_config.SelectboxColumn(
                "Status", options=STATUS_OPTIONS, required=True
            )
        }
    )
    # Write back only the edited 'status' column to the underlying df
    df.loc[mask, "status"] = edited_view["status"].values
except Exception:
    # Fallback to per-row selectboxes if data_editor is unavailable
    st.info("Interactive editor unavailable; using basic controls.")
    edited = []
    for i, row in veh_view.iterrows():
        c1, c2, c3, c4 = st.columns([3, 2, 2, 3])
        with c1:
            st.text(f"Stop {i+1} • {row.get('order_id','')}")
        with c2:
            st.text(f"ETA: {row.get('eta','')}")
        with c3:
            st.text(f"TW: {row.get('tw_start','')}–{row.get('tw_end','')}")
        with c4:
            cur = row.get("status", "Planned")
            edited.append(
                st.selectbox(
                    "Status",
                    STATUS_OPTIONS,
                    index=STATUS_OPTIONS.index(cur) if cur in STATUS_OPTIONS else 0,
                    key=f"stat_{veh}_{i}"
                )
            )
    df.loc[mask, "status"] = edited

# Persist state
st.session_state["dispatch_df"] = ensure_columns(df)

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
st.divider()
st.markdown("### Vehicle summary (live)")

summary = (
    df.groupby("vehicle_id", sort=False)
      .agg(Stops=("order_id", "count"),
           Delivered=("status", lambda s: int((s == "Delivered").sum())),
           EnRoute=("status", lambda s: int((s == "En route").sum())),
           Issues=("status", lambda s: int((s.isin(['Issue', 'Skipped'])).sum())))
      .reset_index()
      .rename(columns={"vehicle_id": "Vehicle"})
      .sort_values("Vehicle", kind="stable")
)

st.dataframe(summary, hide_index=True, use_container_width=True)
st.caption("Tip: Use the quick actions or edit Status directly in the table. Save or reload via the CSV controls above.")
