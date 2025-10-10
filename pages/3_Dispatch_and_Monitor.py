from branding import setup_branding, section
setup_branding("SmartHaul – Dispatch & Monitor")

import streamlit as st
import pandas as pd

section("Dispatch and Monitor")

STATUS_OPTIONS = ["Planned", "En route", "Delivered", "Skipped", "Issue"]

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "status" not in df.columns:
        df["status"] = "Planned"
    for c in ["vehicle_id","order_id","eta","lat","lon","tw_start","tw_end"]:
        if c not in df.columns:
            df[c] = ""
    return df

# Optional: load a plan from CSV
lc, rc = st.columns([2,1])
with lc:
    up = st.file_uploader("Load dispatch CSV (optional)", type="csv")
    if up is not None:
        try:
            st.session_state["dispatch_df"] = ensure_columns(pd.read_csv(up))
            st.success("Loaded dispatch from CSV.")
        except Exception as e:
            st.error(f"Failed to load CSV: {e}")
with rc:
    if "dispatch_df" in st.session_state:
        st.download_button("Download dispatch CSV",
                           data=st.session_state["dispatch_df"].to_csv(index=False).encode("utf-8"),
                           file_name="dispatch_log.csv", mime="text/csv")

if "dispatch_df" not in st.session_state:
    st.warning("No plan yet. Go to ‘Optimize Routes’."); st.stop()

df = ensure_columns(st.session_state["dispatch_df"])

left, right = st.columns([2,1])
with left:  veh = st.selectbox("Vehicle", list(df["vehicle_id"].unique()))
with right: act = st.selectbox("Quick action", ["—","Mark all En route","Mark all Delivered","Reset to Planned"])

gmask = df["vehicle_id"] == veh
g = df[gmask].copy().reset_index(drop=True)

if act == "Mark all En route":   g["status"] = "En route"
elif act == "Mark all Delivered": g["status"] = "Delivered"
elif act == "Reset to Planned":   g["status"] = "Planned"

section(f"Stops for {veh}")
edited = []
for i in g.index:
    c1, c2, c3, c4 = st.columns([3,2,2,3])
    with c1: st.text(f"Stop {i+1} • {g.loc[i,'order_id']}")
    with c2: st.text(f"ETA: {g.loc[i,'eta']}")
    with c3: st.text(f"TW: {g.loc[i,'tw_start']}–{g.loc[i,'tw_end']}")
    with c4:
        edited.append(st.selectbox("Status", STATUS_OPTIONS,
                                   index=STATUS_OPTIONS.index(g.loc[i,"status"]) if g.loc[i,"status"] in STATUS_OPTIONS else 0,
                                   key=f"stat_{veh}_{i}"))
g["status"] = edited
df.loc[gmask, "status"] = g["status"].values
st.session_state["dispatch_df"] = df

st.divider()
section("Vehicle summary (live)")
summary = (
    df.groupby("vehicle_id", sort=False)
      .agg(Stops=("order_id","count"),
           Delivered=("status", lambda s: int((s=="Delivered").sum())),
           EnRoute=("status", lambda s: int((s=="En route").sum())),
           Issues=("status", lambda s: int((s.isin(['Issue','Skipped'])).sum())))
      .reset_index().rename(columns={"vehicle_id":"Vehicle"})
)
st.dataframe(summary, hide_index=True, use_container_width=True)
st.caption("Tip: Use the action dropdown to update all stops quickly. Save or reload dispatch via the CSV controls above.")
