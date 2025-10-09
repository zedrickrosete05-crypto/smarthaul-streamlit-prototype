import streamlit as st
import pandas as pd

st.title("📦 Dispatch and Monitor")

if "dispatch_df" not in st.session_state:
    st.warning("No plan yet. Go to ‘2 Optimize Routes’.")
    st.stop()

df = st.session_state["dispatch_df"].copy()

# simple controls
veh = st.selectbox("Vehicle", sorted(df["vehicle_id"].unique()))
g = df[df["vehicle_id"]==veh].copy().reset_index(drop=True)

# per-stop status
status_options = ["Planned","En route","Delivered","Skipped","Issue"]
for i in g.index:
    g.loc[i,"status"] = st.selectbox(f"Stop {i+1} – {g.loc[i,'order_id']} (ETA {g.loc[i,'eta']})",
                                     status_options, index=status_options.index(g.loc[i,"status"]))

# save back
st.session_state["dispatch_df"].loc[g.index, "status"] = g["status"].values

# quick rollups
st.divider()
st.subheader("Vehicle summary")
done = (g["status"]=="Delivered").sum()
issues = (g["status"].isin(["Skipped","Issue"])).sum()
c1,c2,c3 = st.columns(3)
c1.metric("Total stops", len(g))
c2.metric("Delivered", done)
c3.metric("Issues", issues)

# optional: trigger replan
if st.button("Replan remaining (placeholder)"):
    st.info("Here you’ll call the optimizer again for stops not Delivered/Skipped.")
