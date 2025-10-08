import streamlit as st

st.title("📊 KPIs & Reports")
if "routes_df" not in st.session_state:
    st.warning("Optimize routes first."); st.stop()

df = st.session_state["routes_df"].copy()
def to_min(s): 
    return None if s=="N/A" else int(s[:2])*60+int(s[3:])
on_time = [(to_min(ea) is not None and to_min(ea) <= to_min(te)) for ea,te in zip(df.eta, df.tw_end)]
on_time_pct = 100*sum(on_time)/len(on_time) if len(on_time) else 0

c1,c2=st.columns(2)
c1.metric("On-time delivery % (plan)", f"{on_time_pct:.1f}%")
c2.metric("ETA MAE (needs actuals)", "—")

st.divider()
st.write("Planned routes")
st.dataframe(df, use_container_width=True)
