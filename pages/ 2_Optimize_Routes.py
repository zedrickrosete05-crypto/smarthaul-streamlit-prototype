import streamlit as st, pandas as pd
from datetime import datetime, timedelta
import math

st.title("🧭 Optimize Routes")

# helpers
def haversine_km(a,b,c,d):
    R=6371.0088
    p1,p2=math.radians(a),math.radians(c)
    dphi=math.radians(c-a); dlmb=math.radians(d-b)
    x=math.sin(dphi/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2*R*math.asin(math.sqrt(x))
def travel_min(a,b,c,d,speed=25.0): return (haversine_km(a,b,c,d)/max(speed,5))*60
def t(s): return datetime.strptime(s,"%H:%M")

def greedy(df, depot=(14.5995,120.9842), speed=25.0, max_stops=10):
    orders=df.copy().reset_index(drop=True); orders["done"]=False
    routes=[]; vid=1
    while not orders.done.all():
        lat,lon=depot; now=t("08:00"); route=[]
        while True:
            cand=[]
            for i,row in orders[~orders.done].iterrows():
                drive=travel_min(lat,lon,row.lat,row.lon,speed)
                arr=now+timedelta(minutes=drive)
                start=max(arr,t(row.tw_start))
                if start<=t(row.tw_end): cand.append((i,drive,start))
            if not cand or len(route)>=max_stops: break
            i,drive,start=sorted(cand,key=lambda x:x[1])[0]
            r=orders.loc[i]; leave=start+timedelta(minutes=r.service_min)
            route.append(dict(vehicle_id=f"V{vid}",order_id=r.order_id,lat=r.lat,lon=r.lon,
                              eta=start.strftime("%H:%M"),tw_start=r.tw_start,tw_end=r.tw_end))
            orders.at[i,"done"]=True; lat,lon=r.lat,r.lon; now=leave
        if route: routes.append(pd.DataFrame(route)); vid+=1
        else:
            i=orders[~orders.done].index[0]; r=orders.loc[i]
            routes.append(pd.DataFrame([dict(vehicle_id=f"V{vid}",order_id=r.order_id,lat=r.lat,lon=r.lon,
                                             eta="N/A",tw_start=r.tw_start,tw_end=r.tw_end)]))
            orders.at[i,"done"]=True; vid+=1
    return pd.concat(routes, ignore_index=True)

# require data
if "orders_df" not in st.session_state:
    st.warning("Upload orders first on the Upload page."); st.stop()

speed = st.slider("Average speed (km/h)", 15, 45, 25, 1)
maxst = st.slider("Max stops per route", 5, 20, 10, 1)

if st.button("Compute routes"):
    df = greedy(st.session_state["orders_df"], speed=speed, max_stops=maxst)
    df["alert"] = df.apply(lambda r: "Late risk" if r["eta"]!="N/A" and r["eta"]>r["tw_end"] else "", axis=1)
    st.session_state["routes_df"] = df
    st.success(f"Computed {df['vehicle_id'].nunique()} route(s) for {len(df)} stops.")

if "routes_df" in st.session_state:
    df = st.session_state["routes_df"]
    st.dataframe(df, use_container_width=True)
    try:
    import pydeck as pdk

    # Work on a safe copy
    df = st.session_state["routes_df"].copy()

    # 1) Build an ordered path for each vehicle (list of {lon, lat})
    # Assumes df rows are already in stop order for each vehicle_id
    def path_from_group(g):
        return [{"lon": float(r.lon), "lat": float(r.lat)} for r in g.itertuples(index=False)]

    paths = (
        df.groupby("vehicle_id", sort=False)
          .apply(path_from_group)
          .reset_index(name="path")
    )

    # 2) Path layer (draws the line)
    path_layer = pdk.Layer(
        "PathLayer",
        data=paths,
        get_path="path",
        get_width=3,
        width_min_pixels=2,
        pickable=False,
    )

    # 3) Point layer (draws the stops)
    point_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position='[lon, lat]',
        get_radius=40,
        pickable=True,
    )

    # 4) Initial view centered on your data
    view = pdk.ViewState(
        latitude=float(df.lat.mean()),
        longitude=float(df.lon.mean()),
        zoom=11,
    )

    # 5) Render both layers
    st.pydeck_chart(pdk.Deck(layers=[path_layer, point_layer], initial_view_state=view))

except Exception as e:
    st.info(f"Map rendering needs pydeck. If installed, error was: {e}")

import io
if "routes_df" in st.session_state:
    df = st.session_state["routes_df"]
    st.download_button(
        "Download planned routes (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="routes_plan.csv",
        mime="text/csv",
    )
