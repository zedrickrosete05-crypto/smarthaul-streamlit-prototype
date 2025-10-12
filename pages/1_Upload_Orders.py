# pages/1_Upload_Orders.py
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SmartHaul – Upload Orders", layout="wide")
st.title("Upload Orders")

REQUIRED_COLS = ["order_id", "tw_start", "tw_end", "service_min"]
OPTIONAL_COLS = ["lat", "lon", "place"]  # at least (lat & lon) OR place must be present

# Template with 'place'
template = pd.DataFrame([
    {"order_id":"O-1001","place":"JY Square, Cebu City","tw_start":"08:30","tw_end":"11:00","service_min":7},
    {"order_id":"O-1002","place":"Cebu IT Park","tw_start":"09:00","tw_end":"12:00","service_min":5},
    {"order_id":"O-1003","place":"SM City Cebu","tw_start":"10:00","tw_end":"13:00","service_min":10},
])
st.download_button("Download CSV template (with place)", template.to_csv(index=False).encode(),
                   file_name="orders_template_places.csv", mime="text/csv")

st.caption("Required: order_id, tw_start (HH:MM), tw_end (HH:MM), service_min (min). Provide either lat+lon or place (address/name).")
file = st.file_uploader("Upload Orders CSV", type=["csv"], key="orders_csv")

def _to_minutes(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s.astype(str), format="%H:%M", errors="coerce")
    return (dt.dt.hour * 60 + dt.dt.minute).astype("Int64")

if file:
    try:
        raw = pd.read_csv(file)
        st.markdown("**Preview (raw upload):**")
        st.dataframe(raw.head(), use_container_width=True)

        # Required columns present?
        miss_req = [c for c in REQUIRED_COLS if c not in raw.columns]
        if miss_req:
            st.error(f"Missing required columns: {miss_req}")
            st.stop()

        df = raw.copy()

        # Parse/validate times + service_min
        df["service_min"] = pd.to_numeric(df["service_min"], errors="coerce")
        df["tw_start_min"] = _to_minutes(df["tw_start"])
        df["tw_end_min"]   = _to_minutes(df["tw_end"])

        problems = []
        if df["service_min"].isna().any(): problems.append("service_min must be numeric.")
        bad_parse = df["tw_start_min"].isna() | df["tw_end_min"].isna()
        if bad_parse.any(): problems.append(f"{int(bad_parse.sum())} row(s) have invalid HH:MM in tw_start/tw_end.")
        bad_order = (df["tw_end_min"] < df["tw_start_min"]) & (~df["tw_end_min"].isna()) & (~df["tw_start_min"].isna())
        if bad_order.any(): problems.append(f"{int(bad_order.sum())} row(s) have tw_end earlier than tw_start.")
        dups = df.duplicated(subset=["order_id"], keep=False)
        if dups.any(): problems.append(f"{int(dups.sum())} duplicate order_id value(s) found.")

        # Geocode if needed (when lat/lon missing and place exists)
        if ("place" in df.columns) and (("lat" not in df.columns) or ("lon" not in df.columns) or df[["lat","lon"]].isna().any().any()):
            from utils.geocode import geocode_place
            st.info("Geocoding rows with missing coordinates from 'place'…")
            lat_list, lon_list, misses = [], [], 0
            for _, r in df.iterrows():
                lat = r.get("lat", float("nan"))
                lon = r.get("lon", float("nan"))
                if pd.isna(lat) or pd.isna(lon):
                    coords = geocode_place(str(r.get("place", "")))
                    if coords:
                        lat, lon = coords
                    else:
                        misses += 1
                        lat, lon = float("nan"), float("nan")
                lat_list.append(lat); lon_list.append(lon)
            df["lat"] = pd.to_numeric(lat_list, errors="coerce")
            df["lon"] = pd.to_numeric(lon_list, errors="coerce")
            if misses:
                problems.append(f"{misses} row(s) could not be geocoded from 'place'.")

        # Final coordinate check
        if "lat" not in df.columns or "lon" not in df.columns:
            problems.append("Provide either lat+lon columns or a 'place' column to geocode.")
        elif df["lat"].isna().any() or df["lon"].isna().any():
            problems.append("Some rows are missing valid coordinates (lat/lon).")

        if problems:
            st.error("Validation issues:\n- " + "\n- ".join(problems))
            st.stop()

        # Canonical output for downstream pages
        cleaned = df[["order_id", "lat", "lon", "service_min", "tw_start_min", "tw_end_min"]]
        cleaned = cleaned.rename(columns={"service_min": "service_time_min"})
        st.success(f"Loaded {len(cleaned)} orders ✅")
        st.dataframe(cleaned.head(20), use_container_width=True)

        st.session_state["orders_df"] = cleaned

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Clear uploaded data"):
                st.session_state.pop("orders_df", None); st.rerun()
        with c2:
            st.page_link("pages/2_Optimize_Routes.py", label="Proceed to Optimize Routes →", icon="➡️")

    except Exception as e:
        st.error(f"Could not read CSV: {e}")
else:
    st.info("Tip: use the template above or upload your CSV. You can provide a **place** instead of lat/lon.")
