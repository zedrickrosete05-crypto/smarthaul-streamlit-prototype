# pages/1_Upload_Orders.py

import pandas as pd
import streamlit as st

# ---- Optional branding helpers (safe fallback if module is missing) ----------
try:
    from branding import setup_branding, section
except Exception:
    def setup_branding(title: str):  # no-op fallback
        st.set_page_config(page_title=title, layout="wide")
        st.title(title)
    def section(h: str):
        st.subheader(h)

# ------------------------------------------------------------------------------
setup_branding("SmartHaul – Upload Orders")

REQUIRED = ["order_id", "lat", "lon", "tw_start", "tw_end", "service_min"]
section("Upload Orders")

# ---- Template to download -----------------------------------------------------
tmpl = pd.DataFrame([
    {"order_id":"O-1001","lat":10.3157,"lon":123.8854,"tw_start":"08:30","tw_end":"11:00","service_min":7},
    {"order_id":"O-1002","lat":10.3099,"lon":123.9180,"tw_start":"09:00","tw_end":"12:00","service_min":5},
    {"order_id":"O-1003","lat":10.3070,"lon":123.9011,"tw_start":"10:00","tw_end":"13:00","service_min":10},
])
st.download_button(
    "Download CSV template (lat/lon)",
    tmpl.to_csv(index=False).encode(),
    file_name="orders_template.csv",
    mime="text/csv"
)

st.caption("Required columns: order_id, lat, lon, tw_start (HH:MM), tw_end (HH:MM), service_min")
file = st.file_uploader("Upload Orders CSV", type="csv", key="orders_csv")

# ---- Helpers -----------------------------------------------------------------
def _to_minutes(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series.astype(str), format="%H:%M", errors="coerce")
    return (dt.dt.hour * 60 + dt.dt.minute).astype("Int64")

def _validate(df: pd.DataFrame) -> list[str]:
    problems = []
    # missing columns
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        problems.append(f"Missing columns: {missing}")
        return problems

    # nulls
    for col in REQUIRED:
        if df[col].isna().any():
            problems.append(f"Column '{col}' has empty values.")

    # types / ranges
    if not pd.api.types.is_numeric_dtype(df["lat"]):
        problems.append("lat must be numeric.")
    if not pd.api.types.is_numeric_dtype(df["lon"]):
        problems.append("lon must be numeric.")
    if not pd.api.types.is_numeric_dtype(df["service_min"]):
        problems.append("service_min must be numeric (minutes).")

    # time windows validity
    bad_tw = (df["tw_end_min"] < df["tw_start_min"]).fillna(True)
    if bad_tw.any():
        n = int(bad_tw.sum())
        problems.append(f"{n} row(s) have tw_end earlier than tw_start.")

    # duplicates
    dups = df.duplicated(subset=["order_id"], keep=False)
    if dups.any():
        n = int(dups.sum())
        problems.append(f"{n} duplicate order_id value(s) found.")

    return problems

# ---- Main logic --------------------------------------------------------------
if file:
    try:
        raw = pd.read_csv(file)
        st.write("Preview (raw upload):")
        st.dataframe(raw.head(), use_container_width=True)

        # Check required first to avoid KeyErrors
        missing = [c for c in REQUIRED if c not in raw.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()

        # Build cleaned frame
        df = raw.copy()
        df["tw_start_min"] = _to_minutes(df["tw_start"])
        df["tw_end_min"]   = _to_minutes(df["tw_end"])

        # Basic numeric casting
        for col in ["lat", "lon", "service_min"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Validate and report
        issues = _validate(df)
        if issues:
            st.error("Validation issues:\n- " + "\n- ".join(issues))
            st.stop()

        # Success: keep only canonical columns for downstream pages
        cleaned = df[[
            "order_id", "lat", "lon", "service_min", "tw_start_min", "tw_end_min"
        ]].rename(columns={
            "service_min": "service_time_min"
        })

        st.success(f"Loaded {len(cleaned)} orders ✅")
        st.dataframe(cleaned.head(20), use_container_width=True)

        # Store for other pages (e.g., 2_Optimize_Routes.py)
        st.session_state["orders_df"] = cleaned

        # Quick actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear uploaded data"):
                st.session_state.pop("orders_df", None)
                st.rerun()
        with col2:
            st.page_link("pages/2_Optimize_Routes.py", label="Proceed to Optimize Routes →", icon="➡️")

    except Exception as e:
        st.error(f"Could not read CSV: {e}")
else:
    st.info("Tip: download the sample CSV above, then upload it here.")
