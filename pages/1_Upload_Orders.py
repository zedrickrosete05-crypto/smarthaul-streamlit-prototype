# pages/1_Upload_Orders.py
import pandas as pd
import streamlit as st

# ---- Safe branding import (won't crash if branding.py has issues) ------------
try:
    from branding import setup_branding, section
except Exception:
    def setup_branding(title: str):
        st.set_page_config(page_title=title, layout="wide")
        st.title(title)
    def section(h: str):
        st.subheader(h)

setup_branding("SmartHaul – Upload Orders")
section("Upload Orders")

# ---- CSV template ------------------------------------------------------------
REQUIRED_COLS = ["order_id", "lat", "lon", "tw_start", "tw_end", "service_min"]

template = pd.DataFrame([
    {"order_id":"O-1001","lat":10.3157,"lon":123.8854,"tw_start":"08:30","tw_end":"11:00","service_min":7},
    {"order_id":"O-1002","lat":10.3099,"lon":123.9180,"tw_start":"09:00","tw_end":"12:00","service_min":5},
    {"order_id":"O-1003","lat":10.3070,"lon":123.9011,"tw_start":"10:00","tw_end":"13:00","service_min":10},
])
st.download_button(
    "Download CSV template",
    template.to_csv(index=False).encode(),
    file_name="orders_template.csv",
    mime="text/csv",
)

st.caption("Required columns: order_id, lat, lon, tw_start (HH:MM), tw_end (HH:MM), service_min (minutes)")
file = st.file_uploader("Upload Orders CSV", type=["csv"], key="orders_csv")

# ---- Helpers -----------------------------------------------------------------
def _to_minutes(series: pd.Series) -> pd.Series:
    # Parse HH:MM → minutes from midnight; return Int64 with NA for bad rows
    dt = pd.to_datetime(series.astype(str), format="%H:%M", errors="coerce")
    return (dt.dt.hour * 60 + dt.dt.minute).astype("Int64")

def _validate(df: pd.DataFrame) -> list[str]:
    issues = []

    # Missing columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        issues.append(f"Missing columns: {missing}")
        return issues  # bail early

    # Empty cells
    for col in REQUIRED_COLS:
        if df[col].isna().any():
            issues.append(f"Column '{col}' has empty values.")

    # Numeric expectations
    for col in ["lat", "lon", "service_min"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            issues.append(f"Column '{col}' must be numeric.")

    # Time window logic
    bad_parse = df["tw_start_min"].isna() | df["tw_end_min"].isna()
    if bad_parse.any():
        issues.append(f"{int(bad_parse.sum())} row(s) have invalid time format in tw_start/tw_end (expect HH:MM).")

    if ("tw_start_min" in df) and ("tw_end_min" in df):
        bad_order = (df["tw_end_min"] < df["tw_start_min"]) & (~df["tw_end_min"].isna()) & (~df["tw_start_min"].isna())
        if bad_order.any():
            issues.append(f"{int(bad_order.sum())} row(s) have tw_end earlier than tw_start.")

    # Duplicate order IDs
    dups = df.duplicated(subset=["order_id"], keep=False)
    if dups.any():
        issues.append(f"{int(dups.sum())} duplicate order_id value(s) found.")

    return issues

# ---- Main flow ---------------------------------------------------------------
if file:
    try:
        raw = pd.read_csv(file)
        st.markdown("**Preview (raw upload):**")
        st.dataframe(raw.head(), use_container_width=True)

        # Early missing check to avoid KeyError later
        missing = [c for c in REQUIRED_COLS if c not in raw.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()

        df = raw.copy()

        # Cast numerics (coerce errors to NaN so validator can flag them)
        for col in ["lat", "lon", "service_min"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Parse HH:MM to minutes
        df["tw_start_min"] = _to_minutes(df["tw_start"])
        df["tw_end_min"]   = _to_minutes(df["tw_end"])

        # Validate
        problems = _validate(df)
        if problems:
            st.error("Validation issues:\n- " + "\n- ".join(problems))
            st.stop()

        # Keep canonical columns for downstream pages
        cleaned = df[[
            "order_id", "lat", "lon", "service_min", "tw_start_min", "tw_end_min"
        ]].rename(columns={"service_min": "service_time_min"})

        st.success(f"Loaded {len(cleaned)} orders ✅")
        st.dataframe(cleaned.head(20), use_container_width=True)

        # Store for other pages
        st.session_state["orders_df"] = cleaned

        # Actions
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Clear uploaded data"):
                st.session_state.pop("orders_df", None)
                st.rerun()
        with c2:
            st.page_link("pages/2_Optimize_Routes.py", label="Proceed to Optimize Routes →", icon="➡️")

    except Exception as e:
        st.error(f"Could not read CSV: {e}")
else:
    st.info("Tip: download the sample CSV above, then upload it here.")
