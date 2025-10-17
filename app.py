import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime as dt
import io

st.set_page_config(page_title="SUPT :: GROK Forecast Dashboard", layout="centered")

# ---------------------------------------------------------------
# FINAL FUNCTIONAL BUILD (v5.0) - Guaranteed to run
# ---------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=900)
def load_seismic_data():
    """
    1️⃣ Try INGV (Campi Flegrei)
    2️⃣ Fallback to USGS
    3️⃣ Always return valid DataFrame
    """

    df = pd.DataFrame()
    source_label = "Unknown"
    now = dt.datetime.utcnow()
    end_time = now.strftime("%Y-%m-%dT%H:%M:%S")
    start_time = (now - dt.timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S")

    # ------------- INGV FEED -------------
    try:
        ingv_url = (
            f"https://webservices.ingv.it/fdsnws/event/1/query?"
            f"starttime={start_time}&endtime={end_time}"
            f"&latmin=40.7&latmax=40.9&lonmin=14.0&lonmax=14.3"
            f"&minmag=0&format=text"
        )
        resp = requests.get(ingv_url, timeout=10)
        resp.raise_for_status()

        df_ingv = pd.read_csv(io.StringIO(resp.text), delimiter="|", comment="#", low_memory=False)
        df_ingv.columns = [c.strip() for c in df_ingv.columns]

        # try to locate columns automatically
        time_col = next((c for c in df_ingv.columns if "Time" in c), None)
        mag_col = next((c for c in df_ingv.columns if "Mag" in c), None)
        depth_col = next((c for c in df_ingv.columns if "Depth" in c), None)

        if not all([time_col, mag_col, depth_col]):
            raise ValueError("INGV feed missing required columns")

        df = pd.DataFrame({
            "time": pd.to_datetime(df_ingv[time_col], errors="coerce"),
            "magnitude": pd.to_numeric(df_ingv[mag_col], errors="coerce"),
            "depth_km": pd.to_numeric(df_ingv[depth_col], errors="coerce")
        }).dropna()

        if len(df) > 0:
            return df, "INGV Live"

        raise ValueError("No data returned from INGV")

    except Exception as e:
        st.warning(f"⚠️ INGV feed unavailable ({e}). Trying USGS fallback...")

    # ------------- USGS FEED -------------
    try:
        usgs_url = (
            f"https://earthquake.usgs.gov/fdsnws/event/1/query?"
            f"format=csv&starttime={start_time}&endtime={end_time}"
            f"&minlatitude=40.6&maxlatitude=41.0&minlongitude=13.9&maxlongitude=14.4"
        )
        resp = requests.get(usgs_url, timeout=10)
        resp.raise_for_status()
        df_usgs = pd.read_csv(io.StringIO(resp.text))
        if all(c in df_usgs.columns for c in ["time", "mag", "depth"]):
            df = pd.DataFrame({
                "time": pd.to_datetime(df_usgs["time"], errors="coerce"),
                "magnitude": pd.to_numeric(df_usgs["mag"], errors="coerce"),
                "depth_km": pd.to_numeric(df_usgs["depth"], errors="coerce")
            }).dropna()

        if len(df) > 0:
            return df, "USGS Fallback"

        raise ValueError("No events returned from USGS")

    except Exception as e:
        st.warning(f"⚠️ USGS fallback failed ({e}). Using synthetic dataset...")

    # ------------- SYNTHETIC -------------
    now = dt.datetime.utcnow()
    df = pd.DataFrame({
        "time": [now - dt.timedelta(hours=i) for i in range(24)][::-1],
        "magnitude": np.random.uniform(0.5, 2.0, 24),
        "depth_km": np.random.uniform(1.0, 4.0, 24)
    })
    return df, "Synthetic Continuity"

# ---------------------------------------------------------------
# MAIN DASHBOARD LOGIC
# ---------------------------------------------------------------

st.title("🌋 SUPT :: GROK Forecast Dashboard")
st.caption("v5.0 — Functional Live Build (INGV / USGS / Synthetic)")

with st.spinner("Fetching latest seismic data..."):
    df, data_source = load_seismic_data()

# Badge colors
badge_colors = {
    "INGV Live": "🟢",
    "USGS Fallback": "🔵",
    "Synthetic Continuity": "🔴"
}
badge = badge_colors.get(data_source, "⚪")

st.markdown(f"### {badge} Active Data Feed: `{data_source}`")

if df.empty:
    st.error("No seismic data available — even synthetic fallback failed.")
    st.stop()

# ---------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------

EII = np.clip((df["magnitude"].max() * 0.25 + df["depth_km"].mean() * 0.1), 0, 1)
RPAM_STATUS = "ACTIVE – Collapse Window Initiated" if EII > 0.85 else "MONITORING"

st.metric("Energetic Instability Index (EII)", f"{EII:.3f}")
st.metric("RPAM Status", RPAM_STATUS)

# ---------------------------------------------------------------
# VISUALS
# ---------------------------------------------------------------

st.subheader("24h Harmonic Drift — Depth vs Magnitude")

df_last24 = df[df["time"] > dt.datetime.utcnow() - dt.timedelta(hours=24)]

st.line_chart(df_last24.set_index("time")[["magnitude", "depth_km"]])

st.caption("Powered by SUPT — Sheppard’s Universal Proxy Theory :: v5.0 Functional Core")
