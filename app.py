# ===========================================================
# 🌋 SUPT :: GROK Forecast Dashboard (v6.0 Absolute Functional)
# ===========================================================
# Live Data Sources:
#  - INGV (Italy / Campi Flegrei)
#  - USGS (Global)
#  - EMSC (EU backup)
#  - Synthetic Continuity fallback
# ===========================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime as dt
import io

st.set_page_config(page_title="SUPT :: GROK Forecast Dashboard", layout="wide")

# -----------------------------------------------------------
# HELPERS
# -----------------------------------------------------------
def try_request(url, timeout=10):
    """Safe HTTP fetch with unified error return"""
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200 and len(r.text) > 100:
            return r.text
    except Exception:
        pass
    return None


# -----------------------------------------------------------
# FETCH FUNCTION
# -----------------------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def load_seismic_data():
    now = dt.datetime.utcnow()
    end_time = now.strftime("%Y-%m-%dT%H:%M:%S")
    start_time = (now - dt.timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S")

    df, src = pd.DataFrame(), None

    # 1️⃣ INGV – Italy
    ingv_url = (
        f"https://webservices.ingv.it/fdsnws/event/1/query?"
        f"starttime={start_time}&endtime={end_time}"
        f"&minmag=2.0&format=text"
    )
    raw = try_request(ingv_url)
    if raw:
        try:
            df_ingv = pd.read_csv(io.StringIO(raw), delimiter="|", comment="#", low_memory=False)
            cols = [c.strip() for c in df_ingv.columns]
            time_col = next((c for c in cols if "Time" in c), None)
            mag_col = next((c for c in cols if "Mag" in c), None)
            depth_col = next((c for c in cols if "Depth" in c), None)
            if all([time_col, mag_col, depth_col]):
                df = pd.DataFrame({
                    "time": pd.to_datetime(df_ingv[time_col], errors="coerce"),
                    "magnitude": pd.to_numeric(df_ingv[mag_col], errors="coerce"),
                    "depth_km": pd.to_numeric(df_ingv[depth_col], errors="coerce"),
                    "place": "Italy (INGV)"
                }).dropna()
                if not df.empty:
                    return df, "INGV 🇮🇹"
        except Exception:
            pass

    # 2️⃣ USGS – Global
    usgs_url = (
        f"https://earthquake.usgs.gov/fdsnws/event/1/query?"
        f"format=csv&starttime={start_time}&endtime={end_time}&minmagnitude=4"
    )
    raw = try_request(usgs_url)
    if raw:
        try:
            df_usgs = pd.read_csv(io.StringIO(raw))
            if all(c in df_usgs.columns for c in ["time", "mag", "depth", "place"]):
                df = pd.DataFrame({
                    "time": pd.to_datetime(df_usgs["time"], errors="coerce"),
                    "magnitude": pd.to_numeric(df_usgs["mag"], errors="coerce"),
                    "depth_km": pd.to_numeric(df_usgs["depth"], errors="coerce"),
                    "place": df_usgs["place"]
                }).dropna()
                if not df.empty:
                    return df, "USGS 🌍"
        except Exception:
            pass

    # 3️⃣ EMSC – European feed (JSON)
    emsc_url = "https://www.seismicportal.eu/fdsnws/event/1/query?format=json&limit=50"
    raw = try_request(emsc_url)
    if raw:
        try:
            data = requests.get(emsc_url, timeout=10).json()
            if "features" in data:
                feats = data["features"]
                times = [pd.to_datetime(f["properties"]["time"]) for f in feats]
                mags = [f["properties"]["mag"] for f in feats]
                depths = [f["properties"]["depth"] for f in feats]
                locs = [f["properties"]["flynn_region"] for f in feats]
                df = pd.DataFrame({
                    "time": times,
                    "magnitude": mags,
                    "depth_km": depths,
                    "place": locs
                }).dropna()
                if not df.empty:
                    return df, "EMSC 🇪🇺"
        except Exception:
            pass

    # 4️⃣ Synthetic fallback
    now = dt.datetime.utcnow()
    df = pd.DataFrame({
        "time": [now - dt.timedelta(hours=i) for i in range(24)][::-1],
        "magnitude": np.random.uniform(3.5, 5.5, 24),
        "depth_km": np.random.uniform(5, 15, 24),
        "place": ["Synthetic Continuity"] * 24
    })
    return df, "Synthetic Continuity 🧪"


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
st.title("🌋 SUPT :: GROK Forecast Dashboard")
st.caption("v6.0 — Final Functional Build (INGV / USGS / EMSC / Synthetic)")

with st.spinner("Fetching live seismic data..."):
    df, src = load_seismic_data()

badge = {"INGV 🇮🇹":"🟢","USGS 🌍":"🔵","EMSC 🇪🇺":"🟣","Synthetic Continuity 🧪":"🔴"}.get(src,"⚪")
st.markdown(f"### {badge} Active Data Feed: **{src}**")

if df.empty:
    st.error("No data available. Check connectivity.")
    st.stop()

# -----------------------------------------------------------
# METRICS
# -----------------------------------------------------------
EII = np.clip((df["magnitude"].max() * 0.25 + df["depth_km"].mean() * 0.05), 0, 1)
RPAM = "ACTIVE — Collapse Window Initiated" if EII > 0.85 else "STABLE MONITORING"
st.metric("Energetic Instability Index (EII)", f"{EII:.3f}")
st.metric("RPAM Status", RPAM)

# -----------------------------------------------------------
# VISUALS
# -----------------------------------------------------------
st.subheader("Recent Earthquakes (Live)")
st.dataframe(df[["time", "magnitude", "depth_km", "place"]].sort_values("time", ascending=False).head(20))

st.subheader("Magnitude vs Depth (Last 7 Days)")
st.line_chart(df.set_index("time")[["magnitude", "depth_km"]])

# -----------------------------------------------------------
# FOOTER
# -----------------------------------------------------------
st.caption(f"Powered by SUPT — Sheppard’s Universal Proxy Theory :: v6.0 | Source: {src}")
