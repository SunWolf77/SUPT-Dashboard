# ===============================================================
# 🌞 SunWolf-SUPT v3.6 — GROK-Integrated Live Forecast Dashboard
# Combines INGV + USGS + NOAA + SUPT metrics (ψₛ, EII, RPAM)
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime as dt
import io
import plotly.graph_objs as go

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
API_TIMEOUT = 10
NOAA_GEOMAG_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
LOCAL_FALLBACK = "events_6.csv"  # For local debug use
DEFAULT_SOLAR = {
    "psi_s": 0.72,
    "solar_speed": 688,
    "C_flare": 0.99,
    "M_flare": 0.55,
    "X_flare": 0.15
}

# ---------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------
def compute_eii(md_max, md_mean, shallow_ratio, psi_s):
    """Energetic Instability Index (SUPT core metric)"""
    return np.clip((md_max * 0.2 + md_mean * 0.15 + shallow_ratio * 0.4 + psi_s * 0.25), 0, 1)

def classify_phase(EII):
    if EII >= 0.85:
        return "ACTIVE - Collapse Window Initiated"
    elif EII >= 0.6:
        return "ELEVATED - Pressure Coupling Phase"
    else:
        return "MONITORING"

# ---------------------------------------------------------------
# FETCH NOAA GEOMAGNETIC KP INDEX
# ---------------------------------------------------------------
@st.cache_data(ttl=600)
def fetch_geomag_data():
    try:
        res = requests.get(NOAA_GEOMAG_URL, timeout=API_TIMEOUT)
        res.raise_for_status()
        data = res.json()
        latest = data[-1]
        kp = float(latest[1])
        return {
            "kp_index": kp,
            "time_tag": latest[0],
            "geomag_alert": "HIGH" if kp >= 5 else "LOW"
        }
    except Exception as e:
        st.warning(f"NOAA Geomagnetic fetch failed: {e}")
        return {"kp_index": 0.0, "time_tag": "Fallback", "geomag_alert": "LOW"}

# ---------------------------------------------------------------
# FETCH INGV (LIVE) OR FALLBACK TO USGS OR LOCAL CSV
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=900)
def load_seismic_data():
    try:
        end_time = dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
        start_time = (dt.datetime.utcnow() - dt.timedelta(days=7)).strftime('%Y-%m-%dT%H:%M:%S')
        ingv_url = (
            f"https://webservices.ingv.it/fdsnws/event/1/query?"
            f"starttime={start_time}&endtime={end_time}"
            f"&latmin=40.7&latmax=40.9&lonmin=14.0&lonmax=14.3"
            f"&format=text"
        )
        res = requests.get(ingv_url, timeout=API_TIMEOUT)
        res.raise_for_status()
        df = pd.read_csv(io.StringIO(res.text), delimiter="|", comment="#")
        df.columns = [c.strip() for c in df.columns]

        # Detect time/mag/depth dynamically
        time_col = next((c for c in df.columns if "time" in c.lower()), None)
        mag_col = next((c for c in df.columns if "mag" in c.lower()), None)
        dep_col = next((c for c in df.columns if "depth" in c.lower()), None)

        if not time_col or not mag_col or not dep_col:
            raise KeyError("Missing expected columns in INGV data")

        df["time"] = pd.to_datetime(df[time_col], errors="coerce")
        df["magnitude"] = pd.to_numeric(df[mag_col], errors="coerce")
        df["depth_km"] = pd.to_numeric(df[dep_col], errors="coerce")
        df = df.dropna(subset=["time", "magnitude", "depth_km"])
        return df[df["time"] > dt.datetime.utcnow() - dt.timedelta(days=7)]
    except Exception as e:
        st.warning(f"INGV API fetch failed: {e}. Trying USGS fallback...")
        try:
            usgs_url = (
                f"https://earthquake.usgs.gov/fdsnws/event/1/query?"
                f"format=csv&starttime={start_time}&endtime={end_time}"
                f"&minlatitude=40.7&maxlatitude=40.9&minlongitude=14.0&maxlongitude=14.3"
            )
            r2 = requests.get(usgs_url, timeout=API_TIMEOUT)
            r2.raise_for_status()
            df = pd.read_csv(io.StringIO(r2.text))
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
            df["magnitude"] = pd.to_numeric(df["mag"], errors="coerce")
            df["depth_km"] = pd.to_numeric(df["depth"], errors="coerce")
            return df[df["time"] > dt.datetime.utcnow() - dt.timedelta(days=7)]
        except Exception as e:
            st.warning(f"USGS fallback failed: {e}. Using local CSV sample.")
            try:
                df_local = pd.read_csv(LOCAL_FALLBACK)
                df_local["time"] = pd.to_datetime(df_local["Time"], errors="coerce")
                df_local["magnitude"] = df_local.get("MD", 0.5)
                df_local["depth_km"] = df_local.get("Depth", 1.8)
                return df_local
            except Exception:
                st.error("No valid data source available.")
                return pd.DataFrame()

# ---------------------------------------------------------------
# MAIN DASHBOARD
# ---------------------------------------------------------------
st.set_page_config(layout="centered", page_title="SUPT :: GROK Forecast")
st.title("🌋 SUPT :: GROK Forecast Dashboard")
st.caption("Campi Flegrei Risk & Energetic Instability Monitor :: v3.6")

# ---------------------------------------------------------------
# DATA FETCH
# ---------------------------------------------------------------
with st.spinner("Fetching live seismic data..."):
    df = load_seismic_data()
if df.empty:
    st.error("No seismic data loaded — check live feed or fallback.")
    st.stop()

with st.spinner("Fetching geomagnetic data..."):
    geomag = fetch_geomag_data()

# ---------------------------------------------------------------
# CALCULATIONS
# ---------------------------------------------------------------
md_max = df["magnitude"].max()
md_mean = df["magnitude"].mean()
shallow_ratio = (df["depth_km"] < 2.5).mean()
psi_s = st.sidebar.slider("Solar Pressure Proxy (ψₛ)", 0.0, 1.0, DEFAULT_SOLAR["psi_s"])
EII = compute_eii(md_max, md_mean, shallow_ratio, psi_s)
RPAM_STATUS = classify_phase(EII)
COLLAPSE_WINDOW = "Q1 2026" if RPAM_STATUS.startswith("ACTIVE") else "N/A"

# ---------------------------------------------------------------
# DISPLAY METRICS
# ---------------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("EII", f"{EII:.3f}")
col2.metric("RPAM Status", RPAM_STATUS)
col3.metric("Collapse Window", COLLAPSE_WINDOW)

st.markdown(f"**Mean Depth:** {df['depth_km'].mean():.2f} km  |  **Mean Mag:** {df['magnitude'].mean():.2f}")
st.markdown(f"**Geomagnetic Kp:** {geomag['kp_index']} ({geomag['geomag_alert']})  |  Last Update: {geomag['time_tag']}")

# ---------------------------------------------------------------
# HISTOGRAM
# ---------------------------------------------------------------
st.subheader("Depth Distribution (Past 7 Days)")
fig = go.Figure(data=[go.Histogram(x=df["depth_km"], nbinsx=15, marker_color="#FFA726")])
fig.update_layout(xaxis_title="Depth (km)", yaxis_title="Quake Count", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------
# SOLAR INPUTS
# ---------------------------------------------------------------
st.sidebar.header("Solar Activity")
st.sidebar.number_input("Solar Wind Speed (km/s)", value=DEFAULT_SOLAR["solar_speed"])
st.sidebar.slider("C-Flare Probability", 0.0, 1.0, DEFAULT_SOLAR["C_flare"])
st.sidebar.slider("M-Flare Probability", 0.0, 1.0, DEFAULT_SOLAR["M_flare"])
st.sidebar.slider("X-Flare Probability", 0.0, 1.0, DEFAULT_SOLAR["X_flare"])

# ---------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------
st.caption("Powered by SUPT ψ-Fold • NOAA • INGV • USGS")
st.caption("Harmonically Tuned for GROK • v3.6 Live Integration Build")
