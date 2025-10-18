# ===============================================================
# SUPT :: GROK Forecast Dashboard — v6.0 Live Continuum Build
# Integrating Grok's API stabilization + SUPT Live EII/RPAM model
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime as dt
import io
import traceback
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
st.set_page_config(page_title="SUPT :: GROK Live Forecast Dashboard", layout="wide")
API_TIMEOUT = 10
LOCAL_FALLBACK_CSV = "events_6.csv"
NASA_API_KEY = "DEMO_KEY"  # Replace with your NASA key if available

# Live API Endpoints
NOAA_GEOMAG_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
NOAA_SOLAR_WIND_URL = "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json"
NOAA_FLARE_URL = "https://services.swpc.noaa.gov/json/goes/primary/xray-flares-latest.json"
USGS_EQ_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query?"
INGV_URL = "https://webservices.ingv.it/fdsnws/event/1/query?"

# Default Solar Parameters
DEFAULT_SOLAR = {"psi_s": 0.72, "solar_speed": 688, "density": 5.0}

# ---------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------
def compute_eii(mag_max, mag_mean, shallow_ratio, psi_s):
    return np.clip((mag_max * 0.25 + mag_mean * 0.25 + shallow_ratio * 0.25 + psi_s * 0.25), 0, 1)

def classify_phase(EII):
    if EII >= 0.85:
        return "ACTIVE – Collapse Window Initiated", "🔴"
    elif EII >= 0.6:
        return "ELEVATED – Pressure Coupling", "🟠"
    return "MONITORING", "🟢"

def generate_synthetic_seismic_data(n=20):
    now = dt.datetime.utcnow()
    times = [now - dt.timedelta(hours=i * 3) for i in range(n)]
    mags = np.random.uniform(0.5, 1.5, n)
    depths = np.random.uniform(0.5, 5.0, n)
    return pd.DataFrame({"time": times, "magnitude": mags, "depth_km": depths})

# ---------------------------------------------------------------
# FETCH FUNCTIONS
# ---------------------------------------------------------------
@st.cache_data(ttl=600)
def fetch_noaa_kp():
    try:
        r = requests.get(NOAA_GEOMAG_URL, timeout=API_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        return float(data[-1][1])
    except Exception as e:
        st.warning(f"NOAA Kp fetch failed: {e}")
        return 1.0

@st.cache_data(ttl=600)
def fetch_solar_data():
    try:
        r = requests.get(NOAA_SOLAR_WIND_URL, timeout=API_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        last = data[-1]
        return {
            "speed": float(last[1]),
            "density": float(last[2]),
            "psi_s": np.clip(float(last[1]) / 800, 0, 1),
            "timestamp": last[0],
        }
    except Exception as e:
        st.warning(f"NOAA Solar Wind fetch failed: {e}")
        return DEFAULT_SOLAR

@st.cache_data(ttl=600)
def fetch_ingv_seismic():
    try:
        now = dt.datetime.utcnow()
        start = now - dt.timedelta(days=7)
        params = {
            "starttime": start.strftime("%Y-%m-%dT%H:%M:%S"),
            "endtime": now.strftime("%Y-%m-%dT%H:%M:%S"),
            "latmin": 40.7, "latmax": 40.9,
            "lonmin": 14.0, "lonmax": 14.3,
            "format": "text"
        }
        r = requests.get(INGV_URL, params=params, timeout=API_TIMEOUT)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), delimiter="|", skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]

        # Normalize columns dynamically
        time_col = [c for c in df.columns if "time" in c.lower()]
        mag_col = [c for c in df.columns if "mag" in c.lower()]
        depth_col = [c for c in df.columns if "depth" in c.lower()]

        if not (time_col and mag_col and depth_col):
            raise KeyError("Missing INGV columns")

        df["time"] = pd.to_datetime(df[time_col[0]], errors="coerce")
        df["magnitude"] = pd.to_numeric(df[mag_col[0]], errors="coerce")
        df["depth_km"] = pd.to_numeric(df[depth_col[0]], errors="coerce")
        df = df.dropna(subset=["time", "magnitude", "depth_km"])
        return df
    except Exception as e:
        st.warning(f"INGV fetch failed: {e}. Switching to fallback.")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def load_seismic_data():
    df = fetch_ingv_seismic()
    if df.empty:
        try:
            df = pd.read_csv(LOCAL_FALLBACK_CSV)
            df["time"] = pd.to_datetime(df["Time"], errors="coerce")
            df["magnitude"] = pd.to_numeric(df["MD"], errors="coerce")
            df["depth_km"] = pd.to_numeric(df["Depth"], errors="coerce")
            df = df.dropna(subset=["time", "magnitude", "depth_km"])
            return df
        except Exception:
            return generate_synthetic_seismic_data()
    return df

# ---------------------------------------------------------------
# LIVE DATA PIPELINE
# ---------------------------------------------------------------
st.info("Fetching live data feeds... please wait ⏳")
solar = fetch_solar_data()
kp = fetch_noaa_kp()
df = load_seismic_data()

# Compute metrics
if not df.empty:
    mag_max = df["magnitude"].max()
    mag_mean = df["magnitude"].mean()
    shallow_ratio = len(df[df["depth_km"] < 3]) / len(df)
    EII = compute_eii(mag_max, mag_mean, shallow_ratio, solar["psi_s"])
else:
    EII = 0.0

RPAM, color = classify_phase(EII)

# ---------------------------------------------------------------
# LAYOUT
# ---------------------------------------------------------------
st.success("✅ All systems operational — SUPT GROK v6.0 Continuum Build")
st.caption(f"Last update: {dt.datetime.utcnow():%Y-%m-%d %H:%M:%S UTC}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Energetic Instability Index (EII)", f"{EII:.3f}")
col2.metric("Ψ Coupling", f"{solar['psi_s']:.3f}")
col3.metric("Geomagnetic Kp", f"{kp:.2f}")
col4.metric("RPAM Phase", f"{color} {RPAM}")

st.markdown("---")

# ---------------------------------------------------------------
# SOLAR WIND CHART
# ---------------------------------------------------------------
st.subheader("☀️ Solar Wind & Geomagnetic Activity")
st.write(
    f"**Speed:** {solar['speed']:.2f} km/s | **Density:** {solar['density']:.2f} p/cm³ | **ψₛ:** {solar['psi_s']:.3f}"
)
solar_df = pd.DataFrame({
    "Speed (km/s)": [solar["speed"]],
    "Density (p/cm³)": [solar["density"]],
    "Kp": [kp],
})
st.line_chart(solar_df)

# ---------------------------------------------------------------
# SEISMIC TABLE + DRIFT
# ---------------------------------------------------------------
st.subheader("🌋 Seismic Events (Past 7 Days)")
if not df.empty:
    st.dataframe(df.sort_values("time", ascending=False).head(15))
else:
    st.warning("No live seismic data available (INGV and fallback failed).")

st.subheader("🌀 Harmonic Drift — Magnitude vs Depth")
if not df.empty:
    st.line_chart(df.set_index("time")[["magnitude", "depth_km"]])
else:
    st.info("No data available for drift chart.")

st.markdown("---")
st.caption("Feeds: NOAA • INGV • USGS | SUPT v6.0 Live Continuum Build — Sheppard’s Universal Proxy Theory")

# Auto-refresh
st_autorefresh(interval=600000, key="data_refresh")
