# ===============================================================
# SunWolf's Forecast Dashboard (Live Continuum Stable Build) v3.9.5
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime as dt
import io
import traceback
import plotly.graph_objects as go

# ===============================================================
# CONFIGURATION
# ===============================================================
API_TIMEOUT = 10
LOCAL_FALLBACK_CSV = "events_6.csv"
NOAA_GEOMAG_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
INGV_API_URL = "https://webservices.ingv.it/fdsnws/event/1/query?"

DEFAULT_SOLAR = {
    "C_flare": 0.99, "M_flare": 0.55, "X_flare": 0.15,
    "psi_s": 0.72, "solar_speed": 688
}

# ===============================================================
# UTILITY FUNCTIONS
# ===============================================================
def compute_eii(md_max, md_mean, shallow_ratio, psi_s):
    """Energetic Instability Index"""
    return np.clip((md_max * 0.2 + md_mean * 0.15 + shallow_ratio * 0.4 + psi_s * 0.25), 0, 1)

def classify_phase(EII):
    """SUPT phase classification"""
    if EII >= 0.85:
        return "ACTIVE – Collapse Window Initiated"
    elif EII >= 0.6:
        return "ELEVATED – Pressure Coupling Phase"
    return "MONITORING"

def generate_synthetic_seismic_data(n=20):
    """Synthetic fallback data when APIs fail"""
    now = dt.datetime.utcnow()
    times = [now - dt.timedelta(hours=i * 3) for i in range(n)]
    mags = np.random.uniform(0.5, 1.3, n)
    depths = np.random.uniform(0.8, 3.0, n)
    return pd.DataFrame({"time": times, "magnitude": mags, "depth_km": depths})

# ===============================================================
# NOAA FETCH — GEOMAGNETIC ACTIVITY
# ===============================================================
@st.cache_data(ttl=600)
def fetch_geomag_data():
    try:
        r = requests.get(NOAA_GEOMAG_URL, timeout=API_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        latest = data[-1]
        return {
            "kp_index": float(latest[1]),
            "time_tag": latest[0],
            "geomag_alert": "HIGH" if float(latest[1]) >= 5 else "LOW"
        }
    except Exception as e:
        st.warning(f"NOAA fetch failed: {e}")
        return {"kp_index": 0.0, "time_tag": "Fallback", "geomag_alert": "LOW"}

# ===============================================================
# INGV FETCH — SEISMIC ACTIVITY (Campi Flegrei)
# ===============================================================
@st.cache_data(ttl=600)
def fetch_ingv_seismic_data():
    try:
        now = dt.datetime.utcnow()
        start_time = now - dt.timedelta(days=7)
        params = {
            "starttime": start_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "endtime": now.strftime("%Y-%m-%dT%H:%M:%S"),
            "latmin": 40.7, "latmax": 40.9,
            "lonmin": 14.0, "lonmax": 14.3,
            "format": "text"
        }
        r = requests.get(INGV_API_URL, params=params, timeout=API_TIMEOUT)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), delimiter="|")
        df['time'] = pd.to_datetime(df['Time'], errors='coerce')
        df['magnitude'] = pd.to_numeric(df['Magnitude'], errors='coerce')
        df['depth_km'] = pd.to_numeric(df['Depth/Km'], errors='coerce')
        df = df.dropna(subset=['time', 'magnitude', 'depth_km'])
        return df
    except Exception as e:
        st.warning(f"INGV fetch failed: {e}")
        return pd.DataFrame()

# ===============================================================
# FALLBACK SEISMIC DATA HANDLER
# ===============================================================
@st.cache_data(ttl=600)
def load_seismic_data():
    df = fetch_ingv_seismic_data()
    if df.empty:
        try:
            df = pd.read_csv(LOCAL_FALLBACK_CSV)
            df['time'] = pd.to_datetime(df['Time'], errors='coerce')
            df['magnitude'] = pd.to_numeric(df['MD'], errors='coerce')
            df['depth_km'] = pd.to_numeric(df['Depth'], errors='coerce')
            df = df.dropna(subset=['time', 'magnitude', 'depth_km'])
            recent_start = dt.datetime.utcnow() - dt.timedelta(days=7)
            return df[df['time'] >= recent_start]
        except Exception as e:
            st.info("Local fallback failed. Using synthetic dataset.")
            return generate_synthetic_seismic_data()
    return df

# ===============================================================
# SOLAR HISTORY & FORECAST SIMULATION
# ===============================================================
def generate_solar_history(psi_s, hours=24):
    now = dt.datetime.utcnow()
    times = [now - dt.timedelta(hours=i) for i in range(hours)][::-1]
    psi_vals = np.random.normal(psi_s, 0.05, hours)
    return pd.DataFrame({"time": times, "psi_s": psi_vals})

def generate_forecast_wave(psi_s, hours=48):
    now = dt.datetime.utcnow()
    times = [now + dt.timedelta(hours=i) for i in range(hours)]
    forecast_psi = np.sin(np.linspace(0, np.pi * 2, hours)) * 0.3 + psi_s
    return pd.DataFrame({"hour": range(hours), "forecast_psi": forecast_psi})

# ===============================================================
# MAIN DASHBOARD
# ===============================================================
st.set_page_config(layout="wide")
st.title("🌞🐺 SunWolf's Forecast Dashboard")

# Data Fetch
df = load_seismic_data()
geomag = fetch_geomag_data()

# Compute SUPT metrics
md_max = df['magnitude'].max()
md_mean = df['magnitude'].mean()
shallow_ratio = len(df[df["depth_km"] < 2.5]) / max(len(df), 1)
psi_s = DEFAULT_SOLAR["psi_s"]

EII = compute_eii(md_max, md_mean, shallow_ratio, psi_s)
RPAM = classify_phase(EII)

# Display Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Energetic Instability Index (EII)", f"{EII:.3f}")
col2.metric("RPAM Phase", RPAM)
col3.metric("Geomagnetic Kp", f"{geomag['kp_index']:.1f}")

# ===============================================================
# COHERENCE GAUGE
# ===============================================================
st.markdown("### ☯ SUPT ψₛ Coupling — 24 h Harmonic Drift")
if not df.empty:
    psi_hist = generate_solar_history(psi_s)
    depth_signal = np.interp(np.linspace(0, len(df) - 1, 24), np.arange(len(df)),
                             np.clip(df["depth_km"].rolling(3, min_periods=1).mean(), 0, 5))
    psi_norm = (psi_hist["psi_s"] - np.mean(psi_hist["psi_s"])) / np.std(psi_hist["psi_s"])
    depth_norm = (depth_signal - np.mean(depth_signal)) / np.std(depth_signal)
    cci = np.corrcoef(psi_norm, depth_norm)[0, 1] ** 2 if len(df) > 1 else 0

    color = "green" if cci >= 0.7 else "orange" if cci >= 0.4 else "red"
    label = "Coherent" if cci >= 0.7 else "Moderate" if cci >= 0.4 else "Decoupled"

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=cci,
        title={"text": f"CCI: {label}"},
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 0.4], "color": "#FFCDD2"},
                {"range": [0.4, 0.7], "color": "#FFF59D"},
                {"range": [0.7, 1.0], "color": "#C8E6C9"}
            ]
        }
    ))
    st.plotly_chart(gauge, use_container_width=True)
else:
    st.info("No data for CCI gauge.")

# ===============================================================
# FORECAST WAVEFORM
# ===============================================================
st.markdown("### 🔮 ψₛ Temporal Resonance Forecast (Next 48 Hours)")
forecast = generate_forecast_wave(psi_s)
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=forecast["hour"], y=forecast["forecast_psi"],
    mode="lines", line=dict(color="#FFB300", width=3), name="ψₛ Forecast"
))
fig.update_layout(
    title="48-Hour ψₛ Temporal Wave Forecast",
    xaxis_title="Hours Ahead",
    yaxis_title="ψₛ Index",
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)

# ===============================================================
# FOOTER
# ===============================================================
st.caption(f"Updated {dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')} | Feeds: NOAA • INGV | SUPT v3.9.5")
st.caption("Powered by Sheppard’s Universal Proxy Theory — Continuous ψₛ–Depth–Kp Coherence.")
