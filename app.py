# ===============================================================
# SUPT :: GROK Forecast Dashboard (Tri-Coherence Live Viewer) v4.0
# ψₛ–Depth–Kp Temporal Continuum Analyzer
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime as dt
import io
import plotly.graph_objects as go
import traceback

API_TIMEOUT = 10
LOCAL_FALLBACK_CSV = "events_6.csv"
NOAA_GEOMAG_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"

DEFAULT_SOLAR = {"psi_s": 0.72, "solar_speed": 688, "C_flare": 0.99, "M_flare": 0.55, "X_flare": 0.15}

# ===============================================================
# UTILITY CORE
# ===============================================================
def compute_eii(md_max, md_mean, shallow_ratio, psi_s):
    return np.clip((md_max * 0.2 + md_mean * 0.15 + shallow_ratio * 0.4 + psi_s * 0.25), 0, 1)

def classify_phase(EII):
    if EII >= 0.85:
        return "ACTIVE – Collapse Window Initiated"
    elif EII >= 0.6:
        return "ELEVATED – Pressure Coupling Phase"
    return "MONITORING"

def generate_synthetic_seismic_data(n=24):
    now = dt.datetime.utcnow()
    return pd.DataFrame({
        "time": [now - dt.timedelta(hours=i) for i in range(n)],
        "magnitude": np.random.uniform(0.6, 1.3, n),
        "depth_km": np.random.uniform(0.8, 3.0, n)
    })

# ===============================================================
# NOAA FETCH
# ===============================================================
@st.cache_data(ttl=600)
def fetch_geomag_data():
    try:
        r = requests.get(NOAA_GEOMAG_URL, timeout=API_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data[1:], columns=data[0])
        df["time_tag"] = pd.to_datetime(df["time_tag"])
        df["Kp"] = pd.to_numeric(df["Kp"], errors="coerce")
        latest = df.iloc[-1]
        return {"kp_index": latest["Kp"], "time_tag": latest["time_tag"], "geomag_df": df}
    except Exception as e:
        st.warning(f"NOAA Kp fetch failed: {e}")
        return {"kp_index": 0.0, "time_tag": "Fallback", "geomag_df": pd.DataFrame()}

# ===============================================================
# SEISMIC FETCH (Adaptive)
# ===============================================================
@st.cache_data(show_spinner=False)
def load_seismic_data():
    def normalize_columns(df):
        df.columns = [c.lower().replace("(", "").replace(")", "").replace("/", "").strip() for c in df.columns]
        t = next((c for c in df.columns if "time" in c), None)
        d = next((c for c in df.columns if "depth" in c), None)
        m = next((c for c in df.columns if "mag" in c), None)
        if not all([t, d, m]): raise KeyError("Essential INGV columns missing")
        df["time"] = pd.to_datetime(df[t], errors="coerce")
        df["depth_km"] = pd.to_numeric(df[d], errors="coerce")
        df["magnitude"] = pd.to_numeric(df[m], errors="coerce")
        return df.dropna(subset=["time", "depth_km", "magnitude"])

    try:
        end_time = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
        start_time = (dt.datetime.utcnow() - dt.timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S")
        ingv_url = (f"https://webservices.ingv.it/fdsnws/event/1/query?"
                    f"starttime={start_time}&endtime={end_time}"
                    f"&latmin=40.7&latmax=40.9&lonmin=14.0&lonmax=14.3&minmag=0&format=text")
        r = requests.get(ingv_url, timeout=API_TIMEOUT)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), delimiter="|", comment="#")
        df = normalize_columns(df)
        st.info("✅ INGV live feed active.")
        return df
    except Exception:
        st.warning("INGV feed failed — using synthetic data for continuity.")
        return generate_synthetic_seismic_data()

# ===============================================================
# HARMONIC MODELS
# ===============================================================
def generate_solar_history(psi_s):
    hours = np.arange(0, 24)
    drift = psi_s + 0.02 * np.sin(hours / 3) + np.random.uniform(-0.005, 0.005, len(hours))
    return pd.DataFrame({"hour": hours, "psi_s": np.clip(drift, 0, 1)})

def generate_forecast_wave(psi_s):
    hours = np.arange(0, 48)
    base = psi_s + 0.03 * np.sin(hours / 5) + 0.015 * np.cos(hours / 8)
    noise = np.random.uniform(-0.01, 0.01, len(hours))
    return pd.DataFrame({"hour": hours, "forecast_psi": np.clip(base + noise, 0, 1)})

# ===============================================================
# UI SETUP
# ===============================================================
st.set_page_config(page_title="SUPT :: GROK Forecast Dashboard", layout="wide")
st.title("🌋 SUPT :: GROK Forecast Dashboard")
st.caption("Campi Flegrei Risk & Energetic Instability Monitor :: v4.0 — Tri-Coherence Live Viewer")

with st.spinner("Fetching data..."):
    df = load_seismic_data()
    geomag = fetch_geomag_data()

if df.empty: df = generate_synthetic_seismic_data()

psi_s = st.sidebar.slider("Solar Pressure Proxy (ψₛ)", 0.0, 1.0, DEFAULT_SOLAR["psi_s"])
EII = compute_eii(df["magnitude"].max(), df["magnitude"].mean(), len(df[df["depth_km"] < 2.5]) / max(len(df), 1), psi_s)
RPAM = classify_phase(EII)

col1, col2 = st.columns(2)
col1.metric("EII", f"{EII:.3f}")
col2.metric("RPAM Status", RPAM)

# ===============================================================
# TRI-PANEL CONTINUUM VIEWER
# ===============================================================
st.markdown("### 🌞🌍 SUPT Tri-Coherence Viewer — ψₛ / Kp / Depth Synchronization (48h)")

# Create normalized temporal waveforms
solar_forecast = generate_forecast_wave(psi_s)
geomag_df = geomag["geomag_df"]
depth_series = df["depth_km"].rolling(3, min_periods=1).mean().iloc[:48]

if geomag_df.empty:
    geomag_df = pd.DataFrame({"time_tag": pd.date_range(dt.datetime.utcnow() - dt.timedelta(hours=47), periods=48, freq="H"),
                              "Kp": np.random.uniform(0.5, 4.0, 48)})

geomag_wave = (geomag_df["Kp"].iloc[-48:].reset_index(drop=True) - geomag_df["Kp"].min()) / (geomag_df["Kp"].max() - geomag_df["Kp"].min())
depth_wave = (depth_series - depth_series.min()) / (depth_series.max() - depth_series.min())

fig = go.Figure()
fig.add_trace(go.Scatter(y=solar_forecast["forecast_psi"], mode="lines", name="ψₛ Solar", line=dict(color="#FFA726", width=3)))
fig.add_trace(go.Scatter(y=geomag_wave, mode="lines", name="Kp Geomagnetic", line=dict(color="#42A5F5", width=2)))
fig.add_trace(go.Scatter(y=depth_wave, mode="lines", name="Depth (km, norm)", line=dict(color="#8BC34A", width=2)))
fig.update_layout(
    title="Tri-Coherence Harmonic Alignment (ψₛ ↔ Kp ↔ Depth)",
    xaxis_title="Time (48h window)",
    yaxis_title="Normalized Amplitude",
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)

# ===============================================================
# FOOTER
# ===============================================================
st.caption(f"Updated {dt.datetime.utcnow():%Y-%m-%d %H:%M:%S UTC} | Feeds: NOAA • INGV • USGS | SUPT v4.0 Continuum Engine")
st.caption("Powered by Sheppard’s Universal Proxy Theory — ψₛ–Depth–Kp Resonance Visualizer.")
