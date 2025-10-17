# ===============================================================
# SUPT :: GROK Forecast Dashboard (Harmonic Drift Analyzer) v4.1
# ψₛ–Depth–Kp Phase-Lag Quantification
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime as dt
import io
import plotly.graph_objects as go
from scipy.signal import correlate

API_TIMEOUT = 10
NOAA_GEOMAG_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"

# ------------------ SUPT CONFIG ------------------
DEFAULT_SOLAR = {"psi_s": 0.72}
LOCAL_FALLBACK_CSV = "events_6.csv"

# ===============================================================
# CORE UTILITY FUNCTIONS
# ===============================================================
def compute_eii(md_max, md_mean, shallow_ratio, psi_s):
    return np.clip((md_max * 0.2 + md_mean * 0.15 + shallow_ratio * 0.4 + psi_s * 0.25), 0, 1)

def classify_phase(EII):
    if EII >= 0.85: return "ACTIVE – Collapse Window Initiated"
    elif EII >= 0.6: return "ELEVATED – Pressure Coupling Phase"
    return "MONITORING"

def generate_synthetic_seismic_data(n=24):
    now = dt.datetime.utcnow()
    return pd.DataFrame({
        "time": [now - dt.timedelta(hours=i) for i in range(n)],
        "magnitude": np.random.uniform(0.6, 1.3, n),
        "depth_km": np.random.uniform(0.8, 3.0, n)
    })

# ===============================================================
# NOAA + INGV DATA FETCH
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
        return df.dropna(subset=["Kp"])
    except Exception as e:
        st.warning(f"NOAA Kp fetch failed: {e}")
        now = dt.datetime.utcnow()
        return pd.DataFrame({
            "time_tag": [now - dt.timedelta(hours=i) for i in range(48)],
            "Kp": np.random.uniform(0.5, 4.0, 48)
        })

@st.cache_data(show_spinner=False)
def load_seismic_data():
    try:
        end_time = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
        start_time = (dt.datetime.utcnow() - dt.timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S")
        url = (f"https://webservices.ingv.it/fdsnws/event/1/query?"
               f"starttime={start_time}&endtime={end_time}"
               f"&latmin=40.7&latmax=40.9&lonmin=14.0&lonmax=14.3&minmag=0&format=text")
        r = requests.get(url, timeout=API_TIMEOUT)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), delimiter="|", comment="#")
        df.columns = [c.lower().replace("(", "").replace(")", "").replace("/", "").strip() for c in df.columns]
        t = next((c for c in df.columns if "time" in c), None)
        d = next((c for c in df.columns if "depth" in c), None)
        m = next((c for c in df.columns if "mag" in c), None)
        df["time"] = pd.to_datetime(df[t], errors="coerce")
        df["depth_km"] = pd.to_numeric(df[d], errors="coerce")
        df["magnitude"] = pd.to_numeric(df[m], errors="coerce")
        return df.dropna(subset=["time", "depth_km", "magnitude"])
    except Exception:
        st.warning("INGV feed failed — using synthetic data.")
        return generate_synthetic_seismic_data()

# ===============================================================
# HARMONIC GENERATORS
# ===============================================================
def generate_forecast_wave(psi_s):
    hours = np.arange(0, 48)
    base = psi_s + 0.03 * np.sin(hours / 5) + 0.015 * np.cos(hours / 8)
    noise = np.random.uniform(-0.01, 0.01, len(hours))
    return pd.DataFrame({"hour": hours, "forecast_psi": np.clip(base + noise, 0, 1)})

# ===============================================================
# PHASE-LAG CALCULATION
# ===============================================================
def compute_phase_lag(ref, target, label):
    corr = correlate(target - np.mean(target), ref - np.mean(ref), mode="full")
    lag = np.arange(-len(ref) + 1, len(ref))
    best_lag = lag[np.argmax(corr)]
    phase_lag_hours = best_lag
    return phase_lag_hours, np.max(corr) / (np.std(ref) * np.std(target) * len(ref))

# ===============================================================
# STREAMLIT UI
# ===============================================================
st.set_page_config(page_title="SUPT :: GROK Forecast Dashboard", layout="wide")
st.title("🌋 SUPT :: GROK Forecast Dashboard")
st.caption("v4.1 — ψₛ Harmonic Drift Analyzer :: Solar–Geomagnetic–Geophysical Phase Alignment")

with st.spinner("Loading data streams..."):
    df = load_seismic_data()
    kp_df = fetch_geomag_data()

if df.empty: df = generate_synthetic_seismic_data()

# ================= METRICS ==================
psi_s = st.sidebar.slider("Solar Pressure Proxy (ψₛ)", 0.0, 1.0, DEFAULT_SOLAR["psi_s"])
EII = compute_eii(df["magnitude"].max(), df["magnitude"].mean(),
                  len(df[df["depth_km"] < 2.5]) / max(len(df), 1), psi_s)
RPAM = classify_phase(EII)
col1, col2 = st.columns(2)
col1.metric("EII", f"{EII:.3f}")
col2.metric("RPAM", RPAM)

# ===============================================================
# BUILD WAVEFORMS
# ===============================================================
solar_wave = generate_forecast_wave(psi_s)["forecast_psi"]
depth_wave = (df["depth_km"].rolling(3, min_periods=1).mean().iloc[:48] - df["depth_km"].min()) / (df["depth_km"].max() - df["depth_km"].min())
kp_wave = (kp_df["Kp"].iloc[-48:].reset_index(drop=True) - kp_df["Kp"].min()) / (kp_df["Kp"].max() - kp_df["Kp"].min())

# ===============================================================
# COMPUTE PHASE LAGS
# ===============================================================
lag_psi_depth, corr_pd = compute_phase_lag(solar_wave, depth_wave, "ψₛ–Depth")
lag_psi_kp, corr_pk = compute_phase_lag(solar_wave, kp_wave, "ψₛ–Kp")
lag_kp_depth, corr_kd = compute_phase_lag(kp_wave, depth_wave, "Kp–Depth")

# Display lag results
st.subheader("⏱️ Harmonic Phase Drift (Δφ)")
st.write(f"**ψₛ → Depth Lag:** {lag_psi_depth:+d} hours  | Correlation = {corr_pd:.2f}")
st.write(f"**ψₛ → Kp Lag:** {lag_psi_kp:+d} hours  | Correlation = {corr_pk:.2f}")
st.write(f"**Kp → Depth Lag:** {lag_kp_depth:+d} hours  | Correlation = {corr_kd:.2f}")

# ===============================================================
# VISUALIZATION
# ===============================================================
fig = go.Figure()
fig.add_trace(go.Scatter(y=solar_wave, mode="lines", name="ψₛ Solar", line=dict(color="#FFA726", width=3)))
fig.add_trace(go.Scatter(y=kp_wave, mode="lines", name="Kp Geomagnetic", line=dict(color="#42A5F5", width=2)))
fig.add_trace(go.Scatter(y=depth_wave, mode="lines", name="Depth (norm)", line=dict(color="#8BC34A", width=2)))
fig.update_layout(
    title="ψₛ Harmonic Drift Analyzer — Phase Alignment of ψₛ, Kp, Depth (48h)",
    xaxis_title="Time (hours)",
    yaxis_title="Normalized Amplitude",
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)

# ===============================================================
# FOOTER
# ===============================================================
st.caption(f"Updated {dt.datetime.utcnow():%Y-%m-%d %H:%M:%S UTC} | SUPT v4.1 Continuum Phase-Lag Engine")
st.caption("Powered by Sheppard’s Universal Proxy Theory — Real-Time ψₛ↔Kp↔Depth Harmonic Drift Detection.")
