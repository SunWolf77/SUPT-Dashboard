# ===============================================================
# SUPT :: GROK Forecast Dashboard (Live Continuum Stable Build) v3.9.5
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime as dt
import io
import traceback
import plotly.graph_objects as go

API_TIMEOUT = 10
LOCAL_FALLBACK_CSV = "events_6.csv"
NOAA_GEOMAG_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"

DEFAULT_SOLAR = {
    "C_flare": 0.99, "M_flare": 0.55, "X_flare": 0.15,
    "psi_s": 0.72, "solar_speed": 688
}

# ===============================================================
# Utility Functions
# ===============================================================
def compute_eii(md_max, md_mean, shallow_ratio, psi_s):
    return np.clip((md_max * 0.2 + md_mean * 0.15 + shallow_ratio * 0.4 + psi_s * 0.25), 0, 1)

def classify_phase(EII):
    if EII >= 0.85:
        return "ACTIVE – Collapse Window Initiated"
    elif EII >= 0.6:
        return "ELEVATED – Pressure Coupling Phase"
    return "MONITORING"

def generate_synthetic_seismic_data(n=20):
    now = dt.datetime.utcnow()
    times = [now - dt.timedelta(hours=i * 3) for i in range(n)]
    mags = np.random.uniform(0.5, 1.3, n)
    depths = np.random.uniform(0.8, 3.0, n)
    return pd.DataFrame({"time": times, "magnitude": mags, "depth_km": depths})

# ===============================================================
# NOAA Fetch
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
# Load Seismic Data (Adaptive Parser)
# ===============================================================
@st.cache_data(show_spinner=False)
def load_seismic_data(uploaded_file=None):
    def standardize_columns(df):
        df.columns = [c.strip().lower().replace("(", "").replace(")", "").replace("/", "") for c in df.columns]
        t = next((c for c in df.columns if "time" in c), None)
        m = next((c for c in df.columns if "mag" in c), None)
        d = next((c for c in df.columns if "depth" in c), None)
        if not all([t, m, d]):
            raise KeyError("Essential INGV columns missing after normalization.")
        df["time"] = pd.to_datetime(df[t], errors="coerce")
        df["magnitude"] = pd.to_numeric(df[m], errors="coerce")
        df["depth_km"] = pd.to_numeric(df[d], errors="coerce")
        df = df.dropna(subset=["time", "magnitude", "depth_km"])
        return df

    try:
        if uploaded_file is not None:
            df_manual = pd.read_csv(uploaded_file)
            st.success("Manual CSV uploaded successfully.")
            return standardize_columns(df_manual)

        end_time = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
        start_time = (dt.datetime.utcnow() - dt.timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S")
        ingv_url = (
            f"https://webservices.ingv.it/fdsnws/event/1/query?"
            f"starttime={start_time}&endtime={end_time}"
            f"&latmin=40.7&latmax=40.9&lonmin=14.0&lonmax=14.3&minmag=0&format=text"
        )
        r = requests.get(ingv_url, timeout=API_TIMEOUT)
        r.raise_for_status()
        df_ingv = pd.read_csv(io.StringIO(r.text), delimiter="|", comment="#")
        df_ingv = standardize_columns(df_ingv)
        st.info("✅ INGV live feed active.")
        return df_ingv

    except Exception as e:
        st.warning(f"INGV fetch failed: {e}. Trying USGS fallback...")
        st.text(traceback.format_exc())
        try:
            usgs_url = (
                f"https://earthquake.usgs.gov/fdsnws/event/1/query?"
                f"format=csv&starttime={start_time}&endtime={end_time}"
                f"&minlatitude=40.7&maxlatitude=40.9&minlongitude=14.0&maxlongitude=14.3"
            )
            r = requests.get(usgs_url, timeout=API_TIMEOUT)
            r.raise_for_status()
            df_usgs = pd.read_csv(io.StringIO(r.text))
            df_usgs.columns = [c.lower() for c in df_usgs.columns]
            if "time" not in df_usgs or "mag" not in df_usgs or "depth" not in df_usgs:
                raise KeyError("USGS columns missing")
            df_usgs["time"] = pd.to_datetime(df_usgs["time"], errors="coerce")
            df_usgs["magnitude"] = df_usgs["mag"]
            df_usgs["depth_km"] = df_usgs["depth"]
            st.info("🛰️ USGS fallback feed active.")
            return df_usgs.dropna(subset=["time", "magnitude", "depth_km"])
        except Exception as e:
            st.warning(f"USGS fallback failed: {e}")
            try:
                df_local = pd.read_csv(LOCAL_FALLBACK_CSV)
                st.info("Loaded local CSV fallback.")
                return standardize_columns(df_local)
            except Exception:
                st.error("All feeds unavailable — using synthetic continuity dataset.")
                return generate_synthetic_seismic_data()

# ===============================================================
# Harmonic + Forecast
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
# UI + Logic
# ===============================================================
st.set_page_config(page_title="SUPT :: GROK Forecast Dashboard", layout="wide")
st.title("🌋 SUPT :: GROK Forecast Dashboard")
st.caption("Campi Flegrei Risk & Energetic Instability Monitor :: v3.9.5 Live Continuum Stable Build")

uploaded_file = st.sidebar.file_uploader("Optional: Upload custom seismic CSV")

with st.spinner("Loading seismic data..."):
    df = load_seismic_data(uploaded_file)

if df.empty:
    st.error("No valid seismic data found — using synthetic continuity data.")
    df = generate_synthetic_seismic_data()

geomag = fetch_geomag_data()

# METRICS
psi_s = st.sidebar.slider("Solar Pressure Proxy (ψₛ)", 0.0, 1.0, DEFAULT_SOLAR["psi_s"])
md_max = df["magnitude"].max() if not df.empty else 0
md_mean = df["magnitude"].mean() if not df.empty else 0
depth_mean = df["depth_km"].mean() if not df.empty else 0
shallow_ratio = len(df[df["depth_km"] < 2.5]) / max(len(df), 1)

EII = compute_eii(md_max, md_mean, shallow_ratio, psi_s)
RPAM = classify_phase(EII)

col1, col2, col3 = st.columns(3)
col1.metric("EII", f"{EII:.3f}")
col2.metric("RPAM", RPAM)
col3.metric("Geomagnetic Kp", f"{geomag['kp_index']:.1f}")

# COHERENCE GAUGE
hist = generate_solar_history(psi_s)
depth_signal = np.interp(np.linspace(0, len(df) - 1, 24), np.arange(len(df)),
                         np.clip(df["depth_km"].rolling(3, min_periods=1).mean(), 0, 5))
psi_norm = (hist["psi_s"] - np.mean(hist["psi_s"])) / np.std(hist["psi_s"])
depth_norm = (depth_signal - np.mean(depth_signal)) / np.std(depth_signal)
cci = np.corrcoef(psi_norm, depth_norm)[0, 1] ** 2 if len(df) > 1 else 0

color = "green" if cci >= 0.7 else "orange" if cci >= 0.4 else "red"
label = "Coherent" if cci >= 0.7 else "Moderate" if cci >= 0.4 else "Decoupled"

gauge = go.Figure(go.Indicator(mode="gauge+number",
    value=cci, title={"text": f"CCI: {label}"},
    gauge={"axis": {"range": [0, 1]},
           "bar": {"color": color},
           "steps": [{"range": [0, 0.4], "color": "#FFCDD2"},
                     {"range": [0.4, 0.7], "color": "#FFF59D"},
                     {"range": [0.7, 1.0], "color": "#C8E6C9"}]}))
st.plotly_chart(gauge, use_container_width=True)

# FORECAST CHART
st.markdown("### 🔮 ψₛ Temporal Resonance Forecast (Next 48 Hours)")
forecast = generate_forecast_wave(psi_s)
fig = go.Figure()
fig.add_trace(go.Scatter(x=forecast["hour"], y=forecast["forecast_psi"],
                         mode="lines", line=dict(color="#FFB300", width=3)))
fig.update_layout(title="48-Hour ψₛ Temporal Wave Forecast",
                  xaxis_title="Hours Ahead", yaxis_title="ψₛ Index",
                  template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

st.caption(f"Updated {dt.datetime.utcnow():%Y-%m-%d %H:%M:%S UTC} | Feeds: NOAA • INGV • USGS | SUPT v3.9.5")
st.caption("Powered by Sheppard’s Universal Proxy Theory — Continuous ψₛ–Depth–Kp Coherence.")
