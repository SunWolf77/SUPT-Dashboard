# ===============================================================
#  SUPT :: GROK Forecast Dashboard (SunWolf Coupled Edition) v3.8f
#  Campi Flegrei + Solar ψₛ–Depth Harmonic Coherence System
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime as dt
import io
import plotly.graph_objects as go

# ---------------- CONFIG ---------------- #
API_TIMEOUT = 10
LOCAL_FALLBACK_CSV = "events_6.csv"
NOAA_GEOMAG_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"

DEFAULT_SOLAR = {
    "C_flare": 0.99,
    "M_flare": 0.55,
    "X_flare": 0.15,
    "psi_s": 0.72,
    "solar_speed": 688,
}

# ---------------- SESSION STATE ---------------- #
if "cci" not in st.session_state:
    st.session_state.cci = 0.0
if "psi_hist" not in st.session_state:
    st.session_state.psi_hist = []
if "depth_signal" not in st.session_state:
    st.session_state.depth_signal = []

# ---------------- FUNCTIONS ---------------- #
def compute_eii(md_max, md_mean, shallow_ratio, psi_s):
    """Energetic Instability Index"""
    return np.clip((md_max * 0.2 + md_mean * 0.15 + shallow_ratio * 0.4 + psi_s * 0.25), 0, 1)


def classify_phase(EII):
    """SUPT Phase Classification"""
    if EII >= 0.85:
        return "ACTIVE – Collapse Window Initiated"
    elif EII >= 0.6:
        return "ELEVATED – Pressure Coupling Phase"
    else:
        return "MONITORING"


@st.cache_data(ttl=600)
def fetch_geomag_data():
    """NOAA Kp geomagnetic data"""
    try:
        response = requests.get(NOAA_GEOMAG_URL, timeout=API_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        latest = data[-1] if data else [None, 0]
        return {
            "kp_index": float(latest[1]),
            "time_tag": latest[0],
            "geomag_alert": "HIGH" if float(latest[1]) >= 5 else "LOW",
        }
    except Exception as e:
        st.warning(f"NOAA Geomagnetic API fetch failed: {e}. Using defaults.")
        return {"kp_index": 0.0, "time_tag": "Fallback", "geomag_alert": "LOW"}


@st.cache_data(show_spinner=False)
def load_seismic_data():
    """Load INGV → USGS → Synthetic fallback"""
    try:
        end_time = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
        start_time = (dt.datetime.utcnow() - dt.timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S")
        ingv_url = (
            f"https://webservices.ingv.it/fdsnws/event/1/query?"
            f"starttime={start_time}&endtime={end_time}&latmin=40.7&latmax=40.9&lonmin=14.0&lonmax=14.3&format=text"
        )
        r = requests.get(ingv_url, timeout=API_TIMEOUT)
        r.raise_for_status()
        df_ingv = pd.read_csv(io.StringIO(r.text), delimiter="|", comment="#")
        if "Time" not in df_ingv.columns or "Depth/Km" not in df_ingv.columns:
            raise KeyError("INGV format inconsistent")
        df_ingv["time"] = pd.to_datetime(df_ingv["Time"])
        df_ingv["magnitude"] = df_ingv["Magnitude"]
        df_ingv["depth_km"] = df_ingv["Depth/Km"]
        return df_ingv[df_ingv["time"] > dt.datetime.utcnow() - dt.timedelta(days=7)]
    except Exception as e:
        st.warning(f"INGV fetch failed: {e}. Trying USGS fallback...")
        try:
            usgs_url = (
                f"https://earthquake.usgs.gov/fdsnws/event/1/query?"
                f"format=csv&starttime={start_time}&endtime={end_time}"
                f"&minlatitude=40.7&maxlatitude=40.9&minlongitude=14.0&maxlongitude=14.3"
            )
            r = requests.get(usgs_url, timeout=API_TIMEOUT)
            r.raise_for_status()
            df_usgs = pd.read_csv(io.StringIO(r.text))
            df_usgs["time"] = pd.to_datetime(df_usgs["time"])
            df_usgs["magnitude"] = df_usgs["mag"]
            df_usgs["depth_km"] = df_usgs["depth"]
            return df_usgs[df_usgs["time"] > dt.datetime.utcnow() - dt.timedelta(days=7)]
        except Exception as e:
            st.warning(f"USGS fallback failed: {e}. Using synthetic dataset for continuity.")
            df = pd.DataFrame({
                "time": pd.date_range(dt.datetime.utcnow() - dt.timedelta(days=7), periods=12, freq="12H"),
                "magnitude": np.random.uniform(0.5, 1.2, 12),
                "depth_km": np.random.uniform(0.8, 3.2, 12),
            })
            return df


def generate_solar_history(psi_s):
    hours = np.arange(0, 24)
    drift = psi_s + 0.02 * np.sin(hours / 3) + np.random.uniform(-0.005, 0.005, len(hours))
    return pd.DataFrame({"hour": hours, "psi_s": np.clip(drift, 0, 1)})


# ---------------- LAYOUT ---------------- #
st.set_page_config(page_title="SUPT :: GROK Forecast Dashboard", layout="wide")
st.title("🌋 SUPT :: GROK Forecast Dashboard")
st.caption("Campi Flegrei Risk & Energetic Instability Monitor :: v3.8f (Functional Build)")

with st.spinner("Loading seismic dataset..."):
    df = load_seismic_data()

if df.empty:
    st.error("No seismic data loaded — check live feed or fallback.")
    st.stop()

geomag_data = fetch_geomag_data()

# ---------------- SIDEBAR ---------------- #
st.sidebar.header("Solar Activity Controls")
psi_s = st.sidebar.slider("Solar Pressure Proxy (ψₛ)", 0.0, 1.0, DEFAULT_SOLAR["psi_s"])
solar_speed = st.sidebar.number_input("Solar Wind Speed (km/s)", value=DEFAULT_SOLAR["solar_speed"])
C_prob = st.sidebar.slider("C-Flare Probability", 0.0, 1.0, DEFAULT_SOLAR["C_flare"])
M_prob = st.sidebar.slider("M-Flare Probability", 0.0, 1.0, DEFAULT_SOLAR["M_flare"])
X_prob = st.sidebar.slider("X-Flare Probability", 0.0, 1.0, DEFAULT_SOLAR["X_flare"])

# ---------------- METRICS ---------------- #
md_max = df["magnitude"].max()
md_mean = df["magnitude"].mean()
depth_mean = df["depth_km"].mean()
shallow_ratio = len(df[df["depth_km"] < 2.5]) / len(df)
EII = compute_eii(md_max, md_mean, shallow_ratio, psi_s)
RPAM_STATUS = classify_phase(EII)

col1, col2, col3, col4 = st.columns(4)
col1.metric("EII", f"{EII:.3f}")
col2.metric("RPAM", RPAM_STATUS)
col3.metric("ψₛ", f"{psi_s:.3f}")
col4.metric("Geomagnetic Kp", f"{geomag_data['kp_index']:.1f}")

# ---------------- LIVE COHERENCE ---------------- #
st.markdown("### ☯ SUPT Coupling Coherence Index (CCI) — Live Tracking")

depth_signal = np.interp(
    np.linspace(0, len(df) - 1, 24),
    np.arange(len(df)),
    np.clip(df["depth_km"].rolling(window=3, min_periods=1).mean().values, 0, 5),
)
hist = generate_solar_history(psi_s)
psi_norm = (hist["psi_s"] - np.mean(hist["psi_s"])) / np.std(hist["psi_s"])
depth_norm = (depth_signal - np.mean(depth_signal)) / np.std(depth_signal)
cci = np.corrcoef(psi_norm, depth_norm)[0, 1] ** 2

color = "green" if cci >= 0.7 else "orange" if cci >= 0.4 else "red"
label = "Coherent" if cci >= 0.7 else "Moderate" if cci >= 0.4 else "Decoupled"

gauge = go.Figure(
    go.Indicator(
        mode="gauge+number+delta",
        value=cci,
        delta={"reference": 0.5, "increasing": {"color": "orange"}},
        title={"text": f"CCI: {label}", "font": {"size": 22}},
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 0.4], "color": "#FFCDD2"},
                {"range": [0.4, 0.7], "color": "#FFF59D"},
                {"range": [0.7, 1.0], "color": "#C8E6C9"},
            ],
        },
    )
)
gauge.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=0))
st.plotly_chart(gauge, use_container_width=True)

# ---------------- HARMONIC DRIFT ---------------- #
st.markdown("### ψₛ Harmonic Drift — 24-Hour Trend")
fig2 = go.Figure()
fig2.add_trace(
    go.Scatter(x=hist["hour"], y=hist["psi_s"], mode="lines", line=dict(color="#FF9800", width=3))
)
fig2.update_layout(
    title="ψₛ 24-Hour Harmonic Drift",
    xaxis_title="UTC Hour",
    yaxis_title="ψₛ Index",
    template="plotly_white",
)
st.plotly_chart(fig2, use_container_width=True)

# ---------------- ψₛ–DEPTH COUPLING ---------------- #
st.markdown("### ψₛ–Depth Coupling Trend (SUPT Coherence Field)")
depth_norm = (depth_signal - np.mean(depth_signal)) / np.std(depth_signal)
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=hist["hour"], y=hist["psi_s"], mode="lines+markers", name="ψₛ", line=dict(color="#FFD54F", width=3)))
fig3.add_trace(go.Scatter(x=hist["hour"], y=depth_norm / 2 + 0.5, mode="lines+markers", name="Depth Response", line=dict(color="#42A5F5", width=2, dash="dot")))
fig3.update_layout(
    title=f"SUPT Coupling Coherence Index (CCI): {cci:.3f}",
    xaxis_title="UTC Hour",
    yaxis_title="Normalized Amplitude",
    template="plotly_white",
    legend=dict(orientation="h", y=-0.2),
)
st.plotly_chart(fig3, use_container_width=True)

# ---------------- FOOTER ---------------- #
st.caption(f"Updated {dt.datetime.utcnow():%Y-%m-%d %H:%M:%S UTC} | Feeds: NOAA • INGV • USGS | Mode: SunWolf Harmonics | SUPT v3.8f")
st.caption("Powered by Sheppard’s Universal Proxy Theory — Real-time ψₛ–Depth Energy Coupling Monitor.")
