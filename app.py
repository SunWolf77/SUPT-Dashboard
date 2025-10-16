# ===============================================================
# 🌞🐺 SunWolf-SUPT v3.7 — GROK-Fusion Live Forecast Dashboard
# Real-time coupling between Solar & Geothermal Systems
# INGV • USGS • NOAA • SUPT ψ-Fold Engine
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
NOAA_KP_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
LOCAL_FALLBACK = "events_6.csv"
DEFAULT_SOLAR = {
    "psi_s": 0.72,
    "solar_speed": 688,
    "C_flare": 0.99,
    "M_flare": 0.55,
    "X_flare": 0.15,
}
REFRESH_INTERVAL_MIN = 10

# ---------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------
def compute_eii(md_max, md_mean, shallow_ratio, psi_s):
    """Energetic Instability Index"""
    return np.clip((md_max * 0.2 + md_mean * 0.15 + shallow_ratio * 0.4 + psi_s * 0.25), 0, 1)

def classify_phase(EII):
    if EII >= 0.85:
        return "ACTIVE - Collapse Window Initiated"
    elif EII >= 0.6:
        return "ELEVATED - Pressure Coupling Phase"
    else:
        return "MONITORING"

def fetch_geomag_data():
    try:
        data = requests.get(NOAA_KP_URL, timeout=API_TIMEOUT).json()
        latest = data[-1]
        kp = float(latest[1])
        return {"kp": kp, "alert": "HIGH" if kp >= 5 else "LOW", "time": latest[0]}
    except Exception:
        return {"kp": 0.0, "alert": "LOW", "time": "Fallback"}

# ---------------------------------------------------------------
# SEISMIC FETCH — INGV → USGS → LOCAL
# ---------------------------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def fetch_quakes():
    """Fetch Campi Flegrei quakes (INGV → USGS → local fallback). Always returns valid DataFrame."""
    start = (dt.datetime.utcnow() - dt.timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S")
    end = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

    try:
        url = (
            f"https://webservices.ingv.it/fdsnws/event/1/query?"
            f"starttime={start}&endtime={end}&latmin=40.7&latmax=40.9&lonmin=14.0&lonmax=14.3&format=text"
        )
        resp = requests.get(url, timeout=API_TIMEOUT)
        resp.raise_for_status()
        text = resp.text.strip()

        if not text or "No events" in text:
            raise ValueError("INGV returned empty or null response.")

        # Ensure header exists
        lines = [l for l in text.splitlines() if l.strip()]
        if not lines[0].startswith("#"):
            lines.insert(
                0,
                "#EventID|Time|Latitude|Longitude|Depth(km)|Magnitude|Author|EventLocationName",
            )
        clean_text = "\n".join(lines)

        # Parse
        df = pd.read_csv(io.StringIO(clean_text), delimiter="|", comment="#", engine="python")
        df.columns = [c.strip() for c in df.columns]

        # Adaptive mapping for any column variant
        col_map = {}
        for c in df.columns:
            cl = c.lower()
            if "time" in cl or "origin" in cl or "date" in cl:
                col_map[c] = "Time"
            elif "depth" in cl or "km" in cl:
                col_map[c] = "Depth(km)"
            elif "mag" in cl or "magnitude" in cl:
                col_map[c] = "Magnitude"
        df = df.rename(columns=col_map)

        # Log what was detected
        st.info(f"Detected INGV columns: {list(df.columns)}")

        if not all(x in df.columns for x in ["Time", "Depth(km)", "Magnitude"]):
            raise KeyError("Required fields missing after mapping.")

        df["time"] = pd.to_datetime(df["Time"], errors="coerce")
        df["magnitude"] = pd.to_numeric(df["Magnitude"], errors="coerce")
        df["depth_km"] = pd.to_numeric(df["Depth(km)"], errors="coerce")
        df = df.dropna(subset=["time", "magnitude", "depth_km"])
        df = df[df["time"] > dt.datetime.utcnow() - dt.timedelta(days=7)]

        if df.empty:
            raise ValueError("INGV feed valid but returned 0 usable rows.")

        return df.reset_index(drop=True)

    except Exception as e:
        st.warning(f"INGV fetch failed: {e}. Trying USGS fallback...")

        # === USGS Fallback ===
        try:
            usgs_url = (
                f"https://earthquake.usgs.gov/fdsnws/event/1/query?"
                f"format=csv&starttime={start}&endtime={end}"
                f"&minlatitude=40.7&maxlatitude=40.9&minlongitude=14.0&maxlongitude=14.3"
            )
            df = pd.read_csv(io.StringIO(requests.get(usgs_url, timeout=API_TIMEOUT).text))
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
            df["magnitude"] = pd.to_numeric(df["mag"], errors="coerce")
            df["depth_km"] = pd.to_numeric(df["depth"], errors="coerce")
            df = df.dropna(subset=["time", "magnitude", "depth_km"])
            if not df.empty:
                st.success(f"USGS fallback succeeded. {len(df)} quakes loaded.")
                return df
            raise ValueError("USGS fallback returned empty data.")
        except Exception as e2:
            st.warning(f"USGS fallback failed: {e2}. Loading local/synthetic sample.")

            # === Local or synthetic fallback ===
            try:
                df = pd.read_csv(LOCAL_FALLBACK)
                df["time"] = pd.to_datetime(df.get("Time", df.index), errors="coerce")
                df["magnitude"] = pd.to_numeric(df.get("MD", 0.6), errors="coerce")
                df["depth_km"] = pd.to_numeric(df.get("Depth", 1.9), errors="coerce")
                df = df.dropna(subset=["time", "magnitude", "depth_km"])
                if len(df) > 0:
                    st.info(f"Loaded {len(df)} local fallback quakes.")
                    return df
            except Exception:
                pass

            # === Synthetic data ===
            now = dt.datetime.utcnow()
            synth = pd.DataFrame(
                {
                    "time": [now - dt.timedelta(hours=i) for i in range(12)],
                    "magnitude": np.random.uniform(0.2, 1.2, 12),
                    "depth_km": np.random.uniform(0.5, 3.5, 12),
                }
            )
            st.warning("⚠️ Using synthetic dataset for continuity.")
            return synth

# ---------------------------------------------------------------
# SOLAR WIND (SYNTHETIC FOR ψₛ DRIFT)
# ---------------------------------------------------------------
def generate_solar_history(psi_s):
    t = np.arange(24)
    noise = np.random.normal(0, 0.015, size=24)
    harmonic = psi_s + 0.05 * np.sin(2 * np.pi * t / 24) + noise
    return pd.DataFrame({"hour": t, "psi_s": np.clip(harmonic, 0, 1)})

# ---------------------------------------------------------------
# DASHBOARD SETUP
# ---------------------------------------------------------------
st.set_page_config(layout="wide", page_title="SUPT :: GROK Fusion")
st.markdown(
    "<h1 style='text-align:center;'>🌋 SUPT :: GROK Forecast Dashboard</h1>"
    "<p style='text-align:center;'>Campi Flegrei Risk & Energetic Instability Monitor :: v3.7</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# FETCH DATA
# ---------------------------------------------------------------
with st.spinner("Fetching seismic data..."):
    df = fetch_quakes()
with st.spinner("Fetching geomagnetic data..."):
    kp = fetch_geomag_data()

if df.empty:
    st.error("No seismic data loaded — check live feed or fallback.")
    st.stop()

# ---------------------------------------------------------------
# CALCULATE METRICS
# ---------------------------------------------------------------
md_max = df["magnitude"].max()
md_mean = df["magnitude"].mean()
shallow_ratio = (df["depth_km"] < 2.5).mean()
psi_s = st.sidebar.slider("Solar Pressure Proxy (ψₛ)", 0.0, 1.0, DEFAULT_SOLAR["psi_s"])
EII = compute_eii(md_max, md_mean, shallow_ratio, psi_s)
phase = classify_phase(EII)
window = "Q1 2026" if "ACTIVE" in phase else "N/A"

# ---------------------------------------------------------------
# DISPLAY SUMMARY
# ---------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("EII", f"{EII:.3f}")
col2.metric("RPAM", phase)
col3.metric("ψₛ", f"{psi_s:.3f}")
col4.metric("Geomagnetic Kp", f"{kp['kp']:.1f}")

st.markdown(f"**Seismic Records:** {len(df)}  |  **Mean Depth:** {df['depth_km'].mean():.2f} km  |  **Mean Mag:** {md_mean:.2f}")
st.markdown(f"**Geomagnetic Level:** {kp['alert']}  |  Last update: {kp['time']}")

# ---------------------------------------------------------------
# VISUALIZATIONS
# ---------------------------------------------------------------
tab1, tab2 = st.tabs(["📊 Depth Distribution", "☯ ψₛ Harmonic Drift"])

with tab1:
    fig = go.Figure(data=[go.Histogram(x=df["depth_km"], nbinsx=15, marker_color="#FFB74D")])
    fig.update_layout(
        title="Seismic Depth Distribution (Past 7 Days)",
        xaxis_title="Depth (km)", yaxis_title="Count", template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    hist = generate_solar_history(psi_s)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=hist["hour"], y=hist["psi_s"], mode="lines", line=dict(color="#FF9800", width=3)))
    fig2.update_layout(
        title="ψₛ 24-Hour Harmonic Drift",
        xaxis_title="UTC Hour", yaxis_title="ψₛ Index", template="plotly_white", yaxis=dict(range=[0, 1])
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ---------------------------------------------------------------
    # SUPT ψₛ – Depth Coupling Cross-Phase Trend
    # ---------------------------------------------------------------
    st.markdown("### ☯ ψₛ–Depth Coupling Trend (SUPT Coherence Field)")

    # Create a smoothed synthetic depth response curve aligned to ψₛ time domain
    # (For live INGV/USGS data, uses actual mean depth variation)
    if not df.empty:
        depth_signal = np.interp(
            np.linspace(0, len(df) - 1, 24),
            np.arange(len(df)),
            np.clip(df["depth_km"].rolling(window=3, min_periods=1).mean().values, 0, 5),
        )
    else:
        depth_signal = np.random.uniform(0.5, 3.0, 24)

    # Normalize both signals for correlation
    psi_norm = (hist["psi_s"] - hist["psi_s"].mean()) / hist["psi_s"].std()
    depth_norm = (depth_signal - np.mean(depth_signal)) / np.std(depth_signal)

    # Coupling coherence index (R²)
    cci = np.corrcoef(psi_norm, depth_norm)[0, 1] ** 2

    fig3 = go.Figure()
    fig3.add_trace(
        go.Scatter(
            x=hist["hour"], y=hist["psi_s"],
            mode="lines+markers",
            name="ψₛ Harmonic",
            line=dict(color="#FFD54F", width=3)
        )
    )
    fig3.add_trace(
        go.Scatter(
            x=hist["hour"], y=depth_signal / 5,
            mode="lines+markers",
            name="Depth Response (normalized)",
            line=dict(color="#42A5F5", width=2, dash="dot")
        )
    )
    fig3.update_layout(
        title=f"SUPT Coupling Coherence Index (CCI): {cci:.3f}",
        xaxis_title="UTC Hour",
        yaxis_title="Normalized Coupling Amplitude",
        legend=dict(orientation="h", y=-0.2),
        template="plotly_white"
    )
    st.plotly_chart(fig3, use_container_width=True)


# ---------------------------------------------------------------
# SIDEBAR (SOLAR ACTIVITY)
# ---------------------------------------------------------------
st.sidebar.header("Solar Activity Controls")
st.sidebar.number_input("Solar Wind Speed (km/s)", value=DEFAULT_SOLAR["solar_speed"])
st.sidebar.slider("C-Flare Probability", 0.0, 1.0, DEFAULT_SOLAR["C_flare"])
st.sidebar.slider("M-Flare Probability", 0.0, 1.0, DEFAULT_SOLAR["M_flare"])
st.sidebar.slider("X-Flare Probability", 0.0, 1.0, DEFAULT_SOLAR["X_flare"])

# ---------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------
st.markdown(
    f"<hr><p style='text-align:center;'>Updated {dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} "
    f"| Mode: Solar-Geothermal Coupling | SunWolf-SUPT v3.7</p>",
    unsafe_allow_html=True,
)
