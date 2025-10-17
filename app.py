# ================================================================
# SUPT :: GROK Forecast Dashboard
# v4.6-Final — Stable Functional Build (INGV + USGS + NOAA)
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime as dt
import io
import plotly.graph_objects as go

# ---------------- CONFIG ----------------
API_TIMEOUT = 10
NOAA_KP_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
DEFAULT_PSI = 0.72

# ---------------- UTILITIES ----------------
def compute_eii(md_max, md_mean, shallow_ratio, psi_s):
    """Compute Energetic Instability Index."""
    return np.clip((md_max * 0.2 + md_mean * 0.15 + shallow_ratio * 0.4 + psi_s * 0.25), 0, 1)

def classify_phase(EII):
    """Classify RPAM state from EII."""
    if EII >= 0.85:
        return "ACTIVE – Collapse Window Initiated"
    elif EII >= 0.6:
        return "ELEVATED – Pressure Coupling Phase"
    else:
        return "MONITORING"

def generate_synthetic_data():
    """Create a continuity dataset if all live feeds fail."""
    now = dt.datetime.utcnow()
    return pd.DataFrame({
        "time": [now - dt.timedelta(hours=i) for i in range(24)],
        "magnitude": np.random.uniform(0.6, 1.3, 24),
        "depth_km": np.random.uniform(0.8, 3.0, 24),
        "source": "synthetic"
    })

# ---------------- NOAA FETCH ----------------
@st.cache_data(ttl=600)
def fetch_noaa_kp():
    """Fetch current planetary Kp index."""
    try:
        r = requests.get(NOAA_KP_URL, timeout=API_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data[1:], columns=data[0])
        df["time_tag"] = pd.to_datetime(df["time_tag"])
        df["Kp"] = pd.to_numeric(df["Kp"], errors="coerce")
        return df.dropna(subset=["Kp"])
    except Exception as e:
        st.warning(f"NOAA fetch failed: {e}")
        now = dt.datetime.utcnow()
        return pd.DataFrame({
            "time_tag": [now - dt.timedelta(hours=i) for i in range(48)],
            "Kp": np.random.uniform(1, 5, 48)
        })

# ---------------- INGV / USGS FETCH ----------------
@st.cache_data(show_spinner=False, ttl=600)
def load_seismic_data():
    """Fetch seismic data from INGV (primary), USGS (secondary), or synthetic fallback."""

    source_label = "Unknown"
    df = pd.DataFrame()

    try:
        # --- Primary INGV live query ---
        end_time = dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
        start_time = (dt.datetime.utcnow() - dt.timedelta(days=7)).strftime('%Y-%m-%dT%H:%M:%S')

        ingv_url = (
            f"https://webservices.ingv.it/fdsnws/event/1/query?"
            f"starttime={start_time}&endtime={end_time}"
            f"&latmin=40.6&latmax=41.0&lonmin=13.9&lonmax=14.4"
            f"&minmag=0&format=text"
        )

        response = requests.get(ingv_url, timeout=10)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text), delimiter="|", comment="#")

        if "Time" not in df.columns or "Depth/Km" not in df.columns or "Magnitude" not in df.columns:
            raise KeyError("INGV essential columns missing")

        df["time"] = pd.to_datetime(df["Time"], errors="coerce")
        df["magnitude"] = pd.to_numeric(df["Magnitude"], errors="coerce")
        df["depth_km"] = pd.to_numeric(df["Depth/Km"], errors="coerce")
        df.dropna(subset=["time", "magnitude", "depth_km"], inplace=True)

        if not df.empty:
            source_label = f"INGV Live ({len(df)} events)"
            st.success(source_label)
            return df, source_label
        else:
            st.info("No Campi Flegrei events — widening query region.")
            # --- Regional fallback (Italy box) ---
            fallback_url = (
                f"https://webservices.ingv.it/fdsnws/event/1/query?"
                f"starttime={start_time}&endtime={end_time}"
                f"&minlat=37.0&maxlat=45.0&minlon=10.0&maxlon=16.0"
                f"&minmag=2&format=text"
            )
            response_fb = requests.get(fallback_url, timeout=10)
            response_fb.raise_for_status()
            df = pd.read_csv(io.StringIO(response_fb.text), delimiter="|", comment="#")
            if "Time" in df.columns and "Magnitude" in df.columns and "Depth/Km" in df.columns:
                df["time"] = pd.to_datetime(df["Time"], errors="coerce")
                df["magnitude"] = pd.to_numeric(df["Magnitude"], errors="coerce")
                df["depth_km"] = pd.to_numeric(df["Depth/Km"], errors="coerce")
                df.dropna(subset=["time", "magnitude", "depth_km"], inplace=True)
                if not df.empty:
                    source_label = f"INGV Regional ({len(df)} events)"
                    st.success(source_label)
                    return df, source_label
            raise ValueError("No INGV regional data found.")

    except Exception as e:
        st.warning(f"INGV fetch failed: {e}. Trying USGS fallback...")

        try:
            # --- USGS fallback ---
            usgs_url = (
                f"https://earthquake.usgs.gov/fdsnws/event/1/query?"
                f"format=csv&starttime={start_time}&endtime={end_time}"
                f"&minlatitude=40.6&maxlatitude=41.0&minlongitude=13.9&maxlongitude=14.4"
            )
            usgs_response = requests.get(usgs_url, timeout=10)
            usgs_response.raise_for_status()
            df = pd.read_csv(io.StringIO(usgs_response.text))
            if not df.empty:
                df["time"] = pd.to_datetime(df["time"], errors="coerce")
                df["magnitude"] = pd.to_numeric(df["mag"], errors="coerce")
                df["depth_km"] = pd.to_numeric(df["depth"], errors="coerce")
                df.dropna(subset=["time", "magnitude", "depth_km"], inplace=True)
                source_label = f"USGS Fallback ({len(df)} events)"
                st.info(source_label)
                return df, source_label
            else:
                raise ValueError("USGS returned empty dataset")

        except Exception as e2:
            st.error(f"No live seismic data available. Using synthetic continuity dataset. ({e2})")

            # --- Synthetic continuity dataset ---
            t_now = dt.datetime.utcnow()
            times = [t_now - dt.timedelta(hours=i) for i in range(24)]
            df = pd.DataFrame({
                "time": times[::-1],
                "magnitude": np.random.uniform(0.5, 1.5, size=24),
                "depth_km": np.random.uniform(1.0, 3.0, size=24)
            })
            source_label = "Synthetic Continuity Mode"
            return df, source_label

# ---------------------------------
# In your main dashboard logic (after loading data)
# ---------------------------------
with st.spinner("Loading seismic data..."):
    df, data_source = load_seismic_data()

if df.empty:
    st.error("No seismic data loaded — check live feed or upload below.")
    uploaded = st.file_uploader("Optional: Upload custom seismic CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.success("Custom seismic data loaded successfully.")
        data_source = "Manual Upload"

# Display active data source prominently
st.markdown(f"**🌍 Active Data Feed:** `{data_source}`")

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="SUPT Forecast Dashboard", layout="wide")
st.title("🌋 SUPT :: GROK Forecast Dashboard")
st.caption("v4.6 — Core Functional Build (Live INGV, NOAA, USGS Fallback)")

with st.spinner("Fetching data feeds..."):
    df = load_seismic_data()
    kp_df = fetch_noaa_kp()

# --- Solar Input ---
psi_s = st.sidebar.slider("Solar Pressure Proxy (ψₛ)", 0.0, 1.0, DEFAULT_PSI)

# --- Compute EII & RPAM ---
if not df.empty:
    md_max = df["magnitude"].max()
    md_mean = df["magnitude"].mean()
    shallow_ratio = len(df[df["depth_km"] < 2.5]) / max(len(df), 1)
    EII = compute_eii(md_max, md_mean, shallow_ratio, psi_s)
    RPAM = classify_phase(EII)
else:
    EII = 0.0
    RPAM = "NO DATA"

col1, col2 = st.columns(2)
col1.metric("EII", f"{EII:.3f}")
col2.metric("RPAM", RPAM)

# --- Plot 24h Trend ---
if not df.empty:
    df_plot = df.sort_values("time").tail(24)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot["time"], y=df_plot["depth_km"],
                             mode="lines+markers", name="Depth (km)", line=dict(color="#FFA726")))
    fig.add_trace(go.Scatter(x=df_plot["time"], y=df_plot["magnitude"],
                             mode="lines+markers", name="Magnitude", line=dict(color="#42A5F5")))
    fig.update_layout(title="24h Harmonic Drift — Depth vs Magnitude",
                      xaxis_title="Time (UTC)", yaxis_title="Value", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No data available for plotting. Using fallback synthetic continuity when available.")

# --- NOAA Kp ---
latest_kp = kp_df["Kp"].iloc[-1] if not kp_df.empty else 0
st.metric("Current Geomagnetic Kp", f"{latest_kp:.1f}")

# --- Footer ---
st.caption(f"Updated {dt.datetime.utcnow():%Y-%m-%d %H:%M:%S UTC}")
st.caption("Powered by SUPT — Sheppard’s Universal Proxy Theory (Functional Core)")
