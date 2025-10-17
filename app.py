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
    """Load seismic data from INGV (live), fallback to USGS if needed, then synthetic continuity if all fail."""

    try:
        # --- Primary INGV Live Fetch (Campi Flegrei window)
        end_time = dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
        start_time = (dt.datetime.utcnow() - dt.timedelta(days=7)).strftime('%Y-%m-%dT%H:%M:%S')

        ingv_url = (
            f"https://webservices.ingv.it/fdsnws/event/1/query?"
            f"starttime={start_time}&endtime={end_time}"
            f"&latmin=40.7&latmax=40.9&lonmin=14.0&lonmax=14.3"
            f"&minmag=0&format=text"
        )

        response = requests.get(ingv_url, timeout=10)
        response.raise_for_status()

        df_ingv = pd.read_csv(io.StringIO(response.text), delimiter="|", comment="#")

        # --- Normalize column names ---
        df_ingv.columns = [c.strip() for c in df_ingv.columns]
        cols = df_ingv.columns.tolist()

        if "Time" not in cols or "Depth/Km" not in cols or "Magnitude" not in cols:
            raise KeyError("INGV columns missing")

        df_ingv["time"] = pd.to_datetime(df_ingv["Time"], errors="coerce")
        df_ingv["magnitude"] = pd.to_numeric(df_ingv["Magnitude"], errors="coerce")
        df_ingv["depth_km"] = pd.to_numeric(df_ingv["Depth/Km"], errors="coerce")

        df_ingv = df_ingv.dropna(subset=["time", "magnitude", "depth_km"])

        # --- Empty dataset check ---
        if df_ingv.empty:
            st.info("No Campi Flegrei events; using Italian regional fallback.")
            fallback_url = (
                f"https://webservices.ingv.it/fdsnws/event/1/query?"
                f"starttime={start_time}&endtime={end_time}"
                f"&minlat=37.0&maxlat=45.0&minlon=10.0&maxlon=16.0"
                f"&minmag=2&format=text"
            )
            response_fb = requests.get(fallback_url, timeout=10)
            response_fb.raise_for_status()
            df_ingv = pd.read_csv(io.StringIO(response_fb.text), delimiter="|", comment="#")
            if "Time" in df_ingv.columns and "Depth/Km" in df_ingv.columns and "Magnitude" in df_ingv.columns:
                df_ingv["time"] = pd.to_datetime(df_ingv["Time"], errors="coerce")
                df_ingv["magnitude"] = pd.to_numeric(df_ingv["Magnitude"], errors="coerce")
                df_ingv["depth_km"] = pd.to_numeric(df_ingv["Depth/Km"], errors="coerce")
                st.success(f"INGV live feed active ({len(df_ingv)} events).")
                return df_ingv
            else:
                raise KeyError("Fallback region returned non-standard columns")

        st.success(f"INGV live feed active ({len(df_ingv)} events).")
        return df_ingv

    except Exception as e:
        st.warning(f"INGV fetch failed: {e}. Trying USGS fallback...")

        try:
            # --- USGS Fallback (7-day, Campi Flegrei window)
            usgs_url = (
                f"https://earthquake.usgs.gov/fdsnws/event/1/query?"
                f"format=csv&starttime={start_time}&endtime={end_time}"
                f"&minlatitude=40.7&maxlatitude=40.9&minlongitude=14.0&maxlongitude=14.3"
            )
            r_usgs = requests.get(usgs_url, timeout=10)
            r_usgs.raise_for_status()
            df_usgs = pd.read_csv(io.StringIO(r_usgs.text))
            if df_usgs.empty:
                st.warning("USGS fallback feed active (0 events).")
                raise ValueError("USGS returned empty dataset")
            df_usgs["time"] = pd.to_datetime(df_usgs["time"], errors="coerce")
            df_usgs["magnitude"] = pd.to_numeric(df_usgs["mag"], errors="coerce")
            df_usgs["depth_km"] = pd.to_numeric(df_usgs["depth"], errors="coerce")
            st.info(f"USGS fallback feed active ({len(df_usgs)} events).")
            return df_usgs

        except Exception as e2:
            st.error(f"No live seismic data available. Using synthetic continuity dataset. ({e2})")

            # --- Synthetic fallback (continuity mode) ---
            t_now = dt.datetime.utcnow()
            times = [t_now - dt.timedelta(hours=i) for i in range(24)]
            df_synth = pd.DataFrame({
                "time": times[::-1],
                "magnitude": np.random.uniform(0.5, 1.5, size=24),
                "depth_km": np.random.uniform(1.0, 3.0, size=24)
            })
            return df_synth

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
