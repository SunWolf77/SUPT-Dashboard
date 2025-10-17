# ================================================================
# SUPT :: GROK Forecast Dashboard v4.5-Final
# Core Functional Build — No SciPy, No Patch Dependencies
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime as dt
import io
import plotly.graph_objects as go
import csv

# ---------------- CONFIG ----------------
API_TIMEOUT = 10
NOAA_KP_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
LOCAL_FALLBACK_CSV = "events_6.csv"
DEFAULT_PSI = 0.72

# ---------------- UTILITIES ----------------
def compute_eii(md_max, md_mean, shallow_ratio, psi_s):
    return np.clip((md_max * 0.2 + md_mean * 0.15 + shallow_ratio * 0.4 + psi_s * 0.25), 0, 1)

def classify_phase(EII):
    if EII >= 0.85: return "ACTIVE – Collapse Window Initiated"
    elif EII >= 0.6: return "ELEVATED – Pressure Coupling Phase"
    else: return "MONITORING"

def generate_synthetic_seismic_data(n=24):
    now = dt.datetime.utcnow()
    return pd.DataFrame({
        "time": [now - dt.timedelta(hours=i) for i in range(n)],
        "magnitude": np.random.uniform(0.6, 1.3, n),
        "depth_km": np.random.uniform(0.8, 3.0, n)
    })

# ---------------- NOAA FETCH ----------------
@st.cache_data(ttl=600)
def fetch_noaa_kp():
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
@st.cache_data(show_spinner=False)
def load_seismic_data():
    end_time = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    start_time = (dt.datetime.utcnow() - dt.timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S")

    try:
        # INGV primary feed
        url = (f"https://webservices.ingv.it/fdsnws/event/1/query?"
               f"starttime={start_time}&endtime={end_time}"
               f"&latmin=40.7&latmax=40.9&lonmin=14.0&lonmax=14.3"
               f"&minmag=0&format=text")
        r = requests.get(url, timeout=API_TIMEOUT)
        r.raise_for_status()
        text_data = r.text.strip()

        if len(text_data) < 80:
            raise ValueError("INGV returned empty dataset")

        # Detect delimiter automatically
        first_line = text_data.splitlines()[0]
        delim = "|" if "|" in first_line else "," if "," in first_line else "\t"

        df = pd.read_csv(io.StringIO(text_data), delimiter=delim, comment="#")
        df.columns = [c.lower().strip().replace("/", "").replace("(", "").replace(")", "") for c in df.columns]

        time_col = next((c for c in df.columns if "time" in c or "date" in c), None)
        depth_col = next((c for c in df.columns if "depth" in c), None)
        mag_col = next((c for c in df.columns if "mag" in c and "type" not in c), None)

        if not all([time_col, depth_col, mag_col]):
            raise KeyError("INGV columns missing")

        df["time"] = pd.to_datetime(df[time_col], errors="coerce")
        df["depth_km"] = pd.to_numeric(df[depth_col], errors="coerce")
        df["magnitude"] = pd.to_numeric(df[mag_col], errors="coerce")
        df = df.dropna(subset=["time", "depth_km", "magnitude"])

        if df.empty:
            raise ValueError("INGV returned no valid rows")

        st.success(f"✅ INGV live feed active ({len(df)} events)")
        return df

    except Exception as e:
        st.warning(f"INGV fetch failed: {e}. Trying USGS fallback...")
        try:
            # USGS fallback
            usgs_url = (f"https://earthquake.usgs.gov/fdsnws/event/1/query?"
                        f"format=csv&starttime={start_time}&endtime={end_time}"
                        f"&minlatitude=40.7&maxlatitude=40.9&minlongitude=14.0&maxlongitude=14.3")
            usgs = requests.get(usgs_url, timeout=API_TIMEOUT)
            usgs.raise_for_status()
            df_usgs = pd.read_csv(io.StringIO(usgs.text))
            df_usgs["time"] = pd.to_datetime(df_usgs["time"], errors="coerce")
            df_usgs["depth_km"] = df_usgs["depth"]
            df_usgs["magnitude"] = df_usgs["mag"]
            st.info(f"USGS fallback feed active ({len(df_usgs)} events).")
            return df_usgs[["time", "depth_km", "magnitude"]].dropna()
        except Exception:
            st.warning("All live feeds unavailable — using synthetic continuity dataset.")
            return generate_synthetic_seismic_data()

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="SUPT Forecast Dashboard", layout="wide")
st.title("🌋 SUPT :: GROK Forecast Dashboard")
st.caption("v4.5 — Core Functional Build (Live INGV, NOAA, USGS Fallback)")

with st.spinner("Fetching data feeds..."):
    df = load_seismic_data()
    kp_df = fetch_noaa_kp()

psi_s = st.sidebar.slider("Solar Pressure Proxy (ψₛ)", 0.0, 1.0, DEFAULT_PSI)

# Core Calculations
EII = compute_eii(df["magnitude"].max(), df["magnitude"].mean(),
                  len(df[df["depth_km"] < 2.5]) / max(len(df), 1), psi_s)
RPAM = classify_phase(EII)

col1, col2 = st.columns(2)
col1.metric("EII", f"{EII:.3f}")
col2.metric("RPAM", RPAM)

# Simple 24h harmonic drift
df_plot = df.sort_values("time").tail(24)
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_plot["time"], y=df_plot["depth_km"],
                         mode="lines+markers", name="Depth (km)", line=dict(color="#FFA726")))
fig.add_trace(go.Scatter(x=df_plot["time"], y=df_plot["magnitude"],
                         mode="lines+markers", name="Magnitude", line=dict(color="#42A5F5")))
fig.update_layout(title="24h Harmonic Drift — Depth vs Magnitude",
                  xaxis_title="Time (UTC)", yaxis_title="Value", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# NOAA Kp
latest_kp = kp_df["Kp"].iloc[-1] if not kp_df.empty else 0
st.metric("Current Geomagnetic Kp", f"{latest_kp:.1f}")

# Footer
st.caption(f"Updated {dt.datetime.utcnow():%Y-%m-%d %H:%M:%S UTC}")
st.caption("Powered by SUPT — Sheppard’s Universal Proxy Theory (Functional Core)")
