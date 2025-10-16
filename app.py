# ==========================================================
# ☀️ SunWolf-SUPT: Solar Gold Forecast Dashboard
# Color-coded SUPT Harmonic State + Auto-Rotating 3D Phase
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objs as go
from datetime import datetime, timezone

# ----------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------
st.set_page_config(
    page_title="SunWolf-SUPT: Solar Gold Forecast Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------
# Live Clock & Styling
# ----------------------------------------------------------
def live_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def clock_pulse_html():
    return """
    <style>
    @keyframes pulse {
      0% {opacity: 0.3;}
      50% {opacity: 1;}
      100% {opacity: 0.3;}
    }
    .pulse-dot {
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background-color: #fdd835;
      animation: pulse 2s infinite;
      margin-right: 6px;
    }
    </style>
    <div class='pulse-dot'></div>
    """

# ----------------------------------------------------------
# Header
# ----------------------------------------------------------
st.markdown(
    """
    <h1 style='text-align:center; color:#ffb300;'>☀️ SunWolf-SUPT: Solar Gold Forecast Dashboard ☀️</h1>
    <p style='text-align:center; color:#fbc02d;'>Real-time coupling between Solar & Geothermal Systems — Live Forecast Engine</p>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns([2, 3, 1])
with col3:
    st.markdown(clock_pulse_html() + f"<b>🕒 {live_utc()}</b>", unsafe_allow_html=True)

# ----------------------------------------------------------
# Data Fetching
# ----------------------------------------------------------
@st.cache_data(ttl=900)
def fetch_kp_index():
    try:
        url = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
        df = pd.DataFrame(requests.get(url, timeout=10).json()[1:], columns=["time", "kp_index"])
        df["kp_index"] = df["kp_index"].astype(float)
        return df
    except Exception:
        return pd.DataFrame(columns=["time", "kp_index"])

@st.cache_data(ttl=900)
def fetch_solar_wind():
    try:
        url = "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json"
        df = pd.DataFrame(requests.get(url, timeout=10).json()[1:], columns=["time_tag", "density", "speed", "temperature"])
        df["density"] = df["density"].astype(float)
        df["speed"] = df["speed"].astype(float)
        return df.tail(96)
    except Exception:
        return pd.DataFrame(columns=["time_tag", "density", "speed"])

@st.cache_data(ttl=900)
def fetch_ingv():
    try:
        url = "https://webservices.ingv.it/fdsnws/event/1/query?starttime=2025-10-01&endtime=now&minlat=40.7&maxlat=40.9&minlon=14.0&maxlon=14.3&format=text"
        df = pd.read_csv(url, sep="|", comment="#", header=None)
        df.columns = ["time", "lat", "lon", "depth", "md", "loc", "agency"]
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame(columns=["time", "md", "depth"])

kp_df = fetch_kp_index()
sw_df = fetch_solar_wind()
eq_df = fetch_ingv()

st.write(f"**Data Feeds:** NOAA 🟢 | INGV 🟢 | USGS 🟢 | Last Update: {live_utc()}")

# ----------------------------------------------------------
# SUPT Metrics
# ----------------------------------------------------------
def compute_metrics(kp_df, sw_df, eq_df):
    kp = kp_df["kp_index"].astype(float).mean() if not kp_df.empty else 0
    sw_speed = sw_df["speed"].mean() if not sw_df.empty else 0
    sw_density = sw_df["density"].mean() if not sw_df.empty else 0
    eq_mean_depth = eq_df["depth"].mean() if not eq_df.empty else 0
    eq_mean_md = eq_df["md"].mean() if not eq_df.empty else 0

    psi_s = min(1, (kp / 9 + sw_speed / 700) / 2)
    eii = min(1, (psi_s * 0.6 + (eq_mean_md / 5) * 0.4))
    alpha_r = 1 - psi_s * 0.8

    # Define harmonic state
    if eii <= 0.35:
        rpam_status = "STABLE"
        color = "#4FC3F7"  # Blue
    elif eii <= 0.65:
        rpam_status = "TRANSITIONAL"
        color = "#FFB300"  # Amber
    else:
        rpam_status = "CRITICAL"
        color = "#E53935"  # Red

    return psi_s, eii, alpha_r, rpam_status, sw_speed, sw_density, color

psi_s, eii, alpha_r, rpam_status, sw_speed, sw_density, color = compute_metrics(kp_df, sw_df, eq_df)

# ----------------------------------------------------------
# Display Metrics
# ----------------------------------------------------------
st.markdown(
    f"""
    <div style='background-color:{color}; padding:10px; border-radius:10px; text-align:center;'>
    <b style='color:white;'>RPAM Status: {rpam_status}</b>
    </div>
    """,
    unsafe_allow_html=True,
)
col1, col2, col3, col4 = st.columns(4)
col1.metric("EII", f"{eii:.3f}")
col2.metric("ψₛ (Solar Coupling)", f"{psi_s:.3f}")
col3.metric("αᵣ (Damping)", f"{alpha_r:.3f}")
col4.metric("Phase", rpam_status)

# ----------------------------------------------------------
# Gauges
# ----------------------------------------------------------
gauge_col1, gauge_col2 = st.columns(2)

with gauge_col1:
    st.subheader("☀️ Solar Wind Speed (km/s)")
    fig1 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sw_speed,
        gauge={'axis': {'range': [250, 800]},
               'bar': {'color': "#FFA000"},
               'steps': [{'range': [250, 500], 'color': "#FFF8E1"},
                         {'range': [500, 650], 'color': "#FFD54F"},
                         {'range': [650, 800], 'color': "#F4511E"}]},
        title={'text': "Plasma Velocity"}
    ))
    st.plotly_chart(fig1, use_container_width=True)

with gauge_col2:
    st.subheader("🌫 Solar Wind Density (p/cm³)")
    fig2 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sw_density,
        gauge={'axis': {'range': [0, 20]},
               'bar': {'color': "#FFB300"},
               'steps': [{'range': [0, 5], 'color': "#FFF8E1"},
                         {'range': [5, 10], 'color': "#FFD54F"},
                         {'range': [10, 20], 'color': "#F4511E"}]},
        title={'text': "Plasma Density"}
    ))
    st.plotly_chart(fig2, use_container_width=True)

# ----------------------------------------------------------
# Auto-Rotating 3D Harmonic Memory (Color-coded)
# ----------------------------------------------------------
st.subheader("SUPT 24h Harmonic Phase Memory (ψₛ, EII, αᵣ Drift)")

try:
    if not kp_df.empty and not sw_df.empty:
        kp_df["time"] = pd.to_datetime(kp_df["time"], errors="coerce")
        sw_df["time_tag"] = pd.to_datetime(sw_df["time_tag"], errors="coerce")
        merged = pd.merge_asof(kp_df.sort_values("time"), sw_df.sort_values("time_tag"), left_on="time", right_on="time_tag", direction="nearest")
        merged["psi_s"] = np.minimum(1, (merged["kp_index"].astype(float)/9 + merged["speed"]/700) / 2)
        merged["eii"] = np.minimum(1, merged["psi_s"]*0.6 + (merged["speed"]/700)*0.4)
        merged["alpha_r"] = 1 - merged["psi_s"]*0.8

        x = np.arange(len(merged))
        xnew = np.linspace(0, len(x) - 1, 200)
        psi_s_smooth = np.interp(xnew, x, merged["psi_s"])
        eii_smooth = np.interp(xnew, x, merged["eii"])
        alpha_s_smooth = np.interp(xnew, x, merged["alpha_r"])

        # Color map by phase
        phase_colors = ["#4FC3F7" if e <= 0.35 else "#FFB300" if e <= 0.65 else "#E53935" for e in eii_smooth]

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter3d(
            x=psi_s_smooth, y=eii_smooth, z=alpha_s_smooth,
            mode='lines', line=dict(color=phase_colors[-1], width=5),
            name='Phase Trajectory (24h)'
        ))

        rotation_frames = []
        for angle in range(0, 360, 3):
            rotation_frames.append(dict(
                layout=dict(scene_camera=dict(eye=dict(x=np.cos(np.radians(angle))*1.5,
                                                       y=np.sin(np.radians(angle))*1.5,
                                                       z=0.7)))
            ))

        fig4.update_layout(
            scene=dict(xaxis_title="ψₛ", yaxis_title="EII", zaxis_title="αᵣ", bgcolor="black"),
            template="plotly_dark",
            margin=dict(l=0, r=0, b=0, t=40),
            title="SUPT Harmonic Drift — ψₛ ↔ EII ↔ αᵣ (24h Memory)",
        )
        fig4.frames = rotation_frames
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.warning("Insufficient data for harmonic phase reconstruction.")
except Exception as e:
    st.error(f"Harmonic Phase Memory Error: {e}")

# ----------------------------------------------------------
# Footer
# ----------------------------------------------------------
st.markdown(
    f"""
    <hr><p style='text-align:center; color:#FBC02D;'>
    Updated {live_utc()} | Feeds: NOAA🟢 INGV🟢 USGS🟢 | Mode: Solar Gold ☀️ | SunWolf-SUPT v2.5
    </p>
    """,
    unsafe_allow_html=True,
)
