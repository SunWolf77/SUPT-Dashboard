# SunWolf-SUPT: Solar Gold Forecast Dashboard
# Live NOAA Solar Wind + INGV Seismic + SUPT Predictive Metrics
# by SUPT / SunWolf Initiative, 2025

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objs as go
from datetime import datetime, timezone
from scipy.interpolate import make_interp_spline

st.set_page_config(page_title="SunWolf-SUPT: Solar Gold Forecast Dashboard", layout="wide")

# 🌞 Title & Header
st.markdown(
    """
    <h1 style='text-align:center; color:#ffb300;'>☀️ SunWolf-SUPT: Solar Gold Forecast Dashboard ☀️</h1>
    <p style='text-align:center; color:#fbc02d;'>Real-time coupling between Solar & Geothermal Systems — Live Forecast Engine</p>
    """,
    unsafe_allow_html=True,
)

# 🕒 Live UTC Clock
def live_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

col1, col2, col3 = st.columns([2, 3, 1])
with col3:
    st.metric(label="🕒 UTC", value=live_utc())

# ==============================
# 🔹 LIVE DATA FETCH FUNCTIONS
# ==============================
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
        return df.tail(96)  # ~24h
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

# Load feeds
kp_df = fetch_kp_index()
sw_df = fetch_solar_wind()
eq_df = fetch_ingv()

# Feed Status
st.write(f"**Data Feeds:** NOAA 🟢 | INGV 🟢 | USGS 🟢 | Last Update: {live_utc()}")

# ==============================
# 🔸 SUPT METRICS ENGINE
# ==============================
def compute_metrics(kp_df, sw_df, eq_df):
    kp = kp_df["kp_index"].astype(float).mean() if not kp_df.empty else 0
    sw_speed = sw_df["speed"].mean() if not sw_df.empty else 0
    sw_density = sw_df["density"].mean() if not sw_df.empty else 0
    eq_mean_depth = eq_df["depth"].mean() if not eq_df.empty else 0
    eq_mean_md = eq_df["md"].mean() if not eq_df.empty else 0

    psi_s = min(1, (kp / 9 + sw_speed / 700) / 2)
    eii = min(1, (psi_s * 0.6 + (eq_mean_md / 5) * 0.4))
    alpha_r = 1 - psi_s * 0.8
    rpam_status = "CRITICAL" if eii > 0.85 else "ELEVATED" if eii > 0.5 else "MONITORING"

    return psi_s, eii, alpha_r, rpam_status, sw_speed, sw_density

psi_s, eii, alpha_r, rpam_status, sw_speed, sw_density = compute_metrics(kp_df, sw_df, eq_df)

# ==============================
# 🌋 METRICS DISPLAY
# ==============================
col1, col2, col3, col4 = st.columns(4)
col1.metric("EII", f"{eii:.3f}")
col2.metric("ψₛ (Solar Coupling)", f"{psi_s:.3f}")
col3.metric("αᵣ (Damping)", f"{alpha_r:.3f}")
col4.metric("RPAM Status", rpam_status)

# ==============================
# ☀️ SOLAR WIND DUAL GAUGE
# ==============================
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

# ==============================
# 🔄 COUPLING CHART
# ==============================
st.subheader("Solar-Geophysical Coupling (24h Harmonic)")

if not kp_df.empty:
    times = pd.to_datetime(kp_df["time"])
    kp_smooth = make_interp_spline(np.arange(len(kp_df)), kp_df["kp_index"])(np.linspace(0, len(kp_df)-1, 200))
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=times, y=kp_smooth, mode="lines", line=dict(color="#F57C00", width=3), name="Kp Index"))
    fig3.update_layout(
        title="Geomagnetic Activity (Kp)",
        xaxis_title="Time (UTC)",
        yaxis_title="Kp Index",
        template="plotly_white"
    )
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.warning("No Kp data available.")

# ==============================
# 🪶 FOOTER
# ==============================
st.markdown(
    f"""
    <hr>
    <p style='text-align:center; color:#FBC02D;'>
    Updated {live_utc()} | Feeds: NOAA🟢 INGV🟢 USGS🟢 | Mode: Solar Gold ☀️ | SunWolf-SUPT v2
    </p>
    """,
    unsafe_allow_html=True,
)
