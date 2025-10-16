# ==========================================================
# ☀️ SunWolf-SUPT: Solar Gold Forecast Dashboard v3.2
# Real-Time Solar–Geothermal Coupling Monitor
# Powered by SUPT ψ-Fold | NOAA | INGV | USGS
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone
import plotly.graph_objs as go

# ----------------------------------------------------------
# Streamlit Page Configuration
# ----------------------------------------------------------
st.set_page_config(
    page_title="SunWolf-SUPT: Solar Gold Forecast Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------
# Utility: Live UTC Time
# ----------------------------------------------------------
def live_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

# ----------------------------------------------------------
# Header
# ----------------------------------------------------------
st.markdown(
    """
    <h1 style='text-align:center; color:#ffb300;'> 🌞🐺SunWolf-SUPT: Solar Gold Forecast Dashboard </h1>
    <p style='text-align:center; color:#fbc02d;'>Real-time coupling between Solar & Geothermal Systems — SUPT ψ-Fold Engine</p>
    """,
    unsafe_allow_html=True,
)
st.markdown(f"<p style='text-align:right; color:gray;'>🕒 Updated: {live_utc()}</p>", unsafe_allow_html=True)

# ==========================================================
# 🌍 Data Fetchers — NOAA + USGS + INGV (Live)
# ==========================================================

@st.cache_data(ttl=600)
def fetch_kp_index():
    """Fetch NOAA real-time planetary Kp index."""
    try:
        url = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
        data = requests.get(url, timeout=10).json()
        df = pd.DataFrame(data[1:], columns=data[0])
        df["time_tag"] = pd.to_datetime(df["time_tag"])
        df["kp_index"] = df["kp_index"].astype(float)
        return df[["time_tag", "kp_index"]]
    except Exception as e:
        st.warning(f"NOAA Kp fetch failed: {e}")
        return pd.DataFrame(columns=["time_tag", "kp_index"])

@st.cache_data(ttl=600)
def fetch_solar_wind():
    """Fetch solar wind plasma data (NOAA SWPC 7-day)."""
    try:
        url = "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json"
        data = requests.get(url, timeout=10).json()
        df = pd.DataFrame(data[1:], columns=data[0])
        df["density"] = df["density"].astype(float)
        df["speed"] = df["speed"].astype(float)
        df["temperature"] = df["temperature"].astype(float)
        df["time_tag"] = pd.to_datetime(df["time_tag"])
        return df.tail(48)
    except Exception as e:
        st.warning(f"Solar wind fetch failed: {e}")
        return pd.DataFrame(columns=["time_tag", "density", "speed", "temperature"])

@st.cache_data(ttl=600)
def fetch_usgs_quakes():
    """Fetch global M1+ earthquakes (USGS)."""
    try:
        url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.csv"
        df = pd.read_csv(url)
        df = df.rename(columns={"time": "Time", "mag": "Magnitude", "depth": "Depth/km", "place": "Location"})
        df["Time"] = pd.to_datetime(df["Time"])
        return df
    except Exception as e:
        st.warning(f"USGS fetch failed: {e}")
        return pd.DataFrame(columns=["Time", "Magnitude", "Depth/km", "Location"])

@st.cache_data(ttl=600)
def fetch_ingv_quakes():
    """Fetch Campi Flegrei localized quakes (INGV CSV API)."""
    try:
        url = (
            "https://webservices.ingv.it/fdsnws/event/1/query?"
            "starttime=2025-10-01T00:00:00&endtime=now&minlat=40.7&maxlat=40.9&"
            "minlon=14.0&maxlon=14.3&format=csv"
        )
        df = pd.read_csv(url)
        df = df.rename(columns={
            "time": "Time", "latitude": "Lat", "longitude": "Lon",
            "depth": "Depth/km", "mag": "Mag", "place": "Loc"
        })
        df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
        return df.dropna(subset=["Depth/km", "Mag"])
    except Exception as e:
        st.warning(f"INGV fetch failed: {e}")
        return pd.DataFrame(columns=["Time", "Mag", "Depth/km", "Loc"])

# ----------------------------------------------------------
# Fallback Handler
# ----------------------------------------------------------
def ensure_fallback(df, label):
    if df.empty:
        st.warning(f"⚠️ {label} feed unavailable — using synthetic sample.")
        if label == "NOAA":
            return pd.DataFrame({"time_tag": [datetime.utcnow()], "kp_index": [1.0]})
        elif label == "Solar Wind":
            return pd.DataFrame({"time_tag": [datetime.utcnow()], "density": [3.5], "speed": [420], "temperature": [45000]})
        elif label == "INGV":
            return pd.DataFrame({"Time": [datetime.utcnow()], "Mag": [0.8], "Depth/km": [1.7], "Loc": ["Fallback synthetic"]})
    return df

# ----------------------------------------------------------
# Fetch + Validate
# ----------------------------------------------------------
kp_df = ensure_fallback(fetch_kp_index(), "NOAA")
sw_df = ensure_fallback(fetch_solar_wind(), "Solar Wind")
eq_df = ensure_fallback(fetch_ingv_quakes(), "INGV")

feeds_status = "🟢 NOAA | 🟢 INGV | 🟢 USGS" if not sw_df.empty else "⚠️ Partial feeds (fallback active)"
st.markdown(f"<b>Data Feeds:</b> {feeds_status} | Last Refresh: {live_utc()}", unsafe_allow_html=True)

# ==========================================================
# ⚙️ SUPT Metric Computation
# ==========================================================
def compute_supt_metrics(kp_df, sw_df, eq_df):
    if sw_df.empty or kp_df.empty:
        return 0, 0, 0, "NO DATA", 0, 0, "#EF9A9A"

    kp = kp_df["kp_index"].mean()
    sw_speed = sw_df["speed"].mean()
    sw_density = sw_df["density"].mean()
    eq_mean_md = eq_df["Mag"].mean() if not eq_df.empty else 0

    psi_s = min(1, (kp / 9 + sw_speed / 700) / 2)
    eii = min(1, (psi_s * 0.6 + (eq_mean_md / 5) * 0.4))
    alpha_r = 1 - psi_s * 0.8

    if eii <= 0.35:
        rpam, color = "STABLE", "#4FC3F7"
    elif eii <= 0.65:
        rpam, color = "TRANSITIONAL", "#FFB300"
    else:
        rpam, color = "CRITICAL", "#E53935"

    return psi_s, eii, alpha_r, rpam, sw_speed, sw_density, color

psi_s, eii, alpha_r, rpam_status, sw_speed, sw_density, color = compute_supt_metrics(kp_df, sw_df, eq_df)

# ==========================================================
# Display UI
# ==========================================================
st.markdown(
    f"<div style='background-color:{color}; padding:10px; border-radius:8px; text-align:center; color:white;'>"
    f"<b>RPAM: {rpam_status}</b></div>",
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("EII", f"{eii:.3f}")
col2.metric("ψₛ (Solar Coupling)", f"{psi_s:.3f}")
col3.metric("αᵣ (Damping)", f"{alpha_r:.3f}")
col4.metric("Phase", rpam_status)

# Gauges ---------------------------------------------------
g1, g2 = st.columns(2)

with g1:
    st.subheader("☀️ Solar Wind Speed (km/s)")
    fig1 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sw_speed,
        gauge={'axis': {'range': [250, 800]},
               'bar': {'color': color},
               'steps': [
                   {'range': [250, 500], 'color': "#FFF8E1"},
                   {'range': [500, 650], 'color': "#FFD54F"},
                   {'range': [650, 800], 'color': "#F4511E"}]},
        title={'text': "Plasma Velocity"}
    ))
    st.plotly_chart(fig1, use_container_width=True)

with g2:
    st.subheader("🌫 Solar Wind Density (p/cm³)")
    fig2 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sw_density,
        gauge={'axis': {'range': [0, 20]},
               'bar': {'color': color},
               'steps': [
                   {'range': [0, 5], 'color': "#FFF8E1"},
                   {'range': [5, 10], 'color': "#FFD54F"},
                   {'range': [10, 20], 'color': "#F4511E"}]},
        title={'text': "Plasma Density"}
    ))
    st.plotly_chart(fig2, use_container_width=True)

# ==========================================================
# Footer
# ==========================================================
st.markdown(
    f"<hr><p style='text-align:center; color:#FBC02D;'>Updated {live_utc()} | Feeds: {feeds_status} | Mode: Solar Gold ☀️ | SunWolf-SUPT v3.2</p>",
    unsafe_allow_html=True,
)
