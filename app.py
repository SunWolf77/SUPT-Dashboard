# ==========================================================
# 🌞🐺 SunWolf-SUPT v3.5 — Solar Gold + ψₛ Coupling + INGV Live
# Real-time coupling between Solar and Geothermal Systems
# Powered by SUPT ψ-Fold • NOAA • INGV • USGS
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import datetime as dt
from datetime import datetime, timezone
import plotly.graph_objs as go

# ==========================================================
# Utility
# ==========================================================
def live_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

st.set_page_config(page_title="SunWolf-SUPT v3.5", layout="wide")

# ==========================================================
# NOAA FEEDS
# ==========================================================
@st.cache_data(ttl=600)
def fetch_kp_index():
    try:
        url = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
        data = requests.get(url, timeout=15).json()
        header, rows = data[0], data[1:]
        df = pd.DataFrame(rows, columns=header)
        kp_col = [c for c in df.columns if "kp" in c.lower()][0]
        time_col = [c for c in df.columns if "time" in c.lower()][0]
        df["time"] = pd.to_datetime(df[time_col], errors="coerce")
        df["kp_index"] = pd.to_numeric(df[kp_col], errors="coerce")
        return df.dropna(subset=["kp_index"]).tail(24)
    except Exception as e:
        st.warning(f"NOAA Kp fetch failed: {e}")
        return pd.DataFrame(columns=["time", "kp_index"])

@st.cache_data(ttl=600)
def fetch_solar_wind():
    try:
        url = "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json"
        data = requests.get(url, timeout=15).json()
        df = pd.DataFrame(data[1:], columns=data[0])
        df["density"] = pd.to_numeric(df["density"], errors="coerce")
        df["speed"] = pd.to_numeric(df["speed"], errors="coerce")
        df["time_tag"] = pd.to_datetime(df["time_tag"], errors="coerce")
        return df.dropna(subset=["speed", "density"]).tail(96)
    except Exception as e:
        st.warning(f"Solar wind fetch failed: {e}")
        return pd.DataFrame(columns=["time_tag", "density", "speed"])

# ==========================================================
# INGV LIVE FDSN — CAMPi FLEGREI
# ==========================================================
@st.cache_data(ttl=900, show_spinner=False)
def fetch_ingv_quakes():
    """
    Real-time INGV FDSNWS query for Campi Flegrei (past 7 days).
    """
    try:
        end_time = dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
        start_time = (dt.datetime.utcnow() - dt.timedelta(days=7)).strftime('%Y-%m-%dT%H:%M:%S')
        url = (
            f"https://webservices.ingv.it/fdsnws/event/1/query?"
            f"starttime={start_time}&endtime={end_time}"
            f"&latmin=40.7&latmax=40.9&lonmin=14.0&lonmax=14.3"
            f"&minmag=0.0&maxmag=5.0&format=text"
        )
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text), delimiter="|", comment="#")
        df = df.rename(columns={
            "Time": "Time", "Latitude": "Lat", "Longitude": "Lon",
            "Depth(km)": "Depth/km", "Magnitude": "Mag",
            "Location": "Loc"
        })
        df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
        df["Mag"] = pd.to_numeric(df["Mag"], errors="coerce")
        df["Depth/km"] = pd.to_numeric(df["Depth/km"], errors="coerce")
        df = df.dropna(subset=["Depth/km", "Mag"])
        df = df[df["Time"] > (dt.datetime.utcnow() - dt.timedelta(days=7))]
        if df.empty:
            st.warning("⚠️ INGV returned no recent events — fallback activated.")
            return pd.DataFrame(columns=["Time", "Mag", "Depth/km", "Loc"])
        return df.tail(200)
    except Exception as e:
        st.warning(f"INGV API fetch failed: {e}. Using fallback dataset.")
        try:
            seismic_path = "data/seismic_local.csv"
            df = pd.read_csv(seismic_path)
            df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
            df["Mag"] = pd.to_numeric(df["MD"], errors="coerce")
            df["Depth/km"] = pd.to_numeric(df["Depth"], errors="coerce")
            return df[df["Time"] > (dt.datetime.utcnow() - dt.timedelta(days=7))]
        except Exception:
            return pd.DataFrame({
                "Time": [dt.datetime.utcnow()],
                "Mag": [0.9], "Depth/km": [1.8], "Loc": ["Synthetic Event"]
            })

# ==========================================================
# ETNA COMPARATIVE OVERLAY
# ==========================================================
@st.cache_data(ttl=900)
def fetch_etna_quakes():
    try:
        url = (
            "https://webservices.ingv.it/fdsnws/event/1/query?"
            "catalog=Etna&starttime=2025-09-01T00:00:00Z&endtime=now&format=csv"
        )
        df = pd.read_csv(url)
        df = df.rename(columns={"time": "Time", "depth": "Depth/km"})
        df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
        df["Depth/km"] = pd.to_numeric(df["Depth/km"], errors="coerce")
        return df.dropna(subset=["Depth/km"]).tail(200)
    except Exception:
        return pd.DataFrame(columns=["Time", "Depth/km"])

# ==========================================================
# USGS (Fallback)
# ==========================================================
@st.cache_data(ttl=600)
def fetch_usgs_quakes():
    try:
        df = pd.read_csv("https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.csv")
        df = df.rename(columns={"time": "Time", "mag": "Mag", "depth": "Depth/km", "place": "Loc"})
        df["Time"] = pd.to_datetime(df["Time"])
        return df
    except Exception:
        return pd.DataFrame(columns=["Time", "Mag", "Depth/km", "Loc"])

# ==========================================================
# Fallback Helper
# ==========================================================
def ensure_fallback(df, label):
    if df.empty:
        st.warning(f"⚠️ {label} feed unavailable — using synthetic sample.")
        if label == "NOAA":
            return pd.DataFrame({"time": [datetime.utcnow()], "kp_index": [1.0]})
        if label == "Solar Wind":
            return pd.DataFrame({"time_tag": [datetime.utcnow()], "density": [3.5], "speed": [430]})
        if label == "INGV":
            return pd.DataFrame({"Time": [datetime.utcnow()], "Mag": [0.8], "Depth/km": [1.7]})
    return df

# ==========================================================
# Fetch all live feeds
# ==========================================================
kp_df = ensure_fallback(fetch_kp_index(), "NOAA Kp")
sw_df = ensure_fallback(fetch_solar_wind(), "Solar Wind")
eq_df = ensure_fallback(fetch_ingv_quakes(), "INGV")
etna_df = fetch_etna_quakes()

feeds_status = (
    "🟢 NOAA | 🟢 INGV | 🟢 USGS"
    if not sw_df.empty else "⚠️ Partial feeds (fallback active)"
)

# ==========================================================
# Compute SUPT Metrics
# ==========================================================
def compute_supt_metrics(kp_df, sw_df, eq_df):
    if sw_df.empty or kp_df.empty:
        return 0, 0, 0, "NO DATA", 0, 0, "#EF9A9A"
    kp = kp_df["kp_index"].mean()
    sw_speed = sw_df["speed"].mean()
    sw_density = sw_df["density"].mean()
    eq_mag = eq_df["Mag"].mean() if not eq_df.empty else 0.0
    psi_s = min(1, (kp / 9 + sw_speed / 700) / 2)
    eii = min(1, (psi_s * 0.6 + (eq_mag / 5) * 0.4))
    alpha_r = 1 - psi_s * 0.8
    if eii <= 0.35:
        rpam, color = "STABLE", "#4FC3F7"
    elif eii <= 0.65:
        rpam, color = "TRANSITIONAL", "#FFB300"
    else:
        rpam, color = "CRITICAL", "#E53935"
    return psi_s, eii, alpha_r, rpam, sw_speed, sw_density, color

psi_s, eii, alpha_r, rpam_status, sw_speed, sw_density, color = compute_supt_metrics(
    kp_df, sw_df, eq_df
)

# ==========================================================
# HEADER
# ==========================================================
st.markdown(
    f"<h1 style='text-align:center; color:#FFA000;'>🌞🐺 SunWolf-SUPT: Solar Gold Forecast Dashboard</h1>"
    f"<p style='text-align:center; color:#FBC02D;'>Real-time coupling between Solar & Geothermal Systems — SUPT ψ-Fold Engine</p>"
    f"<p style='text-align:right; color:gray;'>🕒 Updated: {live_utc()}</p>",
    unsafe_allow_html=True,
)
st.markdown(f"<b>Data Feeds:</b> {feeds_status}", unsafe_allow_html=True)

# ==========================================================
# Dashboard Metrics
# ==========================================================
st.markdown(
    f"<div style='background-color:{color}; padding:10px; border-radius:8px; text-align:center; color:white;'>"
    f"<b>RPAM: {rpam_status}</b></div>",
    unsafe_allow_html=True,
)
c1, c2, c3, c4 = st.columns(4)
c1.metric("EII", f"{eii:.3f}")
c2.metric("ψₛ (Solar Coupling)", f"{psi_s:.3f}")
c3.metric("αᵣ (Damping)", f"{alpha_r:.3f}")
c4.metric("Phase", rpam_status)

# ==========================================================
# Gauges
# ==========================================================
g1, g2 = st.columns(2)
with g1:
    st.subheader("☀️ Solar Wind Speed (km/s)")
    fig1 = go.Figure(go.Indicator(
        mode="gauge+number", value=sw_speed,
        gauge={"axis": {"range": [250, 800]},
               "bar": {"color": color},
               "steps": [{"range": [250, 500], "color": "#FFF8E1"},
                         {"range": [500, 650], "color": "#FFD54F"},
                         {"range": [650, 800], "color": "#F4511E"}]},
        title={"text": "Plasma Velocity"}))
    st.plotly_chart(fig1, use_container_width=True)

with g2:
    st.subheader("🌫 Solar Wind Density (p/cm³)")
    fig2 = go.Figure(go.Indicator(
        mode="gauge+number", value=sw_density,
        gauge={"axis": {"range": [0, 20]},
               "bar": {"color": color},
               "steps": [{"range": [0, 5], "color": "#FFF8E1"},
                         {"range": [5, 10], "color": "#FFD54F"},
                         {"range": [10, 20], "color": "#F4511E"}]},
        title={"text": "Plasma Density"}))
    st.plotly_chart(fig2, use_container_width=True)

# ==========================================================
# ψₛ Harmonic Coupling Curve
# ==========================================================
st.markdown("### ☯ SUPT ψₛ Coupling — 24 h Harmonic Drift")
if not sw_df.empty:
    psi_hist = ((sw_df["speed"] / 700) + (sw_df["density"] / 10)) / 2
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=sw_df["time_tag"], y=psi_hist, mode="lines",
        line=dict(color="#FFB300", width=2.5), name="ψₛ Coupling Index"))
    fig_hist.update_layout(
        xaxis_title="UTC Time (last 24 h)", yaxis_title="ψₛ",
        yaxis=dict(range=[0, 1]), template="plotly_white", height=300)
    st.plotly_chart(fig_hist, use_container_width=True)
else:
    st.info("No solar wind history available yet — waiting for feed update.")

# ==========================================================
# Comparative Overlay — Etna vs Campi Flegrei
# ==========================================================
st.markdown("### 🌋 Regional Coupling Overlay — Etna vs Campi Flegrei")
try:
    if not etna_df.empty and not eq_df.empty:
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Box(y=eq_df["Depth/km"], name="Campi Flegrei", marker_color="#FFB300"))
        fig_cmp.add_trace(go.Box(y=etna_df["Depth/km"], name="Etna (Sicily)", marker_color="#81C784"))
        fig_cmp.update_layout(
            yaxis_title="Depth (km)",
            title="Volcanic Depth Distribution vs ψₛ Coupling Phase",
            template="plotly_white", height=300)
        st.plotly_chart(fig_cmp, use_container_width=True)
    else:
        st.info("Regional overlay waiting for feed refresh.")
except Exception:
    st.info("Etna overlay unavailable — retry after feed refresh.")

# ==========================================================
# Footer
# ==========================================================
st.markdown(
    f"<hr><p style='text-align:center; color:#FBC02D;'>Updated {live_utc()} | Feeds: {feeds_status} | Mode: Solar Gold ☀️ | SunWolf-SUPT v3.5</p>",
    unsafe_allow_html=True,
)
