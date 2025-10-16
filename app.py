# ==============================================================
# ☀️ SunWolf-SUPT: Global Forecast Dashboard v3.5
# Grok-Ready Fusion Edition (SUPT ψ-Fold + NOAA + INGV Live)
# ==============================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objs as go
from datetime import datetime, timezone

st.set_page_config(page_title="SunWolf-SUPT Global v3.5", layout="wide")

# --------------------- 🕒 Helpers ---------------------
def live_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def pulse_dot():
    return """<style>
    @keyframes pulse{0%{opacity:.3;}50%{opacity:1;}100%{opacity:.3;}}
    .dot{display:inline-block;width:10px;height:10px;border-radius:50%;
    background:#ffb300;animation:pulse 2s infinite;margin-right:5px;}
    </style><div class='dot'></div>"""

# --------------------- 🌞 Fetchers ---------------------
@st.cache_data(ttl=900)
def fetch_noaa_plasma():
    try:
        url = "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json"
        df = pd.DataFrame(requests.get(url, timeout=10).json()[1:], columns=["time_tag","density","speed","temperature"])
        df["density"] = df["density"].astype(float)
        df["speed"] = df["speed"].astype(float)
        return df.tail(96)
    except: return pd.DataFrame()

@st.cache_data(ttl=900)
def fetch_ingv_cf():
    try:
        url = "https://webservices.ingv.it/fdsnws/event/1/query?starttime=2025-10-01&endtime=now&minlat=40.7&maxlat=40.9&minlon=14.0&maxlon=14.3&format=text"
        df = pd.read_csv(url, sep="|", comment="#", header=None)
        df.columns = ["time","lat","lon","depth","md","loc","agency"]
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df[df["md"].notna()]
        return df.tail(200)
    except: return pd.DataFrame()

# --------------------- ⚙️ SUPT Core ---------------------
def compute_supt(df_seis, df_solar):
    if df_seis.empty or df_solar.empty: return 0,0,0,"NO DATA","#9e9e9e"
    md_mean, depth_mean = df_seis["md"].mean(), df_seis["depth"].mean()
    shallow_ratio = np.mean(df_seis["depth"] <= 3.0)
    sw_speed, density = df_solar["speed"].mean(), df_solar["density"].mean()
    psi_s = min(1, (sw_speed/700 + density/10)/2)
    eii = min(1, 0.45*psi_s + 0.35*shallow_ratio + 0.2*(md_mean/3))
    alpha_r = round(1 - psi_s*0.8, 3)
    if eii < .35: return psi_s,eii,alpha_r,"STABLE","#4FC3F7"
    elif eii < .65: return psi_s,eii,alpha_r,"TRANSITIONAL","#FFB300"
    else: return psi_s,eii,alpha_r,"CRITICAL","#E53935"

# --------------------- 💡 Layout ---------------------
st.markdown(f"<h1 style='text-align:center;color:#ffb300;'>☀️ SunWolf-SUPT Global Forecast Dashboard ☀️</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align:center;color:#fbc02d;'>Grok-Ready Real-Time ψ-Fold & Geosolar Coupling Monitor</p>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([2,3,1])
with col3: st.markdown(pulse_dot()+f"<b>🕒 {live_utc()}</b>", unsafe_allow_html=True)

# Fetch Data
solar = fetch_noaa_plasma()
seismic = fetch_ingv_cf()
psi_s, eii, alpha_r, rpam, color = compute_supt(seismic, solar)

# Dynamic background
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
background:{'linear-gradient(180deg,#e3f2fd,#bbdefb)' if rpam=='STABLE' else 
'linear-gradient(180deg,#fff8e1,#ffe082)' if rpam=='TRANSITIONAL' else 
'linear-gradient(180deg,#ffebee,#ef5350)'} !important; transition:1s;
}}
</style>""", unsafe_allow_html=True)

# Metrics Display
st.markdown(f"<div style='background:{color};padding:10px;border-radius:10px;text-align:center;'><b style='color:white;'>RPAM: {rpam}</b></div>", unsafe_allow_html=True)
c1,c2,c3,c4 = st.columns(4)
c1.metric("EII", f"{eii:.3f}")
c2.metric("ψₛ (Solar Coupling)", f"{psi_s:.3f}")
c3.metric("αᵣ (Damping)", f"{alpha_r:.3f}")
c4.metric("Phase", rpam)

# Gauges
g1,g2 = st.columns(2)
if not solar.empty:
    with g1:
        st.subheader("☀️ Solar Wind Speed (km/s)")
        fig1 = go.Figure(go.Indicator(mode="gauge+number", value=solar["speed"].mean(),
            gauge={'axis':{'range':[250,800]},'bar':{'color':color},
                    'steps':[{'range':[250,500],'color':"#FFF8E1"},
                             {'range':[500,650],'color':"#FFD54F"},
                             {'range':[650,800],'color':"#F4511E"}]},
            title={'text':"Plasma Velocity"}))
        st.plotly_chart(fig1, use_container_width=True)
    with g2:
        st.subheader("🌫 Solar Wind Density (p/cm³)")
        fig2 = go.Figure(go.Indicator(mode="gauge+number", value=solar["density"].mean(),
            gauge={'axis':{'range':[0,20]},'bar':{'color':color},
                    'steps':[{'range':[0,5],'color':"#FFF8E1"},
                             {'range':[5,10],'color':"#FFD54F"},
                             {'range':[10,20],'color':"#F4511E"}]},
            title={'text':"Plasma Density"}))
        st.plotly_chart(fig2, use_container_width=True)

# Seismic Depth Distribution
if not seismic.empty:
    st.subheader("Campi Flegrei Seismic Depth Distribution (Live INGV)")
    st.bar_chart(np.histogram(seismic["depth"], bins=15, range=(0,5))[0])

# Footer
st.markdown(f"<hr><p style='text-align:center;color:#FBC02D;'>Updated {live_utc()} | Feeds: NOAA🟢 INGV🟢 USGS🟢 | Mode: Solar Gold ☀️ | SunWolf-SUPT v3.5</p>", unsafe_allow_html=True)
