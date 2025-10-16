# SUPT :: Live Forecast Dashboard App (Streamlit)
# Version: v2.1 Grok-ready with Live INGV & USGS Fetch

import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime as dt
import matplotlib.pyplot as plt
import io  # For parsing text responses

# -------------------------- CONFIG -----------------------------
API_TIMEOUT = 10  # seconds
DEFAULT_SOLAR = {
    "C_flare": 0.99,
    "M_flare": 0.55,
    "X_flare": 0.15,
    "psi_s": 0.72,
    "solar_speed": 688  # km/s
}
LOCAL_FALLBACK_CSV = "events_6.csv"  # Local CSV fallback path

# ----------------------- UTILITY FUNCTIONS ----------------------

def compute_eii(md_max, md_mean, shallow_ratio, psi_s):
    return np.clip((md_max * 0.2 + md_mean * 0.15 + shallow_ratio * 0.4 + psi_s * 0.25), 0, 1)

def classify_phase(EII):
    if EII >= 0.85:
        return "ACTIVE - Collapse Window Initiated"
    elif EII >= 0.6:
        return "ELEVATED - Pressure Coupling Phase"
    else:
        return "MONITORING"

# --------------------- LOAD SEISMIC DATA ------------------------
@st.cache_data(show_spinner=False)
def load_seismic_data():
    try:
        # Live INGV FDSNWS query for last 7 days in Campi Flegrei box
        end_time = dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
        start_time = (dt.datetime.utcnow() - dt.timedelta(days=7)).strftime('%Y-%m-%dT%H:%M:%S')
        ingv_url = f"https://webservices.ingv.it/fdsnws/event/1/query?starttime={start_time}&endtime={end_time}&latmin=40.7&latmax=40.9&lonmin=14.0&lonmax=14.3&format=text"
        ingv_response = requests.get(ingv_url, timeout=API_TIMEOUT)
        ingv_response.raise_for_status()
        # Parse pipe-separated text from INGV
        df_ingv = pd.read_csv(io.StringIO(ingv_response.text), delimiter="|")
        df_ingv['time'] = pd.to_datetime(df_ingv['Time'])  # Convert 'Time' to datetime
        df_ingv['magnitude'] = df_ingv['Magnitude']
        df_ingv['depth_km'] = df_ingv['Depth/Km']
        return df_ingv[df_ingv['time'] > dt.datetime.utcnow() - dt.timedelta(days=7)]  # Re-filter
    except Exception as e:
        st.warning(f"INGV API fetch failed: {e}. Trying USGS fallback...")

        try:
            # USGS FDSNWS fallback query for last 7 days in Campi Flegrei box (format=csv)
            usgs_url = f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime={start_time}&endtime={end_time}&minlatitude=40.7&maxlatitude=40.9&minlongitude=14.0&maxlongitude=14.3"
            usgs_response = requests.get(usgs_url, timeout=API_TIMEOUT)
            usgs_response.raise_for_status()
            df_usgs = pd.read_csv(io.StringIO(usgs_response.text))
            df_usgs['time'] = pd.to_datetime(df_usgs['time'])  # Convert 'time' to datetime
            df_usgs['magnitude'] = df_usgs['mag']
            df_usgs['depth_km'] = df_usgs['depth']
            return df_usgs[df_usgs['time'] > dt.datetime.utcnow() - dt.timedelta(days=7)]  # Re-filter
        except Exception as e:
            st.warning(f"USGS API fetch failed: {e}. Using local CSV fallback.")
            # Fallback to local CSV
            df_local = pd.read_csv(LOCAL_FALLBACK_CSV)
            df_local['time'] = pd.to_datetime(df_local['Time'])  # Adjust for local CSV format
            df_local['magnitude'] = df_local['MD']
            df_local['depth_km'] = df_local['Depth']
            return df_local[df_local['time'] > dt.datetime.utcnow() - dt.timedelta(days=7)]

# -------------------- MAIN DASHBOARD LOGIC ---------------------
st.set_page_config(layout="centered", page_title="SUPT :: GROK Forecast")
st.title("SUPT :: GROK Forecast Dashboard")
st.caption("Campi Flegrei Risk & Energetic Instability Monitor :: v2.1")

with st.spinner("Loading seismic data..."):
    df = load_seismic_data()

if df.empty:
    st.error("No seismic data loaded. Check API or local source.")
    st.stop()

# METRIC CALCS
md_max = df['magnitude'].max()
md_mean = df['magnitude'].mean()
depth_mean = df['depth_km'].mean()
shallow_ratio = len(df[df['depth_km'] < 2.5]) / len(df) if len(df) > 0 else 0

# SOLAR INPUT (Manual override or live input)
st.sidebar.header("Solar Activity Input")
psi_s = st.sidebar.slider("Solar Pressure Proxy (ψₛ)", 0.0, 1.0, DEFAULT_SOLAR["psi_s"])
solar_speed = st.sidebar.number_input("Solar Wind Speed (km/s)", value=DEFAULT_SOLAR["solar_speed"])
C_prob = st.sidebar.slider("C-Flare Probability", 0.0, 1.0, DEFAULT_SOLAR["C_flare"])
M_prob = st.sidebar.slider("M-Flare Probability", 0.0, 1.0, DEFAULT_SOLAR["M_flare"])
X_prob = st.sidebar.slider("X-Flare Probability", 0.0, 1.0, DEFAULT_SOLAR["X_flare"])

# COMPUTE
EII = compute_eii(md_max, md_mean, shallow_ratio, psi_s)
RPAM_STATUS = classify_phase(EII)
COLLAPSE_WINDOW = "Q1 2026" if RPAM_STATUS == "ACTIVE - Collapse Window Initiated" else "N/A"

# DISPLAY
st.metric("Energetic Instability Index (EII)", f"{EII:.3f}")
st.metric("RPAM Status", RPAM_STATUS)
st.metric("Collapse Window", COLLAPSE_WINDOW)

st.subheader("Seismic Snapshot (past 7 days)")
st.write(df[['time', 'magnitude', 'depth_km']].tail(15))

# PLOT
fig, ax = plt.subplots()
ax.hist(df['depth_km'], bins=15, color='orange', edgecolor='black')
ax.set_xlabel("Depth (km)")
ax.set_ylabel("Quake Count")
ax.set_title("Depth Distribution")
st.pyplot(fig)

# Solar status box
st.subheader("Current Solar Conditions")
st.write(f"**Solar Wind Speed**: {solar_speed} km/s")
st.write(f"**Flare Probabilities**: C: {C_prob}, M: {M_prob}, X: {X_prob}")

# Footer
st.caption("Powered by SUPT - Sheppard's Universal Proxy Theory")
st.caption("Forecast parameters editable. Grok API-ready.")
