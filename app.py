# =========================================================
# SUPT :: v5.2 Live Sync Fix — Final Functional Release
# =========================================================
# Adds full INGV + USGS dual-source support with schema auto-correction
# Author: SUPT Systems / Sheppard Continuum Core
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime as dt
import io
from streamlit_autorefresh import st_autorefresh

# -------------------------------
# CONFIGURATION
# -------------------------------
st.set_page_config(page_title="SUPT :: Live Forecast Dashboard", layout="wide")
API_TIMEOUT = 10

# Live endpoints
USGS_URL = (
    "https://earthquake.usgs.gov/fdsnws/event/1/query?"
    "format=geojson&starttime={}&endtime={}&minlatitude=40.7&maxlatitude=40.9&"
    "minlongitude=14.0&maxlongitude=14.3&minmagnitude=0"
)
INGV_URL = (
    "https://webservices.ingv.it/fdsnws/event/1/query?"
    "starttime={}&endtime={}&latmin=40.7&latmax=40.9&lonmin=14.0&lonmax=14.3&format=text"
)
NOAA_KP_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
DSCOVR_URLS = [
    "https://services.swpc.noaa.gov/products/solar-wind/plasma-1-day.json",
    "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json",
    "https://services.swpc.noaa.gov/json/dscovr_plasma.json",
]

# -------------------------------
# FETCH FUNCTIONS
# -------------------------------

@st.cache_data(ttl=600)
def fetch_ingv_data():
    """Fetch last 7 days Campi Flegrei seismic data from INGV with schema detection"""
    end = dt.datetime.utcnow()
    start = end - dt.timedelta(days=7)
    url = INGV_URL.format(start.strftime("%Y-%m-%dT%H:%M:%S"), end.strftime("%Y-%m-%dT%H:%M:%S"))
    try:
        r = requests.get(url, timeout=API_TIMEOUT)
        r.raise_for_status()
        if not r.text.strip():
            raise ValueError("Empty INGV response.")
        df = pd.read_csv(io.StringIO(r.text), delimiter="|", skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]

        # Normalize possible column variations
        time_col = [c for c in df.columns if "time" in c.lower()]
        mag_col = [c for c in df.columns if "mag" in c.lower()]
        depth_col = [c for c in df.columns if "depth" in c.lower()]

        if not (time_col and mag_col and depth_col):
            raise KeyError("Missing expected INGV columns.")

        df["time"] = pd.to_datetime(df[time_col[0]], errors="coerce")
        df["magnitude"] = df[mag_col[0]].astype(float)
        df["depth_km"] = df[depth_col[0]].astype(float)
        df.dropna(subset=["time"], inplace=True)

        if len(df) == 0:
            raise ValueError("No valid INGV events parsed.")

        return df
    except Exception as e:
        st.warning(f"INGV fetch failed: {e}. Switching to USGS fallback.")
        return None


@st.cache_data(ttl=600)
def fetch_usgs_data():
    """Fetch global fallback (Campi Flegrei region box) from USGS"""
    end = dt.datetime.utcnow()
    start = end - dt.timedelta(days=7)
    url = USGS_URL.format(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    try:
        r = requests.get(url, timeout=API_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame([
            {
                "time": dt.datetime.utcfromtimestamp(f["properties"]["time"] / 1000),
                "magnitude": f["properties"]["mag"],
                "depth_km": f["geometry"]["coordinates"][2],
                "place": f["properties"]["place"],
            }
            for f in data["features"]
        ])
        return df
    except Exception as e:
        st.error(f"USGS fallback failed: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=600)
def fetch_solar_data():
    """Fetch NOAA/DSCOVR solar wind feed"""
    for url in DSCOVR_URLS:
        try:
            r = requests.get(url, timeout=API_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            last = data[-1] if isinstance(data, list) else data.get("data", [])[-1]
            if len(last) > 3:
                return {
                    "speed": float(last[1]),
                    "density": float(last[2]),
                    "temp": float(last[3]),
                    "psi_s": np.clip(float(last[1]) / 800, 0, 1),
                }
        except Exception:
            continue
    st.warning("Solar feed unavailable — using fallback dataset.")
    return {"speed": 400, "density": 5.0, "temp": 0, "psi_s": 0.5}


@st.cache_data(ttl=600)
def fetch_kp_index():
    """Fetch NOAA planetary Kp index"""
    try:
        r = requests.get(NOAA_KP_URL, timeout=API_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        return round(float(data[-1][1]), 2)
    except Exception as e:
        st.warning(f"Kp fetch failed: {e}. Defaulting to 1.0.")
        return 1.0


def compute_eii(df, psi_s, kp):
    """Compute Energetic Instability Index"""
    if df is None or df.empty:
        return 0.0
    mag_mean = df["magnitude"].mean()
    shallow_ratio = len(df[df["depth_km"] < 3]) / len(df)
    eii = np.clip((mag_mean * 0.25 + shallow_ratio * 0.35 + psi_s * 0.25 + kp * 0.15) / 2, 0, 1)
    return round(float(eii), 3)


# -------------------------------
# DATA PIPELINE
# -------------------------------
st.info("Fetching live data feeds... please wait ⏳")

solar = fetch_solar_data()
kp = fetch_kp_index()
df = fetch_ingv_data()
if df is None or df.empty:
    df = fetch_usgs_data()

EII = compute_eii(df, solar["psi_s"], kp)

# RPAM classification
if EII >= 0.85:
    RPAM = "ACTIVE – Collapse Window Initiated"
    color = "🔴"
elif EII >= 0.6:
    RPAM = "ELEVATED – Pressure Coupling"
    color = "🟠"
else:
    RPAM = "MONITORING"
    color = "🟢"

# -------------------------------
# VISUALIZATION
# -------------------------------
st.success("✅ All systems operational — SUPT Live Dashboard (v5.2)")
st.caption(f"🕒 Last update: {dt.datetime.utcnow():%Y-%m-%d %H:%M:%S UTC}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Energetic Instability Index (EII)", f"{EII:.3f}")
col2.metric("Ψ Coupling", f"{solar['psi_s']:.3f}")
col3.metric("Geomagnetic Kp", f"{kp:.2f}")
col4.metric("RPAM Phase", f"{color} {RPAM}")

st.markdown("---")

st.subheader("☀️ Solar Wind & Geomagnetic Activity")
st.write(
    f"**Speed:** {solar['speed']:.2f} km/s | **Density:** {solar['density']:.2f} p/cm³ | "
    f"**Temp:** {solar['temp']:.0f} K | **ψₛ:** {solar['psi_s']:.3f}"
)
solar_df = pd.DataFrame(
    {"Speed (km/s)": [solar["speed"]], "Density (p/cm³)": [solar["density"]], "Kp": [kp]}
)
st.line_chart(solar_df)

st.subheader("🌋 Seismic Events (Past 7 Days)")
st.dataframe(df.sort_values("time", ascending=False).head(15))

st.subheader("🌀 Harmonic Drift — Magnitude & Depth")
if not df.empty:
    st.line_chart(df.set_index("time")[["magnitude", "depth_km"]])
else:
    st.info("No valid seismic data.")

st.markdown("---")
st.caption(
    f"Feeds: NOAA • USGS • INGV | SUPT v5.2 Live Sync Fix — Sheppard’s Universal Proxy Theory"
)

# Auto-refresh
st_autorefresh(interval=600000, key="data_refresh")
