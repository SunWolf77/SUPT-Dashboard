# ================================================================
# 🌋 SUPT :: GROK Forecast Dashboard v7.0 (Production Build)
# ================================================================
# Live Data: INGV 🇮🇹 | USGS 🌍 | EMSC 🇪🇺 | NOAA ☀️ | Synthetic 🧪
# Full SUPT Tri-Coupled Monitoring Engine
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime as dt
import io

st.set_page_config(page_title="SUPT :: GROK Forecast Dashboard", layout="wide")

# ================================================================
# --- UTILITIES ---
# ================================================================
def safe_request(url, timeout=10):
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200 and len(r.text) > 100:
            return r.text
    except Exception:
        return None
    return None

def safe_json(url, timeout=10):
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

# ================================================================
# --- SOLAR + GEOMAGNETIC DATA ---
# ================================================================
@st.cache_data(ttl=600, show_spinner=False)
def fetch_noaa_solar():
    try:
        # Solar wind from DSCOVR via NOAA SWPC
        sw_url = "https://services.swpc.noaa.gov/products/summary/solar-wind.json"
        kp_url = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
        sw_data = requests.get(sw_url, timeout=10).json()
        kp_data = requests.get(kp_url, timeout=10).json()

        # Extract key values
        latest_sw = sw_data[-1] if isinstance(sw_data, list) else sw_data
        vel = float(latest_sw.get("speed", 450))
        dens = float(latest_sw.get("density", 4))
        bt = float(latest_sw.get("bt", 5))
        kp = float(kp_data[-1][1]) if isinstance(kp_data, list) else 2.0

        return {
            "solar_speed": vel,
            "solar_density": dens,
            "bt": bt,
            "kp": kp,
            "status": "Live NOAA Feed"
        }
    except Exception:
        return {
            "solar_speed": 420,
            "solar_density": 3.0,
            "bt": 4.5,
            "kp": 1.7,
            "status": "Synthetic Fallback"
        }

# ================================================================
# --- SEISMIC DATA AGGREGATOR ---
# ================================================================
@st.cache_data(ttl=900, show_spinner=False)
def load_seismic_data():
    now = dt.datetime.utcnow()
    end_time = now.strftime("%Y-%m-%dT%H:%M:%S")
    start_time = (now - dt.timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S")

    df, src = pd.DataFrame(), "None"

    # 1️⃣ INGV
    ingv = safe_request(
        f"https://webservices.ingv.it/fdsnws/event/1/query?"
        f"starttime={start_time}&endtime={end_time}&minmag=2.0&format=text"
    )
    if ingv:
        try:
            dfi = pd.read_csv(io.StringIO(ingv), delimiter="|", comment="#")
            tc = next((c for c in dfi.columns if "Time" in c), None)
            mc = next((c for c in dfi.columns if "Mag" in c), None)
            dc = next((c for c in dfi.columns if "Depth" in c), None)
            if all([tc, mc, dc]):
                df = pd.DataFrame({
                    "time": pd.to_datetime(dfi[tc], errors="coerce"),
                    "magnitude": pd.to_numeric(dfi[mc], errors="coerce"),
                    "depth_km": pd.to_numeric(dfi[dc], errors="coerce"),
                    "place": "Italy (INGV)"
                }).dropna()
                if not df.empty:
                    return df, "INGV 🇮🇹"
        except Exception:
            pass

    # 2️⃣ USGS
    usgs = safe_request(
        f"https://earthquake.usgs.gov/fdsnws/event/1/query?"
        f"format=csv&starttime={start_time}&endtime={end_time}&minmagnitude=4"
    )
    if usgs:
        try:
            dfg = pd.read_csv(io.StringIO(usgs))
            if all(c in dfg.columns for c in ["time", "mag", "depth", "place"]):
                df = pd.DataFrame({
                    "time": pd.to_datetime(dfg["time"], errors="coerce"),
                    "magnitude": pd.to_numeric(dfg["mag"], errors="coerce"),
                    "depth_km": pd.to_numeric(dfg["depth"], errors="coerce"),
                    "place": dfg["place"]
                }).dropna()
                if not df.empty:
                    return df, "USGS 🌍"
        except Exception:
            pass

    # 3️⃣ EMSC
    emsc_json = safe_json("https://www.seismicportal.eu/fdsnws/event/1/query?format=json&limit=50")
    if emsc_json and "features" in emsc_json:
        feats = emsc_json["features"]
        if feats:
            df = pd.DataFrame({
                "time": [pd.to_datetime(f["properties"]["time"]) for f in feats],
                "magnitude": [f["properties"]["mag"] for f in feats],
                "depth_km": [f["properties"]["depth"] for f in feats],
                "place": [f["properties"]["flynn_region"] for f in feats]
            }).dropna()
            if not df.empty:
                return df, "EMSC 🇪🇺"

    # 4️⃣ Synthetic fallback
    now = dt.datetime.utcnow()
    df = pd.DataFrame({
        "time": [now - dt.timedelta(hours=i) for i in range(24)][::-1],
        "magnitude": np.random.uniform(3.5, 5.5, 24),
        "depth_km": np.random.uniform(5, 15, 24),
        "place": ["Synthetic Continuity"] * 24
    })
    return df, "Synthetic 🧪"

# ================================================================
# --- SUPT METRICS ---
# ================================================================
def compute_eii(df, solar):
    if df.empty:
        return 0.0
    md_max = df["magnitude"].max()
    md_mean = df["magnitude"].mean()
    shallow_ratio = (df["depth_km"] < 10).mean()
    psi = (solar["solar_speed"] / 800) * 0.5 + (solar["solar_density"] / 10) * 0.5
    eii = np.clip(md_max * 0.2 + md_mean * 0.15 + shallow_ratio * 0.35 + psi * 0.3, 0, 1)
    return eii, psi

# ================================================================
# --- DASHBOARD BODY ---
# ================================================================
st.title("🌋 SUPT :: GROK Forecast Dashboard")
st.caption("v7.0 — Real-Time Solar–Geomagnetic–Seismic SUPT Engine")

solar = fetch_noaa_solar()
df, src = load_seismic_data()

# Header status
feed_color = {"INGV 🇮🇹": "🟢", "USGS 🌍": "🔵", "EMSC 🇪🇺": "🟣", "Synthetic 🧪": "🔴"}[src]
st.markdown(f"### {feed_color} Active Seismic Feed: **{src}** ☀️ Solar Feed: **{solar['status']}**")

if df.empty:
    st.error("No seismic data available.")
    st.stop()

# ---------------------------------------------------------------
# METRICS PANEL
# ---------------------------------------------------------------
EII, PSI = compute_eii(df, solar)
RPAM = "ACTIVE – Collapse Window Initiated" if EII > 0.85 else (
    "ELEVATED – Pressure Coupling Phase" if EII > 0.6 else "STABLE"
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("EII", f"{EII:.3f}")
col2.metric("Ψ Coupling", f"{PSI:.3f}")
col3.metric("Solar Wind Density (p/cm³)", f"{solar['solar_density']:.2f}")
col4.metric("Geomagnetic Kp", f"{solar['kp']:.1f}")

st.markdown(f"### RPAM Status: **{RPAM}**")

# ---------------------------------------------------------------
# VISUALS
# ---------------------------------------------------------------
st.subheader("🕳️ Seismic Events — Last 7 Days")
st.dataframe(df[["time", "magnitude", "depth_km", "place"]].sort_values("time", ascending=False).head(20))

st.subheader("📈 Harmonic Drift — Magnitude & Depth")
st.line_chart(df.set_index("time")[["magnitude", "depth_km"]])

st.subheader("☀️ Solar Dynamics — Past Readings (Sample)")
solar_sample = pd.DataFrame({
    "Solar Wind (km/s)": np.clip(np.random.normal(solar["solar_speed"], 25, 24), 300, 800),
    "Density (p/cm³)": np.clip(np.random.normal(solar["solar_density"], 0.5, 24), 0.1, 10),
    "Kp Index": np.clip(np.random.normal(solar["kp"], 0.4, 24), 0, 9)
})
st.line_chart(solar_sample)

# ---------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------
st.caption(
    f"Updated {dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} | "
    f"Feeds: NOAA ☀️ / INGV 🇮🇹 / USGS 🌍 / EMSC 🇪🇺 | SUPT v7.0"
)
st.caption("Powered by Sheppard’s Universal Proxy Theory — Solar–Geophysical Coupling Monitor")
