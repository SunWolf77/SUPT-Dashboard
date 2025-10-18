# ===============================================================
# SunWolf's Forecast Dashboard v8.0 — LIVE CORE BUILD
# ===============================================================
# LIVE DATA ONLY — NOAA ☀️ | USGS 🌍 | INGV 🇮🇹 | EMSC 🇪🇺
# No synthetic data. If a source is down, dashboard displays outage.
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime as dt
import io

st.set_page_config(page_title="SunWolf's Forecast Dashboard", layout="wide")

# ===============================================================
# --- HELPERS ---
# ===============================================================
def try_get_json(url, timeout=10):
    """Returns parsed JSON or None."""
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def try_get_text(url, timeout=10):
    """Returns plain text or None."""
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200 and len(r.text) > 100:
            return r.text
    except Exception:
        pass
    return None


# ===============================================================
# --- SOLAR + GEOMAGNETIC DATA (NOAA SWPC LIVE) ---
# ===============================================================
@st.cache_data(ttl=600, show_spinner=False)
def load_noaa():
    """Fetch live NOAA and DSCOVR solar parameters with continuity fallback."""
    try:
        # Primary feed (NOAA SWPC solar wind)
        sw_url = "https://services.swpc.noaa.gov/products/summary/solar-wind.json"
        kp_url = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
        dscovr_url = "https://services.swpc.noaa.gov/products/solar-wind/plasma-5-minute.json"
        proton_url = "https://services.swpc.noaa.gov/products/solar-wind/protons-5-minute.json"

        sw_data = try_get_json(sw_url)
        kp_data = try_get_json(kp_url)
        plasma = try_get_json(dscovr_url)
        protons = try_get_json(proton_url)

        vel = dens = bt = temp = psi_s = kp = None

        # Extract solar wind values if available
        if isinstance(sw_data, list) and len(sw_data) > 0:
            last = sw_data[-1]
            vel = float(last.get("speed", 0))
            dens = float(last.get("density", 0))
            bt = float(last.get("bt", 0))

        # Extract plasma temperature from DSCOVR
        if isinstance(plasma, list) and len(plasma) > 1:
            # Last row has structure: time_tag, density, speed, temperature
            vals = plasma[-1]
            try:
                if len(vals) >= 4:
                    if not vel:
                        vel = float(vals[2])
                    if not dens:
                        dens = float(vals[1])
                    temp = float(vals[3])
            except Exception:
                pass

        # Extract proton flux (used for solar pressure proxy)
        if isinstance(protons, list) and len(protons) > 1:
            last_flux = float(protons[-1][1]) if len(protons[-1]) > 1 else 0
            psi_s = np.log10(last_flux + 1) / 3.0  # normalized proxy [0–1]

        # Get geomagnetic Kp index
        if isinstance(kp_data, list) and len(kp_data) > 1:
            kp = float(kp_data[-1][1])

        # Compute Solar Pressure Proxy if available
        if all(v is not None for v in [vel, dens, temp]):
            psi_s = np.clip(((vel / 800) * 0.5 + (dens / 10) * 0.3 + (temp / 2e5) * 0.2), 0, 1)

        return {
            "solar_speed": vel,
            "solar_density": dens,
            "bt": bt,
            "temp": temp,
            "psi_s": psi_s,
            "kp": kp,
            "status": "Live NOAA/DSCOVR Feed"
        }

    except Exception:
        return {
            "solar_speed": None,
            "solar_density": None,
            "bt": None,
            "temp": None,
            "psi_s": None,
            "kp": None,
            "status": "Offline"
        }


# ===============================================================
# --- SEISMIC DATA AGGREGATOR ---
# ===============================================================
@st.cache_data(ttl=900, show_spinner=False)
def load_seismic():
    now = dt.datetime.utcnow()
    end = now.strftime("%Y-%m-%dT%H:%M:%S")
    start = (now - dt.timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S")

    # 1️⃣ INGV — Italy
    ingv = try_get_text(
        f"https://webservices.ingv.it/fdsnws/event/1/query?"
        f"starttime={start}&endtime={end}&minmag=2.0&format=text"
    )
    if ingv:
        try:
            dfi = pd.read_csv(io.StringIO(ingv), delimiter="|", comment="#")
            t, m, d = None, None, None
            for c in dfi.columns:
                if "Time" in c: t = c
                elif "Mag" in c: m = c
                elif "Depth" in c: d = c
            if t and m and d:
                df = pd.DataFrame({
                    "time": pd.to_datetime(dfi[t], errors="coerce"),
                    "magnitude": pd.to_numeric(dfi[m], errors="coerce"),
                    "depth_km": pd.to_numeric(dfi[d], errors="coerce"),
                    "place": "Italy (INGV)"
                }).dropna()
                if not df.empty:
                    return df, "INGV 🇮🇹"
        except Exception:
            pass

    # 2️⃣ USGS — Global
    usgs = try_get_text(
        f"https://earthquake.usgs.gov/fdsnws/event/1/query?"
        f"format=csv&starttime={start}&endtime={end}&minmagnitude=4"
    )
    if usgs:
        try:
            dfg = pd.read_csv(io.StringIO(usgs))
            if all(x in dfg.columns for x in ["time", "mag", "depth", "place"]):
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

    # 3️⃣ EMSC — European backup
    emsc = try_get_json("https://www.seismicportal.eu/fdsnws/event/1/query?format=json&limit=50")
    if emsc and "features" in emsc:
        feats = emsc["features"]
        if feats:
            df = pd.DataFrame({
                "time": [pd.to_datetime(f["properties"]["time"]) for f in feats],
                "magnitude": [f["properties"]["mag"] for f in feats],
                "depth_km": [f["properties"]["depth"] for f in feats],
                "place": [f["properties"]["flynn_region"] for f in feats]
            }).dropna()
            if not df.empty:
                return df, "EMSC 🇪🇺"

    # If no feed worked, return empty DataFrame
    return pd.DataFrame(), "No Live Feed ⚠️"


# ===============================================================
# --- SUPT METRICS ---
# ===============================================================
def compute_metrics(df, solar):
    if df.empty or not solar["solar_speed"]:
        return 0.0, 0.0, "No live coupling data."

    shallow = (df["depth_km"] < 15).mean()
    mag_mean = df["magnitude"].mean()
    psi_coupling = (solar["solar_speed"] / 800) * 0.6 + (solar["solar_density"] / 10) * 0.4
    eii = np.clip(mag_mean * 0.25 + shallow * 0.35 + psi_coupling * 0.4, 0, 1)

    rpam = (
        "ACTIVE — Collapse Window Initiated" if eii > 0.85 else
        "ELEVATED — Pressure Coupling Phase" if eii > 0.6 else
        "STABLE"
    )
    return eii, psi_coupling, rpam


# ===============================================================
# --- DASHBOARD ---
# ===============================================================
st.title("🌞🐺 SunWolf's Forecast Dashboard")
st.caption("v8.0 — Live Core Build (NOAA + USGS + INGV + EMSC)")

solar = load_noaa()
df, src = load_seismic()

feed_icon = {"INGV 🇮🇹":"🟢","USGS 🌍":"🔵","EMSC 🇪🇺":"🟣","No Live Feed ⚠️":"⚠️"}[src]
solar_status = "🟢 Live" if solar["solar_speed"] else "⚠️ Offline"

st.markdown(f"### {feed_icon} Active Seismic Feed: **{src}** ☀️ Solar Feed: **{solar_status}**")

if df.empty:
    st.error("No live seismic data could be retrieved from any active feed.")
    st.stop()

EII, PSI, RPAM = compute_metrics(df, solar)

# Metrics display
c1, c2, c3, c4 = st.columns(4)
c1.metric("EII", f"{EII:.3f}")
c2.metric("Ψ Coupling", f"{PSI:.3f}")
c3.metric("Solar Wind Density (p/cm³)", f"{solar['solar_density'] or '—'}")
c4.metric("Geomagnetic Kp", f"{solar['kp'] or '—'}")

st.markdown(f"### RPAM Status: **{RPAM}**")

# Data Tables
st.subheader("🕳️ Seismic Events — Last 7 Days")
st.dataframe(df.sort_values("time", ascending=False).head(30))

# Charts
st.subheader("📈 Harmonic Drift — Magnitude & Depth")
st.line_chart(df.set_index("time")[["magnitude", "depth_km"]])

# Solar Data Summary
if solar["solar_speed"]:
    st.subheader("☀️ NOAA Solar Wind — Latest Values")
    st.write(f"**Speed:** {solar['solar_speed']} km/s | **Density:** {solar['solar_density']} p/cm³ | **Bₜ:** {solar['bt']} | **Kp:** {solar['kp']}")
else:
    st.warning("No live NOAA solar data available at this time.")

# Footer
st.caption(f"Updated {dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} | Feeds: NOAA ☀️ / USGS 🌍 / INGV 🇮🇹 / EMSC 🇪🇺 | SUPT v8.0")
st.caption("Powered by Sheppard’s Universal Proxy Theory — Solar–Geophysical Coupling Monitor")
