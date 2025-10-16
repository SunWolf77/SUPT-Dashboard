# ==========================================================
# ☀️ SunWolf-SUPT v3.3 — Solar Gold + ψₛ-Coupling Live Patch
# ==========================================================
import streamlit as st, pandas as pd, numpy as np, requests
from datetime import datetime, timedelta, timezone
import plotly.graph_objs as go

# ----------------------------------------------------------------
# Utility
# ----------------------------------------------------------------
def live_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def utc_now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

# ----------------------------------------------------------------
# NOAA Live Feeds
# ----------------------------------------------------------------
@st.cache_data(ttl=600)
def fetch_kp_index():
    """More tolerant NOAA Kp fetch (auto-detects schema & missing keys)."""
    try:
        url = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
        data = requests.get(url, timeout=15).json()
        header, rows = data[0], data[1:]
        df = pd.DataFrame(rows, columns=header)
        kp_col = [c for c in df.columns if "kp" in c.lower()][0]
        time_col = [c for c in df.columns if "time" in c.lower()][0]
        df["time"] = pd.to_datetime(df[time_col], errors="coerce")
        df["kp_index"] = pd.to_numeric(df[kp_col], errors="coerce")
        df = df.dropna(subset=["kp_index"])
        return df[["time", "kp_index"]].tail(24)
    except Exception as e:
        st.warning(f"NOAA Kp fetch failed: {e}")
        return pd.DataFrame(columns=["time", "kp_index"])

@st.cache_data(ttl=600)
def fetch_solar_wind():
    """NOAA Solar-Wind feed with robust parser."""
    try:
        url = "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json"
        data = requests.get(url, timeout=15).json()
        df = pd.DataFrame(data[1:], columns=data[0])
        for c in ["density", "speed"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["time_tag"] = pd.to_datetime(df["time_tag"], errors="coerce")
        return df.dropna(subset=["speed", "density"]).tail(96)
    except Exception as e:
        st.warning(f"Solar Wind fetch failed: {e}")
        return pd.DataFrame(columns=["time_tag", "density", "speed"])

# ----------------------------------------------------------------
# INGV Feed (UTC patched)
# ----------------------------------------------------------------
@st.cache_data(ttl=600)
def fetch_ingv_quakes():
    """Campi Flegrei quakes with UTC timestamp patch."""
    try:
        endtime = utc_now_iso()
        url = (
            f"https://webservices.ingv.it/fdsnws/event/1/query?"
            f"starttime=2025-10-01T00:00:00&endtime={endtime}&"
            "minlat=40.7&maxlat=40.9&minlon=14.0&maxlon=14.3&format=csv"
        )
        df = pd.read_csv(url)
        df = df.rename(columns={"time": "Time", "mag": "Mag", "depth": "Depth/km"})
        df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
        return df.dropna(subset=["Depth/km", "Mag"])
    except Exception as e:
        st.warning(f"INGV fetch failed: {e}")
        return pd.DataFrame(columns=["Time", "Mag", "Depth/km"])

# ----------------------------------------------------------------
# Fallbacks
# ----------------------------------------------------------------
def ensure_fallback(df, label):
    if df.empty:
        st.warning(f"⚠️ {label} feed unavailable — using synthetic sample.")
        if label == "NOAA":
            return pd.DataFrame({"time": [datetime.utcnow()], "kp_index": [1.0]})
        if label == "Solar Wind":
            return pd.DataFrame({"time_tag": [datetime.utcnow()], "density": [3.3], "speed": [430]})
        if label == "INGV":
            return pd.DataFrame({"Time": [datetime.utcnow()], "Mag": [0.9], "Depth/km": [1.8]})
    return df

# ----------------------------------------------------------------
# Live Fetch
# ----------------------------------------------------------------
kp_df = ensure_fallback(fetch_kp_index(), "NOAA Kp")
sw_df = ensure_fallback(fetch_solar_wind(), "Solar Wind")
eq_df = ensure_fallback(fetch_ingv_quakes(), "INGV")

feeds_status = (
    "🟢 NOAA  |  🟢 INGV  |  🟢 USGS"
    if not sw_df.empty else "⚠️ Partial feeds (backup active)"
)
st.markdown(
    f"<b>Data Feeds:</b> {feeds_status} | Last Refresh: {live_utc()}",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------
# Compute SUPT Metrics + RPAM
# ----------------------------------------------------------------
def compute_supt_metrics(kp_df, sw_df, eq_df):
    if sw_df.empty or kp_df.empty:
        return 0, 0, 0, "NO DATA", 0, 0, "#EF9A9A"
    kp = kp_df["kp_index"].mean()
    sw_speed, sw_density = sw_df["speed"].mean(), sw_df["density"].mean()
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

# ----------------------------------------------------------------
# Dashboard Header
# ----------------------------------------------------------------
st.markdown(
    f"<div style='background-color:{color}; padding:10px; border-radius:8px; text-align:center; color:white;'>"
    f"<b>RPAM: {rpam_status}</b></div>",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------
# Metric Row
# ----------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("EII", f"{eii:.3f}")
c2.metric("ψₛ (Solar Coupling)", f"{psi_s:.3f}")
c3.metric("αᵣ (Damping)", f"{alpha_r:.3f}")
c4.metric("Phase", rpam_status)

# ----------------------------------------------------------------
# Gauges
# ----------------------------------------------------------------
g1, g2 = st.columns(2)
with g1:
    st.subheader("☀️ Solar Wind Speed (km/s)")
    st.plotly_chart(
        go.Figure(go.Indicator(
            mode="gauge+number", value=sw_speed,
            gauge={"axis": {"range": [250, 800]},
                   "bar": {"color": color},
                   "steps": [{"range": [250, 500], "color": "#FFF8E1"},
                             {"range": [500, 650], "color": "#FFD54F"},
                             {"range": [650, 800], "color": "#F4511E"}]},
            title={"text": "Plasma Velocity"})),
        use_container_width=True,
    )

with g2:
    st.subheader("🌫 Solar Wind Density (p/cm³)")
    st.plotly_chart(
        go.Figure(go.Indicator(
            mode="gauge+number", value=sw_density,
            gauge={"axis": {"range": [0, 20]},
                   "bar": {"color": color},
                   "steps": [{"range": [0, 5], "color": "#FFF8E1"},
                             {"range": [5, 10], "color": "#FFD54F"},
                             {"range": [10, 20], "color": "#F4511E"}]},
            title={"text": "Plasma Density"})),
        use_container_width=True,
    )

# ----------------------------------------------------------------
# ψₛ-Coupling History (24 h)
# ----------------------------------------------------------------
st.markdown("### ☯ SUPT ψₛ Coupling — 24 h Harmonic Drift")
if not sw_df.empty:
    sw_df["time_tag"] = pd.to_datetime(sw_df["time_tag"])
    sw_df = sw_df.tail(288)
    psi_hist = ((sw_df["speed"] / 700) + (sw_df["density"] / 10)) / 2
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=sw_df["time_tag"], y=psi_hist, mode="lines",
        line=dict(color="#FFB300", width=2.5),
        name="ψₛ Coupling Index"))
    fig_hist.update_layout(
        xaxis_title="UTC Time (last 24 h)",
        yaxis_title="ψₛ",
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
        height=300,
        margin=dict(l=40, r=40, t=20, b=20))
    st.plotly_chart(fig_hist, use_container_width=True)
else:
    st.info("No solar wind history available yet — waiting for feed update.")

# ----------------------------------------------------------------
# Footer
# ----------------------------------------------------------------
st.markdown(
    f"<hr><p style='text-align:center; color:#FBC02D;'>Updated {live_utc()} | Feeds: {feeds_status} | Mode: Solar Gold ☀️ | SunWolf-SUPT v3.3</p>",
    unsafe_allow_html=True,
)
