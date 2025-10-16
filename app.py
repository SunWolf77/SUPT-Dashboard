import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import plotly.graph_objects as go

# --------------------------------------------
# PAGE CONFIG
# --------------------------------------------
st.set_page_config(page_title="SunWolf-SUPT :: Global Forecast Dashboard", layout="wide")
st.title("🌞🐺 SunWolf-SUPT :: Global Live Forecast Dashboard")

# --------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------
st.sidebar.title("🌎 SunWolf Global-Eye Mode")
mode = st.sidebar.radio(
    "Select Active Region",
    [
        "Campi Flegrei + Vulcano (Italy)",
        "Etna (Sicily)",
        "Klyuchevskoy (Kamchatka)",
        "Drake Passage (South Atlantic)"
    ],
    index=0
)

if "Etna" in mode:
    region_name = "Etna (Sicily)"
    latmin, latmax, lonmin, lonmax = 37.6, 37.8, 14.9, 15.1
elif "Klyuchevskoy" in mode:
    region_name = "Klyuchevskoy (Kamchatka)"
    latmin, latmax, lonmin, lonmax = 56.0, 56.2, 160.5, 160.8
elif "Drake" in mode:
    region_name = "Drake Passage (South Atlantic)"
    latmin, latmax, lonmin, lonmax = -60.5, -55.5, -70.5, -60.5
else:
    region_name = "Campi Flegrei + Vulcano (Italy)"
    latmin, latmax, lonmin, lonmax = 38.38, 40.84, 14.10, 15.05

st.sidebar.success(f"Fetching live data for **{region_name}** region")

# --------------------------------------------
# DATA FETCHERS
# --------------------------------------------
@st.cache_data(ttl=900)
def fetch_ingv(latmin, latmax, lonmin, lonmax):
    """Fetch INGV or fallback to USGS if empty."""
    try:
        url = (
            f"https://webservices.ingv.it/fdsnws/event/1/query?"
            f"starttime=2025-09-01&endtime=now&latmin={latmin}&latmax={latmax}&"
            f"lonmin={lonmin}&lonmax={lonmax}&minmag=-0.5&maxmag=6&format=csv"
        )
        r = requests.get(url, timeout=20)

        if not r.text.strip() or "Error" in r.text:
            st.warning("⚠️ INGV returned no data; switching to USGS fallback.")
            usgs_url = (
                f"https://earthquake.usgs.gov/fdsnws/event/1/query?"
                f"format=geojson&starttime=2025-09-01&endtime=now"
                f"&minlatitude={latmin}&maxlatitude={latmax}"
                f"&minlongitude={lonmin}&maxlongitude={lonmax}&minmagnitude=-0.5"
            )
            usgs_r = requests.get(usgs_url, timeout=20).json()
            features = usgs_r.get("features", [])
            if not features:
                return pd.DataFrame(), "USGS"
            df = pd.DataFrame([
                {
                    "time": pd.to_datetime(f["properties"]["time"], unit="ms", utc=True),
                    "latitude": f["geometry"]["coordinates"][1],
                    "longitude": f["geometry"]["coordinates"][0],
                    "depth": f["geometry"]["coordinates"][2],
                    "magnitude": f["properties"]["mag"],
                    "place": f["properties"]["place"],
                }
                for f in features
            ])
            return df, "USGS"

        df = pd.read_csv(io.StringIO(r.text))
        df.rename(columns=lambda x: x.strip().lower(), inplace=True)
        df["depth"] = pd.to_numeric(df.get("depth", np.nan), errors="coerce")
        df["magnitude"] = pd.to_numeric(df.get("magnitude", np.nan), errors="coerce")
        df["time"] = pd.to_datetime(df.get("time", pd.NaT), utc=True, errors="coerce")
        return df.dropna(subset=["depth", "magnitude"]), "INGV"

    except Exception as e:
        st.error(f"🚨 INGV/USGS feed error: {e}")
        return pd.DataFrame(), "ERROR"


@st.cache_data(ttl=900)
def fetch_noaa_kp():
    """Fetch NOAA Kp Index with fallback."""
    try:
        url = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"
        r = requests.get(url, timeout=15)
        data = pd.DataFrame(r.json())
        data["time_tag"] = pd.to_datetime(data["time_tag"], utc=True, errors="coerce")
        data["kp_index"] = pd.to_numeric(data["kp_index"], errors="coerce")
        data = data[(data["kp_index"].notna()) & (data["kp_index"].between(0, 9))]
        if data.empty:
            return pd.DataFrame({"time_tag": [pd.Timestamp.utcnow()], "kp_index": [1.0]})
        mean_kp = round(float(data.tail(5)["kp_index"].mean()), 2)
        if mean_kp <= 0:
            mean_kp = 1.0
        return pd.DataFrame({"time_tag": [data["time_tag"].iloc[-1]], "kp_index": [mean_kp]})
    except Exception:
        return pd.DataFrame({"time_tag": [pd.Timestamp.utcnow()], "kp_index": [1.0]})


# --------------------------------------------
# NOAA & SOURCE BADGES
# --------------------------------------------
kp_df = fetch_noaa_kp()
kp_value = float(kp_df["kp_index"].iloc[0])
integrity_state = "🟢 Live NOAA Feed" if kp_value > 0 else "🔴 Offline"

st.markdown(
    f"""
    <div style="position:absolute; top:15px; right:25px;
                background-color:{'#00cc66' if kp_value > 0 else '#cc0000'};
                color:white; padding:6px 12px;
                border-radius:10px; font-size:16px;
                font-weight:bold; box-shadow:0px 0px 6px #999;">
        {integrity_state}
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------
# SUPT METRICS
# --------------------------------------------
def compute_supt(df, kp_df):
    if df.empty:
        return 0.0, "NORMAL", 1.0, float(kp_df["kp_index"].iloc[-1])
    mean_depth = df["depth"].mean()
    shallow_ratio = len(df[df["depth"] < 3]) / max(len(df), 1)
    kp = float(kp_df["kp_index"].iloc[-1])
    psi_s = np.clip(kp / 3.5, 0.5, 3.0)
    eii = np.clip((1 / (mean_depth + 0.1)) * shallow_ratio * (psi_s / 2), 0, 1)
    rpam = "ACTIVE" if eii > 0.85 else "ELEVATED" if eii > 0.55 else "NORMAL"
    return eii, rpam, psi_s, kp


# --------------------------------------------
# BUILD DASHBOARD
# --------------------------------------------
def build_dashboard(latmin, latmax, lonmin, lonmax):
    df, source = fetch_ingv(latmin, latmax, lonmin, lonmax)
    eii, rpam, psi_s, kp = compute_supt(df, kp_df)

    # Source indicator
    source_map = {
        "INGV": ("🟣 INGV Feed", "#9933ff"),
        "USGS": ("🟡 USGS Fallback", "#ffaa00"),
        "ERROR": ("⚪ Data Feed Error", "#888888")
    }
    label, color = source_map.get(source, ("⚪ Unknown Source", "#888"))

    st.markdown(
        f"""
        <div style="position:absolute; top:55px; right:25px;
                    background-color:{color};
                    color:white; padding:6px 12px;
                    border-radius:10px; font-size:16px;
                    font-weight:bold; box-shadow:0px 0px 6px #999;">
            {label}
        </div>
        """,
        unsafe_allow_html=True
    )

    fig = go.Figure()
    fig.update_layout(
        title=f"🌋 SunWolf-SUPT Phase Tracking — {region_name}<br>EII={eii:.3f} | RPAM={rpam} | ψₛ={psi_s:.3f} | Kp={kp:.1f}",
        template="plotly_dark", height=700,
        scene=dict(xaxis_title="Longitude", yaxis_title="Latitude", zaxis_title="Depth (km, inverted)", zaxis=dict(range=[-10, 0]))
    )

    if not df.empty:
        fig.add_trace(go.Scatter3d(
            x=df["longitude"], y=df["latitude"], z=-df["depth"],
            mode="markers", name=region_name,
            marker=dict(size=4, color="orange", opacity=0.7),
            hovertext=[f"Md {m:.1f}<br>{t}" for m, t in zip(df["magnitude"], df["time"])]
        ))

    return fig


# --------------------------------------------
# RENDER
# --------------------------------------------
fig = build_dashboard(latmin, latmax, lonmin, lonmax)
st.plotly_chart(fig, use_container_width=True)
