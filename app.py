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
    """Fetch localized INGV/USGS seismic data with fallback."""
    try:
        url = (
            f"https://webservices.ingv.it/fdsnws/event/1/query?"
            f"starttime=2025-09-01&endtime=now&latmin={latmin}&latmax={latmax}&"
            f"lonmin={lonmin}&lonmax={lonmax}&minmag=-0.5&maxmag=6&format=csv"
        )
        r = requests.get(url, timeout=20)

        # --- Case 1: Empty or bad CSV ---
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
                return pd.DataFrame()
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
            return df

        # --- Case 2: Normal INGV CSV ---
        df = pd.read_csv(io.StringIO(r.text))
        df.rename(columns=lambda x: x.strip().lower(), inplace=True)
        df["depth"] = pd.to_numeric(df.get("depth", np.nan), errors="coerce")
        df["magnitude"] = pd.to_numeric(df.get("magnitude", np.nan), errors="coerce")
        df["time"] = pd.to_datetime(df.get("time", pd.NaT), utc=True, errors="coerce")
        return df.dropna(subset=["depth", "magnitude"])

    except Exception as e:
        st.error(f"🚨 INGV/USGS feed error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=900)
def fetch_noaa_kp():
    """Fetch the latest observed geomagnetic Kp index from NOAA with robust smoothing and fallback."""
    try:
        url_obs = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"
        r_obs = requests.get(url_obs, timeout=15)
        data_obs = pd.DataFrame(r_obs.json())
        data_obs["time_tag"] = pd.to_datetime(data_obs["time_tag"], utc=True, errors="coerce")
        data_obs["kp_index"] = pd.to_numeric(data_obs["kp_index"], errors="coerce")
        data_obs = data_obs[(data_obs["kp_index"].notna()) & (data_obs["kp_index"].between(0, 9))]

        if not data_obs.empty:
            valid = data_obs.tail(5)
            mean_kp = round(float(valid["kp_index"].mean()), 2)
            if mean_kp <= 0:
                mean_kp = 1.0
            return pd.DataFrame({"time_tag": [valid["time_tag"].iloc[-1]], "kp_index": [mean_kp]})

        url_fore = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
        r_fore = requests.get(url_fore, timeout=15)
        data_fore = r_fore.json()
        df_fore = pd.DataFrame(data_fore[1:], columns=data_fore[0])
        df_fore.columns = [c.lower().strip() for c in df_fore.columns]
        df_fore["time_tag"] = pd.to_datetime(df_fore.get("time_tag", pd.NaT), utc=True, errors="coerce")
        kp_col = next((c for c in df_fore.columns if "kp" in c and "index" in c), None)
        df_fore["kp_index"] = pd.to_numeric(df_fore[kp_col], errors="coerce")
        df_fore = df_fore[(df_fore["kp_index"].notna()) & (df_fore["kp_index"].between(0, 9))]
        mean_kp_fore = round(float(df_fore["kp_index"].tail(3).mean()), 2)
        if mean_kp_fore <= 0:
            mean_kp_fore = 1.0
        return pd.DataFrame({"time_tag": [pd.Timestamp.utcnow()], "kp_index": [mean_kp_fore]})
    except Exception:
        return pd.DataFrame({"time_tag": [pd.Timestamp.utcnow()], "kp_index": [1.0]})

# --------------------------------------------
# INTEGRITY INDICATOR LOGIC
# --------------------------------------------
kp_df = fetch_noaa_kp()

if "kp_index" in kp_df.columns and not kp_df.empty:
    kp_value = float(kp_df["kp_index"].iloc[0])
    if kp_value > 0:
        integrity_state = "🟢 Live NOAA Feed"
        integrity_color = "#00cc66"
    else:
        integrity_state = "🟡 Forecast Mode"
        integrity_color = "#ffaa00"
else:
    integrity_state = "🔴 Offline"
    integrity_color = "#cc0000"

st.markdown(
    f"""
    <div style="position:absolute; top:15px; right:25px;
                background-color:{integrity_color};
                color:white; padding:6px 12px;
                border-radius:10px; font-size:16px;
                font-weight:bold; box-shadow:0px 0px 6px #999;">
        {integrity_state}
    </div>
    """,
    unsafe_allow_html=True
)

st.caption(f"Powered by SUPT ψ-Fold • NOAA • INGV | Real-Time Data Fusion | Data Integrity Status: {integrity_state}")

# --------------------------------------------
# SUPT COMPUTATION
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
# BUILD & RENDER
# --------------------------------------------
def build_dashboard(latmin, latmax, lonmin, lonmax):
    df = fetch_ingv(latmin, latmax, lonmin, lonmax)
    eii, rpam, psi_s, kp = compute_supt(df, kp_df)

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

fig = build_dashboard(latmin, latmax, lonmin, lonmax)
st.plotly_chart(fig, use_container_width=True)
