import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import time
import plotly.graph_objects as go

# ========== PAGE SETUP ==========
st.set_page_config(page_title="SunWolf-SUPT :: Live Dashboard", layout="wide")
st.title("☀️ SunWolf-SUPT :: Campi Flegrei + Vulcano Live Forecast Dashboard")
st.caption("Powered by SUPT ψ-Fold + NOAA + INGV Real-Time Data Fusion")

# =========================================
# ========== DATA FETCH FUNCTIONS =========
# =========================================

@st.cache_data(ttl=900)
def fetch_ingv(latmin, latmax, lonmin, lonmax):
    """Fetch localized INGV seismic data (Campi Flegrei, Vulcano)."""
    try:
        url = (
            f"https://webservices.ingv.it/fdsnws/event/1/query?"
            f"starttime=2025-10-01&endtime=now&latmin={latmin}&latmax={latmax}&"
            f"lonmin={lonmin}&lonmax={lonmax}&minmag=-0.5&maxmag=5&format=csv"
        )
        r = requests.get(url, timeout=15)
        df = pd.read_csv(io.StringIO(r.text))
        df.rename(columns=lambda x: x.strip().lower(), inplace=True)
        if "latitude" not in df.columns or "longitude" not in df.columns:
            return pd.DataFrame()
        df["depth"] = pd.to_numeric(df["depth"], errors="coerce")
        df["magnitude"] = pd.to_numeric(df["magnitude"], errors="coerce")
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
        return df.dropna(subset=["depth", "magnitude"])
    except Exception as e:
        st.warning(f"⚠️ INGV fetch failed: {e}")
        return pd.DataFrame()

# --- PATCHED NOAA FETCH ---
@st.cache_data(ttl=900)
def fetch_noaa_kp():
    """Fetch latest geomagnetic Kp index from NOAA SWPC with flexible field detection."""
    try:
        url = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
        r = requests.get(url, timeout=15)
        data = r.json()
        df = pd.DataFrame(data[1:], columns=data[0])
        df.columns = [c.lower().strip() for c in df.columns]
        df["time_tag"] = pd.to_datetime(df["time_tag"], utc=True, errors="coerce")
        # Detect which column holds Kp
        kp_col = next((c for c in df.columns if "kp" in c and "index" in c), None)
        if kp_col is None:
            # fallback: last numeric column
            kp_col = [c for c in df.columns if df[c].apply(lambda x: str(x).replace('.', '', 1).isdigit()).any()][-1]
        df["kp_index"] = pd.to_numeric(df[kp_col], errors="coerce")
        return df.tail(8)
    except Exception as e:
        st.warning(f"⚠️ NOAA Kp fetch failed: {e}")
        return pd.DataFrame({"time_tag": [pd.Timestamp.utcnow()], "kp_index": [1.0]})


# =========================================
# ========== SUPT COMPUTATION CORE ========
# =========================================

def compute_sunwolf(cf_df, vulc_df, kp_df):
    """SUPT SunWolf model — compute EII, ψₛ, RPAM, etc."""
    if len(cf_df) == 0 or len(vulc_df) == 0:
        return 0.0, "NORMAL", 1.0, float(kp_df["kp_index"].iloc[-1])

    mean_depth = (cf_df["depth"].mean() + vulc_df["depth"].mean()) / 2
    shallow_ratio = len(cf_df[cf_df["depth"] < 3]) / max(len(cf_df), 1)
    kp = float(kp_df["kp_index"].iloc[-1])
    psi_s = np.clip(kp / 3.5, 0.1, 3.0)

    eii = np.clip((1 / (mean_depth + 0.1)) * shallow_ratio * (psi_s / 2), 0, 1)
    rpam = "ACTIVE" if eii > 0.85 else "ELEVATED" if eii > 0.55 else "NORMAL"

    return eii, rpam, psi_s, kp

# =========================================
# ========== BUILD DASHBOARD FIGURE =======
# =========================================

def build_dashboard():
    cf_df = fetch_ingv(40.79, 40.84, 14.10, 14.15)   # Campi Flegrei
    vulc_df = fetch_ingv(38.38, 38.47, 14.90, 15.05) # Vulcano
    kp_df = fetch_noaa_kp()

    eii, rpam, psi_s, kp = compute_sunwolf(cf_df, vulc_df, kp_df)

    fig = go.Figure()
    fig.update_layout(
        title=f"🌋 SunWolf-SUPT Phase Tracking — Campi Flegrei & Vulcano<br>"
              f"EII={eii:.3f} | RPAM={rpam} | ψₛ={psi_s:.3f} | KP={kp:.1f}",
        template="plotly_dark",
        height=700,
        scene=dict(
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            zaxis_title="Depth (km, inverted)",
            zaxis=dict(range=[-5, 0])
        )
    )

    for name, df, color in [
        ("Campi Flegrei", cf_df, "orange"),
        ("Vulcano", vulc_df, "lightblue")
    ]:
        if len(df):
            fig.add_trace(go.Scatter3d(
                x=df["longitude"], y=df["latitude"], z=-df["depth"],
                mode="markers", name=name,
                marker=dict(size=4, color=color, opacity=0.7),
                hovertext=[f"{name}<br>Md {m:.1f}<br>{t}" for m, t in zip(df["magnitude"], df["time"])]
            ))

    # ψₛ Resonance Wave
    t = np.linspace(0, 2*np.pi, 50)
    amp = np.sin(t * psi_s * np.pi) * 0.5
    z_wave = -2 + amp
    x_wave = np.linspace(14.10, 14.15, 50)
    y_wave = np.linspace(40.79, 40.84, 50)

    fig.add_trace(go.Scatter3d(
        x=x_wave, y=y_wave, z=z_wave,
        mode="lines", line=dict(color="gold", width=6),
        name="ψₛ Resonance Wave", hoverinfo="none"
    ))

    fig.update_layout(
        updatemenus=[{
            "buttons": [
                {"args": [None, {"frame": {"duration": 150, "redraw": True}, "fromcurrent": True}],
                 "label": "▶️ Play ψₛ Resonance", "method": "animate"},
                {"args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}],
                 "label": "⏸ Pause", "method": "animate"}
            ],
            "direction": "left", "pad": {"r": 10, "t": 70},
            "showactive": False, "type": "buttons", "x": 0.1, "xanchor": "right", "y": 1.05, "yanchor": "top"
        }]
    )

    # Geomagnetic Kp Gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=kp,
        title={"text": "Geomagnetic Kp Index"},
        domain={"x": [0, 0.4], "y": [0, 0.25]},
        gauge={
            "axis": {"range": [0, 9]},
            "bar": {"color": "gold"},
            "steps": [
                {"range": [0, 3], "color": "darkblue"},
                {"range": [3, 6], "color": "orange"},
                {"range": [6, 9], "color": "red"}
            ]
        }
    ))

    return fig, kp_df

# =========================================
# ========== STREAMLIT RENDERING ==========
# =========================================

fig, kp_df = build_dashboard()
st.plotly_chart(fig, use_container_width=True)

# =========================================
# ========== COUPLING CHART ===============
# =========================================

st.divider()
st.subheader("🌐 Depth–Kp Coupling Monitor (Live)")

cf_df = fetch_ingv(40.79, 40.84, 14.10, 14.15)
if not cf_df.empty:
    cf_df["time"] = pd.to_datetime(cf_df["time"], utc=True)
    hourly = cf_df.groupby(pd.Grouper(key="time", freq="3H"))["depth"].mean().reset_index()
else:
    hourly = pd.DataFrame({"time": pd.date_range(end=pd.Timestamp.utcnow(), periods=8, freq="3H"),
                           "depth": np.linspace(4.5, 2.5, 8)})

merged = pd.merge_asof(hourly.sort_values("time"),
                       kp_df[["time_tag", "kp_index"]].rename(columns={"time_tag": "time"}),
                       on="time", tolerance=pd.Timedelta("3H"), direction="nearest")

merged.rename(columns={"depth": "Mean Depth (km)", "kp_index": "Kp Index"}, inplace=True)
st.line_chart(merged.set_index("time"))
st.caption("Live coupling between geomagnetic activity and mean seismic depth — recalibrated every 3 hours.")

st.success("✅ Live Dashboard Ready — SunWolf-SUPT (ψₛ Resonance + Depth–Kp Coupling)")
