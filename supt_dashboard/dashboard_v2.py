import pandas as pd
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta

# Existing code...
# (Assuming you already have your NOAA Solar Wind + USGS functions defined above)

# === NEW: SUPT SunWolf Integration ===

def fetch_ingv(latmin, latmax, lonmin, lonmax):
    """Fetch recent Campi Flegrei / Vulcano events."""
    url = (f"https://webservices.ingv.it/fdsnws/event/1/query?"
           f"starttime={datetime.utcnow()-timedelta(days=7):%Y-%m-%d}&endtime=now"
           f"&latmin={latmin}&latmax={latmax}&lonmin={lonmin}&lonmax={lonmax}&format=text")
    try:
        df = pd.read_csv(url, sep="|", comment="#")
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.rename(columns={"mag":"md"}).dropna(subset=["depth", "md"])
        return df
    except Exception as e:
        print("INGV fetch failed:", e)
        return pd.DataFrame(columns=["time","latitude","longitude","depth","md"])

def fetch_kp():
    """Fetch current planetary K-index from NOAA SWPC."""
    try:
        data = requests.get(
            "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json",
            timeout=5).json()
        return float(data[-1][1])
    except Exception:
        return 3.0

def compute_sunwolf(cf_df, vulc_df, kp):
    """Compute SUPT–SunWolf EII and RPAM metrics."""
    shallow = lambda df: (df["depth"] < 3).mean() if len(df) else 0
    cf_sr, vulc_sr = shallow(cf_df), shallow(vulc_df)
    eii = 0.5 * (cf_sr + vulc_sr) * (1 + min(kp/7, 0.25))
    rpam = "ELEVATED" if eii > 0.55 else "NORMAL"
    psi_s = round(1 + min(kp/28, 0.25), 3)
    return eii, rpam, psi_s

def build_dashboard():
    """Extended dashboard integrating SUPT SunWolf model + solar resonance."""
    # Fetch seismic + geomagnetic data
    cf_df = fetch_ingv(40.79, 40.84, 14.10, 14.15)   # Campi Flegrei
    vulc_df = fetch_ingv(38.38, 38.47, 14.90, 15.05) # Vulcano
    kp = fetch_kp()

    eii, rpam, psi_s = compute_sunwolf(cf_df, vulc_df, kp)

    # === PLOTLY DASHBOARD ===
    fig = go.Figure()
    fig.update_layout(
        title=f"☀️ SunWolf-SUPT Phase Tracking — Campi Flegrei & Vulcano<br>"
              f"EII={eii:.3f} | RPAM={rpam} | ψₛ={psi_s} | KP={kp}",
        template="plotly_dark",
        height=700,
        scene=dict(
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            zaxis_title="Depth (km, inverted)",
            zaxis=dict(range=[-5, 0])
        )
    )

    # Add quake scatter traces
    for name, df, color in [
        ("Campi Flegrei", cf_df, "orange"),
        ("Vulcano", vulc_df, "lightblue")
    ]:
        if len(df):
            fig.add_trace(go.Scatter3d(
                x=df["longitude"], y=df["latitude"], z=-df["depth"],
                mode="markers", name=name,
                marker=dict(size=4, color=color, opacity=0.7),
                hovertext=[f"{name}<br>Md {m:.1f}<br>{t}" for m, t in zip(df["md"], df["time"])]
            ))

    # === SOLAR RESONANCE LAYER ===
    import numpy as np
    t = np.linspace(0, 2*np.pi, 50)
    amplitude = np.sin(t * (psi_s * 3.14)) * 0.5
    z_wave = -2 + amplitude  # anchored around 2 km depth
    x_wave = np.linspace(14.10, 14.15, 50)
    y_wave = np.linspace(40.79, 40.84, 50)

    fig.add_trace(go.Scatter3d(
        x=x_wave, y=y_wave, z=z_wave,
        mode="lines",
        line=dict(color="gold", width=6),
        name="ψₛ Resonance Wave",
        hoverinfo="none"
    ))

    # === ANIMATION FRAMES ===
    frames = []
    for phase in np.linspace(0, 2*np.pi, 20):
        z_anim = -2 + np.sin(t * (psi_s * 3.14) + phase) * 0.5
        frames.append(go.Frame(
            data=[go.Scatter3d(x=x_wave, y=y_wave, z=z_anim,
                               mode="lines", line=dict(color="gold", width=6))],
            name=str(phase)
        ))
    fig.frames = frames
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

    # === Add KP gauge ===
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=kp,
        title={"text": "Geomagnetic Kp Index"},
        domain={"x": [0, 0.4], "y": [0, 0.25]},
        gauge={"axis": {"range": [0, 9]},
               "bar": {"color": "gold"},
               "steps": [
                   {"range": [0, 3], "color": "darkblue"},
                   {"range": [3, 6], "color": "orange"},
                   {"range": [6, 9], "color": "red"}]}
    ))

    return fig
