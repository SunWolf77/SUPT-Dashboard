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
    """Extended dashboard integrating SUPT SunWolf model."""
    # Fetch seismic + geomagnetic data
    cf_df = fetch_ingv(40.79, 40.84, 14.10, 14.15)   # Campi Flegrei
    vulc_df = fetch_ingv(38.38, 38.47, 14.90, 15.05) # Vulcano
    kp = fetch_kp()

    eii, rpam, psi_s = compute_sunwolf(cf_df, vulc_df, kp)

    # === PLOTLY DASHBOARD ===
    fig = go.Figure()
    fig.update_layout(
        title=f"SunWolf-SUPT Phase Tracking — Campi Flegrei & Vulcano<br>EII={eii:.3f}, RPAM={rpam}, ψₛ={psi_s}",
        template="plotly_dark",
        height=600
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

    # Add solar coupling bar
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=kp,
        title={"text": "Geomagnetic Kp Index"},
        domain={"x": [0, 0.4], "y": [0, 0.25]},
        gauge={"axis": {"range": [0, 9]},
               "bar": {"color": "gold"}}
    ))

    return fig
