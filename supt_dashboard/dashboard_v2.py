import requests
import plotly.graph_objects as go
from datetime import datetime, timezone

NOAA_URL = "https://services.swpc.noaa.gov/json/solar-wind.json"
USGS_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_hour.geojson"

def fetch_noaa():
    try:
        r = requests.get(NOAA_URL, timeout=10)
        r.raise_for_status()
        data = r.json()
        data = data[-10:]
        times = [datetime.utcfromtimestamp(item["time_tag"]/1000).isoformat() for item in data]
        values = [item.get("density", 0.1) for item in data]
        return times, values
    except Exception as e:
        print(f"[NOAA Fallback] {e}")
        now = datetime.now(timezone.utc).isoformat()
        return [now], [0.1]

def fetch_usgs():
    try:
        r = requests.get(USGS_URL, timeout=10)
        r.raise_for_status()
        data = r.json()
        times, mags = [], []
        for feat in data["features"][:10]:
            ts = feat["properties"]["time"] / 1000.0
            times.append(datetime.utcfromtimestamp(ts).isoformat())
            mags.append(feat["properties"]["mag"] or 0)
        return times, mags
    except Exception as e:
        print(f"[USGS Fallback] {e}")
        now = datetime.now(timezone.utc).isoformat()
        return [now], [0.0]

def build_dashboard():
    noaa_times, noaa_vals = fetch_noaa()
    usgs_times, usgs_mags = fetch_usgs()

    stress = [v - 1.0 for v in noaa_vals]  # simple proxy

    fig = go.Figure()

    # NOAA ΔΦ Drift
    fig.add_trace(go.Scatter(
        x=noaa_times, y=noaa_vals,
        mode="lines+markers",
        name="ΔΦ Drift (NOAA)",
        line=dict(color="orange")
    ))

    # Stress overlay
    fig.add_trace(go.Scatter(
        x=noaa_times, y=stress,
        mode="lines",
        name="Stress k(ΔΦ)",
        line=dict(color="blue", dash="dot")
    ))

    # USGS earthquakes
    fig.add_trace(go.Scatter(
        x=usgs_times, y=usgs_mags,
        mode="markers",
        name="Earthquake Magnitude (USGS)",
        marker=dict(color="red", size=10)
    ))

    # Threshold line
    fig.add_hline(y=-1.0, line=dict(color="purple", dash="dash"),
                  annotation_text="ZFCM Threshold (-1.0)")

    # Check alert condition
    if any(s < -1.0 for s in stress):
        fig.add_annotation(
            text="🚨 ALERT: Stress below ZFCM Threshold! 🚨",
            xref="paper", yref="paper",
            x=0.5, y=1.1, showarrow=False,
            font=dict(color="red", size=16, family="Arial Black"),
            bgcolor="yellow", bordercolor="red", borderwidth=2
        )

    # Layout
    fig.update_layout(
        title=f"SUΨT Dashboard — Live NOAA + USGS @ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        xaxis_title="Time (UTC)",
        yaxis_title="Value",
        template="plotly_white",
        height=700
    )

    return fig
