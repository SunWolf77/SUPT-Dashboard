# data_fetch.py - All live NOAA/SWPC API fetch functions
import requests
import json
import numpy as np
from datetime import datetime

def get_goes_flux_factor():
    """Fetch GOES X-ray flux and calculate boost based on slope and peak."""
    urls = [
        'https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json',
        'https://services.swpc.noaa.gov/json/goes/secondary/xrays-1-day.json'
    ]
    flux_data = []
    for url in urls:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()  # Raise exception on HTTP error
            data = json.loads(resp.text)
            flux_data.extend([d for d in data if d.get('energy') == '0.1-0.8 nm'])
        except Exception as e:
            print(f"GOES fetch failed for {url}: {e}")
            continue

    if not flux_data:
        return 1.0  # Default if no data

    # Take last ~3 hours (180 points)
    recent = flux_data[-180:]
    times = [datetime.fromisoformat(d['time_tag'].replace('Z', '+00:00')) for d in recent]
    fluxes = np.array([d['flux'] for d in recent])
    log_flux = np.log10(fluxes + 1e-10)

    mins = np.array([(t - times[0]).total_seconds() / 60 for t in times])
    if len(mins) > 1:
        slope, _ = np.polyfit(mins, log_flux, 1)
    else:
        slope = 0

    boost = 1.0
    if slope > 0.01 or np.max(fluxes) > 1e-5:  # M-class precursor
        boost += 0.5 * max(0, slope / 0.01)
    return max(1.0, min(2.0, boost))

def get_solar_wind_factor():
    """Fetch solar wind speed and calculate boost for high-speed streams."""
    url = 'https://services.swpc.noaa.gov/products/solar-wind/plasma-6-hours.json'
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = json.loads(resp.text)
        # [time_tag, density, speed, temperature] - speed is index 2
        speeds = np.array([float(row[2]) for row in data[1:] if row[2] != 'n/a' and row[2].isdigit()])
        if len(speeds) == 0:
            return 1.0
        avg_speed = np.mean(speeds[-12:])  # Last ~hour
        boost = 1.0
        if avg_speed > 500:
            boost += (avg_speed - 500) / 500  # Linear scale to 1000 km/s → 2.0
        return max(1.0, min(2.0, boost))
    except Exception as e:
        print(f"Solar wind fetch failed: {e}")
        return 1.0

def get_geomag_storm_factor():
    """Fetch planetary K-index and map to G-scale boost."""
    url = 'https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json'
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = json.loads(resp.text)
        latest_kp = float(data[-1][1])  # Latest Kp value
        g_level = 0
        if latest_kp >= 5:
            g_level = min(5, int(latest_kp - 4))
        boost = 1.0 + (g_level * 0.2)  # 0.2 per G-level
        return max(1.0, min(2.0, boost))
    except Exception as e:
        print(f"Geomag fetch failed: {e}")
        return 1.0

def get_solar_flare_factor():
    """Fetch recent GOES X-ray flux and classify flare boost."""
    url = 'https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json'
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = json.loads(resp.text)
        fluxes = [d['flux'] for d in data[-10:] if d.get('energy') == '0.1-0.8 nm']
        if not fluxes:
            return 1.0
        max_flux = max(fluxes)
        if max_flux > 1e-4:  # X-class
            return 2.0
        elif max_flux > 1e-5:  # M-class
            return 1.5
        elif max_flux > 1e-6:  # C-class
            return 1.2
        return 1.0
    except Exception as e:
        print(f"Solar flare fetch failed: {e}")
        return 1.0
