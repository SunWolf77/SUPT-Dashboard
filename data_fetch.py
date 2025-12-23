import requests
import json
from datetime import datetime

def get_goes_flux_factor():
    urls = [
        'https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json',
        'https://services.swpc.noaa.gov/json/goes/secondary/xrays-1-day.json'
    ]
    flux_data = []
    for url in urls:
        try:
            resp = requests.get(url, timeout=10)
            data = json.loads(resp.text)
            flux_data.extend([d for d in data if d.get('energy') == '0.1-0.8 nm'])
        except Exception as e:
            print(f"GOES fetch failed: {e}")
            continue
    if not flux_data:
        return 1.0
    times = [datetime.fromisoformat(d['time_tag'].replace('Z', '+00:00')) for d in flux_data[-180:]]
    fluxes = np.array([d['flux'] for d in flux_data[-180:]])
    log_flux = np.log10(fluxes + 1e-10)
    mins = np.array([(t - times[0]).total_seconds() / 60 for t in times])
    if len(mins) > 1:
        slope, _ = np.polyfit(mins, log_flux, 1)
    else:
        slope = 0
    boost = 1.0
    if slope > 0.01 or np.max(fluxes) > 1e-5:
        boost += 0.5 * (slope / 0.01)
    return max(1.0, min(2.0, boost))

def get_solar_wind_factor():
    url = 'https://services.swpc.noaa.gov/products/solar-wind/plasma-6-hours.json'
    try:
        resp = requests.get(url, timeout=10)
        data = json.loads(resp.text)
        speeds = np.array([float(row[2]) for row in data[1:] if row[2] != 'n/a'])
        if len(speeds) == 0:
            return 1.0
        avg_speed = np.mean(speeds[-12:])
        boost = 1.0
        if avg_speed > 500:
            boost += (avg_speed - 500) / 500
        return max(1.0, min(2.0, boost))
    except Exception as e:
        print(f"Solar wind fetch failed: {e}")
        return 1.0

def get_geomag_storm_factor():
    url = 'https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json'
    try:
        resp = requests.get(url, timeout=10)
        data = json.loads(resp.text)
        latest_kp = float(data[-1][1])
        g_level = 0
        if latest_kp >= 5:
            g_level = min(5, int(latest_kp - 4))
        boost = 1.0 + (g_level * 0.2)
        return max(1.0, min(2.0, boost))
    except Exception as e:
        print(f"Geomag fetch failed: {e}")
        return 1.0

def get_solar_flare_factor():
    url = 'https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json'
    try:
        resp = requests.get(url, timeout=10)
        data = json.loads(resp.text)
        fluxes = [d['flux'] for d in data[-10:] if d.get('energy') == '0.1-0.8 nm']
        max_flux = max(fluxes) if fluxes else 0
        if max_flux > 1e-4:
            return 2.0  # X-class
        elif max_flux > 1e-5:
            return 1.5  # M-class
        elif max_flux > 1e-6:
            return 1.2  # C-class
        return 1.0
    except Exception as e:
        print(f"Flare fetch failed: {e}")
        return 1.0
