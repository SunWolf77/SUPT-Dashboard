import requests
import json
import numpy as np

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
        except:
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

# (add get_solar_wind_factor, get_geomag_storm_factor, get_solar_flare_factor as in previous versions)
