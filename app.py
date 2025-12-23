import streamlit as st
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from scipy.optimize import curve_fit
from scipy.integrate import odeint
from astropy.coordinates import get_body, SkyCoord, get_body_barycentric
from astropy.time import Time
import astropy.units as u
import matplotlib.pyplot as plt
import requests
import json
from datetime import datetime, timedelta

st.title("SunWolf's Sentinel Forecasting Dashboard")

# Inputs
col1, col2 = st.columns(2)
proxy1 = col1.slider("Proxy 1 (0-1)", 0.0, 1.0, 0.75)
proxy2 = col2.slider("Proxy 2 (0-1)", 0.0, 1.0, 0.7)
proxies = [proxy1, proxy2]
geomag_kp = st.number_input("Current Geomag Kp Index", value=2.0)
schumann_power = st.number_input("Schumann Power (manual from charts)", value=20.0)
domain = st.selectbox("Domain", ['EQ', 'VOLC', 'SOL'])
start_date = st.text_input("Start Date (YYYY-MM-DD)", datetime.now().strftime("%Y-%m-%d"))
ionex_text = st.text_area("Paste IONEX Text (optional for LAIC)")

# Historical matches (expand as needed)
historical_matches = [
    [0.8, 6.9, 1, 'EQ', 3.0, 26.0],
    [0.7, 5.5, 2, 'EQ', 4.0, 20.0],
]

# Resonance fit function
def resonance_fit(x, a, b):
    return a * np.exp(b * x)

# Calibrate resonance
def calibrate_resonance(matches, domain=None):
    if domain:
        filtered = [m for m in matches if m[3] == domain]
    else:
        filtered = matches
    if len(filtered) < 2:
        st.warning("Not enough historical data — using default amplification.")
        return 1.0, 0.0
    proxies, outcomes, _, _, _, _ = zip(*filtered)
    try:
        popt, _ = curve_fit(resonance_fit, proxies, outcomes, p0=[1, 1])
        return popt
    except Exception as e:
        st.error(f"Calibration failed: {e}")
        return 1.0, 0.0

# Duffing-like oscillator
def duffing_oscillator(y, t, gamma, alpha, beta, tau, omega, proxies):
    x, v = y
    folded_proxy = np.mean(proxies)
    dxdt = v
    dvdt = -gamma * v - alpha * x - beta * x**3 - 0.01 * v**2 + tau * np.sin(omega * t) * folded_proxy
    return [dxdt, dvdt]

# Tidal factor
def compute_tidal_factor(t_days, start_date, bodies=['moon', 'mars', 'saturn', 'neptune']):
    tidal_values = []
    for day in t_days:
        t_astropy = Time(start_date) + day * u.day
        total_tidal = 0
        earth_pos = get_body_barycentric('earth', t_astropy)
        for body in bodies:
            body_pos = get_body_barycentric(body, t_astropy)
            d = np.linalg.norm((body_pos - earth_pos).xyz.to(u.au).value) * u.au
            if body == 'moon':
                M = 7.342e22 * u.kg
            elif body == 'mars':
                M = 6.417e23 * u.kg
            elif body == 'saturn':
                M = 5.683e26 * u.kg
            elif body == 'neptune':
                M = 1.024e26 * u.kg
            else:
                continue
            G = 6.67430e-11 * u.m**3 / u.kg / u.s**2
            R_earth = 6371e3 * u.m
            tidal = 2 * G * M * R_earth / d**3
            total_tidal += tidal.decompose().value
        tidal_values.append(total_tidal)
    tidal_norm = np.array(tidal_values) / 1e-6 if np.max(tidal_values) > 0 else np.ones(len(t_days))
    return tidal_norm

# Alignment detection
def detect_alignments(t_days, start_date, base_body='moon', planets=['mars', 'jupiter', 'saturn', 'uranus'], aspects=[0, 60, 90, 120]):
    alignment_factors = np.ones(len(t_days))
    for i, day in enumerate(t_days):
        t_astropy = Time(start_date) + day * u.day
        base_pos = get_body(base_body, t_astropy).icrs
        boost = 1.0
        for planet in planets:
            planet_pos = get_body(planet, t_astropy).icrs
            sep = base_pos.separation(planet_pos).deg
            for aspect in aspects:
                if abs(sep - aspect) < 1.0:
                    boost += 0.2
        alignment_factors[i] = boost
    return alignment_factors

# Low-pass filter
def low_pass_filter(data, cutoff=0.1, fs=1.0, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Critical triplet check
def check_critical_triplet(signal, station_dists=[600], time_int=20):
    peaks, _ = find_peaks(signal)
    if len(peaks) >= 3:
        times = peaks[:3]
        if max(times) - min(times) <= time_int:
            return True
    return False

# GOES X-ray
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

# Solar wind
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
    except:
        return 1.0

# Geomagnetic storm
def get_geomag_storm_factor():
    url = 'https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json'
    try:
        resp = requests.get(url, timeout=10)
        data = json.loads(resp.text)
        latest_kp = float(data[-1][1])
        g_level = 0
        if latest_kp >= 5:
            g_level = min(5, (latest_kp - 4) // 1)
        boost = 1.0 + (g_level * 0.2)
        return max(1.0, min(2.0, boost))
    except:
        return 1.0

# LAIC TEC
def get_laic_tec_factor(ionex_text):
    tec_values = []
    exponent = 0
    in_tec_map = False
    lines = ionex_text.split('\n')
    for line in lines:
        if 'EXPONENT' in line:
            exponent = int(line.split()[0])
        if 'START OF TEC MAP' in line:
            in_tec_map = True
        elif 'END OF TEC MAP' in line:
            in_tec_map = False
        elif in_tec_map and 'LAT/LON' not in line:
            for v in line.split():
                if v.isdigit() or (v.startswith('-') and v[1:].isdigit()):
                    tec_values.append(int(v))
    if tec_values:
        mean_tec = np.mean(tec_values) * (10 ** exponent) * 0.1
        baseline = 20.0
        boost = max(1.0, mean_tec / baseline)
        return min(2.0, boost)
    return 1.0

# Schumann (manual input already in UI)
def get_schumann_factor(power):
    baseline = 20.0
    boost = max(1.0, power / baseline)
    return min(2.0, boost)

# Solar flare
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
    except:
        return 1.0

# Sentinel forecast function
def sentinel_forecast(proxies, geomag_kp=0, schumann_power=20.0, historical_matches=None, domain=None, time_steps=100, start_date=None, ionex_text=None):
    t = np.linspace(0, 10, time_steps)
    tidal_factors = compute_tidal_factor(t, start_date or datetime.now().strftime("%Y-%m-%d"))
    alignment_factors = detect_alignments(t, start_date or datetime.now().strftime("%Y-%m-%d"))
    folded_proxy = np.mean(proxies)
    params = (0.80, 0.019, 0.010, 0.05, 0.025, proxies)
    y0 = [0.0, 0.0]
    sol = odeint(duffing_oscillator, y0, t, args=params)
    signal = sol[:, 0]
    signal *= np.exp(-0.1 * t) * alignment_factors * tidal_factors
    if historical_matches:
        a, b = calibrate_resonance(historical_matches, domain)
        amplification = resonance_fit(folded_proxy, a, b)
        signal *= amplification
    geomag_factor = geomag_kp / 9.0 if geomag_kp > 0 else 1.0
    schumann_factor = get_schumann_factor(schumann_power)
    signal *= (1 + geomag_factor + schumann_factor)
    signal *= get_goes_flux_factor()
    signal *= get_solar_wind_factor()
    signal *= get_geomag_storm_factor()
    signal *= get_solar_flare_factor()
    signal *= get_laic_tec_factor(ionex_text) if ionex_text else 1.0
    anomaly_signal = low_pass_filter(signal)
    alert = check_critical_triplet(anomaly_signal)
    forecast = np.cumsum(signal) * folded_proxy * np.exp(0.01 * t)
    peaks, _ = find_peaks(forecast)
    lyap = np.mean(np.diff(np.log(np.abs(np.diff(forecast) + 1e-10))))
    return t, forecast, peaks, alert, lyap

# Run button
if st.button("Run Forecast"):
    try:
        t, forecast, peaks, alert, lyap = sentinel_forecast(
            proxies=proxies,
            geomag_kp=geomag_kp,
            schumann_power=schumann_power,
            historical_matches=historical_matches,
            domain=domain,
            start_date=start_date,
            ionex_text=ionex_text
        )
        fig, ax = plt.subplots()
        ax.plot(t, forecast, label='Forecast')
        ax.scatter(t[peaks], forecast[peaks], color='red', label='Peaks')
        ax.set_xlabel('Days Ahead')
        ax.set_ylabel('Intensity')
        ax.set_title('Sentinel Forecast')
        ax.legend()
        st.pyplot(fig)
        st.success("Forecast complete!")
        st.write(f"Critical Triplet Alert: {alert}")
        st.write(f"Estimated Lyapunov: {lyap:.3f}")
        st.write("Peaks (Days Ahead):", t[peaks])
    except Exception as e:
        st.error(f"Run failed: {str(e)}")
