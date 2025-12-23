import streamlit as st
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
from scipy.signal import find_peaks, butter, filtfilt
from scipy.optimize import curve_fit
from scipy.integrate import odeint
from astropy.coordinates import get_body_barycentric
from astropy.time import Time
import astropy.units as u
import matplotlib.pyplot as plt
import requests
from datetime import datetime
from PIL import Image
from io import BytesIO
import cv2
import pytesseract

st.title("SunWolf's Sentinel - EQ & Volcano Forecast")

# Inputs
p1, p2 = st.columns(2)
proxies = [p1.slider("Proxy 1", 0.0, 1.0, 0.75), p2.slider("Proxy 2", 0.0, 1.0, 0.7)]
kp = st.number_input("Kp Index", value=2.0)
schumann = st.number_input("Schumann Power (manual)", value=20.0)
domain = st.selectbox("Domain", ['EQ', 'VOLC', 'SOL'])
start = st.text_input("Start Date", datetime.now().strftime("%Y-%m-%d"))
ionex = st.text_area("IONEX Text (optional)")

# Historical (expand with real events)
history = [
    [0.8, 6.9, 1, 'EQ', 3.0, 26.0],
    [0.82, 7.6, 1, 'EQ', 2.5, 28.0],
    [0.95, 1.0, 0, 'VOLC', 3.0, 10.0],  # Example volcano
]

# Resonance fit & calibration
def res_fit(x, a, b): return a * np.exp(b * x)

def calibrate(matches, dom=None):
    f = [m for m in matches if dom is None or m[3] == dom]
    if len(f) < 2: return 1.0, 0.0
    x, y = zip(*[(m[0], m[1]) for m in f])
    try:
        return curve_fit(res_fit, x, y, p0=[1, 1])[0]
    except:
        return 1.0, 0.0

# Duffing oscillator
def duffing(y, t, g, a, b, tau, w, p):
    x, v = y
    return [v, -g*v - a*x - b*x**3 - 0.01*v**2 + tau*np.sin(w*t)*p]

# Tidal factor
def tidal(t_days, start):
    vals = []
    for d in t_days:
        t = Time(start) + d * u.day
        e = get_body_barycentric('earth', t)
        total = 0
        for b, m in zip(['moon'], [7.342e22]):  # Simplified - moon dominant
            pos = get_body_barycentric(b, t)
            dist = np.linalg.norm((pos - e).xyz.to(u.au).value) * u.au
            total += 2 * 6.6743e-11 * m * 6371e3 / dist**3
        vals.append(total)
    return np.array(vals) / 1e-6 if vals else np.ones(len(t_days))

# Alignments
def alignments(t_days, start):
    fac = np.ones(len(t_days))
    for i, d in enumerate(t_days):
        t = Time(start) + d * u.day
        moon = get_body('moon', t).icrs
        boost = 1.0
        for p in ['mars', 'jupiter']:
            sep = moon.separation(get_body(p, t).icrs).deg
            if abs(sep - 180) < 5: boost += 0.3
        fac[i] = boost
    return fac

# Low-pass & triplet
def lowpass(data):
    b, a = butter(3, 0.1 / 0.5, btype='low')
    return filtfilt(b, a, data)

def triplet(signal):
    p, _ = find_peaks(signal)
    return len(p) >= 3 and p[2] - p[0] <= 20

# Live APIs (simplified)
def fetch_goes(): return 1.0  # Replace with real fetch
def fetch_wind(): return 1.0
def fetch_kp(): return 1.0
def fetch_flare(): return 1.0

# Schumann OCR
def fetch_schumann_ocr():
    url = "https://sosrff.tsu.ru/new/sch.png"
    try:
        r = requests.get(url, timeout=15)
        img = Image.open(BytesIO(r.content))
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        crop = gray[int(h*0.65):int(h*0.92), int(w*0.68):int(w*0.98)]
        crop = cv2.convertScaleAbs(crop, alpha=2.0, beta=0)
        text = pytesseract.image_to_string(crop, config='--psm 7 digits')
        nums = [int(s) for s in text.split() if s.isdigit() and 5 < int(s) < 200]
        return nums[0] if nums else 20.0
    except:
        return 20.0

# EQ/Volcano Risk Score (probabilistic)
def risk_score(proxies, boosts, domain):
    base = np.mean(proxies) * 10
    solar = boosts['flare'] * boosts['goes']
    geo = boosts['kp'] * boosts['wind']
    total = base + solar * (3 if domain == 'SOL' else 1) + geo * (2 if domain == 'EQ' else 1)
    if domain == 'VOLC':
        total *= 1.3  # Volcano sensitivity boost
    return min(100, max(0, total))

# Interactive Map (volcanoes + recent EQs)
def create_map():
    m = folium.Map(location=[0, 0], zoom_start=2)
    # Example volcanoes (expand with real data)
    volcanoes = [
        {"name": "Kilauea", "lat": 19.421, "lon": -155.287, "risk": 45},
        {"name": "Etna", "lat": 37.751, "lon": 14.993, "risk": 60},
    ]
    for v in volcanoes:
        folium.Marker(
            [v["lat"], v["lon"]],
            popup=f"{v['name']} - Risk: {v['risk']}%",
            icon=folium.Icon(color="red" if v["risk"] > 50 else "orange")
        ).add_to(m)
    return m

if st.button("Run Forecast"):
    try:
        schumann = fetch_schumann_ocr()
        boosts = {'goes': fetch_goes(), 'wind': fetch_wind(), 'kp': fetch_kp(), 'flare': fetch_flare()}
        t = np.linspace(0, 10, 100)
        tidal = tidal(t, start)
        align = alignments(t, start)
        p = np.mean(proxies)
        sol = odeint(duffing, [0, 0], t, args=(0.8, 0.019, 0.01, 0.05, 0.025, p))
        sig = sol[:, 0] * align * tidal
        sig *= (1 + kp/9.0 + schumann/20.0)
        sig *= boosts['goes'] * boosts['wind'] * boosts['kp'] * boosts['flare']
        anom = lowpass(sig)
        alert = triplet(anom)
        fore = np.cumsum(sig) * p * np.exp(0.01 * t)
        peaks, _ = find_peaks(fore, prominence=0.5)
        lyap = np.mean(np.diff(np.log(np.abs(np.diff(fore) + 1e-10))))
        risk = risk_score(proxies, boosts, domain)

        # Plot
        fig, ax = plt.subplots()
        ax.plot(t, fore, label='Forecast')
        ax.scatter(t[peaks], fore[peaks], color='red', label='Peaks')
        ax.legend()
        st.pyplot(fig)

        st.write(f"Peaks: {', '.join([f'{d:.1f}' for d in t[peaks]])} days" if peaks.size else "No peaks")
        st.write(f"Lyapunov: {lyap:.3f}")
        st.write(f"Triplet Alert: {'YES' if alert else 'No'}")
        st.write(f"Risk Score ({domain}): {risk:.0f}%")

        # Interactive Map
        st.subheader("Volcano & EQ Risk Map")
        m = create_map()
        st_folium(m, width=700, height=500)

    except Exception as e:
        st.error(f"Error: {str(e)}")
