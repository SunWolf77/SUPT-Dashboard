import streamlit as st
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
from scipy.signal import find_peaks, butter, filtfilt
from scipy.integrate import odeint
import requests
from datetime import datetime
from PIL import Image
from io import BytesIO
import cv2
import pytesseract

st.title("SunWolf Sentinel - Live Forecast")

# Inputs
proxies = [st.slider(f"Proxy {i+1}", 0.0, 1.0, 0.75) for i in range(2)]
kp = st.number_input("Kp Index", value=2.0)
schumann = st.number_input("Schumann Power", value=20.0)
domain = st.selectbox("Domain", ['EQ', 'VOLC', 'SOL'])
start = st.text_input("Start Date", datetime.now().strftime("%Y-%m-%d"))

# --- Live NOAA Data ---
def get_goes(): return 1.0  # Simplified; add real fetch if needed
def get_wind(): return 1.0
def get_kp(): return 1.0
def get_flare(): return 1.0

# --- Schumann OCR ---
def fetch_schumann():
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

# --- Forecast (simplified for demo) ---
if st.button("Run Forecast"):
    try:
        sch = fetch_schumann()
        p = np.mean(proxies)
        t = np.linspace(0, 10, 100)
        sig = np.exp(0.1 * t) * p  # Mock signal
        fore = np.cumsum(sig)
        peaks, _ = find_peaks(fore, prominence=0.5)
        fig, ax = plt.subplots()
        ax.plot(t, fore, label='Forecast')
        ax.scatter(t[peaks], fore[peaks], color='red', label='Peaks')
        ax.legend()
        st.pyplot(fig)
        st.write(f"Peaks at: {', '.join([f'{d:.1f}' for d in t[peaks]])} days")
        st.write(f"Schumann (OCR): {sch:.1f}")
        st.success("Forecast complete!")
    except Exception as e:
        st.error(f"Error: {str(e)}")

# --- Live USGS EQ Map ---
st.subheader("Live USGS Earthquakes")
try:
    r = requests.get("https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_day.geojson")
    data = r.json()['features']
    m = folium.Map(location=[0, 0], zoom_start=2)
    for f in data[:20]:
        props = f['properties']
        geo = f['geometry']['coordinates']
        folium.CircleMarker(
            [geo[1], geo[0]],
            radius=props['mag'] * 3,
            color="red" if props['mag'] > 5 else "blue",
            popup=f"M{props['mag']}<br>{props['place']}"
        ).add_to(m)
    st_folium(m, width=700)
except:
    st.info("EQ map unavailable")
