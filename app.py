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
        st.warning("Not enough historical data for calibration — using default amplification.")
        return 1.0, 0.0
    proxies, outcomes, _, _, _, _ = zip(*filtered)
    try:
        popt, _ = curve_fit(resonance_fit, proxies, outcomes, p0=[1, 1])
        return popt
    except Exception as e:
        st.error(f"Calibration failed: {e}")
        return 1.0, 0.0

# (Add remaining full functions: duffing_oscillator, compute_tidal_factor, detect_alignments, low_pass_filter, check_critical_triplet, get_goes_flux_factor, get_solar_wind_factor, get_geomag_storm_factor, get_laic_tec_factor, get_schumann_factor, get_solar_flare_factor, sentinel_forecast)
# ... Paste all of them here from our previous complete versions ...

# Run button
if st.button("Run Forecast"):
    try:
        t, forecast, peaks = sentinel_forecast(
            proxies=proxies,
            geomag_kp=geomag_kp,
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
        st.write("Peaks (Days Ahead):", t[peaks])
    except Exception as e:
        st.error(f"Run failed: {str(e)}")
