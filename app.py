import streamlit as st
from models import sentinel_forecast, calibrate_resonance, resonance_fit
from data_fetch import get_goes_flux_factor, get_solar_wind_factor, get_geomag_storm_factor, get_solar_flare_factor
from utils import low_pass_filter, check_critical_triplet
import matplotlib.pyplot as plt
from datetime import datetime

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

# Historical matches
historical_matches = [
    [0.8, 6.9, 1, 'EQ', 3.0, 26.0],
    [0.7, 5.5, 2, 'EQ', 4.0, 20.0],
]

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
        st.success(f"Peaks at days: {', '.join([f'{d:.1f}' for d in t[peaks]])}" if peaks.size > 0 else "No peaks detected")
        st.info(f"Lyapunov: {lyap:.3f}")
        st.info(f"Critical Triplet: {'YES' if alert else 'No'}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
