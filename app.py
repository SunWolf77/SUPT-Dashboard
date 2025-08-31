import streamlit as st
from supt_dashboard.dashboard_v2 import build_dashboard

st.set_page_config(page_title="SUΨT Dashboard", layout="wide")

st.title("SUΨT Dashboard 🛰️")
st.markdown("Live NOAA Solar Wind + USGS Earthquake Data")

fig = build_dashboard()
st.plotly_chart(fig, use_container_width=True)
