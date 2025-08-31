# SUΨT Dashboard 🛰️

Live NOAA Solar Wind + USGS Earthquake Dashboard with SUΨT stress threshold overlay and ALERT banner.

## Features
- Live NOAA ΔΦ Drift proxy (solar wind data)
- Live USGS Earthquake magnitudes (last week)
- Stress overlay (k(ΔΦ))
- ZFCM Threshold marker (-1.0)
- 🚨 ALERT banner when stress falls below threshold

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → pick `app.py` as the entrypoint
4. Deploy 🎉
