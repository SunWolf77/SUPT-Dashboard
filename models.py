import numpy as np
from scipy.integrate import odeint
from astropy.coordinates import get_body_barycentric
from astropy.time import Time
import astropy.units as u

def duffing_oscillator(y, t, gamma, alpha, beta, tau, omega, folded_proxy):
    x, v = y
    dxdt = v
    dvdt = -gamma * v - alpha * x - beta * x**3 - 0.01 * v**2 + tau * np.sin(omega * t) * folded_proxy
    return [dxdt, dvdt]

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

def sentinel_forecast(proxies, geomag_kp=0, schumann_power=20.0, historical_matches=None, domain=None, time_steps=100, start_date=None, ionex_text=None):
    t = np.linspace(0, 10, time_steps)
    start_date = start_date or datetime.now().strftime("%Y-%m-%d")
    tidal_factors = compute_tidal_factor(t, start_date)
    alignment_factors = detect_alignments(t, start_date)
    folded_proxy = np.mean(proxies)
    params = (0.80, 0.019, 0.010, 0.05, 0.025, folded_proxy)
    y0 = [0.0, 0.0]
    sol = odeint(duffing_oscillator, y0, t, args=params)
    signal = sol[:, 0]
    signal *= np.exp(-0.1 * t) * alignment_factors * tidal_factors
    if historical_matches:
        a, b = calibrate_resonance(historical_matches, domain)
        amplification = resonance_fit(folded_proxy, a, b)
        signal *= amplification
    geomag_factor = geomag_kp / 9.0 if geomag_kp > 0 else 1.0
    schumann_factor = max(1.0, min(2.0, schumann_power / 20.0))
    signal *= (1 + geomag_factor + schumann_factor)
    signal *= get_goes_flux_factor()
    signal *= get_solar_wind_factor()
    signal *= get_geomag_storm_factor()
    signal *= get_solar_flare_factor()
    signal *= get_laic_tec_factor(ionex_text) if ionex_text else 1.0
    anomaly_signal = low_pass_filter(signal)
    alert = check_critical_triplet(anomaly_signal)
    forecast = np.cumsum(signal) * folded_proxy * np.exp(0.01 * t)
    peaks, _ = find_peaks(forecast, prominence=0.5)
    lyap = np.mean(np.diff(np.log(np.abs(np.diff(forecast) + 1e-10))))
    return t, forecast, peaks, alert, lyap
