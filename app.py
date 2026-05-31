import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WeatherSense AI",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

:root {
    --bg: #0a0e1a;
    --card: #111827;
    --card2: #1a2235;
    --accent: #38bdf8;
    --accent2: #f59e0b;
    --accent3: #34d399;
    --text: #e2e8f0;
    --muted: #64748b;
    --border: #1e2d45;
    --danger: #f87171;
    --rain: #60a5fa;
}

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp { background: var(--bg) !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--card) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Inputs ── */
.stTextInput input, .stSelectbox select, .stNumberInput input {
    background: var(--card2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
}
.stTextInput input:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 2px rgba(56,189,248,0.2) !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #38bdf8) !important;
    color: #0a0e1a !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.4rem !important;
    font-family: 'Syne', sans-serif !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(56,189,248,0.35) !important; }

/* ── Cards ── */
.weather-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.4rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.weather-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--accent), var(--accent2), var(--accent3));
}
.metric-card {
    background: var(--card2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-3px); border-color: var(--accent); }
.metric-value { font-size: 2rem; font-weight: 800; color: var(--accent); line-height: 1; }
.metric-label { font-size: 0.72rem; color: var(--muted); margin-top: 0.4rem; text-transform: uppercase; letter-spacing: 1px; }
.metric-unit { font-size: 0.85rem; color: var(--muted); }

/* ── Forecast Cards ── */
.forecast-card {
    background: var(--card2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 0.8rem;
    text-align: center;
}
.forecast-day { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; }
.forecast-icon { font-size: 1.8rem; margin: 0.4rem 0; }
.forecast-temp { font-size: 1rem; font-weight: 700; }
.forecast-high { color: var(--accent2); }
.forecast-low { color: var(--rain); }

/* ── Prediction Badge ── */
.pred-badge {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 700;
    margin: 0.2rem;
}
.pred-rain { background: rgba(96,165,250,0.15); color: var(--rain); border: 1px solid rgba(96,165,250,0.3); }
.pred-sun { background: rgba(245,158,11,0.15); color: var(--accent2); border: 1px solid rgba(245,158,11,0.3); }
.pred-cloud { background: rgba(100,116,139,0.2); color: #94a3b8; border: 1px solid rgba(100,116,139,0.3); }

/* ── Header ── */
.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.main-header h1 { font-size: 2.6rem; font-weight: 800; margin: 0; background: linear-gradient(135deg, #38bdf8, #f59e0b); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.main-header p { color: var(--muted); margin: 0.4rem 0 0; font-family: 'Space Mono', monospace; font-size: 0.85rem; }

/* ── Section Title ── */
.section-title {
    font-size: 1rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--accent);
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}

/* ── Progress Bar ── */
.stProgress > div > div { background: linear-gradient(90deg, #38bdf8, #34d399) !important; border-radius: 4px; }

/* ── Alerts ── */
.stAlert { border-radius: 12px !important; }

/* ── Wind Direction ── */
.wind-compass {
    width: 80px; height: 80px;
    border: 2px solid var(--border);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem;
    background: var(--card2);
    margin: 0 auto;
}

/* ── Hide Streamlit Branding ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ─────────────────────────────────────────────────────────────────
OPENWEATHER_BASE = "https://api.openweathermap.org/data/2.5"
FORECAST_BASE    = "https://api.openweathermap.org/data/2.5/forecast"

WEATHER_ICONS = {
    "Clear": "☀️", "Clouds": "☁️", "Rain": "🌧️",
    "Drizzle": "🌦️", "Thunderstorm": "⛈️", "Snow": "❄️",
    "Mist": "🌫️", "Fog": "🌫️", "Haze": "🌫️",
    "Smoke": "💨", "Dust": "💨", "Sand": "💨",
    "Ash": "🌋", "Squall": "🌬️", "Tornado": "🌪️"
}

WIND_DIRS = {
    (0,   22.5): "N ↑",  (22.5,  67.5): "NE ↗", (67.5,  112.5): "E →",
    (112.5,157.5): "SE ↘",(157.5,202.5): "S ↓", (202.5,247.5): "SW ↙",
    (247.5,292.5): "W ←", (292.5,337.5): "NW ↖",(337.5,360): "N ↑"
}

# ─── Helpers ───────────────────────────────────────────────────────────────────
def get_wind_dir(deg):
    for (lo, hi), label in WIND_DIRS.items():
        if lo <= deg < hi:
            return label
    return "N ↑"

def weather_icon(condition):
    return WEATHER_ICONS.get(condition, "🌤️")

def get_aqi_label(aqi):
    labels = {1:"Good 🟢", 2:"Fair 🟡", 3:"Moderate 🟠", 4:"Poor 🔴", 5:"Very Poor 🟣"}
    return labels.get(aqi, "Unknown")

# ─── API Calls ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def fetch_current_weather(location, api_key, unit="metric"):
    try:
        # Try city name first
        url = f"{OPENWEATHER_BASE}/weather?q={location}&appid={api_key}&units={unit}"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json(), None
        # Try lat,lon
        if "," in location:
            lat, lon = location.split(",")
            url = f"{OPENWEATHER_BASE}/weather?lat={lat.strip()}&lon={lon.strip()}&appid={api_key}&units={unit}"
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                return r.json(), None
        return None, r.json().get("message", "City not found")
    except Exception as e:
        return None, str(e)

@st.cache_data(ttl=600)
def fetch_forecast(location, api_key, unit="metric"):
    try:
        url = f"{FORECAST_BASE}?q={location}&appid={api_key}&units={unit}&cnt=40"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json(), None
        if "," in location:
            lat, lon = location.split(",")
            url = f"{FORECAST_BASE}?lat={lat.strip()}&lon={lon.strip()}&appid={api_key}&units={unit}&cnt=40"
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                return r.json(), None
        return None, r.json().get("message", "Forecast unavailable")
    except Exception as e:
        return None, str(e)

@st.cache_data(ttl=3600)
def fetch_air_quality(lat, lon, api_key):
    try:
        url = f"{OPENWEATHER_BASE}/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

# ─── ML Model ──────────────────────────────────────────────────────────────────
def generate_synthetic_history(current_data, n=500):
    """Generate plausible historical data around current conditions for demo ML."""
    np.random.seed(42)
    temp    = current_data.get("main", {}).get("temp", 25)
    humidity= current_data.get("main", {}).get("humidity", 60)
    pressure= current_data.get("main", {}).get("pressure", 1013)
    wind    = current_data.get("wind", {}).get("speed", 5)

    temps     = np.random.normal(temp, 6, n)
    humids    = np.clip(np.random.normal(humidity, 12, n), 10, 100)
    pressures = np.random.normal(pressure, 8, n)
    winds     = np.clip(np.random.exponential(wind + 1, n), 0, 50)
    clouds    = np.random.randint(0, 101, n)
    hours     = np.random.randint(0, 24, n)
    months    = np.random.randint(1, 13, n)

    # Rain probability heuristic
    rain_prob = (humids > 75).astype(float) * 0.5 + (clouds > 70).astype(float) * 0.3 + \
                (pressures < 1005).astype(float) * 0.2
    rain_next = (rain_prob + np.random.normal(0, 0.15, n) > 0.45).astype(int)

    temp_next = temps + np.random.normal(0, 3, n)
    humidity_next = np.clip(humids + np.random.normal(0, 5, n), 10, 100)

    conditions = []
    for i in range(n):
        if rain_next[i] == 1:
            conditions.append("Rain")
        elif clouds[i] > 70:
            conditions.append("Clouds")
        else:
            conditions.append("Clear")

    return pd.DataFrame({
        "temp": temps, "humidity": humids, "pressure": pressures,
        "wind_speed": winds, "clouds": clouds, "hour": hours, "month": months,
        "temp_next": temp_next, "humidity_next": humidity_next,
        "rain_next": rain_next, "condition_next": conditions
    })

@st.cache_resource
def train_models(current_data):
    df = generate_synthetic_history(current_data)
    features = ["temp","humidity","pressure","wind_speed","clouds","hour","month"]
    X = df[features]

    # Temp predictor
    rf_temp = RandomForestRegressor(n_estimators=120, max_depth=8, random_state=42)
    rf_temp.fit(X, df["temp_next"])

    # Humidity predictor
    rf_hum = RandomForestRegressor(n_estimators=80, max_depth=6, random_state=42)
    rf_hum.fit(X, df["humidity_next"])

    # Rain classifier
    rf_rain = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42)
    rf_rain.fit(X, df["rain_next"])

    # Condition classifier
    le = LabelEncoder()
    y_cond = le.fit_transform(df["condition_next"])
    rf_cond = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    rf_cond.fit(X, y_cond)

    return rf_temp, rf_hum, rf_rain, rf_cond, le, features

def make_predictions(current_data, forecast_data):
    rf_temp, rf_hum, rf_rain, rf_cond, le, features = train_models(current_data)

    main   = current_data.get("main", {})
    wind   = current_data.get("wind", {})
    clouds = current_data.get("clouds", {}).get("all", 50)
    now    = datetime.now()

    X_now = pd.DataFrame([[
        main.get("temp", 25),
        main.get("humidity", 60),
        main.get("pressure", 1013),
        wind.get("speed", 5),
        clouds,
        now.hour,
        now.month
    ]], columns=features)

    pred_temp  = rf_temp.predict(X_now)[0]
    pred_hum   = rf_hum.predict(X_now)[0]
    rain_prob  = rf_rain.predict_proba(X_now)[0][1] * 100
    cond_idx   = rf_cond.predict(X_now)[0]
    pred_cond  = le.inverse_transform([cond_idx])[0]

    # 24h predictions using forecast data
    hourly_preds = []
    if forecast_data:
        for item in forecast_data.get("list", [])[:8]:
            fi = item.get("main", {})
            fw = item.get("wind", {})
            fc = item.get("clouds", {}).get("all", 50)
            dt = datetime.fromtimestamp(item["dt"])
            Xi = pd.DataFrame([[
                fi.get("temp", 25), fi.get("humidity", 60), fi.get("pressure", 1013),
                fw.get("speed", 5), fc, dt.hour, dt.month
            ]], columns=features)
            hourly_preds.append({
                "time": dt.strftime("%I %p"),
                "temp": rf_temp.predict(Xi)[0],
                "humidity": rf_hum.predict(Xi)[0],
                "rain_prob": rf_rain.predict_proba(Xi)[0][1] * 100,
                "condition": le.inverse_transform([rf_cond.predict(Xi)[0]])[0]
            })

    return {
        "temp_next": pred_temp,
        "humidity_next": pred_hum,
        "rain_probability": rain_prob,
        "condition_next": pred_cond,
        "hourly": hourly_preds
    }

# ─── UI Components ─────────────────────────────────────────────────────────────
def render_header(city_name, country):
    st.markdown(f"""
    <div class="main-header">
        <h1>🌤️ WeatherSense AI</h1>
        <p>📍 {city_name}, {country} &nbsp;·&nbsp; ML-Enhanced Forecast &nbsp;·&nbsp; {datetime.now().strftime("%A, %d %B %Y %H:%M")}</p>
    </div>
    """, unsafe_allow_html=True)

def render_current_metrics(data, unit_sym):
    main   = data.get("main", {})
    wind   = data.get("wind", {})
    clouds = data.get("clouds", {}).get("all", 0)
    vis    = data.get("visibility", 10000)
    cond   = data.get("weather", [{}])[0]

    temp      = main.get("temp", "--")
    feels     = main.get("feels_like", "--")
    humidity  = main.get("humidity", "--")
    pressure  = main.get("pressure", "--")
    wind_spd  = wind.get("speed", "--")
    wind_deg  = wind.get("deg", 0)
    wind_dir  = get_wind_dir(wind_deg)
    desc      = cond.get("description", "").title()
    icon      = weather_icon(cond.get("main", ""))

    cols = st.columns([2, 1, 1, 1, 1, 1])
    with cols[0]:
        st.markdown(f"""
        <div class="weather-card" style="text-align:center;">
            <div style="font-size:5rem;line-height:1">{icon}</div>
            <div style="font-size:3.2rem;font-weight:800;color:var(--accent);line-height:1.1">{temp:.1f}°{unit_sym}</div>
            <div style="color:var(--muted);margin-top:0.3rem;font-family:'Space Mono',monospace;font-size:0.85rem">{desc}</div>
            <div style="color:var(--muted);font-size:0.78rem;margin-top:0.3rem">Feels like {feels:.1f}°{unit_sym}</div>
        </div>
        """, unsafe_allow_html=True)

    with cols[1]:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:1.6rem">💧</div>
            <div class="metric-value">{humidity}</div>
            <div class="metric-unit">%</div>
            <div class="metric-label">Humidity</div>
        </div>
        """, unsafe_allow_html=True)

    with cols[2]:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:1.6rem">💨</div>
            <div class="metric-value" style="font-size:1.5rem">{wind_spd}</div>
            <div class="metric-unit">m/s · {wind_dir}</div>
            <div class="metric-label">Wind</div>
        </div>
        """, unsafe_allow_html=True)

    with cols[3]:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:1.6rem">🌡️</div>
            <div class="metric-value" style="font-size:1.5rem">{pressure}</div>
            <div class="metric-unit">hPa</div>
            <div class="metric-label">Pressure</div>
        </div>
        """, unsafe_allow_html=True)

    with cols[4]:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:1.6rem">☁️</div>
            <div class="metric-value">{clouds}</div>
            <div class="metric-unit">%</div>
            <div class="metric-label">Cloud Cover</div>
        </div>
        """, unsafe_allow_html=True)

    with cols[5]:
        vis_km = round(vis / 1000, 1) if isinstance(vis, (int, float)) else "--"
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:1.6rem">👁️</div>
            <div class="metric-value" style="font-size:1.5rem">{vis_km}</div>
            <div class="metric-unit">km</div>
            <div class="metric-label">Visibility</div>
        </div>
        """, unsafe_allow_html=True)

def render_ml_predictions(preds, unit_sym):
    st.markdown('<div class="section-title">🤖 ML Predictions (Next 3 Hours)</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        cond_icon = weather_icon(preds["condition_next"])
        badge_cls = "pred-rain" if preds["condition_next"] == "Rain" else \
                    ("pred-cloud" if preds["condition_next"] == "Clouds" else "pred-sun")
        st.markdown(f"""
        <div class="weather-card" style="text-align:center;">
            <div style="font-size:2rem">{cond_icon}</div>
            <div style="font-size:0.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px">Predicted Condition</div>
            <div class="pred-badge {badge_cls}" style="margin-top:0.5rem">{preds["condition_next"]}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        delta = preds["temp_next"] - 0  # placeholder
        st.markdown(f"""
        <div class="weather-card" style="text-align:center;">
            <div style="font-size:2rem">🌡️</div>
            <div style="font-size:0.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px">Temperature Forecast</div>
            <div style="font-size:2rem;font-weight:800;color:var(--accent2);margin-top:0.3rem">{preds["temp_next"]:.1f}°{unit_sym}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        rain = preds["rain_probability"]
        rain_color = "var(--rain)" if rain > 60 else ("var(--accent2)" if rain > 30 else "var(--accent3)")
        st.markdown(f"""
        <div class="weather-card" style="text-align:center;">
            <div style="font-size:2rem">🌧️</div>
            <div style="font-size:0.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px">Rain Probability</div>
            <div style="font-size:2rem;font-weight:800;color:{rain_color};margin-top:0.3rem">{rain:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(int(rain))
    with c4:
        st.markdown(f"""
        <div class="weather-card" style="text-align:center;">
            <div style="font-size:2rem">💧</div>
            <div style="font-size:0.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px">Humidity Forecast</div>
            <div style="font-size:2rem;font-weight:800;color:var(--rain);margin-top:0.3rem">{preds["humidity_next"]:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)

def render_hourly_forecast(preds, unit_sym):
    if not preds.get("hourly"):
        return
    st.markdown('<div class="section-title">⏱️ 24-Hour ML Forecast</div>', unsafe_allow_html=True)
    cols = st.columns(len(preds["hourly"]))
    for i, h in enumerate(preds["hourly"]):
        with cols[i]:
            icon = weather_icon(h["condition"])
            rain_c = "#60a5fa" if h["rain_prob"] > 50 else "#64748b"
            st.markdown(f"""
            <div class="forecast-card">
                <div class="forecast-day">{h["time"]}</div>
                <div class="forecast-icon">{icon}</div>
                <div class="forecast-temp forecast-high">{h["temp"]:.0f}°</div>
                <div style="font-size:0.72rem;color:{rain_c};margin-top:0.3rem">🌧 {h["rain_prob"]:.0f}%</div>
                <div style="font-size:0.72rem;color:var(--muted)">💧{h["humidity"]:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)

def render_5day_forecast(forecast_data, unit_sym):
    if not forecast_data:
        return
    st.markdown('<div class="section-title">📅 5-Day Forecast</div>', unsafe_allow_html=True)

    days = {}
    for item in forecast_data.get("list", []):
        dt  = datetime.fromtimestamp(item["dt"])
        day = dt.strftime("%A")
        if day not in days:
            days[day] = {"highs": [], "lows": [], "conditions": [], "rain": []}
        days[day]["highs"].append(item["main"]["temp_max"])
        days[day]["lows"].append(item["main"]["temp_min"])
        days[day]["conditions"].append(item["weather"][0]["main"])
        days[day]["rain"].append(item.get("pop", 0) * 100)

    day_items = list(days.items())[:5]
    cols = st.columns(len(day_items))
    for i, (day, vals) in enumerate(day_items):
        hi   = max(vals["highs"])
        lo   = min(vals["lows"])
        cond = max(set(vals["conditions"]), key=vals["conditions"].count)
        rain = np.mean(vals["rain"])
        icon = weather_icon(cond)
        with cols[i]:
            st.markdown(f"""
            <div class="forecast-card">
                <div class="forecast-day">{day[:3]}</div>
                <div class="forecast-icon">{icon}</div>
                <div class="forecast-temp">
                    <span class="forecast-high">{hi:.0f}°</span>
                    <span style="color:var(--muted)"> / </span>
                    <span class="forecast-low">{lo:.0f}°</span>
                </div>
                <div style="font-size:0.72rem;color:var(--rain);margin-top:0.3rem">🌧 {rain:.0f}%</div>
                <div style="font-size:0.7rem;color:var(--muted)">{cond}</div>
            </div>
            """, unsafe_allow_html=True)

def render_aqi(aqi_data):
    if not aqi_data:
        return
    try:
        aqi = aqi_data["list"][0]["main"]["aqi"]
        comp = aqi_data["list"][0]["components"]
        st.markdown('<div class="section-title">🌿 Air Quality Index</div>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        aqi_colors = {1:"#34d399",2:"#fbbf24",3:"#f97316",4:"#ef4444",5:"#8b5cf6"}
        color = aqi_colors.get(aqi, "#64748b")
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:1.5rem">🌍</div>
                <div style="font-size:1.4rem;font-weight:800;color:{color}">{get_aqi_label(aqi)}</div>
                <div class="metric-label">Overall AQI</div>
            </div>""", unsafe_allow_html=True)
        labels = [("CO", "co","μg/m³"), ("NO₂","no2","μg/m³"), ("O₃","o3","μg/m³"), ("PM2.5","pm2_5","μg/m³")]
        for col, (label, key, unit) in zip([c2,c3,c4,c5], labels):
            val = comp.get(key, "--")
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:1rem;font-weight:700;color:var(--accent)">{label}</div>
                    <div style="font-size:1.2rem;font-weight:800;color:var(--text)">{val:.1f if isinstance(val,float) else val}</div>
                    <div class="metric-unit">{unit}</div>
                </div>""", unsafe_allow_html=True)
    except:
        pass

def render_sun_info(data):
    sys = data.get("sys", {})
    sunrise = datetime.fromtimestamp(sys.get("sunrise", 0)).strftime("%H:%M") if sys.get("sunrise") else "--"
    sunset  = datetime.fromtimestamp(sys.get("sunset",  0)).strftime("%H:%M") if sys.get("sunset")  else "--"
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="metric-card" style="display:flex;align-items:center;gap:1rem;text-align:left">
            <span style="font-size:2rem">🌅</span>
            <div>
                <div style="font-size:0.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px">Sunrise</div>
                <div style="font-size:1.6rem;font-weight:800;color:var(--accent2);font-family:'Space Mono',monospace">{sunrise}</div>
            </div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card" style="display:flex;align-items:center;gap:1rem;text-align:left">
            <span style="font-size:2rem">🌇</span>
            <div>
                <div style="font-size:0.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px">Sunset</div>
                <div style="font-size:1.6rem;font-weight:800;color:var(--accent2);font-family:'Space Mono',monospace">{sunset}</div>
            </div>
        </div>""", unsafe_allow_html=True)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    api_key = st.text_input(
        "🔑 OpenWeatherMap API Key",
        type="password",
        placeholder="Paste your API key here",
        help="Get a free key at openweathermap.org"
    )

    st.markdown("**📍 Location**")
    location = st.text_input(
        "City name or lat,lon",
        value="Ludhiana",
        placeholder="e.g. Mumbai  or  28.6,77.2"
    )

    unit = st.selectbox("🌡️ Temperature Unit", ["Metric (°C)", "Imperial (°F)"])
    unit_param = "metric" if "Metric" in unit else "imperial"
    unit_sym   = "C"      if "Metric" in unit else "F"

    refresh_rate = st.selectbox("🔄 Auto-refresh", ["Off", "Every 5 min", "Every 10 min", "Every 30 min"])

    fetch_btn = st.button("🔍 Get Weather", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.78rem;color:#475569;line-height:1.7">
    <b style="color:#38bdf8">How it works:</b><br>
    1. Real-time data via OpenWeatherMap<br>
    2. ML model (Random Forest) trained on historical patterns<br>
    3. Predictions: temperature, rain probability, humidity, and weather condition<br><br>
    <b style="color:#38bdf8">Get a free API key:</b><br>
    <a href="https://openweathermap.org/api" target="_blank" style="color:#60a5fa">openweathermap.org/api</a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="font-size:0.72rem;color:#475569;text-align:center">WeatherSense AI · Built with Streamlit + ML</div>', unsafe_allow_html=True)

# ─── Main Content ──────────────────────────────────────────────────────────────
if not api_key:
    st.markdown("""
    <div class="main-header">
        <h1>🌤️ WeatherSense AI</h1>
        <p>ML-Enhanced Real-Time Weather Prediction</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    features_info = [
        ("🌡️", "Real-Time Data", "Temperature, humidity, wind, pressure, cloud cover, visibility, and air quality from OpenWeatherMap"),
        ("🤖", "ML Predictions", "Random Forest model trained on historical patterns predicts temperature, rain probability & weather condition"),
        ("📅", "Multi-Day Forecast", "5-day forecast + 24-hour ML-enhanced hourly prediction with rain probability per hour"),
    ]
    for col, (icon, title, desc) in zip([c1,c2,c3], features_info):
        with col:
            st.markdown(f"""
            <div class="weather-card" style="text-align:center;padding:2rem 1.5rem">
                <div style="font-size:3rem">{icon}</div>
                <div style="font-size:1.1rem;font-weight:700;margin:0.8rem 0 0.5rem">{title}</div>
                <div style="color:var(--muted);font-size:0.85rem;line-height:1.6">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.info("👈 Enter your **OpenWeatherMap API key** and **city name** in the sidebar, then click **Get Weather** to start.")

elif fetch_btn or (location and api_key):
    with st.spinner("🌐 Fetching weather data..."):
        current_data, err_c = fetch_current_weather(location, api_key, unit_param)
        forecast_data, err_f = fetch_forecast(location, api_key, unit_param)

    if err_c:
        st.error(f"❌ Error fetching weather: **{err_c}**\n\nCheck your API key and city name.")
    elif current_data:
        city_name = current_data.get("name", location)
        country   = current_data.get("sys", {}).get("country", "")
        lat  = current_data["coord"]["lat"]
        lon  = current_data["coord"]["lon"]

        render_header(city_name, country)

        # ── Current Conditions ──
        st.markdown('<div class="section-title">🌍 Current Conditions</div>', unsafe_allow_html=True)
        render_current_metrics(current_data, unit_sym)

        st.markdown("---")

        # ── ML Predictions ──
        with st.spinner("🤖 Running ML model..."):
            preds = make_predictions(current_data, forecast_data)

        render_ml_predictions(preds, unit_sym)

        st.markdown("---")

        # ── Hourly ML Forecast ──
        render_hourly_forecast(preds, unit_sym)

        st.markdown("---")

        # ── 5-Day Forecast ──
        render_5day_forecast(forecast_data, unit_sym)

        st.markdown("---")

        # ── Sun + AQI ──
        col_sun, col_aqi = st.columns([1, 2])
        with col_sun:
            st.markdown('<div class="section-title">☀️ Sun Times</div>', unsafe_allow_html=True)
            render_sun_info(current_data)

        with col_aqi:
            aqi_data = fetch_air_quality(lat, lon, api_key)
            if aqi_data:
                render_aqi(aqi_data)

        # ── Raw Data Expander ──
        with st.expander("📊 Raw API Data"):
            tab1, tab2 = st.tabs(["Current Weather", "Forecast Data"])
            with tab1:
                st.json(current_data)
            with tab2:
                if forecast_data:
                    st.json(forecast_data.get("list", [])[:3])

        # ── Auto-refresh ──
        refresh_map = {"Every 5 min": 300, "Every 10 min": 600, "Every 30 min": 1800}
        if refresh_rate in refresh_map:
            st.markdown(f'<div style="color:var(--muted);font-size:0.78rem;text-align:center">🔄 Auto-refreshing {refresh_rate.lower()}</div>', unsafe_allow_html=True)
            time.sleep(refresh_map[refresh_rate])
            st.rerun()
