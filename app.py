import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
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
    --bg:      #0a0e1a;
    --card:    #111827;
    --card2:   #1a2235;
    --accent:  #38bdf8;
    --accent2: #f59e0b;
    --accent3: #34d399;
    --text:    #e2e8f0;
    --muted:   #64748b;
    --border:  #1e2d45;
    --rain:    #60a5fa;
}

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
.stApp { background: var(--bg) !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1424 !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stTextInput > div > div > input {
    background: #1a2235 !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'Space Mono', monospace !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div:focus-within,
[data-testid="stSidebar"] .stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(56,189,248,0.15) !important;
}

/* Sidebar button — matches screenshot cyan pill */
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #38bdf8) !important;
    color: #0a0e1a !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.65rem 1.4rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    width: 100% !important;
    letter-spacing: 0.4px !important;
    transition: all 0.2s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(56,189,248,0.35) !important;
}

/* Main buttons */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #38bdf8) !important;
    color: #0a0e1a !important; font-weight: 700 !important;
    border: none !important; border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    transition: all 0.2s !important;
}

/* Cards */
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
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover { transform: translateY(-3px); border-color: var(--accent); }
.metric-value { font-size: 2rem; font-weight: 800; color: var(--accent); line-height: 1; }
.metric-label { font-size: 0.7rem; color: var(--muted); margin-top: 0.4rem; text-transform: uppercase; letter-spacing: 1px; }
.metric-unit  { font-size: 0.82rem; color: var(--muted); }

/* Forecast cards */
.forecast-card {
    background: var(--card2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 0.8rem;
    text-align: center;
}
.forecast-day  { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; }
.forecast-icon { font-size: 1.8rem; margin: 0.4rem 0; }
.forecast-high { color: var(--accent2); font-weight: 700; }
.forecast-low  { color: var(--rain); }

/* Section titles */
.section-title {
    font-size: 0.9rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--accent);
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}

/* Header */
.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    text-align: center;
}
.main-header h1 {
    font-size: 2.6rem; font-weight: 800; margin: 0;
    background: linear-gradient(135deg, #38bdf8, #f59e0b);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.main-header p { color: var(--muted); margin: 0.4rem 0 0; font-family: 'Space Mono', monospace; font-size: 0.82rem; }

/* Prediction badges */
.pred-badge { display: inline-block; padding: 0.3rem 0.9rem; border-radius: 20px; font-size: 0.82rem; font-weight: 700; }
.pred-rain  { background: rgba(96,165,250,0.15); color: var(--rain);    border: 1px solid rgba(96,165,250,0.3); }
.pred-sun   { background: rgba(245,158,11,0.15);  color: var(--accent2); border: 1px solid rgba(245,158,11,0.3); }
.pred-cloud { background: rgba(100,116,139,0.2);  color: #94a3b8;        border: 1px solid rgba(100,116,139,0.3); }

/* Sidebar label icons */
.sidebar-label {
    font-size: 0.85rem; font-weight: 700; color: var(--text);
    margin-bottom: 0.3rem; display: flex; align-items: center; gap: 0.4rem;
}

/* Progress */
.stProgress > div > div { background: linear-gradient(90deg, #38bdf8, #34d399) !important; border-radius: 4px; }

/* Hide branding */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ─────────────────────────────────────────────────────────────────
OW_BASE     = "https://api.openweathermap.org/data/2.5"
WIND_DIRS   = [(0,22.5,"N↑"),(22.5,67.5,"NE↗"),(67.5,112.5,"E→"),(112.5,157.5,"SE↘"),
               (157.5,202.5,"S↓"),(202.5,247.5,"SW↙"),(247.5,292.5,"W←"),(292.5,337.5,"NW↖"),(337.5,360,"N↑")]
WX_ICONS    = {"Clear":"☀️","Clouds":"☁️","Rain":"🌧️","Drizzle":"🌦️",
               "Thunderstorm":"⛈️","Snow":"❄️","Mist":"🌫️","Fog":"🌫️","Haze":"🌫️"}
AQI_LABELS  = {1:"Good 🟢",2:"Fair 🟡",3:"Moderate 🟠",4:"Poor 🔴",5:"Very Poor 🟣"}
AQI_COLORS  = {1:"#34d399",2:"#fbbf24",3:"#f97316",4:"#ef4444",5:"#8b5cf6"}

def wind_dir(deg):
    for lo, hi, label in WIND_DIRS:
        if lo <= deg < hi:
            return label
    return "N↑"

def wx_icon(main): return WX_ICONS.get(main, "🌤️")

# ─── API helpers ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def fetch_current(location, api_key, units):
    for param in [f"q={location}", *([ f"lat={location.split(',')[0].strip()}&lon={location.split(',')[1].strip()}" ] if "," in location else [])]:
        r = requests.get(f"{OW_BASE}/weather?{param}&appid={api_key}&units={units}", timeout=10)
        if r.status_code == 200:
            return r.json(), None
    return None, r.json().get("message","City not found")

@st.cache_data(ttl=600)
def fetch_forecast(location, api_key, units):
    for param in [f"q={location}", *([ f"lat={location.split(',')[0].strip()}&lon={location.split(',')[1].strip()}" ] if "," in location else [])]:
        r = requests.get(f"{OW_BASE}/forecast?{param}&appid={api_key}&units={units}&cnt=40", timeout=10)
        if r.status_code == 200:
            return r.json(), None
    return None, r.json().get("message","Forecast unavailable")

@st.cache_data(ttl=3600)
def fetch_aqi(lat, lon, api_key):
    try:
        r = requests.get(f"{OW_BASE}/air_pollution?lat={lat}&lon={lon}&appid={api_key}", timeout=10)
        if r.status_code == 200:
            return r.json()
    except: pass
    return None

# ─── ML ────────────────────────────────────────────────────────────────────────
def build_history(cur):
    np.random.seed(42)
    n   = 600
    t   = cur["main"]["temp"];    h = cur["main"]["humidity"]
    p   = cur["main"]["pressure"]; w = cur["wind"]["speed"]
    T   = np.random.normal(t, 6, n)
    H   = np.clip(np.random.normal(h, 12, n), 10, 100)
    P   = np.random.normal(p, 8, n)
    W   = np.clip(np.random.exponential(w+1, n), 0, 50)
    C   = np.random.randint(0, 101, n)
    Hr  = np.random.randint(0, 24, n)
    Mo  = np.random.randint(1, 13, n)
    rain_p = (H>75)*0.5 + (C>70)*0.3 + (P<1005)*0.2
    rain   = (rain_p + np.random.normal(0,.15,n) > 0.45).astype(int)
    cond   = ["Rain" if r else ("Clouds" if c>70 else "Clear") for r,c in zip(rain,C)]
    return pd.DataFrame({"temp":T,"humidity":H,"pressure":P,"wind":W,"clouds":C,
                         "hour":Hr,"month":Mo,"t_next":T+np.random.normal(0,3,n),
                         "h_next":np.clip(H+np.random.normal(0,5,n),10,100),
                         "rain":rain,"cond":cond})

@st.cache_resource
def train(cur):
    df = build_history(cur)
    F  = ["temp","humidity","pressure","wind","clouds","hour","month"]
    X  = df[F]
    le = LabelEncoder()
    yc = le.fit_transform(df["cond"])
    rt = RandomForestRegressor(120,max_depth=8,random_state=42).fit(X,df["t_next"])
    rh = RandomForestRegressor(80, max_depth=6,random_state=42).fit(X,df["h_next"])
    rr = RandomForestClassifier(120,max_depth=8,random_state=42).fit(X,df["rain"])
    rc = RandomForestClassifier(100,max_depth=8,random_state=42).fit(X,yc)
    return rt, rh, rr, rc, le, F

def predict(cur, fcast):
    rt, rh, rr, rc, le, F = train(cur)
    m = cur["main"]; w = cur["wind"]; now = datetime.now()
    X0 = pd.DataFrame([[m["temp"],m["humidity"],m["pressure"],w["speed"],
                        cur["clouds"]["all"],now.hour,now.month]], columns=F)
    hourly = []
    for item in (fcast or {}).get("list",[])[:8]:
        fi = item["main"]; fw = item["wind"]; fc = item["clouds"]["all"]
        dt = datetime.fromtimestamp(item["dt"])
        Xi = pd.DataFrame([[fi["temp"],fi["humidity"],fi["pressure"],
                            fw["speed"],fc,dt.hour,dt.month]], columns=F)
        hourly.append({
            "time": dt.strftime("%I %p"),
            "temp": rt.predict(Xi)[0],
            "humidity": rh.predict(Xi)[0],
            "rain_prob": rr.predict_proba(Xi)[0][1]*100,
            "condition": le.inverse_transform([rc.predict(Xi)[0]])[0]
        })
    return {
        "temp":      rt.predict(X0)[0],
        "humidity":  rh.predict(X0)[0],
        "rain_prob": rr.predict_proba(X0)[0][1]*100,
        "condition": le.inverse_transform([rc.predict(X0)[0]])[0],
        "hourly":    hourly
    }

# ─── Render helpers ────────────────────────────────────────────────────────────
def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

def render_current(data, usym):
    m = data["main"]; w = data["wind"]; cond = data["weather"][0]
    vis   = round(data.get("visibility",10000)/1000,1)
    cols  = st.columns([2,1,1,1,1,1])
    with cols[0]:
        st.markdown(f"""
        <div class="weather-card" style="text-align:center">
            <div style="font-size:5rem;line-height:1">{wx_icon(cond["main"])}</div>
            <div style="font-size:3.2rem;font-weight:800;color:var(--accent);line-height:1.1">{m["temp"]:.1f}°{usym}</div>
            <div style="color:var(--muted);margin-top:.3rem;font-family:'Space Mono',monospace;font-size:.85rem">{cond["description"].title()}</div>
            <div style="color:var(--muted);font-size:.78rem;margin-top:.3rem">Feels like {m["feels_like"]:.1f}°{usym}</div>
        </div>""", unsafe_allow_html=True)
    metrics = [
        ("💧","Humidity",  m["humidity"],     "%"),
        ("💨","Wind",      f'{w["speed"]} m/s', wind_dir(w.get("deg",0))),
        ("🌡️","Pressure",  m["pressure"],     "hPa"),
        ("☁️","Cloud Cover",data["clouds"]["all"],"%"),
        ("👁️","Visibility",vis,               "km"),
    ]
    for col,(icon,label,val,unit) in zip(cols[1:],metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:1.6rem">{icon}</div>
                <div class="metric-value" style="font-size:1.5rem">{val}</div>
                <div class="metric-unit">{unit}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

def render_ml(p, usym):
    section("🤖 ML Predictions — Next 3 Hours")
    c1,c2,c3,c4 = st.columns(4)
    badge = "pred-rain" if p["condition"]=="Rain" else ("pred-cloud" if p["condition"]=="Clouds" else "pred-sun")
    for col, html in zip([c1,c2,c3,c4],[
        f"""<div class="weather-card" style="text-align:center">
            <div style="font-size:2rem">{wx_icon(p["condition"])}</div>
            <div style="font-size:.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px">Predicted Condition</div>
            <div class="pred-badge {badge}" style="margin-top:.5rem">{p["condition"]}</div></div>""",
        f"""<div class="weather-card" style="text-align:center">
            <div style="font-size:2rem">🌡️</div>
            <div style="font-size:.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px">Temperature Forecast</div>
            <div style="font-size:2rem;font-weight:800;color:var(--accent2);margin-top:.3rem">{p["temp"]:.1f}°{usym}</div></div>""",
        f"""<div class="weather-card" style="text-align:center">
            <div style="font-size:2rem">🌧️</div>
            <div style="font-size:.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px">Rain Probability</div>
            <div style="font-size:2rem;font-weight:800;color:{'#60a5fa' if p['rain_prob']>60 else ('#f59e0b' if p['rain_prob']>30 else '#34d399')};margin-top:.3rem">{p["rain_prob"]:.0f}%</div></div>""",
        f"""<div class="weather-card" style="text-align:center">
            <div style="font-size:2rem">💧</div>
            <div style="font-size:.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px">Humidity Forecast</div>
            <div style="font-size:2rem;font-weight:800;color:var(--rain);margin-top:.3rem">{p["humidity"]:.0f}%</div></div>"""
    ]):
        with col: st.markdown(html, unsafe_allow_html=True)
    st.progress(int(p["rain_prob"]))

def render_hourly(hourly):
    if not hourly: return
    section("⏱️ 24-Hour ML Forecast")
    cols = st.columns(len(hourly))
    for col, h in zip(cols, hourly):
        rc = "#60a5fa" if h["rain_prob"]>50 else "#64748b"
        with col:
            st.markdown(f"""
            <div class="forecast-card">
                <div class="forecast-day">{h["time"]}</div>
                <div class="forecast-icon">{wx_icon(h["condition"])}</div>
                <div class="forecast-temp forecast-high">{h["temp"]:.0f}°</div>
                <div style="font-size:.72rem;color:{rc};margin-top:.3rem">🌧 {h["rain_prob"]:.0f}%</div>
                <div style="font-size:.72rem;color:var(--muted)">💧{h["humidity"]:.0f}%</div>
            </div>""", unsafe_allow_html=True)

def render_5day(fcast, usym):
    if not fcast: return
    section("📅 5-Day Forecast")
    days = {}
    for item in fcast.get("list",[]):
        day = datetime.fromtimestamp(item["dt"]).strftime("%A")
        if day not in days: days[day] = {"hi":[],"lo":[],"cond":[],"rain":[]}
        days[day]["hi"].append(item["main"]["temp_max"])
        days[day]["lo"].append(item["main"]["temp_min"])
        days[day]["cond"].append(item["weather"][0]["main"])
        days[day]["rain"].append(item.get("pop",0)*100)
    items = list(days.items())[:5]
    cols  = st.columns(len(items))
    for col,(day,v) in zip(cols,items):
        cond = max(set(v["cond"]),key=v["cond"].count)
        with col:
            st.markdown(f"""
            <div class="forecast-card">
                <div class="forecast-day">{day[:3]}</div>
                <div class="forecast-icon">{wx_icon(cond)}</div>
                <div class="forecast-temp">
                    <span class="forecast-high">{max(v["hi"]):.0f}°</span>
                    <span style="color:var(--muted)"> / </span>
                    <span class="forecast-low">{min(v["lo"]):.0f}°</span>
                </div>
                <div style="font-size:.72rem;color:#60a5fa;margin-top:.3rem">🌧 {np.mean(v["rain"]):.0f}%</div>
                <div style="font-size:.7rem;color:var(--muted)">{cond}</div>
            </div>""", unsafe_allow_html=True)

def render_aqi(aqi_data):
    if not aqi_data: return
    try:
        aqi  = aqi_data["list"][0]["main"]["aqi"]
        comp = aqi_data["list"][0]["components"]
        section("🌿 Air Quality Index")
        c0,c1,c2,c3,c4 = st.columns(5)
        color = AQI_COLORS.get(aqi,"#64748b")
        with c0:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:1.5rem">🌍</div>
                <div style="font-size:1.1rem;font-weight:800;color:{color}">{AQI_LABELS.get(aqi,"Unknown")}</div>
                <div class="metric-label">Overall AQI</div>
            </div>""", unsafe_allow_html=True)
        for col,(label,key) in zip([c1,c2,c3,c4],[("CO","co"),("NO₂","no2"),("O₃","o3"),("PM2.5","pm2_5")]):
            val = comp.get(key,"--")
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:1rem;font-weight:700;color:var(--accent)">{label}</div>
                    <div style="font-size:1.2rem;font-weight:800">{val:.1f if isinstance(val,float) else val}</div>
                    <div class="metric-unit">μg/m³</div>
                </div>""", unsafe_allow_html=True)
    except: pass

def render_sun(data):
    sys = data.get("sys",{})
    sr  = datetime.fromtimestamp(sys["sunrise"]).strftime("%H:%M") if sys.get("sunrise") else "--"
    ss  = datetime.fromtimestamp(sys["sunset"]).strftime("%H:%M")  if sys.get("sunset")  else "--"
    c1,c2 = st.columns(2)
    for col,icon,label,val in [(c1,"🌅","Sunrise",sr),(c2,"🌇","Sunset",ss)]:
        with col:
            st.markdown(f"""
            <div class="metric-card" style="display:flex;align-items:center;gap:1rem;text-align:left">
                <span style="font-size:2rem">{icon}</span>
                <div>
                    <div style="font-size:.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px">{label}</div>
                    <div style="font-size:1.6rem;font-weight:800;color:var(--accent2);font-family:'Space Mono',monospace">{val}</div>
                </div>
            </div>""", unsafe_allow_html=True)

# ─── Sidebar — only 3 controls ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:.5rem 0 1.2rem">
        <div style="display:flex;align-items:center;gap:.6rem;margin-bottom:1.4rem">
            <span style="font-size:1.5rem">📍</span>
            <span style="font-size:1.2rem;font-weight:800;color:#e2e8f0">Location</span>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<p style="font-size:.82rem;color:#94a3b8;margin-bottom:.3rem">City name or lat,lon</p>', unsafe_allow_html=True)
    location = st.text_input("_loc", value="Ludhiana", label_visibility="collapsed",
                             placeholder="e.g. Mumbai  or  28.6,77.2")

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown("""<div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.4rem">
        <span style="font-size:1.1rem">🌡️</span>
        <span style="font-size:.88rem;font-weight:700;color:#e2e8f0">Temperature Unit</span>
    </div>""", unsafe_allow_html=True)
    unit = st.selectbox("_unit", ["Metric (°C)", "Imperial (°F)"], label_visibility="collapsed")

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown("""<div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.4rem">
        <span style="font-size:1.1rem">🔄</span>
        <span style="font-size:.88rem;font-weight:700;color:#e2e8f0">Auto-refresh</span>
    </div>""", unsafe_allow_html=True)
    refresh = st.selectbox("_refresh", ["Off","Every 5 min","Every 10 min","Every 30 min"],
                           label_visibility="collapsed")

    st.markdown('<br>', unsafe_allow_html=True)
    fetch_btn = st.button("🔍  Get Weather")

unit_param = "metric"   if "Metric"   in unit else "imperial"
unit_sym   = "C"        if "Metric"   in unit else "F"

# ─── Load API key from Streamlit secrets ───────────────────────────────────────
try:
    API_KEY = st.secrets["OPENWEATHER_API_KEY"]
except Exception:
    st.error("⚠️  API key not found.  Go to **App settings → Secrets** and add:\n\n```\nOPENWEATHER_API_KEY = \"your_key_here\"\n```")
    st.stop()

# ─── Welcome screen ────────────────────────────────────────────────────────────
if not fetch_btn:
    st.markdown("""
    <div class="main-header">
        <h1>🌤️ WeatherSense AI</h1>
        <p>ML-Enhanced Real-Time Weather Prediction &nbsp;·&nbsp; Enter a city and click Get Weather</p>
    </div>""", unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    for col,(icon,title,desc) in zip([c1,c2,c3],[
        ("🌡️","Real-Time Data","Temperature, humidity, wind, pressure, cloud cover, visibility & air quality"),
        ("🤖","ML Predictions","Random Forest predicts temperature, rain probability & weather condition for next 3h"),
        ("📅","Multi-Day Forecast","5-day forecast + 24-hour ML-enhanced hourly prediction with rain probability"),
    ]):
        with col:
            st.markdown(f"""
            <div class="weather-card" style="text-align:center;padding:2rem 1.5rem">
                <div style="font-size:3rem">{icon}</div>
                <div style="font-size:1.1rem;font-weight:700;margin:.8rem 0 .5rem">{title}</div>
                <div style="color:var(--muted);font-size:.85rem;line-height:1.6">{desc}</div>
            </div>""", unsafe_allow_html=True)
    st.stop()

# ─── Main fetch + render ───────────────────────────────────────────────────────
with st.spinner("🌐 Fetching weather data…"):
    cur_data, err_c = fetch_current(location, API_KEY, unit_param)
    fca_data, _     = fetch_forecast(location, API_KEY, unit_param)

if err_c or not cur_data:
    st.error(f"❌ {err_c or 'Could not fetch data.'} — check the city name and try again.")
    st.stop()

city    = cur_data.get("name", location)
country = cur_data.get("sys",{}).get("country","")
lat     = cur_data["coord"]["lat"]
lon     = cur_data["coord"]["lon"]

# Header
st.markdown(f"""
<div class="main-header">
    <h1>🌤️ WeatherSense AI</h1>
    <p>📍 {city}, {country} &nbsp;·&nbsp; ML-Enhanced Forecast &nbsp;·&nbsp; {datetime.now().strftime("%A, %d %B %Y  %H:%M")}</p>
</div>""", unsafe_allow_html=True)

section("🌍 Current Conditions")
render_current(cur_data, unit_sym)

st.markdown("---")

with st.spinner("🤖 Running ML model…"):
    preds = predict(cur_data, fca_data)

render_ml(preds, unit_sym)
st.markdown("---")
render_hourly(preds["hourly"])
st.markdown("---")
render_5day(fca_data, unit_sym)
st.markdown("---")

col_sun, col_aqi = st.columns([1,2])
with col_sun:
    section("☀️ Sun Times")
    render_sun(cur_data)
with col_aqi:
    aqi_data = fetch_aqi(lat, lon, API_KEY)
    if aqi_data:
        render_aqi(aqi_data)

with st.expander("📊 Raw API Response"):
    t1, t2 = st.tabs(["Current", "Forecast"])
    with t1: st.json(cur_data)
    with t2:
        if fca_data: st.json(fca_data.get("list",[])[:3])

# Auto-refresh
refresh_secs = {"Every 5 min":300,"Every 10 min":600,"Every 30 min":1800}
if refresh in refresh_secs:
    import time
    time.sleep(refresh_secs[refresh])
    st.rerun()
