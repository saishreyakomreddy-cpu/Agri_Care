from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from PIL import Image

from utils.advice import fertilizer_advice
from utils.auth import create_database


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "crop_model.pkl"
WEATHER_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

st.set_page_config(
    page_title="AgriVision Grow",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

create_database()

if not st.session_state.get("authenticated", False):
    st.title("🌱 AgriVision Grow")
    st.subheader("Welcome")
    st.write("Please log in or register to access all agriculture tools.")
    st.info("Use the Login or Register page from the sidebar to continue.")
    st.stop()

style_path = PROJECT_ROOT / "assets" / "style.css"
if style_path.exists():
    st.markdown(
        f"<style>{style_path.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )

user = st.session_state.get("user", {})
with st.sidebar:
    st.markdown("<div style='text-align:center'><div style='font-size:4rem'>🌱</div><h2>AgriVision Grow</h2><p>Smart Agriculture Platform</p></div>", unsafe_allow_html=True)
    st.divider()
    st.markdown(f"Signed in as **{user.get('name', 'User')}**")
    if st.button("Log out"):
        st.session_state.clear()
        st.rerun()

st.markdown(
    """
    <div class="floating-leaves"><div class="leaf">🍃</div><div class="leaf">🌿</div><div class="leaf">🍃</div><div class="leaf">🌱</div></div>
    <div class="hero">
        <div class="ai-badge">🌱 AI POWERED AGRICULTURE</div>
        <div class="hero-title">Smart Farming<br><span>Better Tomorrow</span></div>
        <div class="hero-subtitle">AI-driven insights to help farmers make better decisions, increase productivity and build a sustainable future.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='glass-card'><h2>🚜 Welcome to Smart Agriculture</h2><p>One dashboard for crop recommendations, disease detection, weather intelligence, fertilizer guidance, and irrigation planning.</p></div>", unsafe_allow_html=True)

st.markdown("<div class='section-title'>🌾 Smart Agriculture Tools</div>", unsafe_allow_html=True)
quick_columns = st.columns(5)
quick_tools = [
    ("🌾", "Crop Recommendation", "Find suitable crops from soil and climate inputs."),
    ("🍃", "Disease Detection", "Upload a leaf image for disease analysis."),
    ("🌤️", "Weather", "Check conditions for your farming location."),
    ("🧪", "Fertilizer", "Review NPK levels and fertilizer needs."),
    ("💧", "Irrigation", "Plan water use from temperature and rainfall."),
]
for column, (icon, title, description) in zip(quick_columns, quick_tools):
    with column:
        st.markdown(
            f"<div class='feature-card'><div class='feature-icon'>{icon}</div><div class='feature-title'>{title}</div><div class='feature-description'>{description}</div></div>",
            unsafe_allow_html=True,
        )
        if st.button(f"Open {title}", key=f"quick_{title}", use_container_width=True):
            st.session_state["selected_tool"] = title

if st.session_state.get("selected_tool"):
    st.info(f"{st.session_state['selected_tool']} selected. Use its tab below to get started.")

crop_tab, disease_tab, weather_tab, fertilizer_tab, irrigation_tab = st.tabs([
    "🌾 Crop Recommendation",
    "🍃 Disease Detection",
    "🌤️ Weather",
    "🧪 Fertilizer",
    "💧 Irrigation",
])

with crop_tab:
    st.header("🌾 Crop Recommendation")
    st.write("Enter soil and environmental conditions to get an AI-based recommendation.")
    if not MODEL_PATH.exists():
        st.error("The crop model has not been trained yet.")
    else:
        model = joblib.load(MODEL_PATH)
        input_col1, input_col2 = st.columns(2)
        with input_col1:
            nitrogen = st.number_input("Nitrogen (N)", 0.0, 200.0, 50.0, key="crop_n")
            phosphorus = st.number_input("Phosphorus (P)", 0.0, 200.0, 50.0, key="crop_p")
            potassium = st.number_input("Potassium (K)", 0.0, 200.0, 50.0, key="crop_k")
            temperature = st.number_input("Temperature (°C)", -10.0, 60.0, 25.0, key="crop_temp")
        with input_col2:
            humidity = st.number_input("Humidity (%)", 0.0, 100.0, 70.0, key="crop_humidity")
            ph = st.number_input("Soil pH", 0.0, 14.0, 6.5, key="crop_ph")
            rainfall = st.number_input("Rainfall (mm)", 0.0, 1000.0, 100.0, key="crop_rainfall")

        nutrient_data = pd.DataFrame({"Nutrient": ["Nitrogen", "Phosphorus", "Potassium"], "Value": [nitrogen, phosphorus, potassium]})
        st.plotly_chart(px.bar(nutrient_data, x="Nutrient", y="Value", title="Soil Nutrient Levels"), use_container_width=True)

        if st.button("🌱 Recommend Crop", type="primary"):
            input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
            prediction = model.predict(input_data)
            crop = prediction[0]
            confidence = np.max(model.predict_proba(input_data)) * 100
            st.success(f"Recommended Crop: **{str(crop).upper()}**")
            st.metric("Prediction Confidence", f"{confidence:.2f}%")
            st.info(f"Based on the supplied conditions, **{crop}** is the predicted suitable crop.")
            st.subheader("Fertilizer advice")
            for message in fertilizer_advice(nitrogen, phosphorus, potassium):
                st.write(f"- {message}")

with disease_tab:
    st.header("🍃 Plant Disease Detection")
    st.write("Upload a plant leaf image to analyze it for possible disease.")
    uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"], key="disease_upload")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Leaf", use_container_width=True)
        st.success("Image uploaded successfully.")
        st.warning("Disease prediction model will be connected after the disease dataset is trained.")

@st.cache_data(ttl=900)
def find_location(city: str) -> dict | None:
    response = requests.get(WEATHER_GEOCODING_URL, params={"name": city, "count": 1, "language": "en", "format": "json"}, timeout=10)
    response.raise_for_status()
    results = response.json().get("results", [])
    return results[0] if results else None


@st.cache_data(ttl=900)
def fetch_weather(latitude: float, longitude: float) -> dict:
    response = requests.get(
        WEATHER_FORECAST_URL,
        params={"latitude": latitude, "longitude": longitude, "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code", "timezone": "auto"},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def weather_condition(code: int) -> str:
    return {0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast", 45: "Foggy", 51: "Light drizzle", 61: "Light rain", 63: "Rainy", 65: "Heavy rain", 80: "Rain showers", 95: "Thunderstorm"}.get(code, "Mixed conditions")


def irrigation_advice(temperature: float, humidity: float, rainfall: float) -> str:
    if rainfall > 150:
        return "Low irrigation required."
    if temperature > 35 and humidity < 50:
        return "High irrigation requirement."
    if temperature > 30:
        return "Moderate irrigation required."
    return "Normal irrigation schedule."

with weather_tab:
    st.header("🌤️ Weather Information")
    city = st.text_input("Enter your city", placeholder="Example: Hyderabad", key="weather_city")
    if st.button("Get Weather", type="primary"):
        if not city.strip():
            st.error("Please enter a city.")
        else:
            try:
                location = find_location(city.strip())
                if location is None:
                    st.error("City not found. Try a nearby city.")
                else:
                    weather = fetch_weather(location["latitude"], location["longitude"])
                    current = weather["current"]
                    units = weather["current_units"]
                    metric1, metric2, metric3, metric4 = st.columns(4)
                    metric1.metric("Temperature", f"{current['temperature_2m']} {units['temperature_2m']}")
                    metric2.metric("Humidity", f"{current['relative_humidity_2m']} {units['relative_humidity_2m']}")
                    metric3.metric("Rainfall", f"{current['precipitation']} {units['precipitation']}")
                    metric4.metric("Wind", f"{current['wind_speed_10m']} {units['wind_speed_10m']}")
                    st.info(f"Weather condition: {weather_condition(current['weather_code'])}")
                    st.success(irrigation_advice(current['temperature_2m'], current['relative_humidity_2m'], current['precipitation']))
            except requests.RequestException:
                st.error("The weather API is unavailable right now. Please try again later.")

with fertilizer_tab:
    st.header("🧪 Fertilizer Advice")
    st.write("Review nutrient levels and identify fertilizer needs.")
    fertilizer_n = st.number_input("Nitrogen (N)", 0.0, 200.0, 50.0, key="fert_n")
    fertilizer_p = st.number_input("Phosphorus (P)", 0.0, 200.0, 50.0, key="fert_p")
    fertilizer_k = st.number_input("Potassium (K)", 0.0, 200.0, 50.0, key="fert_k")
    if st.button("Check fertilizer needs", type="primary"):
        for message in fertilizer_advice(fertilizer_n, fertilizer_p, fertilizer_k):
            st.info(message)

with irrigation_tab:
    st.header("💧 Irrigation Guidance")
    irrigation_temperature = st.number_input("Temperature (°C)", value=25.0, key="irrigation_temp")
    irrigation_humidity = st.number_input("Humidity (%)", 0.0, 100.0, 70.0, key="irrigation_humidity")
    irrigation_rainfall = st.number_input("Rainfall (mm)", 0.0, 1000.0, 100.0, key="irrigation_rainfall")
    if st.button("Get irrigation advice", type="primary"):
        st.info(irrigation_advice(irrigation_temperature, irrigation_humidity, irrigation_rainfall))

st.markdown("<div class='footer'>🌱 <b>AgriVision Grow</b><br>Smart Agriculture powered by Python, AI & Machine Learning<br><br>© 2026 AgriVision Grow</div>", unsafe_allow_html=True)
