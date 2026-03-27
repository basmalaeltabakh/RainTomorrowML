import streamlit as st
import pandas as pd
from src.predict import predict_single
from src.config import MODELS_DIR

st.set_page_config(page_title="Single Prediction", page_icon="🔮", layout="wide")
st.title("🔮 Single Day Prediction")

# ── Model selector ────────────────────────────────────────────
available_models = [
    p.stem for p in MODELS_DIR.glob("*.pkl")
    if p.stem not in ['location_map','oe_wind','oe_season','feature_cols','test_data']
]
model_name = st.sidebar.selectbox("Select Model", available_models, index=0)

st.sidebar.markdown("---")
st.sidebar.info("Fill in the weather details and click **Predict**.")

# ── Input form ────────────────────────────────────────────────
with st.form("prediction_form"):
    st.subheader("📍 Location & Date")
    c1, c2, c3 = st.columns(3)
    location = c1.selectbox("Location", [
        'Albury','BadgerysCreek','Cobar','CoffsHarbour','Moree',
        'Newcastle','NorahHead','NorfolkIsland','Penrith','Richmond',
        'Sydney','SydneyAirport','WaggaWagga','Williamtown','Wollongong',
        'Canberra','Tuggeranong','MountGinini','Ballarat','Bendigo',
        'Sale','MelbourneAirport','Melbourne','Mildura','Nhil',
        'Portland','Watsonia','Dartmoor','Brisbane','Cairns',
        'GoldCoast','Longreach','Mackay','MountGambier','Townsville',
        'Adelaide','MountGambier','Nuriootpa','Woomera','Albany',
        'Witchcliffe','PearceRAAF','PerthAirport','Perth','SalmonGums',
        'Walpole','Hobart','Launceston','AliceSprings','Darwin','Katherine','Uluru'
    ])
    month = c2.selectbox("Month", range(1, 13),
                         format_func=lambda x: ['Jan','Feb','Mar','Apr','May','Jun',
                                                 'Jul','Aug','Sep','Oct','Nov','Dec'][x-1])
    year = c3.number_input("Year", min_value=2008, max_value=2030, value=2024)

    st.subheader("🌡️ Temperature")
    c1, c2, c3, c4 = st.columns(4)
    min_temp  = c1.number_input("Min Temp (°C)",   value=12.0, step=0.1)
    max_temp  = c2.number_input("Max Temp (°C)",   value=24.0, step=0.1)
    temp_9am  = c3.number_input("Temp 9am (°C)",   value=17.0, step=0.1)
    temp_3pm  = c4.number_input("Temp 3pm (°C)",   value=22.0, step=0.1)

    st.subheader("💧 Humidity & Rainfall")
    c1, c2, c3 = st.columns(3)
    humidity_9am = c1.slider("Humidity 9am (%)", 0, 100, 70)
    humidity_3pm = c2.slider("Humidity 3pm (%)", 0, 100, 45)
    rainfall     = c3.number_input("Rainfall (mm)", value=0.0, step=0.1)

    st.subheader("💨 Wind")
    c1, c2, c3 = st.columns(3)
    WIND_DIRS = ['N','NNE','NE','ENE','E','ESE','SE','SSE',
                 'S','SSW','SW','WSW','W','WNW','NW','NNW']
    wind_gust_dir  = c1.selectbox("Wind Gust Dir",  WIND_DIRS, index=14)
    wind_dir_9am   = c2.selectbox("Wind Dir 9am",   WIND_DIRS, index=14)
    wind_dir_3pm   = c3.selectbox("Wind Dir 3pm",   WIND_DIRS, index=12)

    c1, c2, c3 = st.columns(3)
    wind_gust_speed = c1.number_input("Wind Gust Speed (km/h)", value=44.0, step=1.0)
    wind_speed_9am  = c2.number_input("Wind Speed 9am (km/h)",  value=20.0, step=1.0)
    wind_speed_3pm  = c3.number_input("Wind Speed 3pm (km/h)",  value=24.0, step=1.0)

    st.subheader("🌤️ Other")
    c1, c2, c3, c4 = st.columns(4)
    pressure_9am  = c1.number_input("Pressure 9am", value=1017.0, step=0.1)
    pressure_3pm  = c2.number_input("Pressure 3pm", value=1015.0, step=0.1)
    evaporation   = c3.number_input("Evaporation",  value=5.0,    step=0.1)
    sunshine      = c4.number_input("Sunshine (hr)", value=7.0,   step=0.1)

    c1, c2, c3 = st.columns(3)
    cloud_9am  = c1.slider("Cloud 9am (oktas)", 0, 9, 4)
    cloud_3pm  = c2.slider("Cloud 3pm (oktas)", 0, 9, 4)
    rain_today = c3.selectbox("Rain Today?", ["No", "Yes"])

    submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

# ── Result ────────────────────────────────────────────────────
if submitted:
    input_dict = {
        'Location': location, 'Month': month, 'Year': year,
        'MinTemp': min_temp, 'MaxTemp': max_temp,
        'Temp9am': temp_9am, 'Temp3pm': temp_3pm,
        'Humidity9am': humidity_9am, 'Humidity3pm': humidity_3pm,
        'Rainfall': rainfall, 'Evaporation': evaporation,
        'Sunshine': sunshine, 'WindGustDir': wind_gust_dir,
        'WindGustSpeed': wind_gust_speed, 'WindDir9am': wind_dir_9am,
        'WindDir3pm': wind_dir_3pm, 'WindSpeed9am': wind_speed_9am,
        'WindSpeed3pm': wind_speed_3pm, 'Pressure9am': pressure_9am,
        'Pressure3pm': pressure_3pm, 'Cloud9am': cloud_9am,
        'Cloud3pm': cloud_3pm,
        'RainToday': 1.0 if rain_today == 'Yes' else 0.0,
    }

    with st.spinner("Predicting..."):
        result = predict_single(input_dict, model_name)

    st.divider()
    if result['prediction'] == 1:
        st.error(f"## 🌧️ Rain Tomorrow — {result['confidence']}% confidence")
    else:
        st.success(f"## ☀️ No Rain Tomorrow — {result['confidence']}% confidence")

    c1, c2, c3 = st.columns(3)
    c1.metric("Prediction",  result['label'])
    c2.metric("Probability", f"{result['probability']:.4f}")
    c3.metric("Model Used",  model_name)