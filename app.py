import streamlit as st
import joblib
import pandas as pd
import numpy as np
from PIL import Image

# Set up the app layout and background
st.set_page_config(page_title="Weather Prediction App", page_icon="🌞", layout="centered")
st.markdown("<style>body {background-color: #f0f8ff;}</style>", unsafe_allow_html=True)

# Load and display an image (sun icon as background)
image = Image.open('sun_icon.jpg')
st.image(image, use_column_width=True)

# Load the saved model and preprocessing objects
model_data = joblib.load('aussie_rain.joblib')
model = model_data['model']
imputer = model_data['imputer']
scaler = model_data['scaler']
encoder = model_data['encoder']
numeric_cols = model_data['numeric_cols']
categorical_cols = model_data['categorical_cols']
encoded_cols = model_data['encoded_cols']

# Function to preprocess user input data
def preprocess_input(data):
    # Create a DataFrame from the input data
    input_df = pd.DataFrame([data])

    # Impute missing values
    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])

    # Scale numeric features
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Encode categorical features
    input_df[encoded_cols] = encoder.transform(input_df[categorical_cols]).toarray()

    # Combine numeric and encoded categorical features
    X_input = input_df[numeric_cols + encoded_cols]
    
    return X_input

# Function for prediction
def predict_weather(input_data):
    # Preprocess the data
    processed_input = preprocess_input(input_data)
    
    # Prediction
    prediction = model.predict(processed_input)[0]
    probability = model.predict_proba(processed_input)[0][list(model.classes_).index(prediction)]
    
    return prediction, probability

# Streamlit interface
st.title("🌦️ Will it Rain Tomorrow? 🌦️")
st.markdown("""
### Enter the Weather Data below and get the forecast!
Fill in the weather parameters and get the prediction whether it will rain tomorrow along with the probability.
""")

# Create a form for data input
with st.form('weather_form'):
    # User data input
    date = st.text_input('Date (e.g., 2021-06-19)', '2021-06-19')
    location = st.selectbox('Location', ['Launceston', 'Brisbane', 'Sydney', 'Melbourne'])
    min_temp = st.number_input('Minimum Temperature', value=23.2)
    max_temp = st.number_input('Maximum Temperature', value=33.2)
    rainfall = st.number_input('Rainfall Amount', value=10.2)
    evaporation = st.number_input('Evaporation', value=4.2)
    sunshine = st.number_input('Sunshine Hours', value=np.nan)
    wind_gust_dir = st.selectbox('Wind Gust Direction', ['NNW', 'NW', 'NNE', 'SSE', 'SW'])
    wind_gust_speed = st.number_input('Wind Gust Speed', value=52.0)
    wind_dir_9am = st.selectbox('Wind Direction at 9:00 AM', ['NW', 'NNE', 'NNW'])
    wind_dir_3pm = st.selectbox('Wind Direction at 3:00 PM', ['NW', 'NNE', 'NNW'])
    wind_speed_9am = st.number_input('Wind Speed at 9:00 AM', value=13.0)
    wind_speed_3pm = st.number_input('Wind Speed at 3:00 PM', value=20.0)
    humidity_9am = st.number_input('Humidity at 9:00 AM', value=89.0)
    humidity_3pm = st.number_input('Humidity at 3:00 PM', value=58.0)
    pressure_9am = st.number_input('Pressure at 9:00 AM', value=1004.8)
    pressure_3pm = st.number_input('Pressure at 3:00 PM', value=1001.5)
    cloud_9am = st.number_input('Cloud Cover at 9:00 AM', value=8.0)
    cloud_3pm = st.number_input('Cloud Cover at 3:00 PM', value=5.0)
    temp_9am = st.number_input('Temperature at 9:00 AM', value=25.7)
    temp_3pm = st.number_input('Temperature at 3:00 PM', value=33.0)
    rain_today = st.selectbox('Did it Rain Today?', ['Yes', 'No'])
    
    # Button for prediction
    submit_button = st.form_submit_button('Predict')

# If the user clicked the prediction button
if submit_button:
    # Create a dictionary with the input data
    new_input = {
        'Date': date,
        'Location': location,
        'MinTemp': min_temp,
        'MaxTemp': max_temp,
        'Rainfall': rainfall,
        'Evaporation': evaporation,
        'Sunshine': sunshine,
        'WindGustDir': wind_gust_dir,
        'WindGustSpeed': wind_gust_speed,
        'WindDir9am': wind_dir_9am,
        'WindDir3pm': wind_dir_3pm,
        'WindSpeed9am': wind_speed_9am,
        'WindSpeed3pm': wind_speed_3pm,
        'Humidity9am': humidity_9am,
        'Humidity3pm': humidity_3pm,
        'Pressure9am': pressure_9am,
        'Pressure3pm': pressure_3pm,
        'Cloud9am': cloud_9am,
        'Cloud3pm': cloud_3pm,
        'Temp9am': temp_9am,
        'Temp3pm': temp_3pm,
        'RainToday': rain_today
    }

    # Get the prediction and probability
    prediction, probability = predict_weather(new_input)

    # Display the result
    if prediction == 'Yes':
        st.success(f"🌧️ It will rain tomorrow with a probability of {probability * 100:.2f}%.")
    else:
        st.info(f"🌞 No rain tomorrow. The probability is {probability * 100:.2f}%.")

# Add footer design
st.markdown("""
<style>
footer {visibility: hidden;}
footer:after {
    content:'Weather Prediction App - Built with Streamlit 🌤️';
    visibility: visible;
    display: block;
    position: relative;
    padding: 5px;
    top: 2px;
}
</style>
""", unsafe_allow_html=True)
