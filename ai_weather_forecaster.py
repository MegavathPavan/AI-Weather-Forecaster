import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
import requests
import json
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Ensure NLTK data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# Set NLTK data path
nltk.data.path.append(os.path.join(os.getcwd(), 'nltk_data'))

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load('en_core_web_sm')

app = FastAPI()

# Generate synthetic weather data
def generate_synthetic_weather_data(days=365):
    date_range = pd.date_range(start=datetime.now(), periods=days)
    temperatures = np.sin(np.arange(days) * 2 * np.pi / 365) * 10 + 20
    precipitation = np.random.exponential(scale=2, size=days)
    
    df = pd.DataFrame({
        'date': date_range,
        'temperature': temperatures,
        'precipitation': precipitation
    })
    df.set_index('date', inplace=True)
    return df

# Generate synthetic data
df = generate_synthetic_weather_data()

# Prepare data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 30
X, y = create_sequences(scaled_data, seq_length)

# Build and train LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(seq_length, 2)),
    Dense(2)
])
model.compile(optimizer=Adam(), loss='mse')
model.fit(X, y, epochs=50, batch_size=32, verbose=0)

# Initialize NLTK SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

def generate_summary(forecast_text):
    return f"Summary: {forecast_text[:100]}..."  # Simple summary, just first 100 characters

class WeatherRequest(BaseModel):
    days: int

@app.post("/forecast")
async def get_weather_forecast(request: WeatherRequest):
    try:
        last_sequence = scaled_data[-seq_length:]
        forecast = []
        
        for _ in range(request.days):
            next_pred = model.predict(last_sequence.reshape(1, seq_length, 2))
            forecast.append(next_pred[0])
            last_sequence = np.roll(last_sequence, -1, axis=0)
            last_sequence[-1] = next_pred
        
        forecast = scaler.inverse_transform(np.array(forecast))
        
        current_date = datetime.now()
        forecast_df = pd.DataFrame({
            "Date": [(current_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(request.days)],
            "Temperature": forecast[:, 0],
            "Precipitation": forecast[:, 1]
        })
        
        forecast_text = f"The weather forecast for the next {request.days} days: " + ", ".join([f"Day {i+1}: {temp:.2f}°C, {precip:.2f}mm precipitation" for i, (temp, precip) in enumerate(forecast)])
        
        try:
            tokens = word_tokenize(forecast_text)
        except LookupError:
            tokens = forecast_text.split()  # Fallback to simple splitting if NLTK fails
        
        doc = nlp(forecast_text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        sentiment_scores = sia.polarity_scores(forecast_text)
        sentiment = "Positive" if sentiment_scores['compound'] > 0 else "Negative"
        
        summary = generate_summary(forecast_text)
        
        return {
            "forecast": forecast_df.to_dict(orient="records"),
            "nlp_analysis": {
                "tokens": tokens,
                "entities": entities,
                "sentiment": sentiment,
                "summary": summary
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Streamlit UI
FASTAPI_URL = "http://localhost:8000/forecast"

def main():
    st.set_page_config(page_title="AI Weather Forecaster", page_icon="🌦️", layout="wide")
    
    st.title("🌦️ AI Weather Forecaster")
    st.write("Welcome to the AI-powered Weather Forecasting tool!")

    days = st.slider("Select number of days for forecast", 1, 30, 7)
    
    if st.button("Get Forecast"):
        try:
            response = requests.post(FASTAPI_URL, json={"days": days})
            if response.status_code == 200:
                result = response.json()
                display_forecast(result, days)
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error: Could not connect to FastAPI server. Is it running? Details: {e}")

def display_forecast(result, days):
    forecast = result['forecast']
    nlp_analysis = result['nlp_analysis']

    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Weather Forecast")
        
        # Create a line chart for temperature and precipitation
        fig = go.Figure()
        dates = [datetime.strptime(f['Date'], '%Y-%m-%d') for f in forecast]
        
        fig.add_trace(go.Scatter(x=dates, y=[f['Temperature'] for f in forecast], name="Temperature (°C)"))
        fig.add_trace(go.Scatter(x=dates, y=[f['Precipitation'] for f in forecast], name="Precipitation (mm)", yaxis="y2"))
        
        fig.update_layout(
            title="Temperature and Precipitation Forecast",
            xaxis_title="Date",
            yaxis_title="Temperature (°C)",
            yaxis2=dict(title="Precipitation (mm)", overlaying="y", side="right")
        )
        
        st.plotly_chart(fig)

        # Display forecast data in a table
        st.write("Detailed Forecast:")
        forecast_df = pd.DataFrame(forecast)
        forecast_df = forecast_df.set_index('Date')
        forecast_df = forecast_df.round(2)
        st.dataframe(forecast_df)

    with col2:
        st.subheader("🧠 NLP Analysis")
        
        st.write(f"**Sentiment:** {nlp_analysis['sentiment']}")
        st.write(f"**Summary:** {nlp_analysis['summary']}")
        
        st.write("**Named Entities:**")
        entities_df = pd.DataFrame(nlp_analysis['entities'], columns=['Entity', 'Type'])
        st.dataframe(entities_df)

        st.write("**Tokens:**")
        st.write(", ".join(nlp_analysis['tokens']))

if __name__ == "__main__":
    main()
