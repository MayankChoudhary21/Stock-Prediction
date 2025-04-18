import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import datetime
import os

# Set TensorFlow backend for Keras
os.environ["KERAS_BACKEND"] = "tensorflow"
from tensorflow.keras.models import load_model  # Critical change

st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title('📈 Stock Trend Prediction using GRU')

@st.cache_data
def get_sp500_tickers():
    df = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
    return sorted(df['Symbol'].tolist())

tickers = get_sp500_tickers()
user_input = st.selectbox("Select Stock Ticker", tickers)

@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, start="2010-01-01", end="2024-12-31")
    df.reset_index(inplace=True)
    df.set_index("Date", inplace=True)
    return df

if user_input:
    df = load_data(user_input)
    if not df.empty and 'Close' in df.columns:
        close_prices = df['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)

        X, y = [], []
        for i in range(100, len(scaled_data)):
            X.append(scaled_data[i-100:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        split = int(0.7 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model_path = f"{user_input}_gru_model.h5"
        if os.path.exists(model_path):
            try:
                # Load model with custom object scope
                model = load_model(model_path, compile=False)
                
                # Test model with dummy input
                test_input = np.random.rand(1, 100, 1)
                _ = model.predict(test_input)
                
                # Rest of your processing...
                y_predict = model.predict(X_test)
                y_predict = scaler.inverse_transform(y_predict.reshape(-1, 1))
                y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

                # ... [rest of your visualization code remains the same] ...

            except Exception as e:
                st.error(f"Model loading failed: {str(e)}")
                st.markdown("""
                **Troubleshooting Tips:**
                1. Ensure you're using TensorFlow 2.15.0 or later
                2. Check if the model file is corrupted
                3. Verify the model was originally saved with TensorFlow Keras
                """)
        else:
            st.error(f"❌ Model file '{model_path}' not found! Ensure the .h5 file exists.")
