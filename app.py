# ─────────────────────────────────────────────
# 🔴 MUST BE AT TOP (fix yfinance chrome issue)
# ─────────────────────────────────────────────
import os
os.environ["YF_USE_CURL_CFFI"] = "0"

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from keras.layers import GRU
import yfinance as yf
import datetime
import keras, tensorflow as tf

# Optional: Groq chatbot
try:
    from groq import Groq
except ImportError:
    st.warning("⚠️ Groq package not found. Install via: `pip install groq`")

# ─────────────────────────────────────────────
# 🔧 GRU PATCH (old models compatibility)
# ─────────────────────────────────────────────
def patched_gru(*args, **kwargs):
    kwargs.pop("time_major", None)
    return GRU(*args, **kwargs)

# ─────────────────────────────────────────────
# STREAMLIT CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Stock Predictor + Chatbot", layout="wide")
st.title("📈 Stock Trend Prediction using GRU + 💬 Chatbot")

# ─────────────────────────────────────────────
# 🏢 MANUAL COMPANY → TICKER MAP
# ─────────────────────────────────────────────
COMPANY_MAP = {
    "Advanced Micro Devices Inc.": "AMD",
    "Apple Inc.": "AAPL",
    "Microsoft Corporation": "MSFT",
    "Amazon.com Inc.": "AMZN",
    "Alphabet Inc. (Google)": "GOOGL",
    "NVIDIA Corporation": "NVDA",
    "Meta Platforms Inc.": "META",
    "Netflix Inc.": "NFLX",
    "Adobe Inc.": "ADBE",
    "Intel Corporation": "INTC",
    "IBM Corporation": "IBM"
}

@st.cache_data
def get_available_companies():
    available = {}
    for company, ticker in COMPANY_MAP.items():
        if os.path.exists(f"{ticker}_gru_model.h5"):
            available[company] = ticker
    return available

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.header("🧭 Navigation")
show_chatbot = st.sidebar.checkbox("💬 Open Chatbot")

# ─────────────────────────────────────────────
# CHATBOT
# ─────────────────────────────────────────────
if show_chatbot:
    st.header("🤖 Chat with Groq LLM")

    api_key = st.text_input("🔑 Enter Groq API Key", type="password")
    model_name = st.selectbox(
        "🧠 Choose model",
        ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"]
    )

    if not api_key:
        st.stop()

    client = Groq(api_key=api_key)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": "You are a helpful AI assistant."}
        ]

    for msg in st.session_state.messages:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    user_msg = st.chat_input("Type your message...")

    if user_msg:
        st.session_state.messages.append({"role": "user", "content": user_msg})
        with st.chat_message("assistant"):
            response = client.chat.completions.create(
                model=model_name,
                messages=st.session_state.messages
            )
            reply = response.choices[0].message.content
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = [
            {"role": "system", "content": "You are a helpful AI assistant."}
        ]
        st.rerun()

    st.divider()

# ─────────────────────────────────────────────
# STOCK SELECTION
# ─────────────────────────────────────────────
companies = get_available_companies()

if not companies:
    st.error("❌ No GRU models found in directory.")
    st.stop()

selected_company = st.selectbox("🏢 Select Company", sorted(companies.keys()))
ticker = companies[selected_company]

st.caption(f"📌 Ticker: {ticker}")
st.caption("📡 Data source: Yahoo Finance (safe mode)")

# ─────────────────────────────────────────────
# DATA LOADING (FIXED)
# ─────────────────────────────────────────────
@st.cache_data
def load_data(ticker):
    df = yf.download(
        ticker,
        start="2010-01-01",
        end="2024-12-31",
        progress=False,
        auto_adjust=False
    )

    if df.empty:
        return pd.DataFrame()

    df.reset_index(inplace=True)
    df.set_index("Date", inplace=True)
    return df

df = load_data(ticker)

if df.empty:
    st.error(f"❌ Failed to fetch stock data for {ticker}.")
    st.stop()

if "Close" not in df.columns:
    st.error("❌ Closing price not available.")
    st.stop()

# ─────────────────────────────────────────────
# DATA PREPARATION
# ─────────────────────────────────────────────
close_prices = df["Close"].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

X, y = [], []
for i in range(100, len(scaled_data)):
    X.append(scaled_data[i - 100:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = X.reshape(X.shape[0], X.shape[1], 1)

split = int(0.7 * len(X))
X_test = X[split:]
y_test = y[split:]

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
model_path = f"{ticker}_gru_model.h5"
st.caption(f"🧠 Keras {keras.__version__} | TensorFlow {tf.__version__}")

model = load_model(model_path, custom_objects={"GRU": patched_gru})

y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# ─────────────────────────────────────────────
# VISUALS
# ─────────────────────────────────────────────
st.subheader("📊 Closing Price History")
fig = plt.figure(figsize=(14, 6))
plt.plot(df["Close"])
st.pyplot(fig)

st.subheader("📈 Moving Averages")
fig = plt.figure(figsize=(14, 6))
plt.plot(df["Close"].rolling(100).mean(), label="100-day MA")
plt.plot(df["Close"].rolling(200).mean(), label="200-day MA")
plt.plot(df["Close"], alpha=0.5)
plt.legend()
st.pyplot(fig)

st.subheader("📉 Predicted vs Actual")
fig = plt.figure(figsize=(14, 6))
plt.plot(y_test, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.legend()
st.pyplot(fig)

# ─────────────────────────────────────────────
# CUSTOM PREDICTION
# ─────────────────────────────────────────────
st.subheader("🎯 Custom Prediction")

input_price = st.number_input(
    "Previous Closing Price",
    value=float(df["Close"].iloc[-1]),
    min_value=0.01
)

if st.button("Predict Next Day Price"):
    last_99 = scaled_data[-99:]
    new_val = scaler.transform([[input_price]])[0][0]
    seq = np.append(last_99, new_val).reshape(1, 100, 1)

    prediction = model.predict(seq)
    predicted_price = scaler.inverse_transform(prediction)[0][0]

    st.success(f"📈 Predicted Closing Price: **${predicted_price:.2f}**")
