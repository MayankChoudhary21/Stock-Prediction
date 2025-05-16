import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from keras.layers import GRU  # Required for loading GRU from .h5
import yfinance as yf
import datetime
import os
import keras, tensorflow as tf

# Optional: Only import if chatbot is used
try:
    from groq import Groq
except ImportError:
    st.warning("⚠️ Groq package not found. Install via: `pip install groq`")

st.set_page_config(page_title="Stock Predictor + Chatbot", layout="wide")
st.title('📈 Stock Trend Prediction using GRU + 💬 Chatbot')

# ───────────────────────────────────────────────────────────────
# 📍 SIDEBAR CONTROLS
# ───────────────────────────────────────────────────────────────
st.sidebar.header("🧭 Navigation")
show_chatbot = st.sidebar.checkbox("💬 Open Chatbot")

# ───────────────────────────────────────────────────────────────
# 📍 CHATBOT SECTION (Groq)
# ───────────────────────────────────────────────────────────────
if show_chatbot:
    st.header("🤖 Chat with Groq LLM")
    st.markdown("Ask anything using **Groq API** (LLaMA 3 / Mixtral)")

    api_key = st.text_input("🔑 Enter Groq API Key", type="password")
    model = st.selectbox("🧠 Choose model", ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"])

    if not api_key:
        st.warning("Please enter your Groq API key above.")
        st.stop()

    try:
        client = Groq(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize Groq client: {e}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": "You are a helpful AI assistant."}]

    for msg in st.session_state.messages:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    user_input = st.chat_input("Type your message...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=st.session_state.messages
                    )
                    reply = response.choices[0].message.content
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                except Exception as e:
                    st.error(f"Groq API Error: {e}")

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
        st.experimental_rerun()

    st.divider()

# ───────────────────────────────────────────────────────────────
# 📍 STOCK PREDICTION SECTION
# ───────────────────────────────────────────────────────────────
@st.cache_data
def get_sp500_tickers():
    df = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
    return sorted(df['Symbol'].tolist())

tickers = get_sp500_tickers()
user_input = st.selectbox("🔎 Select Stock Ticker", tickers)

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
            X.append(scaled_data[i - 100:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        split = int(0.7 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model_path = f"{user_input}_gru_model.h5"
        if os.path.exists(model_path):
            st.caption(f"🧠 Keras {keras.__version__} | TensorFlow {tf.__version__}")
            try:
                model = load_model(model_path, custom_objects={"GRU": GRU})
            except Exception as e:
                st.error(f"❌ Failed to load model: {e}")
                st.stop()

            y_predict = model.predict(X_test)
            y_predict = scaler.inverse_transform(y_predict.reshape(-1, 1))
            y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

            st.subheader(f"{user_input} Historical Data")
            st.write(df.describe())

            st.subheader('📊 Closing Price History')
            fig = plt.figure(figsize=(14, 6))
            plt.plot(df['Close'])
            plt.ylabel('Price')
            st.pyplot(fig)

            st.subheader('📈 Moving Averages (100 & 200 Days)')
            fig = plt.figure(figsize=(14, 6))
            ma100 = df['Close'].rolling(100).mean()
            ma200 = df['Close'].rolling(200).mean()
            plt.plot(ma100, 'r', label='100-day MA')
            plt.plot(ma200, 'g', label='200-day MA')
            plt.plot(df['Close'], alpha=0.5, label='Closing Price')
            plt.legend()
            st.pyplot(fig)

            st.subheader("📉 Predicted vs Actual Prices")
            fig = plt.figure(figsize=(14, 6))
            plt.plot(df.index[-len(y_test):], y_test, 'b', label='Actual')
            plt.plot(df.index[-len(y_predict):], y_predict, 'r', label='Predicted')
            plt.legend()
            st.pyplot(fig)

            st.subheader("🎯 Custom Prediction")
            input_date = st.date_input("Prediction Date", value=datetime.date.today())
            input_price = st.number_input("Previous Closing Price", 
                                          value=float(df['Close'].iloc[-1]),
                                          min_value=0.01)

            if st.button("Predict Next Day Price"):
                last_99 = scaled_data[-99:]
                new_input = scaler.transform([[input_price]])[0][0]
                seq = np.append(last_99, new_input).reshape(100, 1)
                seq = np.reshape(seq, (1, 100, 1))

                prediction = model.predict(seq)
                predicted_price = scaler.inverse_transform(prediction)[0][0]

                st.success(f"📅 Predicted Closing Price for {input_date.strftime('%Y-%m-%d')}: **${predicted_price:.2f}**")
                change = ((predicted_price - input_price) / input_price) * 100
                st.metric("Expected Change", f"{change:.2f}%", delta_color="inverse")

                st.subheader("🔍 Input Sequence")
                seq_dates = pd.date_range(end=input_date, periods=100, freq='D')
                seq_prices = scaler.inverse_transform(seq.reshape(100, 1))
                st.line_chart(pd.DataFrame(seq_prices, index=seq_dates, columns=['Price']))
        else:
            st.error(f"❌ Model file '{model_path}' not found! Train and save your GRU model for this ticker.")
