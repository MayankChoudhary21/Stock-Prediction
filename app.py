import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from keras.layers import GRU
import keras, tensorflow as tf

# Optional: Groq chatbot
try:
    from groq import Groq
except ImportError:
    st.warning("⚠️ Groq package not found. Install via: `pip install groq`")

# ─────────────────────────────────────────────
# 🔧 Patch for old GRU models
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
# COMPANY MAP (UNCHANGED)
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

# ─────────────────────────────────────────────
# ✅ SAFE CSV PATH (minimal fix)
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "stock_details_5_years.csv")

# ─────────────────────────────────────────────
# LOAD CSV (FIXED ONLY HERE)
# ─────────────────────────────────────────────
@st.cache_data
def load_full_csv():
    df = pd.read_csv(CSV_PATH)

    # ✅ Fix column spacing issues
    df.columns = df.columns.str.strip()

    # ✅ FIX: robust datetime parsing (THIS SOLVES YOUR ERROR)
    df["Date"] = df["Date"].astype(str)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Drop invalid dates
    df = df.dropna(subset=["Date"])

    return df

# ─────────────────────────────────────────────
# FILTER VALID COMPANIES
# ─────────────────────────────────────────────
@st.cache_data
def get_valid_companies():
    df = load_full_csv()

    if "Company" not in df.columns:
        st.error("❌ CSV must contain a 'Company' column.")
        st.stop()

    csv_tickers = set(df["Company"].astype(str).unique())

    valid = {}
    for company_name, ticker in COMPANY_MAP.items():
        if (
            ticker in csv_tickers
            and os.path.exists(f"{ticker}_gru_model.h5")
        ):
            valid[company_name] = ticker

    return valid

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.header("🧭 Navigation")
show_chatbot = st.sidebar.checkbox("💬 Open Chatbot")

# ─────────────────────────────────────────────
# CHATBOT (UNCHANGED)
# ─────────────────────────────────────────────
if show_chatbot:
    st.header("🤖 Chat with Groq LLM")

    api_key = st.text_input("🔑 Enter Groq API Key", type="password")
    model_name = st.selectbox(
        "🧠 Choose model",
        ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"]
    )

    if api_key:
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
                st.session_state.messages.append(
                    {"role": "assistant", "content": reply}
                )

        if st.button("🧹 Clear Chat"):
            st.session_state.messages = [
                {"role": "system", "content": "You are a helpful AI assistant."}
            ]
            st.rerun()

    st.divider()

# ─────────────────────────────────────────────
# COMPANY SELECTION
# ─────────────────────────────────────────────
companies = get_valid_companies()

if not companies:
    st.error("❌ No valid companies found (CSV + model mismatch).")
    st.stop()

selected_company = st.selectbox("🏢 Select Company", sorted(companies.keys()))
ticker = companies[selected_company]

st.caption(f"📌 Ticker: {ticker}")

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df_full = load_full_csv()
df = df_full[df_full["Company"] == ticker].copy()

df.sort_values("Date", inplace=True)
df.set_index("Date", inplace=True)

# ─────────────────────────────────────────────
# PREP DATA
# ─────────────────────────────────────────────
close_prices = df["Close"].values.reshape(-1, 1)
scaler = MinMaxScaler()
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
model = load_model(f"{ticker}_gru_model.h5", custom_objects={"GRU": patched_gru})

y_pred = model.predict(X_test)

y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────
st.subheader("📊 Closing Price History")
fig = plt.figure(figsize=(14, 6))
plt.plot(df["Close"])
st.pyplot(fig)

st.subheader("📈 Moving Averages")
fig = plt.figure(figsize=(14, 6))
plt.plot(df["Close"].rolling(100).mean(), label="100 MA")
plt.plot(df["Close"].rolling(200).mean(), label="200 MA")
plt.plot(df["Close"], alpha=0.5)
plt.legend()
st.pyplot(fig)

st.subheader("📉 Predicted vs Actual Prices")
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
