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
# 🔧 GRU PATCH
# ─────────────────────────────────────────────
def patched_gru(*args, **kwargs):
    kwargs.pop("time_major", None)
    return GRU(*args, **kwargs)

# ─────────────────────────────────────────────
# STREAMLIT CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("📈 Stock Trend Prediction using GRU")

# ─────────────────────────────────────────────
# PATH
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "stock_details_5_years.csv")

# ─────────────────────────────────────────────
# AUTO DETECT COMPANY MAP 🔥
# ─────────────────────────────────────────────
def get_model_tickers():
    files = os.listdir(BASE_DIR)
    tickers = []

    for f in files:
        if f.endswith("_gru_model.h5"):
            ticker = f.replace("_gru_model.h5", "")
            tickers.append(ticker)

    return sorted(tickers)

# ─────────────────────────────────────────────
# LOAD CSV (FINAL FIX)
# ─────────────────────────────────────────────
@st.cache_data
def load_full_csv():
    df = pd.read_csv(CSV_PATH)

    df.columns = df.columns.str.strip()

    df["Date"] = df["Date"].astype(str)

    # 🔥 FINAL FIX (timezone-safe)
    df["Date_parsed"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
        utc=True
    )

    mask = df["Date_parsed"].isna()

    df.loc[mask, "Date_parsed"] = pd.to_datetime(
        df.loc[mask, "Date"],
        errors="coerce",
        unit="s",
        utc=True
    )

    df["Date"] = df["Date_parsed"]

    df = df.dropna(subset=["Date"])
    df = df.drop(columns=["Date_parsed"])

    return df

# ─────────────────────────────────────────────
# GET VALID COMPANIES (AUTO)
# ─────────────────────────────────────────────
@st.cache_data
def get_valid_companies():
    df = load_full_csv()

    if "Company" not in df.columns:
        st.error("❌ CSV must contain 'Company'")
        st.stop()

    csv_tickers = set(df["Company"].astype(str).unique())
    model_tickers = get_model_tickers()

    # Only keep intersection
    valid = sorted(list(set(model_tickers) & csv_tickers))

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

    if api_key:
        client = Groq(api_key=api_key)

        if "messages" not in st.session_state:
            st.session_state.messages = []

        user_msg = st.chat_input("Ask something...")

        if user_msg:
            st.session_state.messages.append({"role": "user", "content": user_msg})

            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=st.session_state.messages
            )

            reply = response.choices[0].message.content
            st.write(reply)

            st.session_state.messages.append(
                {"role": "assistant", "content": reply}
            )

# ─────────────────────────────────────────────
# COMPANY SELECTION
# ─────────────────────────────────────────────
companies = get_valid_companies()

if not companies:
    st.error("❌ No valid companies found")
    st.stop()

ticker = st.selectbox("🏢 Select Company (Ticker)", companies)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df_full = load_full_csv()
df = df_full[df_full["Company"] == ticker].copy()

df.sort_values("Date", inplace=True)
df.set_index("Date", inplace=True)

if df.empty:
    st.error("❌ No data found for this ticker")
    st.stop()

# ─────────────────────────────────────────────
# PREP DATA
# ─────────────────────────────────────────────
data = df["Close"].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

X, y = [], []
for i in range(100, len(scaled)):
    X.append(scaled[i - 100:i])
    y.append(scaled[i])

X, y = np.array(X), np.array(y)

split = int(0.7 * len(X))
X_test = X[split:]
y_test = y[split:]

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
model_path = os.path.join(BASE_DIR, f"{ticker}_gru_model.h5")

model = load_model(model_path, custom_objects={"GRU": patched_gru})

# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────
y_pred = model.predict(X_test)

y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)

# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────
st.subheader("📊 Closing Price")
fig = plt.figure(figsize=(14, 6))
plt.plot(df["Close"])
st.pyplot(fig)

st.subheader("📉 Prediction vs Actual")
fig = plt.figure(figsize=(14, 6))
plt.plot(y_test, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.legend()
st.pyplot(fig)

# ─────────────────────────────────────────────
# CUSTOM PREDICTION
# ─────────────────────────────────────────────
st.subheader("🎯 Next Day Prediction")

input_price = st.number_input(
    "Enter Last Closing Price",
    value=float(df["Close"].iloc[-1])
)

if st.button("Predict"):
    last_99 = scaled[-99:]
    new_val = scaler.transform([[input_price]])

    seq = np.append(last_99, new_val).reshape(1, 100, 1)

    pred = model.predict(seq)
    price = scaler.inverse_transform(pred)[0][0]

    st.success(f"📈 Predicted Price: ${price:.2f}")
