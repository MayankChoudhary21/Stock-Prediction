import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import datetime
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import GRU

# Add custom GRU layer to handle legacy parameters
class CompatibleGRU(GRU):
    def __init__(self, *args, **kwargs):
        # Remove unsupported argument before parent initialization
        kwargs.pop('time_major', None)
        super().__init__(*args, **kwargs)

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
            X.append(scaled_data[i - 100:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        split = int(0.7 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model_path = f"{user_input}_gru_model.h5"
        if os.path.exists(model_path):
            try:
                # Load model with custom GRU compatibility
                custom_objects = {'GRU': CompatibleGRU}
                model = load_model(model_path, 
                                  compile=False,
                                  custom_objects=custom_objects)
                
                # Test model with dummy input
                test_input = np.random.rand(1, 100, 1)
                _ = model.predict(test_input)
                
                # Prediction
                y_predict = model.predict(X_test)
                y_predict = scaler.inverse_transform(y_predict.reshape(-1, 1))
                y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

                # Display original data
                st.subheader(f"{user_input} Historical Data")
                st.write(df.describe())

                # Plot closing price history
                st.subheader('📊 Closing Price History')
                fig = plt.figure(figsize=(14, 6))
                plt.plot(df['Close'])
                plt.ylabel('Price')
                st.pyplot(fig)

                # Plot moving averages
                st.subheader('📈 Moving Averages (100 & 200 Days)')
                fig = plt.figure(figsize=(14, 6))
                ma100 = df['Close'].rolling(100).mean()
                ma200 = df['Close'].rolling(200).mean()
                plt.plot(ma100, 'r', label='100-day MA')
                plt.plot(ma200, 'g', label='200-day MA')
                plt.plot(df['Close'], alpha=0.5, label='Closing Price')
                plt.legend()
                st.pyplot(fig)

                # Plot predicted vs actual prices
                st.subheader("📉 Predicted vs Actual Prices")
                fig = plt.figure(figsize=(14, 6))
                plt.plot(df.index[-len(y_test):], y_test, 'b', label='Actual')
                plt.plot(df.index[-len(y_predict):], y_predict, 'r', label='Predicted')
                plt.legend()
                st.pyplot(fig)

                # Display table of predictions vs actual
                st.subheader("📋 Real vs Predicted Table (Last 20 Samples)")
                comparison_df = pd.DataFrame({
                    'Date': df.index[-len(y_test):][-20:],
                    'Actual Price': y_test.flatten()[-20:],
                    'Predicted Price': y_predict.flatten()[-20:]
                })
                comparison_df['Difference'] = comparison_df['Predicted Price'] - comparison_df['Actual Price']
                comparison_df['% Error'] = (comparison_df['Difference'] / comparison_df['Actual Price']) * 100
                comparison_df.set_index('Date', inplace=True)
                st.dataframe(comparison_df.style.format({
                    'Actual Price': '${:.2f}',
                    'Predicted Price': '${:.2f}',
                    'Difference': '${:.2f}',
                    '% Error': '{:.2f}%'
                }))

                # Custom input prediction
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

            except Exception as e:
                st.error(f"Model loading failed: {str(e)}")
                st.markdown("""
                **Final Compatibility Solution:**
                1. Add this code before loading the model:
                ```python
                from tensorflow.keras.layers import GRU
                class CompatibleGRU(GRU):
                    def __init__(self, *args, **kwargs):
                        kwargs.pop('time_major', None)
                        super().__init__(*args, **kwargs)
                ```
                2. Load with `custom_objects={'GRU': CompatibleGRU}`
                """)
        else:
            st.error(f"❌ Model file '{model_path}' not found!")
