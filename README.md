Stock Trend Prediction Application

Overview

This is a Streamlit-based web application that predicts stock trends using historical data and a pre-trained LSTM (Long Short-Term Memory) model. The application allows users to visualize historical stock prices, compare them with moving averages, and predict future trends.

Features

Stock Data Retrieval:

Retrieves historical stock data from Yahoo Finance using the provided stock ticker symbol.

Default stock ticker: AAPL (Apple Inc.).

Visualization:

Displays historical closing prices from 2010 to 2023.

Visualizes the closing price with:

100-day Moving Average.

Both 100-day and 200-day Moving Averages.

Prediction:

Predicts stock prices based on the pre-trained LSTM model.

Compares predicted prices with the original prices.

Requirements

To run this application, ensure you have the following dependencies installed:

Python 3.x

Streamlit

Numpy

Pandas

Matplotlib

Pandas DataReader

yfinance

Keras

scikit-learn

Installation

Clone this repository:

git clone <repository_url>

Navigate to the project directory:

cd <project_directory>

Install the required dependencies:

pip install -r requirements.txt

Usage

Run the Streamlit application:

streamlit run app.py

Open your browser and navigate to the URL provided by Streamlit (e.g., http://localhost:8501).

Enter a stock ticker symbol (e.g., GOOGL, MSFT) in the input field and view the results.

How It Works

Data Retrieval:

The app fetches historical stock data for the specified ticker symbol from Yahoo Finance.

Visualization:

The app creates various charts to show the stock's historical performance and trends.

Prediction:

The pre-trained LSTM model (stock_lstm.h5) is used to predict stock prices.

The model is trained on 70% of the historical data, and predictions are made on the remaining 30%.

Comparison:

The predicted prices are compared with the original prices to visualize the model's performance.

File Structure

app.py: Main application file containing the Streamlit code.

stock_lstm.h5: Pre-trained LSTM model file.

Example Screenshots

Closing Price vs. Time Chart

Closing Price with Moving Averages

Prediction vs. Original Price Comparison

Future Improvements

Add support for multiple stock models.

Include additional technical indicators (e.g., RSI, Bollinger Bands).

Enable real-time stock predictions.

License

This project is open-source and available under the MIT License.

Contributors

Mayank Choudhary
