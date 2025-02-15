
# Stock Trend Prediction Application

## Overview
This is a Streamlit-based web application that predicts stock trends using historical data and a pre-trained LSTM (Long Short-Term Memory) model. The application allows users to visualize historical stock prices, compare them with moving averages, and predict future trends.

## Features
1. **Stock Data Retrieval**:
   - Retrieves historical stock data from Yahoo Finance using the provided stock ticker symbol.
   - Default stock ticker: `AAPL` (Apple Inc.).

2. **Visualization**:
   - Displays historical closing prices from 2010 to 2023.
   - Visualizes the closing price with:
     - 100-day Moving Average.
     - Both 100-day and 200-day Moving Averages.

3. **Prediction**:
   - Predicts stock prices based on the pre-trained LSTM model.
   - Compares predicted prices with the original prices.

## Requirements
To run this application, ensure you have the following dependencies installed:

- Python 3.x
- Streamlit
- Numpy
- Pandas
- Matplotlib
- Pandas DataReader
- yfinance
- Keras
- scikit-learn

## Installation
1. Clone this repository:
   ```bash
   git clone <repository_url>
   ```
2. Navigate to the project directory:
   ```bash
   cd <project_directory>
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
2. Open your browser and navigate to the URL provided by Streamlit (e.g., `http://localhost:8501`).

3. Enter a stock ticker symbol (e.g., `GOOGL`, `MSFT`) in the input field and view the results.

## How It Works
1. **Data Retrieval**:
   - The app fetches historical stock data for the specified ticker symbol from Yahoo Finance.

2. **Visualization**:
   - The app creates various charts to show the stock's historical performance and trends.

3. **Prediction**:
   - The pre-trained LSTM model (`stock_lstm.h5`) is used to predict stock prices.
   - The model is trained on 70% of the historical data, and predictions are made on the remaining 30%.

4. **Comparison**:
   - The predicted prices are compared with the original prices to visualize the model's performance.

## File Structure
- `app.py`: Main application file containing the Streamlit code.
- `stock_lstm.h5`: Pre-trained LSTM model file.

## Visualisation
![image](https://github.com/user-attachments/assets/f53cf5c7-0d3e-4e88-9762-6965c40897bb)
![image](https://github.com/user-attachments/assets/47445c8d-f02c-4ced-9b03-3632647c4971)
![image](https://github.com/user-attachments/assets/f99b2c28-7ac8-4ee8-8c87-fd6242e7f541)



## Future Improvements
- Add support for multiple stock models.
- Include additional technical indicators (e.g., RSI, Bollinger Bands).
- Enable real-time stock predictions.

## License
This project is open-source and available under the [MIT License](LICENSE).

## Contributors
- **Mayank Choudhary**
- **Shambhavi Gunda**

## Code Deployed
- Link:https://mayank-stock-prediction.streamlit.app/

