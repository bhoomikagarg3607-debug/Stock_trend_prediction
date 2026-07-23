import streamlit as st

st.set_page_config(
    page_title="Stock Predictor",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    /* Main background */
    .main {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        color: white;
    }

    /* Title */
    .title {
        font-size: 40px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 20px;
        color: #38bdf8;
    }

    /* Card style */
    .card {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
        margin-bottom: 15px;
    }

    /* Metric text */
    .metric {
        font-size: 20px;
        font-weight: 600;
    }

    /* Buttons */
    .stButton > button {
        background-color: #38bdf8;
        color: black;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
    }

    .stButton > button:hover {
        background-color: #0ea5e9;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>📊 Stock Trend Prediction App</div>", unsafe_allow_html=True)


import numpy as np
import pandas as pd
import yfinance as yf
import pickle
from keras.models import load_model

# -------------------------
# LOAD MODEL + SCALER (CACHE FOR SPEED)
# -------------------------
from sklearn.preprocessing import MinMaxScaler

@st.cache_resource
def load_assets():
    model = load_model("model/multi_stock_lstm_model.h5")

    with open("model/scalers.pkl", "rb") as f:
        scalers = pickle.load(f)  # this is now a dict: {ticker: scaler}

    return model, scalers

model, scalers = load_assets()

# -------------------------
# UI
# -------------------------
st.title("📈 Professional Stock Predictor (LSTM)")
st.write("Enter a stock symbol to get next-day prediction")

ticker = st.text_input("Stock Symbol (e.g. AAPL, TSLA, INFY.NS)")

# -------------------------
# CONSTANTS
# -------------------------
LOOKBACK = 100

if ticker:

    # -------------------------
    # LOAD DATA
    # -------------------------
    df = yf.download(ticker, start="2015-01-01", end=None)

    data = df["Close"].values.reshape(-1, 1)

    if ticker in scalers:
        # Ticker was part of training - use its saved scaler
        scaler = scalers[ticker]
        scaled_data = scaler.transform(data)
    else:
        # Unseen ticker - fit a fresh scaler on the fly.
        # This works reasonably well because the model learned general
        # price-MOVEMENT patterns (0-1 shape of the curve), not absolute
        # price levels, so it can generalize to tickers it never saw.
        st.info(f"{ticker} wasn't in the training set — using an on-the-fly scaler.")
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)


    if df.empty:
        st.error("Invalid ticker or no data found!")
    else:

        st.subheader(f"{ticker} Closing Price History")
        st.line_chart(df["Close"])

        # -------------------------
        # -------------------------
        # PREP DATA (FIXED)
        # -------------------------
        data = df["Close"].values.reshape(-1, 1)

        if ticker in scalers:
            scaler = scalers[ticker]
            scaled_data = scaler.transform(data)
        else:
            st.info(f"{ticker} wasn't in training set — using fresh scaler")
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)

        if len(scaled_data) < LOOKBACK:
            st.error("Not enough data for prediction")
        else:

            # last 100 days → input
            x_input = scaled_data[-LOOKBACK:].reshape(1, LOOKBACK, 1)

            # -------------------------
            # PREDICT NEXT DAY
            # -------------------------
            pred_scaled = model.predict(x_input, verbose=0)
            pred_price = scaler.inverse_transform(pred_scaled)[0][0]

            # latest actual price
            current_price = data[-1][0]

            # -------------------------
            # SIGNAL
            # -------------------------
            threshold = 0.005

            if pred_price > current_price * (1 + threshold):
                signal = "BUY 📈"
            elif pred_price < current_price * (1 - threshold):
                signal = "SELL 📉"
            else:
                signal = "HOLD ⏸"

            # -------------------------
            # CONFIDENCE
            # -------------------------
            confidence = 100 - abs((pred_price - current_price) / current_price) * 100
            confidence = float(max(0, min(100, confidence)))

            # -------------------------
            # DISPLAY
            # -------------------------
            st.subheader("Trading Signal")

            if signal == "BUY 📈":
                st.success(signal)
            elif signal == "SELL 📉":
                st.error(signal)
            else:
                st.warning(signal)

            st.write(f"Current Price: {current_price:.2f}")
            st.write(f"Predicted Next Day Price: {pred_price:.2f}")
            st.write(f"Confidence: {confidence:.2f}%")

            # -------------------------
            # OUTPUT
            # -------------------------
            st.success(f"📊 Predicted Next Price: {pred_price:.2f}")

            # optional comparison
            st.write("Last Actual Price:", df["Close"].iloc[-1])











