"""
Train ONE LSTM model on MANY stock tickers instead of just AAPL.

Approach:
  - Each ticker gets its OWN MinMaxScaler, fit only on that ticker's prices.
    (Different stocks trade at wildly different price levels — e.g. AAPL ~$200
    vs BRK.A ~$600,000 — so a single shared scaler would badly distort most of them.)
  - After scaling, every ticker's history becomes a comparable 0-1 pattern.
  - We build (100-day-window -> next-day) sequences per ticker, then combine
    ALL tickers' sequences into one big training set.
  - A single LSTM is trained on this combined set, so it learns general
    price-MOVEMENT patterns rather than memorizing one company's price range.
  - We save a dictionary of {ticker: scaler} so the app can pull the right
    scaler back out at prediction time (and can fit a fresh one on the fly
    for tickers that weren't in the training set).
"""

import numpy as np
import pandas as pd
import yfinance as yf
import pickle

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# -------------------------
# CONFIG
# -------------------------
TICKERS = [
    "AAPL", "GOOGL", "AMZN",
    "TSLA", "NVDA",
    "BPCL" ,     
    "RELIANCE.NS",
    "SUNPHARMA.NS",
    "STARHEALTH.NS",
    "IRME.NS"
]

START_DATE = "2015-01-01"
END_DATE = None
LOOKBACK = 100
EPOCHS = 30
BATCH_SIZE = 32

# -------------------------
# BUILD SEQUENCES PER TICKER
# -------------------------
def create_dataset(dataset, lookback=LOOKBACK):
    x, y = [], []
    for i in range(lookback, len(dataset)):
        x.append(dataset[i - lookback:i])
        y.append(dataset[i, 0])
    return np.array(x), np.array(y)


all_x = []
all_y = []
scalers = {}  # ticker -> fitted MinMaxScaler

for ticker in TICKERS:
    print(f"Downloading {ticker}...")
    df = yf.download(ticker, start=START_DATE, end=END_DATE)

    if df.empty or len(df) < LOOKBACK + 1:
        print(f"  Skipping {ticker} (not enough data)")
        continue

    close_prices = df["Close"].values.reshape(-1, 1)

    # Each ticker gets its own scaler, fit only on its own price history
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close_prices)
    scalers[ticker] = scaler

    x, y = create_dataset(scaled)
    if len(x) == 0:
        continue

    all_x.append(x)
    all_y.append(y)
    print(f"  {ticker}: {len(x)} training sequences")

# -------------------------
# COMBINE ALL TICKERS INTO ONE TRAINING SET
# -------------------------
x_train = np.concatenate(all_x, axis=0)
y_train = np.concatenate(all_y, axis=0)

# reshape for LSTM: (samples, timesteps, features)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

# shuffle so batches contain a mix of different companies, not one at a time
shuffle_idx = np.random.permutation(len(x_train))
x_train = x_train[shuffle_idx]
y_train = y_train[shuffle_idx]

print(f"\nTotal combined training sequences: {len(x_train)}")

# -------------------------
# SAVE THE PER-TICKER SCALERS (as a dict, not a single scaler)
# -------------------------
with open("scalers.pkl", "wb") as f:
    pickle.dump(scalers, f)

# -------------------------
# MODEL
# -------------------------
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(LOOKBACK, 1)))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mean_squared_error")

# -------------------------
# TRAIN
# -------------------------
model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1)

# -------------------------
# SAVE MODEL
# -------------------------
model.save("multi_stock_lstm_model.h5")

print("\nModel + per-ticker scalers saved successfully!")
print(f"Trained on {len(scalers)} tickers: {list(scalers.keys())}")

import pickle

with open("model/scalers.pkl", "rb") as f:
    scalers = pickle.load(f)

print("Tickers in training set:")
for ticker in scalers.keys():
    print(ticker)

