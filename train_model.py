import numpy as np
import pandas as pd
import yfinance as yf
import pickle

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# -----------------------
# 1. LOAD DATA
# -----------------------
ticker = "AAPL"
df = yf.download(ticker, start="2015-01-01", end="2025-01-01")

data = df["Close"].values.reshape(-1, 1)

# -----------------------
# 2. SCALE DATA
# -----------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# SAVE SCALER (VERY IMPORTANT)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# -----------------------
# 3. CREATE SEQUENCES
# -----------------------
def create_dataset(dataset):
    x, y = [], []
    for i in range(100, len(dataset)):
        x.append(dataset[i-100:i])
        y.append(dataset[i, 0])
    return np.array(x), np.array(y)

x_train, y_train = create_dataset(scaled_data)

# reshape for LSTM
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

# -----------------------
# 4. MODEL
# -----------------------
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mean_squared_error")

# -----------------------
# 5. TRAIN
# -----------------------
model.fit(x_train, y_train, epochs=10, batch_size=32)

# -----------------------
# 6. SAVE MODEL
# -----------------------
model.save("stock_lstm_model.h5")

print("Model + scaler saved successfully!")