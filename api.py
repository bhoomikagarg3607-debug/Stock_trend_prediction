from fastapi import FastAPI
import yfinance as yf
import numpy as np
import pickle
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from pymongo import MongoClient
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LOAD MODEL
model = load_model("model/multi_stock_lstm_model.h5")

with open("model/scalers.pkl", "rb") as f:
    scalers = pickle.load(f)

LOOKBACK = 100


client = MongoClient("mongodb://localhost:27017/")
db = client["stock_db"]
collection = db["predictions"]

collection.insert_one({
    "stock":"AAPL" ,
    "prediction":215.5
})

@app.get("/predict")
def predict(ticker: str):

    df = yf.download(ticker, period="10y")

    if df.empty:
        return {"error": "Invalid ticker"}

    data = df["Close"].values.reshape(-1, 1)

    # scaler handling
    if ticker in scalers:
        scaler = scalers[ticker]
        scaled = scaler.transform(data)
    else:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(data)

    x_input = scaled[-LOOKBACK:].reshape(1, LOOKBACK, 1)

    pred_scaled = model.predict(x_input, verbose=0)
    pred_price = scaler.inverse_transform(pred_scaled)[0][0]

    current_price = float(df["Close"].iloc[-1])

    collection.insert_one({
    "ticker": ticker,
    "current_price": current_price,
    "predicted_price": float(pred_price)
    })

    return {
        "ticker": ticker,
        "current_price": current_price,
        "predicted_price": float(pred_price)
    }


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
