# Replace the load_assets() function and the "PREP DATA" section
# of your app.py with the versions below.

from sklearn.preprocessing import MinMaxScaler

@st.cache_resource
def load_assets():
    model = load_model("model/multi_stock_lstm_model.h5")

    with open("model/scalers.pkl", "rb") as f:
        scalers = pickle.load(f)  # this is now a dict: {ticker: scaler}

    return model, scalers

model, scalers = load_assets()

# ... later, inside `if ticker:` after downloading df ...

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

# rest of the prediction logic (LOOKBACK check, x_input, model.predict,
# scaler.inverse_transform, signal, confidence, display) stays exactly
# the same as before.