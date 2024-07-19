# LSTM Predictions

This repository demonstrates the use of Long Short-Term Memory (LSTM) networks for predicting stock prices. LSTM is a type of Recurrent Neural Network (RNN) that is particularly effective for sequence prediction tasks, such as time series forecasting.

## What is LSTM?

Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) designed to model sequences and time series data. Unlike traditional RNNs, LSTMs are capable of learning long-term dependencies and capturing temporal patterns in data. This makes them ideal for tasks where the order of data points is crucial, such as predicting future stock prices based on historical data.

### Key Components of LSTM:

1. **Cell State**: The memory of the network that carries information across time steps.
2. **Forget Gate**: Decides what information from the cell state should be discarded.
3. **Input Gate**: Updates the cell state with new information.
4. **Output Gate**: Controls what information is output from the cell state.

## Why Use LSTM for Stock Price Prediction?

Stock price prediction is inherently a time series problem, where past prices are used to forecast future prices. LSTM networks offer several advantages for this type of problem:

1. **Handling Long-Term Dependencies**: Stock prices are influenced by long-term trends and patterns. LSTM's ability to remember long-term dependencies helps capture these trends more effectively than traditional RNNs.

2. **Mitigating Vanishing Gradient Problem**: Standard RNNs often struggle with the vanishing gradient problem, where gradients become too small to effectively update weights during training. LSTM’s architecture, with its gates and cell state, helps mitigate this issue, allowing the network to learn over longer sequences.

3. **Temporal Sequence Learning**: LSTMs are designed to learn and predict sequences. This makes them particularly suited for modeling the temporal aspects of stock prices, where the sequence of past prices is critical for predicting future prices.

4. **Flexibility with Input Lengths**: LSTM networks can handle variable input lengths and sequences, which is useful when dealing with historical stock data that may vary in length.

## How LSTM is Used for Stock Price Prediction

In stock price prediction, LSTM networks are used to forecast future prices based on historical data. The typical workflow involves:

1. **Data Collection**: Gather historical stock price data, including features such as `Open`, `High`, `Low`, `Close`, and `Volume`.
2. **Data Preprocessing**: Normalize the data and create sequences of historical prices to use as input for the LSTM model.
3. **Model Building**: Construct and compile an LSTM model with appropriate layers.
4. **Model Training**: Train the LSTM model using historical data sequences.
5. **Prediction**: Use the trained model to predict future stock prices based on the input data.


## Sample Code Snippet

Here’s a sample code of building and training an LSTM model:

```python
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Download stock data
data = yf.download('AAPL', start='2010-01-01', end='2023-01-01')
data = data[['Close']]

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Prepare sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 60
X, y = create_sequences(scaled_data, seq_length)

# Split data into training and testing sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    tf.keras.layers.LSTM(50, return_sequences=False),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))
