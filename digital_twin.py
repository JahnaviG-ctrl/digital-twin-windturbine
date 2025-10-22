import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Generate synthetic vibration data (simulating gearbox sensor output)
time = np.arange(0, 200, 1)
vibration = 0.05 * time + np.sin(time / 15) + np.random.normal(0, 0.1, size=time.size)

# Normalize data for better training
scaler = MinMaxScaler(feature_range=(0, 1))
vibration_scaled = scaler.fit_transform(vibration.reshape(-1,1))

# Prepare data sequences for LSTM
def create_sequences(data, seq_length=20):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 20
X, y = create_sequences(vibration_scaled, seq_length)
X = X.reshape((X.shape[0], X.shape[1], 1))  # [samples, timesteps, features]

# Build LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(seq_length, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(X, y, epochs=30, batch_size=16, verbose=1)

# Predict next 50 time steps
input_seq = X[-1]
predictions = []
for _ in range(50):
    pred = model.predict(input_seq.reshape(1, seq_length, 1), verbose=0)
    predictions.append(pred[0,0])
    input_seq = np.append(input_seq[1:], pred).reshape(seq_length,1)

# Inverse transform predictions
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1,1)).flatten()

# Plot actual and predicted data
plt.plot(time, vibration, label='Actual Vibration')
plt.plot(np.arange(time[-1]+1, time[-1]+51), predictions, label='Predicted Vibration', linestyle='dashed')
plt.xlabel('Time')
plt.ylabel('Vibration')
plt.legend()
plt.title('Digital Twin: LSTM-based Vibration Prediction')
plt.show()
