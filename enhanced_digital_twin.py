import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Simulate time vector
steps = 200
time = np.arange(0, steps, 1)

# Generate synthetic vibration data (normal trend + noise)
vibration = 0.05 * time + np.sin(time / 15) + np.random.normal(0, 0.1, size=steps)

# Generate synthetic temperature data (normal trend + noise)
temperature = 40 + 2 * np.sin(time / 50) + np.random.normal(0, 0.5, size=steps)

# Inject anomalies to vibration (simulate faults from t=150 to 160)
vibration[150:160] += 2  # Large spike to simulate failure

# Normalize vibration data
scaler_v = MinMaxScaler()
vibration_scaled = scaler_v.fit_transform(vibration.reshape(-1, 1))

# Prepare sequences for LSTM
sequence_length = 20
def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        x = data[i:(i + seq_len)]
        y = data[i + seq_len]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

X, y = create_sequences(vibration_scaled, sequence_length)
X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for LSTM

# Build LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(sequence_length, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X, y, epochs=30, batch_size=16, verbose=1)

# Make rolling predictions for next 50 timesteps
input_seq = X[-1]
predictions = []
for _ in range(50):
    pred = model.predict(input_seq.reshape(1, sequence_length, 1), verbose=0)
    predictions.append(pred[0, 0])
    input_seq = np.append(input_seq[1:], pred).reshape(sequence_length, 1)

# Convert predictions back to original scale
predictions = scaler_v.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

# Plot measurements and predictions
plt.figure(figsize=(10, 5))
plt.plot(time, vibration, label='Actual Vibration')
plt.plot(time, temperature, label='Actual Temperature')
plt.plot(np.arange(steps, steps + 50), predictions, linestyle='dashed', label='Predicted Vibration')

# Confidence bounds (dummy example with +/- 0.2)
plt.fill_between(np.arange(steps, steps + 50), predictions - 0.2, predictions + 0.2,
                 color='orange', alpha=0.2, label='Prediction Confidence Bound')

plt.xlabel('Time (arbitrary units)')
plt.ylabel('Sensor Values')
plt.title('Enhanced Digital Twin: Multi-Sensor Vibration Prediction')
plt.legend()
plt.show()
