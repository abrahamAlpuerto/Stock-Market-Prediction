import numpy as np
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
import psutil
from data_formating import get_train_data
import time
from LSTM import AttentionLayer, input_length, prediction_length


# Load the trained model
model = tf.keras.models.load_model('LSTM_40.keras', custom_objects={"AttentionLayer": AttentionLayer})

print(f"Memory usage: {psutil.Process().memory_info().rss / 1024 ** 2} MB")

(X_train, X_test, y_train, y_test), scalers = get_train_data(inputL=input_length, outputL=prediction_length)


# Function to plot predictions vs actual values
def plot_predictions(predictions_rescaled, y_test_rescaled, num_samples=1):
    plt.figure(figsize=(12, 6))

    # Select a subset of the predictions and actual values
    for i in range(min(num_samples, len(predictions_rescaled))):
        plt.plot(predictions_rescaled[i], label=f'Prediction {i+1}', linestyle='--', alpha=0.6)
        plt.plot(y_test_rescaled[i], label=f'Actual {i+1}', alpha=0.6)

    plt.title('Predicted vs Actual Stock Prices')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Predict using the saved model
start_time = time.time()
predictions = model.predict(X_test)
end_time = time.time()

print(f"Inference time: {end_time - start_time} seconds")
# Inverse transform the predictions and actual values using the saved scalers
y_test_rescaled = []
predictions_rescaled = []

for i in range(len(predictions)):
    stock_idx = i % len(scalers)
    stock_name = list(scalers.keys())[stock_idx]
    scaler_close = scalers[stock_name]['Close']

    # Rescale the data back to its original scale
    y_test_rescaled.append(scaler_close.inverse_transform(y_test[i].reshape(-1, 1)).flatten())
    predictions_rescaled.append(scaler_close.inverse_transform(predictions[i].reshape(-1, 1)).flatten())

# Convert lists to arrays
y_test_rescaled = np.array(y_test_rescaled)
predictions_rescaled = np.array(predictions_rescaled)

# Plot the predictions vs actual values
plot_predictions(predictions_rescaled, y_test_rescaled)

# Calculate the rescaled MAPE
mape_rescaled = np.mean(np.abs((y_test_rescaled - predictions_rescaled) / y_test_rescaled)) * 100
print(f'Rescaled MAPE: {mape_rescaled}%')