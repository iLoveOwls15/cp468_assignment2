import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = "WeatherData_Q3.csv"  # Update with the correct path if needed
data = pd.read_csv(file_path)

# Rename columns for clarity
data.columns = ['temp', 'humid', 'rain']  # x1 = Temperature, x2 = Humidity, y = Rain (0 or 1)

# Split into training (first 15 rows) and test (last 5 rows)
train_data = data.iloc[:15]
test_data = data.iloc[15:]

X_train = train_data[['temp', 'humid']].values
y_train = train_data['rain'].values

X_test = test_data[['temp', 'humid']].values
y_test = test_data['rain'].values

# Add bias term (x0 = 1) to input features
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

# Initialize weights randomly between -0.5 and 0.5
np.random.seed(42)
weights = np.random.uniform(-0.5, 0.5, X_train.shape[1])

# Learning rate and epochs
alpha = 0.1
epochs = 1000

# Activation function (Step function)
def activation(z):
    return 1 if z >= 0 else 0

# Train the Perceptron model
for epoch in range(epochs):
    error_count = 0
    for i in range(len(X_train)):
        z = np.dot(weights, X_train[i])
        y_pred = activation(z)
        error = y_train[i] - y_pred
        if error != 0:
            weights += alpha * error * X_train[i]  # Weight update
            error_count += 1
    if error_count == 0:
        break  # Stop early if no errors

# Print final weights
print(f"Final Weights:\nBias: {weights[0]:.4f}\nTemperature Weight: {weights[1]:.4f}\nHumidity Weight: {weights[2]:.4f}")

# Scatter plot of the data points
plt.figure(figsize=(8, 6))
plt.scatter(data[data['rain'] == 0]['temp'], data[data['rain'] == 0]['humid'], 
            color='blue', marker='s', label='No Rain (y=0)')

plt.scatter(data[data['rain'] == 1]['temp'], data[data['rain'] == 1]['humid'], 
            color='red', marker='o', label='Rain (y=1)')

# Decision boundary equation: w0 + w1*x1 + w2*x2 = 0  =>  x2 = (-w0 - w1*x1) / w2
x1_vals = np.linspace(0, 1, 100)  # Temperature range
x2_vals = (-weights[0] - weights[1] * x1_vals) / weights[2]  # Compute corresponding humidity values

# Plot the decision boundary
plt.plot(x1_vals, x2_vals, 'k--', label='Decision Boundary')

# Labels, title, and legend
plt.xlabel('Temperature (x1)')
plt.ylabel('Humidity (x2)')
plt.title('Perceptron Decision Boundary for Weather Prediction')
plt.legend()
plt.grid(True)
plt.xlim(0, 1)
plt.ylim(0, 1)

# Show the plot
plt.show()
