import random as rand
import numpy as np
import pandas as pd 

# Load the dataset
df = pd.read_csv("WeatherData_Q3.csv")

# Split data set into 15 iterations for training, 5 for testing
train_df = df.iloc[:15]
tes_df = df.iloc[15:]

# Inputs
X_train = train_df[["temp", "humid"]].values
y_train = train_df["rain"].values

X_test = tes_df[["temp", "humid"]].values
y_test = tes_df["rain"].values

# Parameters
learning_rate = 0.1
max_iteration = 1000
weights = [rand.uniform(-0.5, 0.5) for i in range(2)]
bias = rand.uniform(-0.5, 0.5)  # Random bias initialization between -0.5 and 0.5

# Activation function: Step function
def activation(weighted_sum):
    return 1 if weighted_sum >= 0 else 0

# Prediction function
def predict(inputs):
    weighted_sum = np.dot(inputs, weights) + bias
    return activation(weighted_sum)

# Training loop
for iteration in range(max_iteration):
    errors = 0
    for inputs, label in zip(X_train, y_train):
        prediction = predict(inputs)
        error = label - prediction

        # Update weights and bias if there is an error
        if error != 0:
            # Update rule for weights and bias
            weights[0] += learning_rate * error * inputs[0]  # temp
            weights[1] += learning_rate * error * inputs[1]  # humid
            bias += learning_rate * error                   # bias
            errors += 1

    # If no errors, training is complete
    if errors == 0:
        print(f"Training converged after {iteration+1} iterations.")
        break
else:
    print(f"Reached max iterations ({max_iteration}).")

# Function to calculate accuracy
def calculate_accuracy(X, y):
    correct = 0
    for inputs, label in zip(X, y):
        prediction = predict(inputs)
        if prediction == label:
            correct += 1
    accuracy = correct / len(y)
    return accuracy

# Report training and test accuracy
train_accuracy = calculate_accuracy(X_train, y_train)
test_accuracy = calculate_accuracy(X_test, y_test)

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Print final weights and bias
print("Final Weights:", weights)
print("Final Bias:", bias)
