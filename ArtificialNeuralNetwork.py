import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid (backpropagation)
def sigmoid_derivative(x):
    return x * (1 - x)

# Training data (input): Predict next number from a pair
# Example [1, 2] -> 3, [2, 3] -> 4, etc
X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6],
    [6, 7]
])

# Expected output (y): next number in the sequence
y = np.array([
    [3],
    [4],
    [5],
    [6],
    [7],
    [8]
])

# Set random seed for reproducibility
np.random.seed(1)

# Number of neurons in each layer
input_layer_neurons = 2   # each input has a pair
hidden_layer_neurons = 3
output_neurons = 1        # one output number

# Randomly initialize weights and biases
weights_input_hidden = np.random.rand(input_layer_neurons, hidden_layer_neurons)
bias_hidden = np.random.rand(1, hidden_layer_neurons)

weights_hidden_output = np.random.rand(hidden_layer_neurons, output_neurons)
bias_output = np.random.rand(1, output_neurons)

# Learning rate
learning_rate = 0.01

# Training the neural network
epochs = 1000
for epoch in range(epochs):
    #  FORWARD PROPAGATION
    # Step 1: Input to hidden layer
    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)

    # Step 2: Hidden to output layer
    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    final_output = sigmoid(final_input)

    #  ERROR CALCULATION
    error = y - final_output

    # Optional: Print loss every 100 epochs
    if epoch % 100 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {epoch} - Loss: {loss:.6f}")

    #  BACKPROPAGATION
    # Step 1: Calculate delta at output
    d_output = error * sigmoid_derivative(final_output)

    # Step 2: Calculate error and delta for hidden layer
    error_hidden = d_output.dot(weights_hidden_output.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    #  UPDATE WEIGHTS AND BIASES
    # Output layer updates
    weights_hidden_output += hidden_output.T.dot(d_output) * learning_rate
    bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate

    # Hidden layer updates
    weights_input_hidden += X.T.dot(d_hidden) * learning_rate
    bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

# PREDICTION
# Test the trained model
test_input = np.array([[7, 8]])  # Expected output: 9
hidden_layer_test = sigmoid(np.dot(test_input, weights_input_hidden) + bias_hidden)
predicted_output = sigmoid(np.dot(hidden_layer_test, weights_hidden_output) + bias_output)

print(f"\nPrediction for input {test_input.tolist()}: {predicted_output[0][0]:.4f}")
