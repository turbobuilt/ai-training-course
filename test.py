import random

# Define the neural network architecture
input_size = 20
hidden_size = 20
output_size = 20
learning_rate = 0.0001
epochs = 5000

# Initialize weights and biases randomly
weights_input_hidden = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(input_size)]
weights_hidden_output = [[random.uniform(-1, 1) for _ in range(output_size)] for _ in range(hidden_size)]

biases_hidden = [random.uniform(-1, 1) for _ in range(hidden_size)]
biases_output = [random.uniform(-1, 1) for _ in range(output_size)]

# Input data (just for demonstration)
input_data = [random.uniform(-1, 1) for _ in range(input_size)]

# Target data (just for demonstration)
target_data = [random.uniform(-1, 1) for _ in range(output_size)]

# Training loop
for epoch in range(epochs):
    # Forward pass
    hidden_layer_input = [0] * hidden_size
    hidden_layer_output = [0] * hidden_size
    output_layer_input = [0] * output_size
    output_layer_output = [0] * output_size

    # Calculate hidden layer input and output
    for i in range(hidden_size):
        hidden_layer_input[i] = biases_hidden[i]
        for j in range(input_size):
            hidden_layer_input[i] += input_data[j] * weights_input_hidden[j][i]
        hidden_layer_output[i] = max(0, hidden_layer_input[i])  # ReLU activation

    # Calculate output layer input and output
    for i in range(output_size):
        output_layer_input[i] = biases_output[i]
        for j in range(hidden_size):
            output_layer_input[i] += hidden_layer_output[j] * weights_hidden_output[j][i]
        output_layer_output[i] = output_layer_input[i]

    # Calculate Mean Squared Error (MSE)
    mse = sum([(target_data[i] - output_layer_output[i]) ** 2 for i in range(output_size)]) / output_size

    # Backpropagation
    # Compute gradients
    output_layer_gradient = [-(target_data[i] - output_layer_output[i]) for i in range(output_size)]
    hidden_layer_gradient = [0] * hidden_size

    for i in range(hidden_size):
        for j in range(output_size):
            hidden_layer_gradient[i] += output_layer_gradient[j] * weights_hidden_output[i][j]
        hidden_layer_gradient[i] *= 1 if hidden_layer_input[i] > 0 else 0  # ReLU derivative

    # Update weights and biases using gradient descent
    for i in range(hidden_size):
        biases_hidden[i] -= learning_rate * hidden_layer_gradient[i]
        for j in range(input_size):
            weights_input_hidden[j][i] -= learning_rate * hidden_layer_gradient[i] * input_data[j]

    for i in range(output_size):
        biases_output[i] -= learning_rate * output_layer_gradient[i]
        for j in range(hidden_size):
            weights_hidden_output[j][i] -= learning_rate * output_layer_gradient[i] * hidden_layer_output[j]

    # Print the MSE every 100 epochs
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, MSE: {mse}')

# After training, the weights and biases will be adjusted to approximate the target data
