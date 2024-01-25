import random
import torch





def relu(x):
    return max(0, x)

def relu_derivative(x):
    return x > 0
    # return 1 if x > 0 else 0

# y = x every time you increase x by one, you increase y by 1
# a
# y = 2x + b
# x x x x x

# x x x x x   [[input[i]*self.weights_input_hidden[output_neuron_index][i] for input_index in range(5)] for output_neuron_index in range(5)]

# x x x x x


# 0 4
# 1 2 4 3 7

# y = m1 + b
# derviative of y = 1x + b wrt m is x  x (-1)


#2   beta 1     2 * .1 + .9 * 1.1 
#1   beta 2     (1 * .01)^2 + ().9 * 1.1)^2

# print(torch.rand((2,5)))  # y,x      x,y
# exit()


class NeuralNetwork:
    def __init__(self):
        # Seed the random number generator for reproducibility
        random.seed(42)

        # Initialize weights and biases for the input to hidden layer
        self.weights_input_hidden = torch.rand((5,5))*2 - 1 # [[random.uniform(-1, 1) for _ in range(5)] for _ in range(5)]
        self.bias_hidden = torch.zeros(5) # [0 for _ in range(5)]

        # Initialize weights and biases for the hidden to output layer
        self.weights_hidden_output = torch.rand((5,5)) * 2 -1 # [[random.uniform(-1, 1) for _ in range(5)] for _ in range(5)]
        self.bias_output = torch.zeros(5) # [0 for _ in range(5)]

    def forward(self, X):
        # Forward pass through the network
        
        # Input layer to hidden layer

        
        self.hidden_layer_output = self.weights_input_hidden * X
        self.hidden_layer_output = self.hidden_layer_output.sum(axis=-1) + self.bias_hidden

         #[sum(X[j] * self.weights_input_hidden[j][k] for j in range(5)) + self.bias_hidden[k] for k in range(5)]
        self.hidden_layer_output = torch.max(torch.tensor(0), self.hidden_layer_output) # [relu(x) for x in self.hidden_layer_input]

        # Hidden layer to output layer
        self.output_layer_input = (self.weights_hidden_output * self.hidden_layer_output).sum(axis=-1) + self.bias_output
        #[sum(self.hidden_layer_output[j] * self.weights_hidden_output[j][k] for j in range(5)) + self.bias_output[k] for k in range(5)]
        self.predicted_output = self.output_layer_input #[relu(x) for x in self.output_layer_input]
        # exit()
        return self.predicted_output

    def backward(self, X, y, learning_rate):
        # Backward pass through the network

        # Calculate the loss
        loss = y - self.predicted_output # [y[i] - self.predicted_output[i] for i in range(5)]

        # Output layer to hidden layer
        output_error = loss.clone() #[loss[i] for i in range(5)]


        # input = [x,x,x,x,x]
        # weights input hidden
        # [
        #     [w,w,w,w,w], # in order to compute output 
        #     [w,w,w,w,w],
        #     [w,w,w,w,w],
        #     [w,w,w,w,w],
        #     [w,w,w,w,w],
        # ]

        # y = mx + b
        # derviative of y = mx + b with respect to m is x

        derivative = relu_derivative(self.hidden_layer_output)
        hidden_layer_error = (output_error * self.weights_hidden_output).sum(axis=-1) * derivative

        # hidden_layer_error = [sum(output_error[j] * self.weights_hidden_output[j][i] for j in range(5)) * relu_derivative(self.hidden_layer_output[i]) for i in range(5)]

        # Update weights and biases
        self.weights_hidden_output = self.weights_hidden_output + (self.hidden_layer_output * output_error * learning_rate)
        self.bias_output = self.bias_output + (output_error * learning_rate)

        self.weights_input_hidden = self.weights_input_hidden + (X * hidden_layer_error * learning_rate)
        self.bias_hidden = self.bias_hidden + (hidden_layer_error * learning_rate)

        # for i in range(5): # y = 2x + b
        #     self.weights_hidden_output[i] = [self.weights_hidden_output[i][j] + self.hidden_layer_output[j] * output_error[i] * learning_rate for j in range(5)]
        #     self.bias_output[i] += output_error[i] * learning_rate

        #     self.weights_input_hidden[i] = [self.weights_input_hidden[i][j] + X[j] * hidden_layer_error[i] * learning_rate for j in range(5)]
        #     self.bias_hidden[i] += hidden_layer_error[i] * learning_rate

    def train(self, X, y, epochs, learning_rate):
        # Train the neural network

        for epoch in range(epochs):
            # Forward pass
            predicted_output = self.forward(X)
            # print("input", X)
            # print("target output", y)
            # print("predicted output", predicted_output)
            # exit()

            # Backward pass
            self.backward(X, y, learning_rate)

            # Print the loss every 100 epochs
            if epoch % 1000 == 0:
                loss = sum((y[i] - predicted_output[i])**2 for i in range(5)) / 5
                print("input", X)
                print("target output", y)
                print("predicted output", predicted_output)
                print(f"Epoch {epoch}, Loss: {loss}")

# Example usage:
# Create a neural network
nn = NeuralNetwork()

# Generate some random input data and labels
X = torch.rand(5) # [random.uniform(0, 1) for _ in range(5)]
y = torch.rand(5) # [random.uniform(0, 1) for _ in range(5)]

# Train the neural network
nn.train(X, y, epochs=100000, learning_rate=0.0001)


# 10**-1 = .1
# 8e-1 
#     = 8 * 10^-2
#     = 8 * .01
#     = .08