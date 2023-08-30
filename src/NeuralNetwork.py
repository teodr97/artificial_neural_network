import numpy as np

class Layer:
    def __init__(self, input_size, output_size, activation_function):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_function = activation_function

        # Initialize weights and biases with Xavier initialization
        "Weights is a matrix of size (output_size, input_size)"
        self.weights = np.random.normal(0, np.sqrt(2 / output_size + input_size), (output_size, input_size))
        "Biases is a vector of size (output_size, 1)"
        self.biases = np.zeros((output_size, 1))

    "Forward pass of the layer"
    def forward(self, x):
        # Implement forward pass
        "X is a vector of size (input_size, 1)"
        self.x = x
        "Z is a vector of size (output_size, 1)"
 
        self.z = self.weights @ x + self.biases
        "A is a vector of size (output_size, 1)"
        self.a = self.activation_function_choice(self.z)
        return self.a

    def activation_function_choice(self, z):
        # Implement activation function
        if self.activation_function == "sigmoid":
            return 1/(1+np.exp(-z))
        elif self.activation_function == "tanh":
            return np.tanh(z)
        elif self.activation_function == "relu":
            return np.maximum(0, z)
        elif self.activation_function == "leaky_relu":
            return np.maximum(0.01*z, z)
        elif self.activation_function == "softmax":
            return self.softmax(z)
        else:
            raise ValueError("Activation function not supported")

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def activation_function_derivative(self, z):
        # Implement derivative of activation function
        if self.activation_function == "sigmoid":
            return self.activation_function_choice(z) * (1 - self.activation_function_choice(z))
        elif self.activation_function == "tanh":
            return 1 - np.power(self.activation_function_choice(z), 2)
        elif self.activation_function == "relu":
            return np.where(z > 0, 1, 0)
        elif self.activation_function == "leaky_relu":
            return np.where(z > 0, 1, 0.01)
        elif self.activation_function == "softmax":
            return self.activation_function_choice(z) * (1 - self.activation_function_choice(z))
        else:
            raise ValueError("Activation function not supported")


    "Backward pass of the output layer"
    def backward_output(self, y_hat, y, learning_rate):
        # Implement backward pass of the output layer (with softmax)
        "y_hat is a vector of size (output_size, 1)"
        "loss is a vector of size (output_size, 1)"

        if self.activation_function == "softmax":
            loss = y_hat - y
        else:
            loss = (y_hat - y) * self.activation_function_derivative(self.z) * 2

        delta = loss

        # Update biases and weights
        self.biases -= learning_rate * delta
        self.weights -= learning_rate * np.dot(delta, self.x.T)

        # Calculate delta for the previous layer
        delta_prev = np.dot(self.weights.T, delta)
        return delta_prev

    "Backward pass of the hidden layers"
    def backward_hidden(self, delta, learning_rate):

        loss = delta * self.activation_function_derivative(self.z)
        self.biases -= loss * learning_rate
        self.weights -= loss @ self.x.T * learning_rate
        delta = np.dot(self.weights.T, loss)
        return delta


class ANN:
    def __init__(self, layers):
        self.layers = layers # List of layers
        self.output_size = layers[-1].output_size

    def forward(self, x):
        # Implement forward pass of the network
        for layer in self.layers:
            x = layer.forward(x)
        return x

    "y is a vector of size (elements_size, output_size)"
    def train(self, x, y, epochs, learning_rate):
        accuracy = []
        error = []
        for epoch in range(epochs):
            # Implement training of the network
            accuracy_cur_epoch = 0
            error_cur_epoch = 0
            for input in range(len(x)):
                # Implement forward pass
                "x[input] is a vector of size (input_size, 1)"
                "y_hat is a vector of size (output_size, 1)"
                y_hat = self.forward(x[input].reshape(-1, 1))
                # Implement backward pass
                "y[input] is a vector of size (output_size, 1)"
                self.backward(y_hat, y[input].reshape(-1, 1), learning_rate)

                if(y_hat.argmax() != y[input].argmax()):
                    error_cur_epoch += 1
                else:
                    accuracy_cur_epoch += 1
            accuracy.append(accuracy_cur_epoch)
            error.append(error_cur_epoch/len(x))
        return error

    def train_with_validation(self, x, y, x_val, y_val, epochs, learning_rate):
        accuracy = []
        error_train = []
        error_val = []
        for epoch in range(epochs):
            # Implement training of the network
            accuracy_cur_epoch = 0
            error_cur_epoch = 0
            for input in range(len(x)):
                # Implement forward pass
                "x[input] is a vector of size (input_size, 1)"
                "y_hat is a vector of size (output_size, 1)"
                y_hat = self.forward(x[input].reshape(-1, 1))
                # Implement backward pass
                "y[input] is a vector of size (output_size, 1)"
                self.backward(y_hat, y[input].reshape(-1, 1), learning_rate)
                if(y_hat.argmax() != y[input].argmax()):
                    error_cur_epoch += 1
                else:
                    accuracy_cur_epoch += 1
            error_train.append(error_cur_epoch/len(x))

            error_cur_epoch = 0
            for input in range(len(x_val)):
                y_hat_val = self.forward(x_val[input].reshape(-1, 1))
                if(y_hat_val.argmax() != y_val[input].argmax()):
                    error_cur_epoch += 1
                else:
                    accuracy_cur_epoch += 1
            accuracy.append(accuracy_cur_epoch)
            error_val.append(error_cur_epoch/len(x_val))
        return error_train, error_val


    def predict(self, x):
        y_hats = []
        for i in range(len(x)):
            x_i = x[i].reshape(-1, 1)
            y_hat = self.forward(x_i)
            y_hats.append(y_hat)
        return np.array(y_hats)


    def backward(self, y_hat, y, learning_rate):
        out_layer = self.layers[-1]
        # Implement backward pass of the output layer
        delta = out_layer.backward_output(y_hat, y, learning_rate)
        # Implement backward pass of the hidden layers
        for layer in reversed(self.layers[:-1]):
            delta = layer.backward_hidden(delta, learning_rate)


class Perceptron:
    def __init__(self, size):
        self.weights = np.random.rand(size) - 0.5
        self.bias = np.random.uniform(-0.5, 0.5)

    def activation_function(self, x):
        if x >= 0:
            return 1
        else:
            return 0

    def predict(self, inputs):
        return self.activation_function(np.dot(inputs, self.weights) + self.bias)

    def train(self, training_inputs, labels, epochs, learning_rate):
        accuracy = []
        for epoch in range(epochs):
            accuracy_rate = 0
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                self.weights += error * inputs * learning_rate
                self.bias += error * learning_rate
                if error != 0:
                    accuracy_rate += 1
            accuracy.append(accuracy_rate/len(training_inputs))

        return accuracy

