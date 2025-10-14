import numpy as np 

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # Initialize weights and biases
        self.learning_rate = learning_rate
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def forward(self, X):
        # Forward propagation
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2

    def compute_loss(self, Y, Y_hat):
        # Compute the loss using Mean Squared Error
        m = Y.shape[0]
        loss = np.sum((Y - Y_hat) ** 2) / (2 * m)
        return loss

    def backward(self, X, Y):
        # Backward propagation
        m = Y.shape[0]

        dZ2 = self.A2 - Y
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(self.A1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Update weights and biases
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def train(self, X, Y, epochs=1000):
        for epoch in range(epochs):
            Y_hat = self.forward(X)
            loss = self.compute_loss(Y, Y_hat)
            self.backward(X, Y)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')   


# Example usage:
if __name__ == "__main__":
    # Sample data (XOR problem) for demonstration
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    Y = np.array([[0],
                  [1],
                  [1],
                  [0]])
    nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=1, learning_rate=0.1)
    nn.train(X, Y, epochs=1000)