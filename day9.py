import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,1])

weights = np.random.rand(2)
bias = np.random.rand(1)
learning_rate = 0.1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

for epoch in range(1000):
    for i in range(len(X)):
        z = np.dot(X[i],weights) + bias
        prediction = sigmoid(z)

        error = y[i] - prediction

        weights += learning_rate * error * X[i]
        bias += learning_rate * error

for i in range(len(X)):
    z = np.dot(X[i], weights) + bias
    prediction = sigmoid(z)
    print(f"Input {X[i]}, Predicted: {prediction[0]:.4f}, Actual: {y[i]}")
