# Brain.py by Hyperdraw
# A NumPy-powered one-file library for machine learning
# Includes common algorithms and a framework for making more

import numpy as np
import math
import multiprocessing



# === Models ===

# Dense linear model
# * units: the number of outputs
def dense(X, units, params):
    # Reshape the parameters from a flat array to a matrix of weights and biases
    reshaped = params.reshape(units, X.shape[1] + 1)
    # Apply the weights and biases matrix
    return np.dot(X, reshaped[:, :-1].transpose()) + reshaped[:, -1]

# Dense linear classification model
# Useful for predicting classification, but do not do gradient descent with this - use dense for gradient descent instead
# * units: the number of outputs
def dense_classification(X, units, params):
    # Calculate the output of `dense` and return the signs of the outputs
    return np.sign(dense(X, units, params))

# Dense logistic (sigmoid) model
# * units: the number of outputs
def logistic(X, units, params):
    # Calculate the output of `dense` and apply sigmoid activation
    return 1 / (1 + math.e ** (-dense(X, units, params)))

# Multi-layer dense neural network with ReLU activation on all layers except the last
# * widths - sequence of integers which are the number of units in each layer
def dense_relu_net(X, widths, params):
    # Keep track of the starting index of the parameters for the current layer
    i = 0
    # Keep track of the current values
    values = X

    # For each layer...
    for j, width in enumerate(widths):
        # Determine the number of parameters this layer needs
        layer_params_size = width * (values.shape[1] + 1)
        # Calculate the output of `dense`
        values = dense(values, width, params[i:i + layer_params_size])

        if j < len(widths) - 1:
            # If this is not the last layer, apply ReLU activation
            values = np.maximum(0, values)


        # Move to the next parameters for the next layer
        i += layer_params_size

    return values

# Multi-layer dense neural network with sigmoid activation on all layers
# * widths - sequence of integers which are the number of units in each layer
def dense_sigmoid_net(X, widths, params):
    # Keep track of the starting index of the parameters for the current layer
    i = 0
    # Keep track of the current values
    values = X

    # For each layer...
    for j, width in enumerate(widths):
        # Determine the number of parameters this layer needs
        layer_params_size = width * (values.shape[1] + 1)
        # Calculate the output of `dense` and apply sigmoid activation
        values = 1 / (1 + math.e ** (-dense(values, width, params[i:i + layer_params_size])))
        # Move to the next parameters for the next layer
        i += layer_params_size

    return values



# === Losses ===

# Mean squared error
def mse(X, Y):
    return ((Y - X) ** 2).mean()

# Perceptron
def perceptron(X, Y):
    return np.maximum(-Y * X, 0).mean()

# Logistic loss
def logloss(X, Y):
    return -((Y * np.log(X) + (1 - Y) * np.log(1 - X)).mean())



# === Algorithms ===

# Calculate the delta for a particular parameter by finding the derivative using the slope formula (numeric differentiation)
def gradient(X, Y, model, extra, loss, params_plus, params_minus, learning_rate, differentiation_ratio):
    return -learning_rate * (loss(model(X, extra, params_plus), Y) - loss(model(X, extra, params_minus), Y)) / (learning_rate * differentiation_ratio * 2)

# Generic gradient descent function
# See the docs for descriptions of each argument
def gradient_descent(X, Y, model, extra, loss, params, iterations=1, learning_rate=0.001, differentiation_ratio=0.0001, pool_size=4):
    # Make a multiprocessing pool to calculate deltas concurrently
    with multiprocessing.Pool(pool_size) as pool:
        # Eye matrix is used to create params_plus and params_minus
        eye = np.eye(len(params)) * learning_rate * differentiation_ratio

        for _ in range(iterations):
            # A matrix where every row is the params vector - used to create params_plus and params_minus
            params_repeated = np.tile(params, params.size).reshape((params.size, params.size))
            # A matrix where every row is the params vector except the corresponding parameter is increased slightly
            params_plus = params_repeated + eye
            # A matrix where every row is the params vector except the corresponding parameter is decreased slightly
            params_minus = params_repeated - eye
            # Arguments for the `gradient` function
            args = [(X, Y, model, extra, loss, params_plus[i], params_minus[i], learning_rate, differentiation_ratio) for i in range(len(params))]
            # Change the params by the deltas calculated by the `gradient` function (`starmap` does multiple `gradient` calls concurrently)
            params += np.array(pool.starmap(gradient, args))

            if _ % 100 == 0:
                print('Iteration: ' + str(_) + ', Loss: ' + str(loss(model(X, extra, params), Y)))