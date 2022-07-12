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

# Multi-layer dense neural network with ReLU activation on all layers except the last and sigmoid activation on the last
# * widths - sequence of integers which are the number of units in each layer
def dense_relu_net_sigmoid(X, widths, params):
    return 1 / (1 + math.e ** (-dense_relu_net(X, widths, params)))

# Multi-layer dense neural network with ReLU activation on all layers except the last and softmax activation on the last
# * widths - sequence of integers which are the number of units in each layer
def dense_relu_net_softmax(X, widths, params):
    ez = math.e ** dense_relu_net(X, widths, params)
    return ez / np.repeat(ez.sum(axis=1), ez.shape[1]).reshape(ez.shape)

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

# Multi-layer dense neural network with sigmoid activation on all layers except the last and softmax activation on the last
# * widths - sequence of integers which are the number of units in each layer
def dense_sigmoid_net_softmax(X, widths, params):
    ez = math.e ** dense_sigmoid_net(X, widths, params)
    return ez / np.repeat(ez.sum(axis=1), ez.shape[1]).reshape(ez.shape)



# === Losses ===

# Mean squared error
def mse(X, Y):
    return ((Y - X) ** 2).mean()

# Perceptron
def perceptron(X, Y):
    return np.maximum(-Y * X, 0).mean()

# Logistic loss
def logloss(X, Y):
    return -((Y * np.log(np.maximum(10 ** -10, X)) + (1 - Y) * np.log(np.maximum(10 ** -10, 1 - X))).mean())

# Crossentropy loss
def crossentropy(X, Y):
    return (-np.log(X))[range(len(X)), np.argmax(Y, axis=1)].mean()



# === Algorithms ===

# Calculate the delta for a particular parameter by finding the derivative using the slope formula (numeric differentiation)
def gradient(X, Y, model, extra, loss, params_plus, params_minus, learning_rate, differentiation_ratio):
    return (loss(model(X, extra, params_plus), Y) - loss(model(X, extra, params_minus), Y)) / (learning_rate * differentiation_ratio * 2)

# Generic gradient descent function
# See the docs for descriptions of each argument
def gradient_descent(X, Y, model, extra, loss, params, iterations=1, learning_rate=0.001, differentiation_ratio=0.0001, pool_size=4, batch_size=32, print_period=1):
    # Make a multiprocessing pool to calculate deltas concurrently
    with multiprocessing.Pool(pool_size) as pool:
        # Eye matrix is used to create params_plus and params_minus
        eye = np.eye(len(params)) * differentiation_ratio
        # Calculate the number of batches
        batch_count = math.ceil(len(X) / batch_size)
        # Create V and S vectors for Adam optimization
        v = np.zeros_like(params)
        s = np.zeros_like(params)
        # B1, B2, and E are hyperparameters for Adam optimization. These are the recommended values.
        b1 = 0.9
        b2 = 0.999
        e = 10 ** -8

        # If learning_rate is a scalar convert it to a sequence
        if isinstance(learning_rate, float):
            learning_rate = [learning_rate]

        for _ in range(iterations):
            # Shuffle the the dataset and split it into batches
            i = np.array(range(len(X)))
            np.random.shuffle(i)
            X_batches = np.array_split(X[i], batch_count)
            Y_batches = np.array_split(Y[i], batch_count)
            # A matrix where every row is the params vector - used to create params_plus and params_minus
            params_repeated = np.tile(params, params.size).reshape((params.size, params.size))
            # A matrix where every row is the params vector except the corresponding parameter is increased slightly
            params_plus = params_repeated + eye
            # A matrix where every row is the params vector except the corresponding parameter is decreased slightly
            params_minus = params_repeated - eye
            # Arguments for the `gradient` function
            args = [(X, Y, model, extra, loss, params_plus[i], params_minus[i], learning_rate[_ % len(learning_rate)], differentiation_ratio) for i in range(len(params))]
            # Compute the gradient (`starmap` does multiple `gradient` calls concurrently)
            gradients = np.array(pool.starmap(gradient, args))
            # Update V and S for Adam optimization
            v = v * b1 + gradients * (1 - b1)
            s = s * b2 + gradients ** 2 * (1 - b2)
            v_hat = v / (1 - b1 ** (_ + 1))
            s_hat = s / (1 - b2 ** (_ + 1))


            for b in range(batch_count):
                # A matrix where every row is the params vector - used to create params_plus and params_minus
                params_repeated = np.tile(params, params.size).reshape((params.size, params.size))
                # A matrix where every row is the params vector except the corresponding parameter is increased slightly
                params_plus = params_repeated + eye
                # A matrix where every row is the params vector except the corresponding parameter is decreased slightly
                params_minus = params_repeated - eye
                # Arguments for the `gradient` function
                args = [(X_batches[b], Y_batches[b], model, extra, loss, params_plus[i], params_minus[i], learning_rate[_ % len(learning_rate)], differentiation_ratio) for i in range(len(params))]
                # Compute the gradient (`starmap` does multiple `gradient` calls concurrently)
                gradients = np.array(pool.starmap(gradient, args))
                # Change the params
                params -= learning_rate[_ % len(learning_rate)] * v_hat / (np.sqrt(s_hat) + e)

            if _ % print_period == 0:
                print('Iteration: ' + str(_) + ', Loss: ' + str(loss(model(X, extra, params), Y)))