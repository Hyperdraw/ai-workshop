# Brain.py by Hyperdraw
# A NumPy-powered one-file library for machine learning
# Includes common algorithms and a framework for making more

import numpy as np
import math
import multiprocessing



# === Models ===

# Dense linear model
def dense(X, units, params):
    reshaped = params.reshape(units, X.shape[1] + 1)
    return np.dot(X, reshaped[:, :-1].transpose()) + reshaped[:, -1]

# Dense linear classification model
def dense_classification(X, units, params):
    return np.sign(dense(X, units, params))

# Dense logistic (sigmoid) model
def logistic(X, units, params):
    return 1 / (1 + math.e ** dense(X, units, params))

# Multi-layer dense neural network with ReLU activation on all layers except the last
def dense_relu_net(X, widths, params):
    i = 0
    values = X

    for j, width in enumerate(widths):
        layer_params_size = width * (values.shape[1] + 1)
        values = dense(values, width, params[i:i + layer_params_size])

        if j < len(widths) - 1:
            values = np.maximum(0, values)

        i += layer_params_size

    return values



# === Losses ===

# Mean squared error
def mse(X, Y):
    return ((Y - X) ** 2).mean()

# Perceptron
def perceptron(X, Y):
    return np.maximum(-Y * X, 0).mean()

def logloss(X, Y):
    return -((Y * np.log(X) + (1 - Y) * np.log(1 - X)).mean())



# === Algorithms ===

def gradient(X, Y, model, extra, loss, params_plus, params_minus, learning_rate, differentiation_ratio):
    return -learning_rate * (loss(model(X, extra, params_plus), Y) - loss(model(X, extra, params_minus), Y)) / (learning_rate * differentiation_ratio * 2)

def gradient_descent(X, Y, model, extra, loss, params, iterations=1, learning_rate=0.001, differentiation_ratio=0.0001, pool_size=4):
    with multiprocessing.Pool(pool_size) as pool:
        eye = np.eye(len(params)) * learning_rate * differentiation_ratio

        for _ in range(iterations):
            params_repeated = np.array([list(params)] * len(params))
            params_plus = params_repeated + eye
            params_minus = params_repeated - eye
            args = [(X, Y, model, extra, loss, params_plus[i], params_minus[i], learning_rate, differentiation_ratio) for i in range(len(params))]
            params += np.array(pool.starmap(gradient, args))

            if _ % 100 == 0:
                print('Iteration: ' + str(_) + ', Loss: ' + str(loss(model(X, extra, params), Y)))