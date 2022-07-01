# Brain.py by Hyperdraw
# A NumPy-powered one-file library for machine learning
# Includes common algorithms and a framework for making more

import numpy as np
import math



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



# === Losses ===

# Mean squared error
def mse(X, Y):
    return ((Y - X) ** 2).mean()

# Perceptron
def perceptron(X, Y):
    return np.maximum(-Y * X, 0).mean()



# === Algorithms ===

def gradient_descent(X, Y, model, extra, loss, params, iterations=1, learning_rate=0.001):
    eye = np.eye(len(params)) * learning_rate / 100
    param_changes = np.zeros_like(params)

    for _ in range(iterations):
        params_repeated = np.array([list(params)] * len(params))
        params_plus = params_repeated + eye
        params_minus = params_repeated - eye

        for i in range(len(params)):
            param_changes[i] = -learning_rate * (loss(model(X, extra, params_plus[i]), Y) - loss(model(X, extra, params_minus[i]), Y)) / (learning_rate / 50)

        params += param_changes

        if _ % 100 == 0:
            print('Iteration: ' + str(_) + ', Loss: ' + str(loss(model(X, extra, params), Y)))