# Hyperdraw's Brain.py

Brain.py is a NumPy-powered one-file library for machine learning.
It includes common algorithms and a framework for making more.

## Features

* Generic gradient descent function
* Common models
* Common loss functions
* Ready to handle custom models and losses

## Usage With Google Colab

Go to [the code](brain.py), right click the "Raw" button on the side, and click "Save Link As." Save the file and upload it to Colab. You can now `import brain`.

## Models API

A model is a function which takes 3 parameters:

* `X`: 2D NumPy array: the dataset input rows
* `extra`: any type - extra model parameters (can be ignored)
* `params`: 1D NumPy array - The trainable model parameters

and returns a 1D NumPy array with the outputs.

## Loss API

A loss is a function which takes 2 parameters:

* `X`: 2D NumPy array - the predicted values
* `Y`: 2D NumPy array - the actual values

and returns the loss as a single number.

## Gradient Descent API

`def gradient_descent(X, Y, model, extra, loss, params, iterations=1, learning_rate=0.001, differentiation_ratio=0.0001, pool_size=4, batch_size=32, print_period=100)`

* `X`: 2D NumPy array - the input rows
* `Y`: 2D NumPy array - the output rows
* `model`: function - the model function
* `extra`: any type - extra model parameters
* `loss`: function - the loss function
* `params`: 1D NumPy array - the initial model parameters (will be changed)
* `iterations`: number - the number of iterations (default 1)
* `learning_rate`: number - the learning rate (default 0.001)
* `differentiation_ratio`: number - the proportion of the learning rate to use for numeric differentiation (the smaller the better) (default 0.0001)
* `pool_size`: number - the number of worker processes to use when computing gradients (default 4)
* `batch_size`: number - size of each batch (default 32)
* `print-period`: number - number of iterations to wait between log prints (default 1)

## Usage Example (Two-Input, Two-Output Linear Regression)

```python
import brain
import numpy as np
X = np.array([[1, 2], [3, 4], [5, 6]]) # The input rows (two inputs each)
Y = np.array([[2, 4], [4, 8], [6, 12]]) # The output rows (two outputs each)
params = np.zeros(6) # Number of parameters is equal to the number of outputs, times the number of inputs plus one
brain.gradient_descent(X, Y, brain.dense, 2, brain.mse, params, iterations=1000) # For dense model, the extra parameter is the number of outputs
print(brain.dense(np.array([[3, 4]]), 2, params)) # Make a prediction (again, the extra parameter should be the number of outputs)
```

## Models

### `dense`

A dense linear model. Extra parameter is the number of outputs.
Expects a number of parameters equal to the number of outputs, times the number of inputs plus one.

### `dense_classification`

Dense linear model that returns the *sign* of outputs (i.e. only 1s and -1s).
Only use for prediction. Use `dense` when training. Extra parameter is the number of outputs.
Expects a number of parameters equal to the number of outputs, times the number of inputs plus one.

### `logistic`

Logistic (sigmoid) activated linear model. Extra parameter is the number of outputs.
Expects a number of parameters equal to the number of outputs, times the number of inputs plus one.

### `dense_relu_net`

Multi-layer dense neural network with ReLU activation on all layers except the last. Extra parameter is a list of widths. The length of that list will be the number of layers. Each value is the number of units (outputs) of that layer. Therefore, the last value in the widths list is the number of outputs of the network.

Expects a number of parameters equal to the sum of the numbers of parameters required for each dense layer. (See `dense` layer for how to calculate the number of required parameters.)

For example, for widths `[16, 16, 4]` expecting 8 inputs, the first two layers need `16 * (8 + 1) = 144` parameters and the last layer needs `4 * (8 + 1) = 36` parameters, so in total the model needs `144 * 2 + 36 = 324` parameters.

# `dense_relu_net_sigmoid`

Multi-layer dense neural network with ReLU activation on all layers except the last and sigmoid activation on the last.
Same interface as `dense_relu_net`.

# `dense_relu_new_softmax`

Multi-layer dense neural network with ReLU activation on all layers except the last and softmax activation on the last.
Same interface as `dense_relu_net`.

# `dense_sigmoid_net`

Multi-layer dense neural network with sigmoid activation on all layers.
Same interface as `dense_relu_net`.

# `dense_sigmoid_net_softmax`

Multi-layer dense neural network with sigmoid activation on all layers except the last and softmax activation on the last.
Same interface as `dense_relu_net`.

## Losses

* `mse` - Mean squared error
* `perceptron` - Perceptron
* `logloss` - Logistic loss
* `crossentropy` - Crossentropy loss