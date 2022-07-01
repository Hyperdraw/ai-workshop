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

`def gradient_descent(X, Y, model, extra, loss, params, iterations=1, learning_rate=0.001)`

* `X`: 2D NumPy array - the input rows
* `Y`: 2D NumPy array - the output rows
* `model`: function - the model function
* `extra`: any type - extra model parameters
* `loss`: function - the loss function
* `params`: 1D NumPy array - the initial model parameters (will be changed)
* `iterations`: number - the number of iterations (default 1)
* `learning_rate`: number - the learning rate (default 0.001)

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

### `dense_classification`

Dense linear model that returns the *sign* of outputs (i.e. only 1s and -1s).
Only use for prediction. Use `dense` when training. Extra parameter is the number of outputs. 

### `logistic`

Logistic (sigmoid) activated linear model. Extra parameter is the number of outputs.

## Losses

* `mse` - Mean squared error
* `perceptron` - Perceptron