# by MJH

# implemented L2 regularization to Deep Neural Network model

import numpy as np
import random


def sigmoid(z):
    return 1 / (1 + np.exp(-z)), z


def relu(z):
    return np.maximum(z, 0), z


def sigmoid_backward(dA, activation_cache):
    # sigmoid'(z) = sigmoid *(1 - sigmoid)
    # activation_cache is Z
    A = sigmoid(activation_cache)[0]
    dZ = dA * A * (1-A)
    return dZ


def relu_backward(dA, activation_cache):
    # relu'(z) = 1 if z > 0, 0 otherwise
    # activation_cache is Z
    dZ = dA * (activation_cache > 0)
    return dZ


def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
                  ex. layer_dims=[5,5,3,1]

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        # implemented 'He initialization'
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2. / layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """

    Z = np.dot(W, A) + b
    # Z = W * A + b

    cache = (A, W, b)
    # this is 'linear_cache'

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """

    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network (# of parameters is L * 2(W amd b)

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
        caches.append(cache)

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
    caches.append(cache)

    assert (AL.shape == (1, X.shape[1]))   # (1, m)

    return AL, caches


def compute_cost(AL, Y, parameters, m, lambd=0.1, regularization=True):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """

    m = Y.shape[1]

    # Compute loss from aL and y.
    cost = - 1 / m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))

    if regularization:
        for l in range(1, len(parameters) // 2 + 1):
            cost += lambd / (2 * m) * np.sum(np.square(parameters['W' + str(l)]))

    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert (cost.shape == ())

    return cost


def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1 / m * np.dot(dZ, A_prev.T)
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches, parameters, regularization=True, lambd=0.1):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    # Initializing the backpropagation
    dAL = - np.divide(Y, AL) + np.divide(1 - Y, 1 - AL)

    # Lth layer (SIGMOID -> LINEAR) gradients.
    # Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,
                                                                                                  current_cache,
                                                                                                  'sigmoid')

    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 2)], caches".
        # Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, 'relu')
        grads["dA" + str(l + 1)] = dA_prev_temp
        if regularization:
            grads['dW' + str(l+1)] = dW_temp + lambd / m * parameters['W' + str(l+1)]
        else:
            grads["dW" + str(l+1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """

    L = len(parameters) // 2  # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l + 1)] -= learning_rate * grads['dW' + str(l + 1)]
        parameters["b" + str(l + 1)] -= learning_rate * grads['db' + str(l + 1)]

    return parameters


def L_layer_model_regularized(X, Y, layer_dims, learning_rate=0.0075, num_iterations=3000, regularization=True, lambd=0.1,
                  print_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layer_dims -- list containing the input size and each layer size, of length (number of layers + 1).
                  ex. if n0 = 2, n1 = 3, n2 = 1, thene layer_dims = [2,3,1]
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    m = X.shape[1]
    costs = []  # keep track of cost

    # Parameters initialization.
    parameters = initialize_parameters_deep(layer_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)

        # Compute cost.
        cost = compute_cost(AL, Y, parameters, m, lambd=lambd, regularization=regularization)

        if np.isnan(cost):
            break

        # Backward propagation.
        grads = L_model_backward(AL, Y, caches, parameters, regularization=regularization, lambd=lambd)

        # Update parameters.i
        parameters = update_parameters(parameters, grads, learning_rate )

        # Print the cost every 100 training example
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if i % 100 == 0:
            costs.append(cost)

    return parameters, costs


def predict(parameters, X):
    AL, cache = L_model_forward(X, parameters)
    prediction = (AL > 0.5)
    return prediction


def test_L_layer_model(m_train=3000, m_test=500, layer_dims=[2, 5, 1], learning_rate=0.01, num_iterations=20000,
                       regularization=True, lambd=0.1, print_cost=True):

    # prepare training and test sets
    X_train = np.zeros((2, m_train))
    Y_train = np.zeros((1, m_train))

    for i in range(m_train):
        x1 = random.uniform(-5, 5)
        x2 = random.uniform(-5, 5)

        X_train[0, i] = x1
        X_train[1, i] = x2

        # optimal decision boundary is circle
        if x1 ** 2 + x2 ** 2 > 15:
            Y_train[0, i] = 1
        else:
            Y_train[0, i] = 0

    X_test = np.zeros((2, m_test))
    Y_test = np.zeros((1, m_test))

    for i in range(m_test):
        x1 = random.uniform(-5, 5)
        x2 = random.uniform(-5, 5)

        X_test[0, i] = x1
        X_test[1, i] = x2

        if x1 ** 2 + x2 ** 2 > 15:
            Y_test[0, i] = 1
        else:
            Y_test[0, i] = 0

    # use nn_model to get parameters
    params, costs = L_layer_model_regularized(X_train, Y_train, layer_dims=layer_dims, num_iterations=num_iterations,
                                              regularization=regularization, lambd=lambd,
                                              learning_rate=learning_rate, print_cost=print_cost)
    # if n0 is small (i.e. data is not complex) like this example, shallow neural network works better
    # since it learns much faster

    print("Parameters :", params)

    prediction_train = predict(params, X_train)
    prediction_test = predict(params, X_test)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(prediction_test - Y_test)) * 100))

    import matplotlib.pyplot as plt
    plt.plot(costs)
    plt.show()

    return params


if __name__ == '__main__':
    test_L_layer_model()