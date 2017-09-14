import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0.0

    return w, b


def propagate(w, b, X, Y):
    """Compute cost, A = sigmoid(w.T * X + b), and gradient

    Arguments:
    w -- weight. numpy array with shape (dim, 1)
    b -- bias. real number
    X -- training X. numpy array with shape (dim, m)
    Y -- training Y (true label). numpy array with shape (1, m)

    * dim is number of features excluding 1s
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_traing - Y_train)) * 100))
    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient with respect to w
    db -- gradient with respect to b

    """
    m = X.shape[1]
    # m is number of training sets

    # Forward Propagation
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -1 / m * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A))

    # Backward Propagation
    dw = 1 / m * np.dot(X, (A - Y).T)  # A-Y is dz
    db = 1 / m * np.sum(A - Y)

    cost = np.squeeze(cost)

    grads = {'dw': dw,
             'db': db}

    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):

    """

    Optimizes w and b by running gradient descent num_iteration times

    :param w: weights. numpy array of shape (dim, 1)
    :param b: bias, real number
    :param X: training X, numpy array of shape (dim, m)
    :param Y: training Y (true label), numpy array of shape (1, m)
    :param num_iterations: number of times gradient descent algorithm is ran
    :param learning_rate: learning rate of gradient descent update rule
    :param print_cost: whether to print loss every 100 steps
    :return: params - dictionary containing w and b. grads - dictionary containing gradients of weights and bias at the end of optimizztion. costs - list of all costs at every 100 steps

    """

    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads['dw']
        db = grads['db']

        # update using dw and db

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)

            if print_cost:
                print("Cost after iteration %d : %f" % (i, cost))

    params = {'w': w,
              'b': b}

    grads = {'dw': dw,
             'db': db}

    return params, grads, cost


def predict(w, b, X):
    """

    :param w: weights. numpy array of shape (dim, 1)
    :param b: bias. scalar/real number
    :param X: numpy array of shape (dim, m)
    :return: prediction of Y (a.k.a Y hat)
    """

    m = X.shape[1]
    Y_prediction = np.zeros((1, m))

    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        if A[0, i] >= 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0

    assert(Y_prediction.shape == (1, m))

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=True):
    """
    Builds Logistic Regression Model and test model using X_test and Y_test

    :param X_train: training set. numpy array of shape (dim, m)
    :param Y_train: training labels. numpy array of shape (1, m)
    :param X_test: testing set.
    :param Y_test: testing labels
    :param num_iterations: number of iterations to optimize
    :param learning_rate: learning rate of optimization process
    :param print_cost: whether to print cost at every 100 steps
    :return: d - dictionary containing information about the model(w, b, costs, etc)
    """

    w, b = initialize_with_zeros(X_train.shape[0])

    params, grads, cost = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    w = params['w']
    b = params['b']

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {'cost': cost,
         'Y_prediction_test': Y_prediction_test,
         'Y_prediction_train': Y_prediction_train,
         'w': w,
         'b': b,
         'learning_rate': learning_rate,
         'num_iterations': num_iterations}

    return d


