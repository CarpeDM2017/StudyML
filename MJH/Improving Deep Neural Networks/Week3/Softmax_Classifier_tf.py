# MJH

import numpy as np
import tensorflow as tf
import sklearn.datasets as ds


def one_hot_matrix(labels, C):
    """
    Input:
        labels -- vector containing the labels
        C -- number of classes

    Output:
        one_hot -- labels in one hot matrix
    """

    one_hot_converter = tf.one_hot(labels, depth=C, axis=0)  # axis=0 means that vectors are columns of the matrix

    with tf.Session() as sess:
        one_hot = sess.run(one_hot_converter)

    return one_hot


def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, shape=(n_x, None), name="X")
    Y = tf.placeholder(tf.float32, shape=(n_y, None), name="Y")

    return X, Y


def initialize_parameters(layer_dims):
    L = len(layer_dims)

    parameters = {}
    for l in range(1, L):
        parameters['W'+str(l)] = tf.get_variable('W'+str(l), [layer_dims[l], layer_dims[l-1]],
                                                 initializer=tf.contrib.layers.xavier_initializer())
        parameters['b'+str(l)] = tf.get_variable('b'+str(l), [layer_dims[l], 1],
                                                 initializer=tf.zeros_initializer())

    return parameters


def forward_propagation(X, parameters, dropout=True, keep_prob=0.7):
    L = len(parameters) // 2

    A = X
    for l in range(1, L):
        A_prev = A
        Z = tf.matmul(parameters['W' + str(l)], A_prev) + parameters['b' + str(l)]
        A = tf.nn.relu(Z)
        if dropout:
            A = tf.nn.dropout(A, keep_prob=keep_prob)

    ZL = tf.matmul(parameters['W' + str(L)], A) + parameters['b' + str(L)]

    return ZL


def compute_cost(ZL, Y):

    logits = tf.transpose(ZL)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    return cost


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    np.random.seed(seed)  # To make your "random" minibatches the same as ours
    m = X.shape[1]  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = m // mini_batch_size
    # number of mini batches of size mini_batch_size in your partitionning (excluding last small mini batch)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, mini_batch_size * k: mini_batch_size * (k + 1)]
        mini_batch_Y = shuffled_Y[:, mini_batch_size * k: mini_batch_size * (k + 1)]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, mini_batch_size * (k + 1) + 1:]
        mini_batch_Y = shuffled_Y[:, mini_batch_size * (k + 1) + 1:]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def model(X_train, Y_train, X_test, Y_test, layer_dims=[100, 100, 40, 20], learning_rate=0.0001, num_epochs=4000,
          minibatch_size=64, dropout=True, keep_prob=0.7, print_cost=True):

    """
    Args:
        X_train:
        Y_train:
        X_test:
        Y_test:
        layer_dims: dimension of hidden layers (thus, excluding dimension of x and y)
        learning_rate:
        num_epochs:
        minibatch_size:
        dropout: whether to use dropout regularization
        keep_prob: keep probability when using dropout regularization
        print_cost:

    Returns:

    """

    tf.reset_default_graph()

    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    seed = 0
    costs = []

    X, Y = create_placeholders(n_x, n_y)

    parameters = initialize_parameters([n_x] + layer_dims + [n_y])

    ZL = forward_propagation(X, parameters, dropout=dropout, keep_prob=keep_prob)

    cost = compute_cost(ZL, Y)

    correct_prediction = tf.equal(tf.argmax(ZL), tf.argmax(Y))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))

    # for tensorboard
    tf.summary.scalar("cost", cost)
    tf.summary.scalar("accuracy", accuracy)
    for param in parameters:
        tf.summary.tensor_summary(param, parameters[param])

    summary_op = tf.summary.merge_all()

    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter(logdir="tensorlog")

        for epoch in range(num_epochs):

            epoch_cost = 0
            t = 0
            num_minibatches = int(m / minibatch_size) + 1

            seed += 1

            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                t += 1

                (minibatch_X, minibatch_Y) = minibatch

                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                epoch_cost += minibatch_cost / num_minibatches

            summary = sess.run(summary_op, feed_dict={X:X_train, Y:Y_train})
            writer.add_summary(summary, epoch)

            if print_cost and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))

            if print_cost and epoch % 5 == 0:
                costs.append(epoch_cost)

        parameters = sess.run(parameters)
        print("Parameters Saved!")

        print("Train Accuracy:", accuracy.eval({X: X_train, Y:Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters, costs


def test_model(test_ratio=0.3, layer_dims=[200, 100, 100, 40, 20], learning_rate=0.0001, num_epochs=5000,
               minibatch_size=128, dropout=True, keep_prob=0.7, print_cost=True):

    X, Y = ds.load_digits(return_X_y=True)
    m_total = len(X)

    permutation = list(np.random.permutation(m_total))
    X = X[permutation, :]
    Y = Y[permutation]
    # shuffle X and Y so that train set and test set are from approximately same distribution

    m_train = int(m_total * (1 - test_ratio))

    X_train = X[:m_train].T
    Y_train = one_hot_matrix(Y[:m_train], 10)

    X_test = X[m_train:].T
    Y_test = one_hot_matrix(Y[m_train:], 10)

    params, costs = model(X_train, Y_train, X_test, Y_test, layer_dims=layer_dims, learning_rate=learning_rate,
                          num_epochs=num_epochs, minibatch_size=minibatch_size, dropout=dropout, keep_prob=keep_prob,
                          print_cost=print_cost)

    import matplotlib.pyplot as plt

    plt.plot(costs)
    plt.show()


if __name__ == '__main__':
    test_model()