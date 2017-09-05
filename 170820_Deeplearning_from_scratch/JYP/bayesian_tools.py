import os
import sys
import numpy as np
sys.path.append(os.pardir)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer

def train_network(dropout_ratio = 0, weight_decay = 0, learning_rate = 0.01, if_dropout=True,
                  x_train = None, t_train = None, x_test = None, t_test = None):
    """train individual networks with given parameters"""
    network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100],
                                  output_size=10, activation='sigmoid', weight_init_std='xavier',
                                  weight_decay_lambda=weight_decay, use_dropout=if_dropout, dropout_ration=dropout_ratio)
    trainer = Trainer(network, x_train, t_train, x_test, t_test, epochs=5, mini_batch_size=500,
                      optimizer='adam', optimizer_param={'lr': learning_rate}, verbose=False)
    trainer.train()
    test_acc = trainer.test_acc_list[-1]
    results = {"test_acc": test_acc, "dropout_ratio": dropout_ratio, "weight_decay": weight_decay, "learning_rate": learning_rate}

    del network
    return results

def gaussian_process(X, y, prediction_range):
    """ Return Results of gaussian regression accordng to given data X, y """
    # Kernel to use
    kernel = Matern(nu=3/5)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1)
    gp.fit(X, y)

    # Return Prediction
    y_pred, y_std = gp.predict(prediction_range, return_std=True)

    # Flatten both array into 1D array
    y_pred = y_pred.flatten()
    y_std = y_std.flatten()

    return y_pred, y_std

def next_parameter_by_ei(current_best, y_pred, y_std, x_choices):
    """
    Returns which parameter to explore next according to expected improvement(EI)
    current_best : best observation so far
    y_pred, y_std, x_choices : output of gaussian process
    """

    # Calculate expected improvement with 95% confidence interval
    improvement_max = (y_pred + 1.96 * y_std)
    improvement_max[improvement_max > 1] = 1
    expected_improvement = improvement_max - current_best

    # Select next choice
    max_index = expected_improvement.argmax()
    next_parameter = x_choices[max_index,:]
    return next_parameter

def create_3d_grid(axes):
    grid = []
    for i in axes :
        for j in axes :
            for k in axes :
                slice = [i,j,k]
                grid.append(slice)
    grid = np.array(grid)
    return grid