# coding: utf-8
import os
import sys
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer
from bayes_opt import BayesianOptimization


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
scale=1e3

def mnist(lam, drop, lr):
    network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100],
                                  output_size=10, activation='relu', weight_init_std='he', weight_decay_lambda=lam/scale, use_dropout=True, dropout_ration=drop/scale)
    trainer = Trainer(network, x_train, t_train, x_test, t_test,
                      epochs=10, mini_batch_size=100,
                      optimizer='adam', optimizer_param={'lr': lr/scale}, verbose=False)
    trainer.train()
    test_acc = trainer.test_acc_list[-1]
    del network
    return test_acc

gp_params={"alpha":1e-5*scale}
model = BayesianOptimization(mnist, {'lam': [0,scale], 'drop': [0,scale], 'lr': [0,scale]}, verbose=True)
model.maximize(init_points=1, n_iter=99, **gp_params)
print("Results: ", model.res['max']['max_val'])