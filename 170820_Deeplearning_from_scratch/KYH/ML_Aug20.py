# coding: utf-8
import os
import sys
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer
import progressbar

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

use_dropout = True
dropout_ratio = np.linspace(0,0.5,10)
weight_decay = np.linspace(0,0.5,10)
lr = np.linspace(0.01,0.5,10)

best_hp = []

# ====================================================
with progressbar.ProgressBar(max_value=1000) as bar:
    for i in range(0,len(dropout_ratio)):
        for j in range(0,len(weight_decay)):
            for k in range(0, len(lr)):
                network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100],
                                              output_size=10, activation='relu', weight_init_std='relu', weight_decay_lambda=weight_decay[j], use_dropout=use_dropout, dropout_ration=dropout_ratio[i])
                trainer = Trainer(network, x_train, t_train, x_test, t_test,
                                  epochs=1, mini_batch_size=100,
                                  optimizer='adam', optimizer_param={'lr': lr[k]}, verbose=False)
                trainer.train()
                test_acc = trainer.test_acc_list[-1]
                
                if (len(best_hp)<100)or(test_acc>best_hp[-1][0]):
                    best_hp.append((test_acc, dropout_ratio[i], weight_decay[j], lr[k]))
                    best_hp.sort(key=lambda x: x[0], reverse=True)
                    best_hp = best_hp[:100]
                del network
                bar.update(100*i+10*j+k)


x = [i[1] for i in best_hp]
y = [i[2] for i in best_hp]
z = [i[3] for i in best_hp]
fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlim(-0.1,0.6)
ax.set_ylim(-0.1,0.6)
ax.set_zlim(-0.1,0.6)
ax.scatter(x,y,z)
plt.show()