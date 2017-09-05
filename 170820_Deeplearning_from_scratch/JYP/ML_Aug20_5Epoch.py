# coding: utf-8
import os
import sys
sys.path.append(os.pardir)
import numpy as np
import pandas as pd
from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer
import progressbar

# Load mnist Data
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# Set Global Variables and hyper parameters
use_dropout = True
dropout_ratio = np.linspace(0, 0.2, 5)
weight_decay = np.geomspace(0.001, 0.2, num=5)
learning_rate = np.geomspace(0.001, 0.2, num=5)
best_hp = []

# Training begins
with progressbar.ProgressBar(max_value=125) as bar:
    for i in range(0, len(dropout_ratio)):
        for j in range(0, len(weight_decay)):
            for k in range(0, len(learning_rate)):
                network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100],
                                              output_size=10, activation='sigmoid', weight_init_std='xavier',
                                              weight_decay_lambda=weight_decay[j],
                                              use_dropout=use_dropout, dropout_ration=dropout_ratio[i])
                trainer = Trainer(network, x_train, t_train, x_test, t_test,
                                  epochs=5, mini_batch_size=500,
                                  optimizer='adam', optimizer_param={'lr': learning_rate[k]}, verbose=False)
                trainer.train()
                test_acc = trainer.test_acc_list[-1]
                best_hp.append(
                    {"test_acc": test_acc, "dropout_ratio": dropout_ratio[i], "weight_decay": weight_decay[j],
                     "learning_rate": learning_rate[k]})
                del network
                bar.update(25 * i + 5 * j + k)

# Exporting Results
# Sort Values by descending order
best_hp.sort(key=lambda x : x["test_acc"], reverse=True)

# Export Training Results to CSV format
df = pd.DataFrame(best_hp)
df = df[["test_acc", "dropout_ratio", "weight_decay", "learning_rate"]]
df.to_csv("Training Results.csv", index_label = "Accuracy Order")