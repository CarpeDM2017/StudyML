# coding: utf-8
import os
import sys
sys.path.append(os.pardir)
import numpy as np
import pandas as pd
from dataset.mnist import load_mnist
import progressbar
import bayesian_tools as tools

# Load mnist Data
(xTrain, tTrain), (xTest, tTest) = load_mnist(normalize=True)

# Read Previously Trained Data for gaussian process regression
train_history = pd.read_csv("Training Results.csv")
x_gaussian = train_history.as_matrix(columns = ["dropout_ratio","weight_decay","learning_rate"])
y_gaussian = train_history.as_matrix(columns = ["test_acc"])

# Start Optimization
iteration_count = 30
axes = np.linspace(0, 0.2, num=15)
#todo : There needs to be more effective way to explore through grids
search_grid = tools.create_3d_grid(axes)
training_history = []

with progressbar.ProgressBar(max_value=iteration_count) as bar:
    for i in np.arange(iteration_count):
        # Update best accuracy for each iteration
        best_accuracy = np.argmax(y_gaussian)
        y_pred, y_std = tools.gaussian_process(x_gaussian, y_gaussian, search_grid)
        next_parameter = tools.next_parameter_by_ei(best_accuracy, y_pred, y_std, search_grid)

        # Decide which point of parameter to explore next
        (dr, wd, lr) = next_parameter
        train_result = tools.train_network(dropout_ratio=dr, weight_decay=wd, learning_rate=lr,
                                  x_train = xTrain, t_train = tTrain, x_test = xTest, t_test = tTest)
        acc = train_result["test_acc"]

        # Update prior from recent results
        x_gaussian = np.vstack((x_gaussian, [dr, wd, lr]))
        y_gaussian = np.vstack((y_gaussian, [acc]))

        # Export Training history
        training_history.append({"count" : i, "test_acc": acc, "dropout_ratio" : dr,
                                 "weight_decay" : wd, "learning_rate" : lr})
        bar.update(i+1)

# Export Updated result as CSV
updated_result = np.hstack((y_gaussian, x_gaussian))
df_updated_result = pd.DataFrame(data = updated_result,
                                 columns =["test_acc", "dropout_ratio", "weight_decay", "learning_rate"])
df_updated_result.sort_values(by = "test_acc", ascending = False, inplace=True)
df_updated_result.to_csv("Updated Results_Bayesian.csv", index = False)

# Export Training History in CSV
df_train_history = pd.DataFrame(training_history)
df_train_history.to_csv("Train History_Bayesian.csv", index = False)