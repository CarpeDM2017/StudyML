# mjh
# session04

import pickle
from sklearn.linear_model import SGDRegressor
import numpy as np

path = '../../session04.pkl'

with open(path, 'rb') as f:
    data = pickle.load(f)

X = data['X']
Y = data['Y']

model = SGDRegressor(loss='squared_loss', max_iter=5000, eta0=0.01, learning_rate='constant',
                     penalty=None)

model.fit(X, Y)
print(model.coef_)
print(model.score(X, Y))  # 0.969137312717


# for score of 1.0
new_X = np.zeros((X.shape[0], X.shape[1] + X.shape[1] ** 2)) # new_x has shape of (10, 20)
for i in range(4):
    for j in range(4):
        new_X[:, 4 + 4 * i + j] = X[:, i] * X[:, j]

new_model = SGDRegressor(loss='squared_loss', max_iter=10000, eta0=0.01, learning_rate='constant',
                         penalty=None)

model.fit(new_X, Y)
print(model.coef_)
print(model.score(new_X, Y))

