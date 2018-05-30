import pickle
import numpy as np

path = '../../session04.pkl'
with open(path, 'rb') as f:
    data = pickle.load(f)
X = data['X']
Y = data['Y']

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import SGDRegressor

X = np.random.randn(10,10)
Y = np.random.randn(10)

model = SGDRegressor(loss='squared_loss', penalty=None, max_iter=1000, eta0=0.1, learning_rate='constant', verbose=True)

model.fit(X,Y)
print(model.coef_)
print(model.intercept_)
print(model.score(X,Y))
