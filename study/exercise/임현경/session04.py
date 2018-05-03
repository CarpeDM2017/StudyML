from sklearn.linear_model import SGDRegressor
import numpy as np
import pickle

path = 'study/session04.pkl'
with open(path, 'rb') as f:
   data = pickle.load(f)
X = data['X']
Y = data['Y']

model = SGDRegressor(loss='squared_loss' , penalty='l2' , max_iter=5000, eta0=0.1, learning_rate='constant', verbose=False)
model.fit(X,Y)
print(model.coef_)
print(model.score(X,Y))
