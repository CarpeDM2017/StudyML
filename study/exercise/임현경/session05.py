import pickle

path = 'study/session05.pkl'
with open(path, 'rb') as f:
    data = pickle.load(f)
X = data['X']
Y = data['Y']

from sklearn.linear_model import SGDClassifier
import numpy as np

model = SGDClassifier(loss='log', penalty='l1', max_iter=2000, eta0=0.05, learning_rate='constant', verbose=False)

model.fit(X,Y)
print(model.score(X,Y))
