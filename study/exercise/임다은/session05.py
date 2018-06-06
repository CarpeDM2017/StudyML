import pickle

path = '../../session05.pkl'
with open(path, 'rb') as f:
    data = pickle.load(f)
X = data['X']
Y = data['Y']
