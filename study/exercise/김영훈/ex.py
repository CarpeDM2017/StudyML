import pickle

path = '../../session05.pkl'     # session05.pkl이 있는 경로를 넣어주세요
with open(path, 'rb') as f:
    data = pickle.load(f)
X = data['X']
Y = data['Y']
