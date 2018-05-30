# MJH
# session 05 exercise

import pickle
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

path = '../../session05.pkl'

with open(path, 'rb') as f:
    data = pickle.load(f)

X = data['X']
Y = data['Y']

print(X.shape)  # (569, 30)
print(Y.shape)  # (569, )

kernels = ['linear', 'poly', 'rbf']
tree_criterions = ['gini', 'entropy']

results = {}

# SGDClassifier
model = SGDClassifier(loss='log', penalty=None, max_iter=10000, eta0=0.1, learning_rate='constant', verbose=False)
model.fit(X, Y)
results['SGDClassifier'] = model.score(X, Y)

# SVC
for kernel in kernels:
    model = SVC(C=1.0, kernel=kernel, max_iter=1000, verbose=False)
    model.fit(X, Y)
    results['SVC_'+kernel] = model.score(X, Y)

# DecisionTreeClassifier
for criterion in tree_criterions:
    model = DecisionTreeClassifier(criterion=criterion, splitter='best', max_depth=20, max_leaf_nodes=200)
    model.fit(X, Y)
    results['DecisionTreeClassifier_'+criterion] = model.score(X, Y)

# sort and print
print()
print('Results')
for key, score in reversed(sorted(results.items(), key=lambda x: x[1])):
    print(key, score)

# Results
# DecisionTreeClassifier_gini 1.0
# DecisionTreeClassifier_entropy 1.0
# SVC_rbf 1.0
# SGDClassifier 0.9420035149384886
# SVC_poly 0.35676625659050965
# SVC_linear 0.23550087873462214
