import pickle
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

path = '../../session05.pkl'     # session05.pkl이 있는 경로를 넣어주세요
with open(path, 'rb') as f:
    data = pickle.load(f)
X = data['X']  # 569 by 30 matrix
Y = data['Y']

Results = []  # A result containing list, [Logit, MultiLogit, SVC, Tree]
"""
SGDClassifier의 입력값

loss : 오차함수의 형태 (hinge, log, squared_loss 등)
penalty : 제약식의 형태 (l2, l1, elasticnet)
max_iter : 경사하강법의 반복횟수
eta0 : 학습률의 초깃값
learning_rate : 학습률의 변화방법 (constant, invscaling 등)
verbose : 학습과정의 출력여부

"""
model = SGDClassifier(
    loss='log', penalty=None,
    max_iter=1000000, eta0=0.1, learning_rate='invscaling',  # maxiter 10000정도로는 수렴하기 전에 끝나는 듯하다.
    verbose=False)

model.fit(X, Y)                      # 로지스틱 모형을 학습시킵니다.
print(model.score(X, Y))             # 모형의 평균 정확도 출력
Results.append(model.score(X, Y))
poly = PolynomialFeatures(2, include_bias=False)
X_poly = poly.fit_transform(X)      # 2차항과 교차항을 추가합니다.

model.fit(X_poly, Y)
print(model.score(X_poly, Y))
Results.append(model.score(X_poly, Y))
"""
SVC의 입력값

C : 정규항의 비례상수
kernel : X변수에 대해 적용할 phi 함수의 형태 (linear, poly, rbf, sigmoid)
degree : kernel=poly인 경우 다항식의 차수
gamma : phi 함수의 모수 (kernel의 형태에 따라 다릅니다)
max_iter : libsvm 알고리즘의 반복횟수
verbose : 학습과정의 출력여부

"""

model2 = SVC(C=1.0, kernel='rbf', gamma=0.10000000000000001, max_iter=1000000, verbose=False)

model2.fit(X, Y)
print(model2.score(X, Y))     # 모형의 평균 정확도를 출력합니다.
Results.append(model2.score(X, Y))

"""
DecisionTreeClassifier의 입력값

criterion : 학습의 기준이 되는 오차함수 (gini, entropy)
splitter : 하위 노드로의 분기 기준 (best, random)
max_depth : 나무의 최대 길이
max_leaf_nodes : 말단 노드의 최대 개수

"""
model3 = DecisionTreeClassifier(
    criterion='gini',
    splitter='best',
    max_depth=10,
    max_leaf_nodes=100
    )

model3.fit(X, Y)
print(model3.score(X, Y))