from sklearn.neural_network import MLPClassifier, MLPRegressor
import numpy as np

# X 데이터는 정규분포, Y 데이터는 베르누이분포로 추출하였습니다.
X = np.random.randn(10,10)
Y = np.random.binomial(1,0.5,10)

"""
MLPClassifier, MLPRegressor의 입력값

hidden_layer_sizes : 중간층의 뉴런 개수. 아래와 같이 튜플 형태로 입력해야 합니다.
                     e.g. (100,200,100)
                     이렇게 입력하면 중간층의 개수는 3개, 뉴런 개수는 순서대로 100, 200, 100입니다.
activation : 활성화 함수의 형태 (identity, logistic, tanh, relu)
solver : 경사하강법 알고리즘의 종류 (lbfgs, sgd, adam)
learning_rate : 학습률의 변화방법 (constant, invscaling 등)
learning_rate_init : 학습률의 초깃값
max_iter : 경사하강법의 반복횟수
verbose : 학습과정의 출력여부

"""
args = {'hidden_layer_sizes':(100,),
        'activation':'relu',
        'solver':'sgd',
        'verbose':True}
# 중간층의 개수가 하나, 중간층의 뉴런 개수가 100개인 다층 퍼셉트론입니다.
model = MLPClassifier(**args)
model.fit(X,Y)
print(model.score(X,Y))     # 모형의 평균 정확도를 출력합니다.

model = MLPRegressor(**args)
model.fit(X,Y)
print(model.score(X,Y))     # 모형의 R-squared값을 출력합니다.
