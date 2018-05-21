from sklearn.linear_model import SGDClassifier
import numpy as np

# X, Y 데이터를 정규분포로부터 랜덤하게 생성해보았습니다.
X = np.random.randn(10,10)
Y = np.random.randn(10)

"""
SGDClassifier의 입력값

loss : 오차함수의 형태 (hinge, log, squared_loss 등)
penalty : 제약식의 형태 (l2, l1, elasticnet)
max_iter : 경사하강법의 반복횟수
eta0 : 학습률의 초깃값
learning_rate : 학습률의 변화방법 (constant, invscaling 등)
verbose : 학습과정의 출력여부

"""
# 오차함수를 'log'로 설정하면 로지스틱 회귀모형이 됩니다.
model = SGDClassifier(loss='log', penalty=None,
                     max_iter=1000, eta0=0.1, learning_rate='constant',
                     verbose=True)
