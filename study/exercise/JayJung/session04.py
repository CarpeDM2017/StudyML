# session04.py

import pickle

# Ordinary Least Squares
from sklearn.linear_model import LinearRegression
# Huber 오차함수를 이용하는 선형회귀모형
from sklearn.linear_model import HuberRegressor
# 경사하강법을 활용한 선형회귀모형
from sklearn.linear_model import SGDRegressor

path = '../../session04.pkl'     # session04.pkl이 있는 경로를 넣어주세요
with open(path, 'rb') as f:
  data = pickle.load(f)
X = data['X']
Y = data['Y']

# Containers for the results
modelCoef = []
R_squared = []

# penality options for SGDRegressor, 'l2' = Ridge, 'l1' = Lasso, 'elasticnet' = elasticnet
penality = ('None', 'l2', 'l1', 'elasticnet') 
for p in penality:
	print('Start SGDRegressor with penality = %s' % p)
	model = SGDRegressor(loss='squared_loss', penalty=p, alpha = 0.7,
                     max_iter=10000, eta0=0.01, learning_rate='optimal',
                     verbose=False)

	model.fit(X,Y)              # 선형회귀모형을 학습시킵니다.
	modelCoef.append(model.coef_)
	R_squared.append(model.score(X,Y))
	print('Using penality = %s' % p)
	print('Beta coefficients are: ')
	print(model.coef_)          # 회귀계수 출력. 순서는 X와 같습니다.
	

	print('R-squared is: ')
	print(model.score(X,Y))     # 모형의 R-squared값 출력
	print('Regression Ended')
	print('-'*len('Beta coefficients are: '))