from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import numpy as np

# X 데이터는 정규분포, Y 데이터는 베르누이분포로 추출하였습니다.
X = np.random.randn(10,10)
Y = np.random.binomial(1,0.5,10)

model = DecisionTreeClassifier(criterion='gini',
                               splitter='best',
                               max_depth=10,
                               max_leaf_nodes=100
                               )

model.fit(X,Y)
model.feature_importances_
model.score(X,Y)
