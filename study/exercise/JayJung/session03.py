# session03.py

import numpy as np

class GradientDescent:
    def __init__(self, learning_rate=0.1):
        self.lr = learning_rate
        self.eps = 1e-3

    def train(self, F, x):
        # 만약 미분의 형태를 이미 알고 있는 경우 바로 미분계수를 넣으면 됩니다
        dim = x.shape[0]
        grad = np.zeros(x.shape)
        eps = np.zeros(x.shape)
        for i in range(len(dim)):
            eps[i] = self.eps
            grad[i] = (F(x+eps) - F(x))/self.eps
            eps[i] = 0
        x = x - self.lr*grad
        return x
"""
GradientDesent class copied that of session 3
"""

class Nesterov:
    def __init__(self, velocity, learning_rate=0.1, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.eps = 1e-6
        self.velocity = velocity
    def train(self, F, x):
        dim = x.shape[0]
        grad = np.zeros(x.shape)
        eps = np.zeros(x.shape)
        x_tmp = np.zeros(x.shape)
        x_tmp = x - self.velocity*self.momentum
        for i in range(dim):
            eps[i] = self.eps # to change only i+1 th element
            grad[i] = (F(x_tmp+eps) - F(x_tmp))/self.eps
        self.velocity = self.momentum*self.velocity + self.lr*grad
        x = x - self.velocity
        return x, self.velocity

def F(input):
    x = input[0]
    y = input[1]
    F = np.sin((1/2)*(x**2) - (1/4)*(y**2) + 3)*np.cos(2*x + 1 - np.exp(y))
    return F

trial = 10
xSol = np.zeros((2,trial))
print(xSol)
for i in range(trial):
    init = np.array(4*np.random.rand(2,1))
    velocity = np.ones((2,1))
    while any(velocity> 1e-6):
        print(i)
        Nest = Nesterov(velocity)
        velocity = Nest.train(F, init)[1]
        init = Nest.train(F, init)[0]

    np.transpose(xSol)[i] = np.transpose(init)      
    print(np.transpose(xSol))    
      