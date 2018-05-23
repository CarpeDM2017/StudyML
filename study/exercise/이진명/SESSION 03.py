# Nesterov Accelerated Gradient 구현
import numpy as np
import matplotlib.pyplot as plt


class GradientDescent:
    def _init_(self, learning_rate=0.1):
        self.lr = learning_rate
        self.eps = 1e-3

class NAG(GradientDescent):
    def f(x,y):sin(1/2*x^2 - 1/4*y^2+3)*cos(2*x +1-e^y)
    def _init_(self, learning_rate=0.1, momentum=0.9):
        super(momentu,self)._init_(learning_rate)
        self.momentum=momentum
        self.velocity = 0

    def train(self, F, x, y):
        dim_x = x.shape[0]
        dim_y = y.shape[0]
        grad_x = np.zeros(x.shape)
        grad_y = np.zeros(y.shape)
        eps_x = np.zeros(x.shape)
        eps_y = np.zeros(y.shape)

        for i in range(dim_x):
            eps_x[i] = self.eps
            grad_x[i] =(F(x-velocity+eps)-F(x-velocity))/self.eps_x
            eps_x[i]=0
        velocity = momentum*velocity + learning_rate*grad_x
        x=x-velocity
        return x

        for i in range(dim_y):
            eps_y[i] = self.eps
            grad_y[i] =(F(y-velocity+eps)-F(y-velocity))/self.eps_y
            eps_y[i]=0
            velocity = momentum*velocity + learning_rate*grad_y
            y=y-velocity
        return y

    def F(x,y): return np.sum(x,y)
    x= np.random.randn(100)
    y= np.random.rand(100)

    print (F())
