"""
2018.05.01
My answers for problems in session03

"""
import numpy as np


class GradientDescent:
    def __init__(self, learning_rate=0.1):
        self.lr = learning_rate
        self.eps = 1e-3

    def train(self, F, x):
        dim = x.shape[0]
        grad = np.zeros(x.shape)
        eps = np.zeros(x.shape)
        for i in range(dim):
            eps[i] = self.eps
            grad[i] = (F(x+eps) - F(x))/self.eps
            eps[i] = 0
        x = x - self.lr*grad
        return x


class Nesterov(GradientDescent):
    def __init__(self, learning_rate=0.1, momentum=0.9):
        super(Nesterov, self).__init__(learning_rate=learning_rate)
        self.momentum = momentum
        self.velocity = 0

    def train(self, F, x):
        dim = x.shape[0]
        grad = np.zeros(x.shape)
        eps = np.zeros(x.shape)
        for i in range(dim):
            eps[i] = self.eps
            grad[i] = (F(x - self.momentum*self.velocity + eps) \
                     - F(x - self.momentum*self.velocity)) / self.eps
            eps[i] = 0
        self.velocity = self.momentum*self.velocity + self.lr*grad
        x = x - self.velocity
        return x


def Q1():
    """
    Implement the Nesterov Method
    """
    optimizer = Nesterov()


def Q2():
    """
    Optimize the given function F by the Nesterov Method
    """
    def F(x_in):
        assert np.size(x_in) == 2, "The function must be of 2 dimension"
        x = x_in[0]
        y = x_in[1]
        x_out = np.sin(x**2/2 - y**2/4 + 3) * np.cos(2*x + 1 - np.exp(y))
        return x_out

    def get_init(seed):
        np.random.seed(seed)        # 시행마다 같은 결과를 다시 얻을 수 있도록 random seed를 고정합니다.
        return np.random.randn(2)   # 평균 0, 표준편차 1인 다변량 정규분포로부터 초깃값을 추출합니다.

    num_seed = 3
    num_step = 10000
    for seed in range(num_seed):
        x = get_init(seed)
        optimizer = Nesterov()
        print("Initial x for seed {} : {}".format(seed, x))
        for step in range(num_step):
            x = optimizer.train(F, x)
        print("Final x for seed {} : {}".format(seed, x))
