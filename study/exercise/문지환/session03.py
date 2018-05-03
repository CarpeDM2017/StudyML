# mjh

import numpy as np

## 3차시 과제


class GradientDescent:
    def __init__(self, learning_rate=0.1):
        self.lr = learning_rate
        self.eps = 1e-3

    def train(self, F, x):
        # 만약 미분의 형태를 이미 알고 있는 경우 바로 미분계수를 넣으면 됩니다
        dim = x.shape[0]
        grad = np.zeros(x.shape)
        eps = np.zeros(x.shape)
        for i in range(dim):
            eps[i] = self.eps
            grad[i] = (F(x+eps) - F(x))/self.eps
            eps[i] = 0
        x = x - self.lr*grad
        return x


# Q1

class NAG(GradientDescent):
    def __init__(self, learning_rate=0.1, momentum=0.9):
        super(NAG, self).__init__(learning_rate)
        self.momentum = momentum
        self.velocity = 0

    def train(self, F, x):
        dim = x.shape[0]
        grad = np.zeros(x.shape)
        eps = np.zeros(x.shape)
        x_after_momentum = x - self.momentum * self.velocity

        for i in range(dim):
            eps[i] = self.eps
            grad[i] = (F(x_after_momentum + eps) - F(x_after_momentum))/self.eps
            eps[i] = 0

        self.velocity = self.momentum * self.velocity + self.lr * grad
        return x - self.velocity


# Q2

def F(X):  # X = [x, y]
    x = X[0]
    y = X[1]

    return np.sin(1/2*x**2 - 1/4*y**2 + 3) * np.cos(2*x + 1 - np.exp(y))


def Q2():
    results = []
    for i in range(3):
        init_X = np.random.randn(2) * 0.5  # init_val.shape = (2,)
        X = init_X
        optim = NAG()

        print("initial value #%d" % (i+1))
        for j in range(100):
            if j % 10 == 0:
                print(j, F(X))
            X = optim.train(F, X)

        print('100', F(X))
        results.append((init_X, X, F(X)))
        print()

    for i in range(len(results)):
        result = results[i]
        print('#%d' % (i+1), 'initial X:', result[0], 'final X:', result[1], 'F(X):', result[2])


if __name__ == '__main__':
    Q2()

