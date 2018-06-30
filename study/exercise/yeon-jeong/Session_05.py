from sklearn.neural_network import MLPClassifier, MLPRegressor
import numpy as np
import torch
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



data = datasets.MNIST(root='./data',download=True)

image, label = data[0]

print(type(image))
print(type(label))

image.show()
print(label)

X = np.random.randn(10,10)
Y = np.random.binomial(1,0.5,10)

args = {'hidden_layer_sizes':(100,),'activation':'relu','solver':'sgd','verbose':True}

model = MLPClassifier(**args)
model.fit(X,Y)
print(model.score(X,Y))

model = MLPRegressor(**args)
model.fit(X,Y)
print(model.score(X,Y))





XX = np.random.randn(10,10)
YY = np.random.randn(10)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.perceptron1 = nn.Linear(10,100)
        self.perceptron2 = nn.Linear(100,100)
        self.perceptron3 = nn.Linear(100,10)

    def forward(self, x):
        x = self.perceptron1(x)
        x = F.sigmoid(x)
        x = self.perceptron2(x)
        x = F.sigmoid(x)
        out = self.perceptron3(x)
        return Out

