import torch
from torchvision import datasets

data = datasets.MNIST(root='study\exercise\임현경\data', download=True)

image, label = data[0]

print(type(image))
print(type(label))

image.show()
print(label)
