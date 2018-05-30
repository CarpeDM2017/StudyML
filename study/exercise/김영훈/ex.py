import torch
from torchvision import datasets

mnist = datasets.MNIST(root='./data')

image, label = mnist[0]

print(type(image))
print(type(label))

image.show()
print(label)
