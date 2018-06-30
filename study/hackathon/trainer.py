import numpy as np
from dataloader import Loader
from models import *


class Trainer:
    def __init__(self, model, batch_size=16):
        self.train_loader = Loader(val=False)
        self.val_loader = Loader(val=True)
        self.model = model
        self.batch_size = batch_size

    def train(self, epochs):
        for epoch in range(epochs):
            images = []
            labels = []
            for i, (image, label) in enumerate(self.train_loader):
                images += [image]
                labels += [label]
                if len(images) == self.batch_size:
                    images = np.array(images)
                    labels = np.array(labels)
                    self.model.train(images, labels)
                    images = []
                    labels = []

    def validate(self):
        score = 0
        for i, (image, label) in enumerate(self.val_loader):
            image = image.reshape(1, -1)
            label = label.reshape(1, -1)
            score += self.model.val(image, label)
        score /= len(self.val_loader)
        return score


# 아래는 학습 과정의 예시 코드입니다.
trainer = Trainer(MyModel())
trainer.train(1)
