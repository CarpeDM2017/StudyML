import requests
import pickle
import os
from PIL import Image
import numpy as np


class Loader:
    def __init__(self, val=False):
        self.val = val
        self.shape = [3, 32, 32]
        self.download()

    def download(self):
        url = "https://github.com/CarpeDM2017/StudyML/raw/CarpeDM2018/study/hackathon/hackathon.pkl"
        r = requests.get(url, allow_redirects=True)
        with open('data', 'wb') as f:
            f.write(r.content)
        with open('data', 'rb') as f:
            data = pickle.load(f)
        os.remove('data')
        del r

        self.download = False
        if self.val:
            print("Validation Data Downloaded")
            self.data = data['val']
        else:
            print("Training Data Downloaded")
            self.data = data['train']

    def __getitem__(self, index):
        return self.data[0][index], self.data[1][index]

    def __len__(self):
        return len(self.data[1])

    def view(self, index):
        arr = self.data[0][index].reshape(*self.shape)*255
        image = Image.fromarray(np.rollaxis(arr,0,3).astype(np.uint8), 'RGB')
        return image

    def label(self, index):
        label = 'Ship' if self.data[1][index] else 'Plane'
        return label
