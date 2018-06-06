import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

DATA_PATH = './data'
BATCH_SIZE = 128
LOG_STEP = 100

def train(model, epochs) :
    global DATA_PATH
    global BATCH_SIZE
    global  
