import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(5000, 4000)
        self.actication = nn.ReLU()
        # the output layer is kind of classification layer, where we have 18 classes
        # but the difference is, that each class is represent by number of occurences
        self.fc2 = nn.Linear(4000, 18)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.actication(x)
        x = self.fc2(x)
        return x

