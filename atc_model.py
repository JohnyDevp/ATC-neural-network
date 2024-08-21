import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class BaseNN(nn.Module):
    def __init__(self, input_size=5000, output_size=18):
        super(BaseNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 4000)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(4000, 2000)
        self.fc3 = nn.Linear(2000, 1000)
        self.fc4 = nn.Linear(1000, 500)
        # the output layer is kind of classification layer, where we have 18 classes
        # but the difference is, that each class is represent by number of occurences
        self.fc5 = nn.Linear(500, out_features=output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation(x)
        x = self.fc4(x)
        x = self.activation(x)
        x = self.fc5(x)
        return x

