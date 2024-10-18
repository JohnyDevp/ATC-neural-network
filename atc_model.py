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


class BetterNN(nn.Module):
    """
    This is a better neural network model ready to use feature vector
    """
    def __init__(self, input_size=40, output_size=18):
        super(BetterNN, self).__init__()
        
        # use relu as activation function, cause any other is not suitable for this task
        self.activation = nn.ReLU()
        
        # Conv1d layer
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3)
        
        # Calculate the size after the Conv1d layer
        # Assuming input_size is the length of the sequence and the input is of shape (batch_size, 1, input_size)
        conv_output_size = (input_size - 3 + 1)  # since kernel_size=3 and stride=1, padding=0
        
        # First linear layer should take conv_output_size as input
        self.fc1 = nn.Linear(conv_output_size, 200)
        
        # the hidden part of linear layers
        self.fc2 = nn.Linear(200, 500)  # Change input size according to fc1 output
        self.fc3 = nn.Linear(500, 500)
        self.fc4 = nn.Linear(500, 300)
        self.fc5 = nn.Linear(300, 100)
        
        # The output layer for classification
        self.fc6 = nn.Linear(100, output_size)

    def forward(self, x):
        # Reshape x to fit Conv1d input requirements
        x = x.unsqueeze(1)  # remove a channel dimension: (batch_size, 1, input_size)
        x = self.conv1d(x)  # Apply Conv1d
        x = x.squeeze(1)     # Remove the channel dimension after conv: (batch_size, conv_output_size)
        
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation(x)
        x = self.fc4(x)
        x = self.activation(x)
        x = self.fc5(x)
        x = self.activation(x)
        x = self.fc6(x)
        return x

    
