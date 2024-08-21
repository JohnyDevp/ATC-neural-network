### Main script to run the training and testing of the model
### Author: Jan Holan
### Date: 2024-07-31

import atc_dataloader, atc_model
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_model(model, dataset_train, dataset_val, optimizer, epochs=10):
    
    loss = torch.nn.MSELoss()
    
    for epoch in range(epochs):
        # set the model to train mode
        model.train()
        for batch in dataset_train:
            # get the data from the batch
            data = batch['data'].to(device)
            target = batch['target'].to(device)
            
            # zero the gradients
            optimizer.zero_grad()
            
            # forward pass
    

def test_model(model, dataset_input, dataset_val):    
    pass

if __name__ == "__main__":
    # load data
    PATH_INPUT_DATA="data/train_in.csv"
    PATH_RESULTS="data/train_out.csv"
    
    atc_data = atc_dataloader.ATCDataset(in_data_path=PATH_INPUT_DATA, out_data_path=PATH_RESULTS)
    
    model = atc_model.BaseNN()
    
    
    