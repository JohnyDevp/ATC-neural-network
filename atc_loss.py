import torch 
from torch import nn

class PredictionLoss_COS_MSE(nn.Module):
    """
    This class is used to calculate the loss between two vectors according to some found loss function (it is not mine)
    """
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    mse = nn.MSELoss()
    
    def __init__(self):
        super(PredictionLoss_COS_MSE, self).__init__()
    
    def forward(self, pred, target):
        # we are using the mean squared error as loss function
        
        # Define weights for functions for Cos and MSE.
        w1 = 5
        w2 = 15
        
        # Apply cumulative sum to both tensors and calculate loss.
        cos_sim = torch.abs(self.cos(torch.cumsum(pred, dim=-1), torch.cumsum(target, dim=-1))).mean()
        mse_loss = self.mse(torch.cumsum(pred, dim=-1), torch.cumsum(target, dim=-1))
        loss = (w1 * mse_loss) / (w2 * cos_sim)
        return loss

class PredictionLoss_BOX_Wise(nn.Module):
    """
    Calculating loss function box-wise. Meaning total number of boxes and difference between each box type is considered.
    """
    
    mse = nn.MSELoss()
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, pred, target):
        # you have some boxes 1-18
        # let decide the loss considering two params    -> total number of boxes - correct meaning loss 0, 
        #                                               -> number per box - correct meaning loss 0 for the current field
        
        loss = 0
        # calculate difference between each type of box 0
        for i in range(len(pred)):
            loss += self.mse(pred[i], target[i])
        
        # add the loss of the total number of boxes
        loss += torch.abs(pred.sum() - target.sum())                      
        
        return loss