import torch
import torch.nn as nn

class SharpeRatioLoss(nn.Module):
    def __init__(self, risk_free_rate=0.0):
        super(SharpeRatioLoss, self).__init__()
        self.risk_free_rate = risk_free_rate

    def forward(self, 
                pred_weights: torch.Tensor, 
                X: torch.Tensor, 
                y: torch.Tensor):

        # Calculate the weighted input data (equal weights)
        equal_weights = torch.full(X.shape, 1 / X.shape[1])
        weighted_X = X * equal_weights

        # Calculate the weighted returns (predicted weights)
        weighted_y = y * pred_weights

        # Combine the weighted returns
        weighted_returns = torch.cat((weighted_X, weighted_y.unsqueeze(0)), dim=0)

        # Calculate the mean and standard deviation of the weighted returns
        means = torch.mean(weighted_returns, dim=0)
        stds = torch.std(weighted_returns, dim=0)

        # Calculate the Sharpe ratio
        sharpe_ratios = (means - self.risk_free_rate) / (stds + 1e-8)
        sharpe_ratio = torch.mean(sharpe_ratios)

        # Return the negative Sharpe ratio as the loss
        return -sharpe_ratio
