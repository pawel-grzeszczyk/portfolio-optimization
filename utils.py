import torch

def sharpe(current_weights, current_returns, past_weights, past_returns):
    """
    Calculate the Sharpe ratio for a given portfolio.

    Parameters:
    output (torch.Tensor): Tensor of weights chosen by the network for the current portfolio.
    current_returns (torch.Tensor): Tensor of today's returns data.
    past_weights (torch.Tensor): Tensor of weights used in the past.
    past_returns (torch.Tensor): Tensor of past returns data.

    Returns:
    torch.Tensor: The mean Sharpe ratio.
    """
    # Calculate weighted returns for today
    weighted_returns = current_weights * today_data
    
    # Get past returns
    weighted_past_returns = past_returns * past_weights
    
    # Calculate cumulative returns from the past
    cumulative_weighted_past_returns = torch.prod(1 + weighted_past_returns, dim=0)
    
    # Calculate total returns including today's data
    total = (cumulative_weighted_past_returns * (weighted_returns + 1)) - 1
    
    # Calculate standard deviation of the returns
    weighted_returns_all = torch.cat([weighted_past_returns, weighted_returns], dim=0)
    stddev = torch.std(weighted_returns_all, dim=0)
    
    # Calculate Sharpe ratio
    sharpe_ratio = total / stddev
    return torch.mean(sharpe_ratio)
