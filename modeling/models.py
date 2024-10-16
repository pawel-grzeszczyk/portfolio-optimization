import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTMModel(nn.Module): 
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_rate=0.5): 
        super(LSTMModel, self).__init__() 
        self.input_size = input_size #input size 
        self.hidden_size = hidden_size #hidden state 
        self.output_size = output_size #number of classes 
        self.num_layers = num_layers #number of layers 

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout_rate) #lstm 
        self.fc =  nn.Linear(hidden_size, hidden_size) #fully connected 1 
        self.fc_out = nn.Linear(hidden_size, output_size) #fully connected 2 
        self.relu = nn.ReLU() 
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x): 
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state 
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state 

        # LSTM layer 
        lstm_out, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state 
        x = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        x = self.relu(x) 
        x = self.fc(x) 
        x = self.relu(x) 
        x = self.fc_out(x) 

        # Pass the output through the softmax function (to get sum equal to 1) 
        out = self.softmax(x)
        
        return out



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
