import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import ScalarFormatter

from sklearn.model_selection import train_test_split
import torch

def generate_data(end_date, days, num_ascending_start, num_descending_start, swap_count):
    if (days) % swap_count != 0:
        raise ValueError('Number of days must be divisible by swap_count')
    seq_len = days // swap_count

    # Generate a list of dates
    date_list = pd.date_range(end=end_date, periods=days).strftime("%Y-%m-%d")

    # Generate list of values
    def generate_values(ascending_start):
        lst = [100]
        previous_step = None
        for i in range(swap_count):
            # If it's the first iteration and it's an ascending start or the previous step was descending
            if (i == 0 and ascending_start) or previous_step == 'descending':
                for _ in range(seq_len):
                    lst.append(np.round(lst[-1] * np.random.uniform(1, 1.1), 2))
                    previous_step = 'ascending'
            else:
                for _ in range(seq_len):
                    lst.append(np.round(lst[-1] * np.random.uniform(0.9, 1), 2))
                    previous_step = 'descending'

        return lst[:days]

    # Generate all values
    all_values = {}
    for i in range(num_ascending_start):
        all_values[f'ascending_{i+1}'] = generate_values(ascending_start=True)
    for i in range(num_descending_start):
        all_values[f'descending_{i+1}'] = generate_values(ascending_start=False)

    # Create a DataFrame
    data = pd.DataFrame(all_values)
    data['Date'] = date_list
    data.set_index('Date', inplace=True)

    # Insert a new row at the top of the DataFrame
    first_row = data.iloc[0]
    first_date = pd.to_datetime(data.index[0])
    previous_date = (first_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    data.loc[previous_date] = first_row
    data.sort_index(inplace=True)

    return data, seq_len

def create_sequences(data_returns, X_seq_len, Y_seq_len, test_size, device):
    # Calculate batch size and input size
    batch_size = len(data_returns) - X_seq_len - Y_seq_len + 1
    input_size = len(data_returns.columns)

    # Convert DataFrame to NumPy for easier slicing
    data_returns_np = data_returns.values

    # Create sequences
    X = []
    Y = []
    for i in range(batch_size):
        first_y_index = i + X_seq_len
        
        X.append(data_returns_np[i:first_y_index])

        # Get the index of the highest return for the next day
        next_day_returns = data_returns_np[first_y_index:first_y_index + Y_seq_len]
        Y.append(next_day_returns)

    # Convert to NumPy arrays
    X = np.array(X)  # Shape: (batch_size, X_seq_len, input_size)
    Y = np.array(Y)  # Shape: (batch_size, Y_seq_len, input_size)

    # Split into train and test sets
    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    Y_train, Y_test = Y[:split_index], Y[split_index:]
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    Y_train = torch.tensor(Y_train, dtype=torch.float32).to(device)
    Y_test = torch.tensor(Y_test, dtype=torch.float32).to(device)

    return X_train, X_test, Y_train, Y_test


def get_Y_max_one(Y):
    # Y_avg = torch.mean(Y, dim=1, keepdim=True).squeeze()
    Y_cum_ret = (Y + 1).prod(dim=1, keepdim=True).squeeze() - 1

    # Get the index of the highest return
    Y_max_one_id = Y_cum_ret.argmax(axis=1)

    # Change the shape
    Y_max_one = torch.zeros_like(Y_cum_ret)
    Y_max_one[torch.arange(Y_max_one_id.shape[0]), Y_max_one_id] = 1

    return Y_max_one

def get_Y_max_light(Y):
    # Y_avg = torch.mean(Y, dim=1, keepdim=True).squeeze()
    Y_cum_ret = (Y + 1).prod(dim=1, keepdim=True).squeeze() - 1

    # Replace negative values with 0
    Y_cum_ret_positive = torch.where(Y_cum_ret > 0, Y_cum_ret, torch.zeros_like(Y_cum_ret))
    # Sum all positive values
    Y_cum_ret_positive_sums = torch.sum(Y_cum_ret_positive, dim=1).unsqueeze(1)
    # Calculate shares of each positive value in the sum
    Y_cum_ret_positive_shares = Y_cum_ret_positive / Y_cum_ret_positive_sums

    # Create a tensor of zeros with a 1 at the last position ()
    do_not_invest = torch.zeros_like(Y_cum_ret_positive_shares)
    do_not_invest[:, -1] = 1  # Set the last column to 1 for all rows

    # Replace all shares recomendation with "do not invest" where the sum of positive values is 0
    Y_max_light = torch.where(Y_cum_ret_positive_sums == 0, do_not_invest, Y_cum_ret_positive_shares)

    return Y_max_light

def get_Y_sharpe_one(X, Y):
    Y_cum_ret = (Y + 1).prod(dim=1, keepdim=True).squeeze() - 1

    # Standard deviation of the returns
    X_and_Y = torch.cat([X, Y], dim=1)
    X_and_Y_std = torch.std(X_and_Y, dim=1, keepdim=True).squeeze()

    # Sharpe ratio 
    Y_sharpe = torch.nan_to_num(Y_cum_ret / X_and_Y_std)

    # Get the index of the highest sharpe ratio
    Y_sharpe_one_id = Y_sharpe.argmax(axis=1)

    # Change the shape
    Y_sharpe_one = torch.zeros_like(Y_cum_ret)
    Y_sharpe_one[torch.arange(Y_sharpe_one_id.shape[0]), Y_sharpe_one_id] = 1

    return Y_sharpe_one

def get_Y_sharpe_light(X, Y):
    Y_cum_ret = (Y + 1).prod(dim=1, keepdim=True).squeeze() - 1

    # Standard deviation of the returns
    X_and_Y = torch.cat([X, Y], dim=1)
    X_and_Y_std = torch.std(X_and_Y, dim=1, keepdim=True).squeeze()

    # Sharpe ratio 
    Y_sharpe = Y_cum_ret / X_and_Y_std

    # Replace negative values with 0
    Y_sharpe_positive = torch.where(Y_sharpe > 0, Y_sharpe, torch.zeros_like(Y_sharpe))
    # Sum all positive values
    Y_sharpe_positive_sums = torch.sum(Y_sharpe_positive, dim=1).unsqueeze(1)
    # Calculate shares of each positive value in the sum
    Y_sharpe_positive_shares = Y_sharpe_positive / Y_sharpe_positive_sums

    # Create a tensor of zeros with a 1 at the last position ()
    do_not_invest = torch.zeros_like(Y_sharpe_positive_shares)
    do_not_invest[:, -1] = 1  # Set the last column to 1 for all rows

    # Replace all shares recomendation with "do not invest" where the sum of positive values is 0
    Y_sharpe_light = torch.where(Y_sharpe_positive_sums == 0, do_not_invest, Y_sharpe_positive_shares)

    return Y_sharpe_light


def calculate_max_return(Y):
    next_day_Y = Y[:, 0, :] 
    max_daily_returns = torch.max(next_day_Y, dim=1).values + 1
    max_total_returns = torch.prod(max_daily_returns) - 1

    return max_daily_returns, max_total_returns

def calculate_portfolio_return(output_weights, Y):
    next_day_Y = Y[:, 0, :] 
    portfolio_daily_returns = (next_day_Y * output_weights).sum(dim=1) + 1
    portfolio_total_returns = torch.prod(portfolio_daily_returns) - 1

    return portfolio_daily_returns, portfolio_total_returns


def plot_loss_curves(results):    
    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()


def plot_asset_share(df):
    plt.figure(figsize=(10, 6))
    for column in df.columns:
        plt.plot(df.index, df[column], label=column)

    # Customize x-axis to display the year only once
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())

    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=False))

    plt.title("Asset Share Over Time")
    plt.xlabel("Year")
    plt.ylabel("Value")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()