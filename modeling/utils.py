import pandas as pd
import numpy as np

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