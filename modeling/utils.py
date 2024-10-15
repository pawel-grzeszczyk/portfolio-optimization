import pandas as pd
import numpy as np

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