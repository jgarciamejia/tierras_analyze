import numpy as np 
from bisect import bisect_left, bisect_right

def median_filter_uneven(x, y, window_duration):
    ''' Performs median filtering of unevenly sampled time series data. 
    '''

    result_x = []
    result_y = []

    for i in range(len(x)):
        current_time = x[i]
        start_index, end_index = get_window_indices_binary(x,current_time, window_duration)

        window_data = y[start_index:end_index]

        median_value = np.median(window_data)
        result_x.append(current_time)
        result_y.append(median_value)

    return np.array(result_x), np.array(result_y)

def get_window_indices_binary(x, current_time, window_duration):
    ''' Finds the start/stop indices of the window at the current time step. 
    '''
    start_index = bisect_left(x, current_time-window_duration/2)
    end_index = bisect_right(x, current_time+window_duration/2)

    return start_index, end_index 
