'''
Function: Preprocess the features extracted by Matlab
'''
import numpy
import numpy as np
import pandas as pd


def normalize_array(input_array):
    '''
    Input the adarray array, normalize the data in each column of the array, and output the normalized array and the mean variance of each column
    '''
    column_means = np.mean(input_array, axis=0)
    column_stddevs = np.std(input_array, axis=0)
    normalized_array = (input_array - column_means) / column_stddevs
    return normalized_array, column_means, column_stddevs


def excel_data_normalize(filename, time_start):
    '''
    Normalize the Excel table features extracted from Matlab
    '''
    data = pd.read_excel(filename, sheet_name='Sheet1')
    datas = data.values
    datas[:, 0] += time_start
    datas[:, 0] %= 1440
    normalized_array, column_means, column_stddevs = normalize_array(datas)
    return normalized_array, column_means, column_stddevs


def min_max_normalize_array(input_array):
    '''
    Input the adarray array, perform maximum and minimum normalization on the data in each column of the array, and output the normalized array and the maximum and minimum values of each column
    '''
    column_mins = np.min(input_array, axis=0)
    column_maxs = np.max(input_array, axis=0)
    normalized_array = (input_array - column_mins) / (column_maxs - column_mins)
    return normalized_array, column_mins, column_maxs


def min_max_other_array(input_array, last_min, last_max):
    '''
    Process the data in each column of input_array
    '''
    _, column_mins, column_maxs = min_max_normalize_array(input_array)
    mins_array = np.zeros((1, 4))
    maxs_array = np.zeros((1, 4))
    for i in range(3):
        mins_array[0, i] = column_mins[i + 1]
        maxs_array[0, i] = column_maxs[i + 1]
    mins_array[0, 3] = last_min
    maxs_array[0, 3] = last_max
    return mins_array, maxs_array


def excel_data_max_min_normalize(filename, time_start):
    '''
    Normalize the Excel table features extracted from Matlab
    '''
    data = pd.read_excel(filename, sheet_name='Sheet1')
    datas = data.values
    datas[:, 0] += time_start
    datas[:, 0] %= 1440
    normalized_array, column_mins, column_maxs = min_max_normalize_array(datas)
    return normalized_array, column_mins, column_maxs


def deta_array(array_data):
    '''
    Obtain the transformation amount of array values
    '''
    num = np.shape(array_data)[0]
    deta_array_data = np.zeros((num - 1, 1))
    for i in range(num - 1):
        deta_array_data[i, 0] = array_data[i + 1] - array_data[i]
    return deta_array_data


def deta_arrays(array_datas):
    '''
    Obtain the transformation amount of array values
    '''
    num1 = np.shape(array_datas)[0]
    num2 = np.shape(array_datas)[1]
    deta_array_datas = np.zeros((num1 - 1, 0))
    for i in range(num2):
        deta_array_data = deta_array(array_datas[:, i])
        deta_array_datas = np.hstack((deta_array_datas, deta_array_data))
    return deta_array_datas


def Unified_processing(filename):
    '''
    Return the input data features, including the current gap time, gap width, and the duration of the gap time from the previous few gaps
    '''
    data = np.array(pd.read_excel(filename, sheet_name="Sheet1", header=None))
    data = data[1:, :]
    features = []
    feature = []
    for i in range(len(data)):
        if i >= 1:
            feature.append((1440 + data[i, 0] - data[i - 1, 0]) % 1440)
            feature.append(data[i, 0])
            feature.append(data[i, 1])
            features.append(feature)
            feature = []
    features = np.array(features)
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    features_norm = (features - mean) / std

    return features, features_norm, mean, std


def get_T_num(data):
    data = np.array(data)
    T_list = []
    num_list = []
    for i in range(len(data)):
        if i > 0:
            t = data[i, 1] + data[i, 2] / 2
            t_ = data[i - 1, 1] + data[i - 1, 2] / 2
            if t < t_:
                num_list.append(1)
                T = t + 1440 - t_
                T_list.append(T)
            else:
                num_list.append(0)
                T = t - t_
                T_list.append(T)
    T_np = np.array(T_list).reshape(-1, 1)
    num_np = np.array(num_list).reshape(-1, 1)
    new_data = data[1:, :]
    new_data = np.concatenate((new_data, T_np), axis=1)
    new_data = np.concatenate((new_data, num_np), axis=1)

    Time = 0
    lushu = 0
    flag = 1
    idx = 0
    for i in range(1, len(new_data) + 1):
        if new_data[-i, -1] == 0:
            t = new_data[-i, -2]
            new_data[-i, -1] = lushu
            new_data[-i, -2] = Time
            Time += t
            lushu += 1
        else:
            new_data[-i, -1] = lushu
            new_data[-i, -2] = Time
            lushu = 0
            Time = 0
            if flag:
                idx = len(new_data) - i
                flag = 0
    new_data = new_data[:idx, :]

    return new_data
