# data preprocessing
import numpy as np

def get_T_num(data):
    data = np.array(data)
    T_list = []
    num_list = []
    for i in range(len(data)):
        if i > 0:
            t = data[i, 0] + data[i, 1]
            t_ = data[i - 1, 0] + data[i - 1, 1]
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
