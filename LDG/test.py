'''
Iterative prediction file without contrastive learning model,
not iterating all day, but updating to true after multiple iterations to continue iterating
'''
import torch
import torch.nn as nn
import numpy as np
import random
import data_pretreatment
import pandas as pd


# Define fully connected neural network model prediction model class
class MLP(nn.Module):
    def __init__(self, width_dim, depth, state_dim, action_dim):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(state_dim, width_dim))  # Input layer
        layers.append(nn.Tanh())  # Add ReLU activation function
        for _ in range(depth):
            layers.append(nn.Linear(width_dim, width_dim))  # Hidden layer
            layers.append(nn.Tanh())  # Add ReLU activation function
        layers.append(nn.Linear(width_dim, action_dim))  # Output layer
        self.net = nn.Sequential(*layers)
        self.initialize_weights()

    def forward(self, state):
        action_mean = self.net(state)
        return action_mean

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


# Calculate errors
def calculate_errors(y_true, y_pred):
    # Ensure the input arrays are numpy arrays
    y_true = y_true.detach().numpy()
    y_pred = y_pred.detach().numpy()
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse}


# Obtain the starting index of each day in steelmaking data
def get_day_idx(data):
    days = []
    days_idx = []
    for i in range(len(data)):
        if i == 0:
            days.append([0])
            continue
        if (data[i, 1] < data[i - 1, 1]):
            days.append([1])
            days_idx.append(i)
        else:
            days.append([0])
    new_data = np.concatenate((data, np.array(days)), axis=1)
    days_idx = np.array(days_idx)
    return new_data, days_idx


# Data segmentation
def data_segmentation(data, idxs):
    data_list = []
    for i in range(len(idxs) - 1):
        data_list.append((data[idxs[i]:idxs[i + 1], :]).tolist())
    data_list.append((data[idxs[-1]:, :]).tolist())
    return data_list


# The input here is not for training, but for testing, therefore it includes the last batch of data
def data_trans(data_np):
    inputs1 = []
    inputs2 = []
    inputs3 = []
    for i in range(len(data_np)):
        input1 = torch.FloatTensor(data_np[i][0]).requires_grad_()
        input2 = torch.FloatTensor(data_np[i][1]).requires_grad_()
        input3 = torch.FloatTensor(data_np[i][2]).requires_grad_()
        a1 = torch.cat((input1[:, :3].clone(), input1[:, 3].unsqueeze(1).clone(), input1[:, 4].unsqueeze(1).clone()),
                       dim=1)
        inputs1.append(a1)
        a2 = torch.cat((input2[:, :3].clone(), input2[:, 3].unsqueeze(1).clone(), input2[:, 4].unsqueeze(1).clone()),
                       dim=1)
        inputs2.append(a2)
        a3 = torch.cat((input3[:, :3].clone(), input3[:, 3].unsqueeze(1).clone(), input3[:, 4].unsqueeze(1).clone()),
                       dim=1)
        inputs3.append(a3)
    return inputs1, inputs2, inputs3


# Enhance/standardize the original data
def data_augmentation(data_np, mean):
    for i in range(len(data_np)):
        for j in range(3):
            data_np[i][j] = data_np[i][j] / mean
    return data_np


# Calculate loss function
def net_f(x, T, n, y, model, model_c):
    width = x[:, 2].reshape(-1, 1)
    feature = x[:, :2]
    feature_h = model_c.get_feature_core(torch.cat([feature, width], dim=1))
    out = model(torch.cat([feature_h, T, n], dim=1))

    criterion = nn.MSELoss()
    loss_data = criterion(y, out)
    loss = loss_data
    return out, loss


# Prediction model testing
def test(inputs, model):
    out = model(inputs)
    return out


# Processing data
def save_tensors_to_excel(data, file1, file2):
    # Extract and concatenate the first tensor list
    tensor_list1 = []
    tensor_list2 = []
    for item in data:
        tensor_list1.append(item[0].detach().numpy().tolist())
        tensor_list2.append(item[1].detach().numpy().tolist())
    # Convert data to a DataFrame
    df1 = pd.DataFrame(tensor_list1)
    df2 = pd.DataFrame(tensor_list2)
    # Saved to excel
    df1.to_excel(file1, index=False)
    df2.to_excel(file2, index=False)


def calculate_differences(file1, file2):
    # Read Excel file
    df1 = pd.read_excel(file1)
    df2 = pd.read_excel(file2)
    # Check if the shapes of two DataFrames are the same
    if df1.shape != df2.shape:
        raise ValueError("The two Excel files must have the same shape.")
    differences = []
    # Traverse every line
    for i in range(len(df1)):
        if (df1.iloc[i] == 0).all() and (df2.iloc[i] == 0).all():
            if i > 0:
                prev_row_diff = df1.iloc[i - 1, 1] - df2.iloc[i - 1, 1]
                differences.append(prev_row_diff)
    # Calculate the difference in the last row
    last_row_diff = df1.iloc[-1, 1] - df2.iloc[-1, 1]
    differences.append(last_row_diff)
    return differences


if __name__ == "__main__":
    # read data
    features1, _, mean1, std1 = data_pretreatment.Unified_processing(
        "C:/Users/49829/Desktop/thesis/paper_song/contrastive_pinn_ldg/contrastive_pinn_ldg/data/three_ldg_data1.xlsx")
    new_features1 = data_pretreatment.get_T_num(features1)
    features2, _, mean2, std2 = data_pretreatment.Unified_processing(
        "C:/Users/49829/Desktop/thesis/paper_song/contrastive_pinn_ldg/contrastive_pinn_ldg/data/three_ldg_data1.xlsx")
    new_features2 = data_pretreatment.get_T_num(features2)
    features3, _, mean3, std3 = data_pretreatment.Unified_processing(
        "C:/Users/49829/Desktop/thesis/paper_song/contrastive_pinn_ldg/contrastive_pinn_ldg/data/three_ldg_data1.xlsx")
    new_features3 = data_pretreatment.get_T_num(features3)
    mean = (np.concatenate((new_features1, new_features2, new_features3), axis=0)).mean(axis=0)
    std = (np.concatenate((new_features1, new_features2, new_features3), axis=0)).std(axis=0)
    # data preparation,The new_feature mentioned earlier starts from the second one, so here we start from the first one
    _, days_1 = get_day_idx(features1[1:, :])
    _, days_2 = get_day_idx(features2[1:, :])
    _, days_3 = get_day_idx(features3[1:, :])
    days_idx1 = [0] + list(days_1[:-1])
    days_idx2 = [0] + list(days_2[:-1])
    days_idx3 = [0] + list(days_3[:-1])
    data_list1 = data_segmentation(new_features1, days_idx1)
    data_list2 = data_segmentation(new_features2, days_idx2)
    data_list3 = data_segmentation(new_features3, days_idx3)
    combined_list = [[row1, row2, row3] for row1, row2, row3 in zip(data_list1, data_list2, data_list3)]
    combined_np = [[np.array(row1), np.array(row2), np.array(row3)] for row1, row2, row3 in
                   zip(data_list1, data_list2, data_list3)]
    data_np = data_augmentation(combined_np, mean)
    inputs1, inputs2, inputs3 = data_trans(data_np)
    # Define predictive models
    width_dim_p = 128
    depth_p = 5
    state_dim_p = 5
    action_dim_p = 1
    dnn1 = MLP(width_dim_p, depth_p, state_dim_p, action_dim_p)
    dnn2 = MLP(width_dim_p, depth_p, state_dim_p, action_dim_p)
    dnn3 = MLP(width_dim_p, depth_p, state_dim_p, action_dim_p)
    dnn1.load_state_dict(torch.load(
        "./model/dnn_model_xiaorong_noco_pde_1"))
    dnn2.load_state_dict(torch.load(
        "./model/dnn_model_xiaorong_noco_pde_2"))
    dnn3.load_state_dict(torch.load(
        "./model/dnn_model_xiaorong_noco_pde_3"))
    # Iterative prediction testing
    mean = torch.FloatTensor(mean)
    lu1_pred_true_lists = []
    lu2_pred_true_lists = []
    lu3_pred_true_lists = []
    day_num = len(inputs1)
    separator = [torch.tensor([0, 0, 0, 0, 0]), torch.tensor([0, 0, 0, 0, 0])]
    for i in range(day_num):
        iter_num = 4
        window_num = int(min(len(inputs1[i]), len(inputs2[i]), len(inputs3[i])) / iter_num)
        for j in range(window_num):  # Find 8 starting points for iterative prediction
            lu1_pred_true_list = []
            lu2_pred_true_list = []
            lu3_pred_true_list = []
            # Single day data preparation
            lu1_true_list_input = inputs1[i]
            lu2_true_list_input = inputs2[i]
            lu3_true_list_input = inputs3[i]
            j1 = j * iter_num
            j2 = j * iter_num
            j3 = j * iter_num
            lu1_input = lu1_true_list_input[j1]
            lu2_input = lu2_true_list_input[j2]
            lu3_input = lu3_true_list_input[j3]
            lu1_input_true = lu1_input.clone()
            lu2_input_true = lu2_input.clone()
            lu3_input_true = lu3_input.clone()
            while max(j1, j2, j3) < iter_num + j * iter_num:  # End after 4 iterations
                lu1_input_true = lu1_true_list_input[j1]
                lu1_input_ = lu1_input * mean
                lu1_input_true_ = lu1_input_true * mean
                lu1_pred_true_list.append([lu1_input_.clone(), lu1_input_true_.clone()])
                out1 = test(lu1_input, dnn1)
                out1_ = out1 * mean[0]
                lu1_input_[0] = out1_
                lu1_input_[1] = lu1_input_[1] + out1_
                lu1_input_[2] = (out1_ - (lu1_input_[2] / 2 + 15)) * 2
                lu1_input_[3] = lu1_input_[3] - 15 - lu1_input_[2]
                lu1_input_[4] = lu1_input_[4] - 1
                lu1_input = lu1_input_ / mean
                j1 += 1

                lu2_input_true = lu2_true_list_input[j2]
                lu2_input_ = lu2_input * mean
                lu2_input_true_ = lu2_input_true * mean
                lu2_pred_true_list.append([lu2_input_.clone(), lu2_input_true_.clone()])
                out2 = test(lu2_input, dnn2)
                out2_ = out2 * mean[0]
                lu2_input_[0] = out2_
                lu2_input_[1] = lu2_input_[1] + out2_
                lu2_input_[2] = (out2_ - (lu2_input_[2] / 2 + 15)) * 2
                lu2_input_[3] = lu2_input_[3] - 15 - lu2_input_[2]
                lu2_input_[4] = lu2_input_[4] - 1
                lu2_input = lu2_input_ / mean
                j2 += 1

                lu3_input_true = lu3_true_list_input[j3]
                lu3_input_ = lu3_input * mean
                lu3_input_true_ = lu3_input_true * mean
                lu3_pred_true_list.append([lu3_input_.clone(), lu3_input_true_.clone()])
                out3 = test(lu3_input, dnn3)
                out3_ = out3 * mean[0]
                lu3_input_[0] = out3_
                lu3_input_[1] = lu3_input_[1] + out3_
                lu3_input_[2] = (out3_ - (lu3_input_[2] / 2 + 15)) * 2
                lu3_input_[3] = lu3_input_[3] - 15 - lu3_input_[2]
                lu3_input_[4] = lu3_input_[4] - 1
                lu3_input = lu3_input_ / mean
                j3 += 1

            lu1_pred_true_lists += lu1_pred_true_list
            lu2_pred_true_lists += lu2_pred_true_list
            lu3_pred_true_lists += lu3_pred_true_list
        lu1_pred_true_lists.append(separator)
        lu2_pred_true_lists.append(separator)
        lu3_pred_true_lists.append(separator)
    print('lu1_pred_true_lists:', lu1_pred_true_lists)
    print('lu2_pred_true_lists:', lu2_pred_true_lists)
    print('lu3_pred_true_lists:', lu3_pred_true_lists)

    # str = 'seg_noco_pinn'
    # str_ = 'segment_result/xiaorong'
    # save_tensors_to_excel(lu1_pred_true_lists, f"./result/{str_}/pred_data1_{str}.xlsx",
    #                       f"./result/{str_}/true_data1_{str}.xlsx")
    # save_tensors_to_excel(lu2_pred_true_lists, f"./result/{str_}/pred_data2_{str}.xlsx",
    #                       f"./result/{str_}/true_data2_{str}.xlsx")
    # save_tensors_to_excel(lu3_pred_true_lists, f"./result/{str_}/pred_data3_{str}.xlsx",
    #                       f"./result/{str_}/true_data3_{str}.xlsx")
