'''
test
'''
import torch
import torch.nn as nn
import numpy as np
import data_pretreatment
import pandas as pd


# Definition of Comparative Learning Models
class Contrastive_model(nn.Module):
    def __init__(self, width_dim, depth, state_dim, action_dim):
        super(Contrastive_model, self).__init__()
        layers = []
        layers.append(nn.Linear(state_dim, width_dim))
        layers.append(nn.Tanh())
        for _ in range(depth):
            layers.append(nn.Linear(width_dim, width_dim))
            layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        # Add MLP decoding network
        mlp_layers = []
        mlp_layers.append(nn.Linear(width_dim, width_dim))
        mlp_layers.append(nn.Tanh())
        mlp_layers.append(nn.Linear(width_dim, action_dim))
        self.mlp = nn.Sequential(*mlp_layers)
        self.initialize_weights()

    def forward(self, state):
        feature_core = self.net(state)
        action_mean = self.mlp(feature_core)
        return action_mean

    def get_feature_core(self, state):
        feature_core = self.net(state)
        return feature_core

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


# Define fully connected neural network model prediction model class
class MLP(nn.Module):
    def __init__(self, width_dim, depth, state_dim, action_dim):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(state_dim, width_dim))
        layers.append(nn.Tanh())
        for _ in range(depth):
            layers.append(nn.Linear(width_dim, width_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(width_dim, action_dim))
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


# Get the starting index
def get_day_idx(data):
    days = []
    days_idx = []
    for i in range(len(data)):
        if i == 0:
            days.append([0])
            continue
        if (data[i, 0] < data[i - 1, 0]):
            days.append([1])
            days_idx.append(i)
        else:
            days.append([0])
    new_data = np.concatenate((data, np.array(days)), axis=1)
    days_idx = np.array(days_idx)
    return new_data, days_idx


# Split the data into multiple sub fragments according to the given index list idxs
def data_segmentation(data, idxs):
    data_list = []
    for i in range(len(idxs) - 1):
        data_list.append((data[idxs[i]:idxs[i + 1], :]).tolist())
    data_list.append((data[idxs[-1]:, :]).tolist())
    return data_list


# Data transformation
def data_trans(data_np):
    inputs1 = []
    inputs2 = []
    inputs3 = []
    inputs4 = []
    for i in range(len(data_np)):
        input1 = torch.FloatTensor(data_np[i][0]).requires_grad_()
        input2 = torch.FloatTensor(data_np[i][1]).requires_grad_()
        input3 = torch.FloatTensor(data_np[i][2]).requires_grad_()
        input4 = torch.FloatTensor(data_np[i][3]).requires_grad_()

        a1 = torch.cat((input1[:, :5].clone(), input1[:, 5].unsqueeze(1).clone(), input1[:, 6].unsqueeze(1).clone()),
                       dim=1)
        inputs1.append(a1)

        a2 = torch.cat((input2[:, :5].clone(), input2[:, 5].unsqueeze(1).clone(), input2[:, 6].unsqueeze(1).clone()),
                       dim=1)
        inputs2.append(a2)

        a3 = torch.cat((input3[:, :5].clone(), input3[:, 5].unsqueeze(1).clone(), input3[:, 6].unsqueeze(1).clone()),
                       dim=1)
        inputs3.append(a3)

        a4 = torch.cat((input4[:, :5].clone(), input4[:, 5].unsqueeze(1).clone(), input4[:, 6].unsqueeze(1).clone()),
                       dim=1)
        inputs4.append(a4)

    return inputs1, inputs2, inputs3, inputs4


# standardize the original data
def data_augmentation(data_np, mean):
    for i in range(len(data_np)):
        for j in range(4):
            data_np[i][j] = data_np[i][j] / mean
    return data_np


# Prediction model testing
def test(inputs, model):
    out = model(inputs)
    return out


# save data
def save_tensors_to_excel(data, file1, file2):
    tensor_list1 = []
    tensor_list2 = []
    for item in data:
        tensor_list1.append(item[0].detach().numpy().tolist())
        tensor_list2.append(item[1].detach().numpy().tolist())
    df1 = pd.DataFrame(tensor_list1)
    df2 = pd.DataFrame(tensor_list2)
    df1.to_excel(file1, index=False)
    df2.to_excel(file2, index=False)


if __name__ == "__main__":
    # read data
    filename1 = "./data/feature_1.xls"
    filename2 = "./data/feature_2.xls"
    filename3 = "./data/feature_3.xls"
    filename4 = "./data/feature_4.xls"
    data1 = np.array(pd.read_excel(filename1, sheet_name="Sheet1", header=None))
    data2 = np.array(pd.read_excel(filename2, sheet_name="Sheet1", header=None))
    data3 = np.array(pd.read_excel(filename3, sheet_name="Sheet1", header=None))
    data4 = np.array(pd.read_excel(filename4, sheet_name="Sheet1", header=None))
    data1[:, 0] = data1[:, 0] % 1440
    data2[:, 0] = data2[:, 0] % 1440
    data3[:, 0] = data3[:, 0] % 1440
    data4[:, 0] = data4[:, 0] % 1440
    features1 = data1[:, :5]
    features2 = data2[:, :5]
    features3 = data3[:, :5]
    features4 = data4[:, :5]

    # Return a new dataset with furnace number and remaining duration
    new_features1 = data_pretreatment.get_T_num(features1)
    new_features2 = data_pretreatment.get_T_num(features2)
    new_features3 = data_pretreatment.get_T_num(features3)
    new_features4 = data_pretreatment.get_T_num(features4)

    # normalization
    mean1 = new_features1.mean(axis=0)
    mean2 = new_features2.mean(axis=0)
    mean3 = new_features3.mean(axis=0)
    mean4 = new_features4.mean(axis=0)
    mean = (mean1 + mean2 + mean3 + mean4) / 4
    std1 = new_features1.std(axis=0)
    std2 = new_features2.std(axis=0)
    std3 = new_features3.std(axis=0)
    std4 = new_features4.std(axis=0)
    std = (std1 + std2 + std3 + std4) / 4
    features1_norm = new_features1 / mean
    features2_norm = new_features2 / mean
    features3_norm = new_features3 / mean
    features4_norm = new_features4 / mean

    # data prepare
    _, days_1 = get_day_idx(features1[1:, :])
    _, days_2 = get_day_idx(features2[1:, :])
    _, days_3 = get_day_idx(features3[1:, :])
    _, days_4 = get_day_idx(features4[1:, :])
    days_idx1 = [0] + list(days_1[:-1])
    days_idx2 = [0] + list(days_2[:-1])
    days_idx3 = [0] + list(days_3[:-1])
    days_idx4 = [0] + list(days_4[:-1])

    data_list1 = data_segmentation(new_features1, days_idx1)
    data_list2 = data_segmentation(new_features2, days_idx2)
    data_list3 = data_segmentation(new_features3, days_idx3)
    data_list4 = data_segmentation(new_features4, days_idx4)
    combined_list = [[row1, row2, row3, row4] for row1, row2, row3, row4 in
                     zip(data_list1, data_list2, data_list3, data_list4)]  # data integration
    combined_np = [[np.array(row1), np.array(row2), np.array(row3), np.array(row4)] for row1, row2, row3, row4 in
                   zip(data_list1, data_list2, data_list3, data_list4)]
    data_np = data_augmentation(combined_np, mean)
    inputs1, inputs2, inputs3, inputs4 = data_trans(data_np)

    # predict model define
    width_dim_p = 128
    depth_p = 5
    state_dim_p = 7
    action_dim_p = 4
    dnn1 = MLP(width_dim_p, depth_p, state_dim_p, action_dim_p)
    dnn2 = MLP(width_dim_p, depth_p, state_dim_p, action_dim_p)
    dnn3 = MLP(width_dim_p, depth_p, state_dim_p, action_dim_p)
    dnn4 = MLP(width_dim_p, depth_p, state_dim_p, action_dim_p)

    dnn1.load_state_dict(torch.load("./model/dnn_model_noco_pde_1"))
    dnn2.load_state_dict(torch.load("./model/dnn_model_noco_pde_2"))
    dnn3.load_state_dict(torch.load("./model/dnn_model_noco_pde_3"))
    dnn4.load_state_dict(torch.load("./model/dnn_model_noco_pde_4"))

    # Iterative prediction testing
    mean = torch.FloatTensor(mean)
    # Lists to store prediction vs. ground truth results for 4 units (heat1–heat4)
    heat1_pred_true_lists = []
    heat2_pred_true_lists = []
    heat3_pred_true_lists = []
    heat4_pred_true_lists = []
    # number of days (data is grouped by day)
    day_num = len(inputs1)
    # Zero tensor separator (used when concatenating results later)
    separator = [torch.tensor([0, 0, 0, 0, 0, 0, 0]), torch.tensor([0, 0, 0, 0, 0, 0, 0])]
    for i in range(day_num):
        iter_num = 4 # number of iterative prediction steps within each window
        window_num = int(min(len(inputs1[i]), len(inputs2[i]), len(inputs3[i]), len(inputs4[i])) / iter_num) # number of windows per day (use the shortest among 4 input series)
        for j in range(window_num):
            # Temporary lists to store predictions and ground truth for the current window
            heat1_pred_true_list = []
            heat2_pred_true_list = []
            heat3_pred_true_list = []
            heat4_pred_true_list = []
            # Ground truth sequences for this day
            heat1_true_list_input = inputs1[i]
            heat2_true_list_input = inputs2[i]
            heat3_true_list_input = inputs3[i]
            heat4_true_list_input = inputs4[i]
            # Initialize starting indices for the current window
            j1 = j * iter_num
            j2 = j * iter_num
            j3 = j * iter_num
            j4 = j * iter_num
            # Initialize model inputs with the first element of the window
            heat1_input = heat1_true_list_input[j1]
            heat2_input = heat2_true_list_input[j2]
            heat3_input = heat3_true_list_input[j3]
            heat4_input = heat4_true_list_input[j4]
            # Keep a copy of the true input for comparison
            heat1_input_true = heat1_input.clone()
            heat2_input_true = heat2_input.clone()
            heat3_input_true = heat3_input.clone()
            heat4_input_true = heat4_input.clone()
            while max(j1, j2, j3, j4) < iter_num + j * iter_num: # Iterative prediction loop (roll forward within the window)
                # Get ground truth at current step
                heat1_input_true = heat1_true_list_input[j1]

                # De-normalize the current input and the ground truth
                heat1_input_ = heat1_input * mean
                heat1_input_true_ = heat1_input_true * mean

                # Save prediction input vs. ground truth pair
                heat1_pred_true_list.append([heat1_input_.clone(), heat1_input_true_.clone()])

                # --- Model prediction step ---
                out1 = test(heat1_input, dnn1)
                out1_ = out1 * mean[1:5]

                # --- Update input features for next iteration ---
                heat1_input_[0] = heat1_input_[0] + heat1_input_[1] + 1 # Update time
                heat1_input_[1:5] = out1_ # replace features with predicted
                heat1_input_[5] = heat1_input_[5] - heat1_input_[1] - 1 # remaining duration
                heat1_input_[6] = heat1_input_[6] - 1 # update heat number

                # Normalize again before feeding into the model in the next step
                heat1_input = heat1_input_ / mean

                # Move to next time index
                j1 += 1

                # 2
                heat2_input_true = heat2_true_list_input[j2]
                heat2_input_ = heat2_input * mean
                heat2_input_true_ = heat2_input_true * mean
                heat2_pred_true_list.append([heat2_input_.clone(), heat2_input_true_.clone()])
                out2 = test(heat2_input, dnn2)
                out2_ = out2 * mean[1:5]
                heat2_input_[0] = heat2_input_[0] + heat2_input_[1] + 1
                heat2_input_[1:5] = out2_
                heat2_input_[5] = heat2_input_[5] - heat2_input_[1] - 1
                heat2_input_[6] = heat2_input_[6] - 1
                heat2_input = heat2_input_ / mean
                j2 += 1
                # 3
                heat3_input_true = heat3_true_list_input[j3]
                heat3_input_ = heat3_input * mean
                heat3_input_true_ = heat3_input_true * mean
                heat3_pred_true_list.append([heat3_input_.clone(), heat3_input_true_.clone()])
                out3 = test(heat3_input, dnn3)
                out3_ = out3 * mean[1:5]
                heat3_input_[0] = heat3_input_[0] + heat3_input_[1] + 1
                heat3_input_[1:5] = out3_
                heat3_input_[5] = heat3_input_[5] - heat3_input_[1] - 1
                heat3_input_[6] = heat3_input_[6] - 1
                heat3_input = heat3_input_ / mean
                j3 += 1
                # 4
                heat4_input_true = heat4_true_list_input[j4]
                heat4_input_ = heat4_input * mean
                heat4_input_true_ = heat4_input_true * mean
                heat4_pred_true_list.append([heat4_input_.clone(), heat4_input_true_.clone()])
                out4 = test(heat4_input, dnn4)
                out4_ = out4 * mean[1:5]
                heat4_input_[0] = heat4_input_[0] + heat4_input_[1] + 1
                heat4_input_[1:5] = out4_
                heat4_input_[5] = heat4_input_[5] - heat4_input_[1] - 1
                heat4_input_[6] = heat4_input_[6] - 1
                heat4_input = heat4_input_ / mean
                j4 += 1

            # Collect predictive data
            heat1_pred_true_lists += heat1_pred_true_list
            heat2_pred_true_lists += heat2_pred_true_list
            heat3_pred_true_lists += heat3_pred_true_list
            heat4_pred_true_lists += heat4_pred_true_list

        # Unified dimension
        heat1_pred_true_lists.append(separator)
        heat2_pred_true_lists.append(separator)
        heat3_pred_true_lists.append(separator)
        heat4_pred_true_lists.append(separator)

    print('heat1_pred_true_lists:', heat1_pred_true_lists)
    print('heat2_pred_true_lists:', heat2_pred_true_lists)
    print('heat3_pred_true_lists:', heat3_pred_true_lists)
    print('heat4_pred_true_lists:', heat4_pred_true_lists)

    # str = 'seg_noco_pinn'
    # str_ = 'segment_result/xiaorong'
    # save_tensors_to_excel(heat1_pred_true_lists, f"./result/{str_}/pred_data1_{str}.xlsx",
    #                       f"./result/{str_}/true_data1_{str}.xlsx")
    # save_tensors_to_excel(heat2_pred_true_lists, f"./result/{str_}/pred_data2_{str}.xlsx",
    #                       f"./result/{str_}/true_data2_{str}.xlsx")
    # save_tensors_to_excel(heat3_pred_true_lists, f"./result/{str_}/pred_data3_{str}.xlsx",
    #                       f"./result/{str_}/true_data3_{str}.xlsx")
    # save_tensors_to_excel(heat4_pred_true_lists, f"./result/{str_}/pred_data4_{str}.xlsx",
    #                       f"./result/{str_}/true_data4_{str}.xlsx")
