'''
PINN prediction model training
'''
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import data_pretreatment


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


# data segmentation
def data_segmentation(data, idxs):
    data_list = []
    for i in range(len(idxs) - 1):
        data_list.append((data[idxs[i]:idxs[i + 1], :]).tolist())
    data_list.append((data[idxs[-1]:, :]).tolist())
    return data_list


# Data feature transformation
def data_trans(data_np, n=1):
    inputs = []
    Ts = []
    lushus = []
    labels = []
    for i in range(len(data_np)):
        input1 = torch.FloatTensor(data_np[i][0]).requires_grad_()
        input2 = torch.FloatTensor(data_np[i][1]).requires_grad_()
        input3 = torch.FloatTensor(data_np[i][2]).requires_grad_()
        input4 = torch.FloatTensor(data_np[i][3]).requires_grad_()

        if n == 1:
            input = input1[:-1, :5]
            T = input1[:-1, 5].unsqueeze(1)
            lushu = input1[:-1, 6].unsqueeze(1)
            label = input1[1:, 1:5]

        if n == 2:
            input = input2[:-1, :5]
            T = input2[:-1, 5].unsqueeze(1)
            lushu = input2[:-1, 6].unsqueeze(1)
            label = input2[1:, 1:5]

        if n == 3:
            input = input3[:-1, :5]
            T = input3[:-1, 5].unsqueeze(1)
            lushu = input3[:-1, 6].unsqueeze(1)
            label = input3[1:, 1:5]

        if n == 4:
            input = input4[:-1, :5]
            T = input4[:-1, 5].unsqueeze(1)
            lushu = input4[:-1, 6].unsqueeze(1)
            label = input4[1:, 1:5]

        inputs.append(input.clone())
        Ts.append(T.clone())
        lushus.append(lushu.clone())
        labels.append(label.clone())

    inputs = torch.cat(inputs, dim=0)
    Ts = torch.cat(Ts, dim=0)
    lushus = torch.cat(lushus, dim=0)
    labels = torch.cat(labels, dim=0)

    return inputs, Ts, lushus, labels


# Enhance/standardize the original data
def data_augmentation(data_np, mean):
    for i in range(len(data_np)):
        for j in range(4):
            data_np[i][j] = data_np[i][j] / mean

    return data_np


# Model forward calculation and loss calculation
def net_f(x, T, n, y, model, model_c, mean):
    a_mean = mean[1]
    T_mean = mean[5]
    n_mean = mean[6]
    width_mean = mean[1]

    time_feature = x[:, 0].reshape(-1, 1)
    width = x[:, 1].reshape(-1, 1)
    feature = x[:, 2:5]
    feature_h = model_c.get_feature_core(torch.cat((time_feature, width, feature), dim=1))
    out = model(torch.cat([feature_h, T, n], dim=1))
    out_ = out[:, 0]

    a_n = torch.autograd.grad(out_, n, grad_outputs=torch.ones_like(out_), retain_graph=True, create_graph=True)[0]
    a_w = torch.autograd.grad(out_, width, grad_outputs=torch.ones_like(out_), retain_graph=True, create_graph=True)[0]
    f_1 = a_n * (a_mean.item() / n_mean.item()) + (
            (T * T_mean.item()) / (2 * n * n_mean.item() * n * n_mean.item()))
    f_2 = a_w * (a_mean / width_mean) - 0.5

    criterion = nn.MSELoss()
    loss_data = criterion(y, out)
    loss_pde1 = criterion(f_1, torch.zeros_like(f_1))
    loss_pde2 = criterion(f_2, torch.zeros_like(f_2))

    loss = loss_data * 10 + loss_pde1 + loss_pde2
    return out, loss, (loss_pde1 + loss_pde2)


# train
def train(num_epochs, batch_size, data_np, model, model_c, optimizer, mean, lu_num):
    loss_list = []
    inputs, Ts, lushus, labels = data_trans(data_np, lu_num)
    for epoch in range(num_epochs):
        permutation = torch.randperm(inputs.size(0))
        inputs_shuffled = inputs[permutation]
        labels_shuffled = labels[permutation]
        Ts_shuffled = Ts[permutation]
        lushus_shuffled = lushus[permutation]

        num_batches = inputs.size(0) // batch_size
        for batch_index in range(num_batches):
            start_index = batch_index * batch_size
            end_index = (batch_index + 1) * batch_size
            batch_inputs = inputs_shuffled[start_index:end_index]
            T_inputs = Ts_shuffled[start_index:end_index]
            lushu_inputs = lushus_shuffled[start_index:end_index]
            batch_labels = labels_shuffled[start_index:end_index]

            out, loss, loss_pde = net_f(batch_inputs, T_inputs, lushu_inputs, batch_labels, model, model_c, mean)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, loss_pde:{loss_pde.item()}")
        loss_list.append(loss.item())
        if epoch != 0 and epoch % 100 == 0:
            lr = 0.001 / (epoch / 100)
            optimizer = torch.optim.Adam(dnn.parameters(), lr=lr)

    return loss_list


# test
def test(batch_size, data_np, model, model_c, lu_num):
    inputs, Ts, lushus, labels = data_trans(data_np, lu_num)

    # Prediction model testing
    permutation = torch.randperm(inputs.size(0))
    inputs_shuffled = inputs[permutation]
    labels_shuffled = labels[permutation]
    Ts_shuffled = Ts[permutation]
    lushus_shuffled = lushus[permutation]

    start_index = 1 * batch_size
    end_index = (1 + 1) * batch_size
    batch_inputs = inputs_shuffled[start_index:end_index]
    T_inputs = Ts_shuffled[start_index:end_index]
    lushu_inputs = lushus_shuffled[start_index:end_index]
    batch_labels = labels_shuffled[start_index:end_index]

    time_feature = batch_inputs[:, 0].reshape(-1, 1)
    width = batch_inputs[:, 1].reshape(-1, 1)
    feature = batch_inputs[:, 2:5]
    feature_h = model_c.get_feature_core(torch.cat((time_feature, width, feature), dim=1))
    out = model(torch.cat([feature_h, T_inputs, lushu_inputs], dim=1))

    return out, batch_labels


if __name__ == "__main__":
    lu_num = 3
    # Read data
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

    new_features1 = data_pretreatment.get_T_num(features1)
    new_features2 = data_pretreatment.get_T_num(features2)
    new_features3 = data_pretreatment.get_T_num(features3)
    new_features4 = data_pretreatment.get_T_num(features4)

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
                     zip(data_list1, data_list2, data_list3, data_list4)]
    combined_np = [[np.array(row1), np.array(row2), np.array(row3), np.array(row4)] for row1, row2, row3, row4 in
                   zip(data_list1, data_list2, data_list3, data_list4)]
    data_np = data_augmentation(combined_np, mean)
    inputs, Ts, lushus, labels = data_trans(data_np, lu_num)

    # Define contrastive learning models
    width_dim_c = 56
    depth_c = 4
    state_dim_c = 5
    action_dim_c = 10
    model_c = Contrastive_model(width_dim_c, depth_c, state_dim_c, action_dim_c)
    model_c.load_state_dict(torch.load("./model/contrastive_model"))

    # Define predictive models
    width_dim_p = 128
    depth_p = 5
    state_dim_p = width_dim_c + 2
    action_dim_p = 4
    dnn = MLP(width_dim_p, depth_p, state_dim_p, action_dim_p)

    # Model training and its parameters
    epoch_p = 1000
    batch_size = 200
    optimizer = torch.optim.Adam(dnn.parameters(), lr=0.001)
    loss_list = train(epoch_p, batch_size, data_np, dnn, model_c, optimizer, mean, lu_num)
    torch.save(dnn.state_dict(), f'./model/dnn_model_c_pde_{lu_num}')
