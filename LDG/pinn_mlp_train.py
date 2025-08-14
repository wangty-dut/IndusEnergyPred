'''
PINN network training
'''
import torch
import torch.nn as nn
import numpy as np
import random
import data_pretreatment



# ##Generate negative sample indices corresponding to different indices of three furnace data
def generate_vectors(n, threshold):
    vectors = []
    for _ in range(n):  # Generate three different positive integers
        while True:
            vec = random.sample(range(1, threshold + 1), 3)
            if len(set(vec)) == 3:
                break
        vectors.append(vec)
    return vectors


# Comparative Learning Network
class Contrastive_model(nn.Module):
    def __init__(self, width_dim, depth, state_dim, action_dim):
        super(Contrastive_model, self).__init__()
        layers = []
        layers.append(nn.Linear(state_dim, width_dim))  # Input layer
        layers.append(nn.Tanh())  # Add ReLU activation function
        for _ in range(depth):
            layers.append(nn.Linear(width_dim, width_dim))  # Hidden layer
            layers.append(nn.Tanh())  # Add ReLU activation function
        self.net = nn.Sequential(*layers)  # Add MLP decoding network
        mlp_layers = []
        mlp_layers.append(nn.Linear(width_dim, width_dim))  # First layer MLP
        mlp_layers.append(nn.Tanh())  # Add Tanh activation function
        mlp_layers.append(nn.Linear(width_dim, action_dim))  # Second layer MLP output layer
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
    y_true = y_true.detach().numpy()  # Ensure the input arrays are numpy arrays
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


# Convert data according to converter number
def data_trans(data_np, n=1):
    inputs = []
    Ts = []
    lushus = []
    labels = []
    for i in range(len(data_np)):
        input1 = torch.FloatTensor(data_np[i][0]).requires_grad_()
        input2 = torch.FloatTensor(data_np[i][1]).requires_grad_()
        input3 = torch.FloatTensor(data_np[i][2]).requires_grad_()
        if n == 1:
            input = input1[:-1, :3]
            T = input1[:-1, 3].unsqueeze(1)
            lushu = input1[:-1, 4].unsqueeze(1)
            label = input1[1:, 0].unsqueeze(1)

        if n == 2:
            input = input2[:-1, :3]
            T = input2[:-1, 3].unsqueeze(1)
            lushu = input2[:-1, 4].unsqueeze(1)
            label = input2[1:, 0].unsqueeze(1)

        if n == 3:
            input = input3[:-1, :3]
            T = input3[:-1, 3].unsqueeze(1)
            lushu = input3[:-1, 4].unsqueeze(1)
            label = input3[1:, 0].unsqueeze(1)
        inputs.append(input.clone())
        Ts.append(T.clone())
        lushus.append(lushu.clone())
        labels.append(label.clone())
    # Merge Tensors from the List
    inputs = torch.cat(inputs, dim=0)
    Ts = torch.cat(Ts, dim=0)
    lushus = torch.cat(lushus, dim=0)
    labels = torch.cat(labels, dim=0)
    return inputs, Ts, lushus, labels


# Enhance/standardize the original data
def data_augmentation(data_np, mean):
    for i in range(len(data_np)):
        for j in range(3):
            data_np[i][j] = data_np[i][j] / mean
    return data_np


# Model forward propagation and Loss function calculation
def net_f(x, T, n, y, model, model_c, mean):
    a_mean = mean[0]
    T_mean = mean[3]
    n_mean = mean[4]
    width_mean = mean[2]
    width = x[:, 2].reshape(-1, 1)
    feature = x[:, :2]
    feature_h = model_c.get_feature_core(torch.cat([feature, width], dim=1))
    out = model(torch.cat([feature_h, T, n], dim=1))
    a_n = torch.autograd.grad(out, n, grad_outputs=torch.ones_like(out), retain_graph=True, create_graph=True)[0]
    a_w = torch.autograd.grad(out, width, grad_outputs=torch.ones_like(out), retain_graph=True, create_graph=True)[0]
    f_1 = a_n * (a_mean.item() / n_mean.item()) + (
            (T * T_mean.item()) / (2 * n * n_mean.item() * n * n_mean.item()))
    f_2 = a_w * (a_mean / width_mean) - 0.5
    criterion = nn.MSELoss()
    loss_data = criterion(y, out)
    loss_pde1 = criterion(f_1, torch.zeros_like(f_1))
    loss_pde2 = criterion(f_2, torch.zeros_like(f_2))
    loss = loss_data * 100 + loss_pde1 + loss_pde2
    return out, loss, (loss_pde1 + loss_pde2), loss_data


# Prediction model training
def train(num_epochs, batch_size, data_np, model, model_c, optimizer, mean, lu_num):
    loss_list = []
    loss_data_list = []
    loss_pde_list = []
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
            out, loss, loss_pde, loss_data = net_f(batch_inputs, T_inputs, lushu_inputs, batch_labels, model, model_c,
                                                   mean)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, loss_pde:{loss_pde.item()}")
        loss_list.append(loss.item())
        loss_data_list.append(loss_data.item())
        loss_pde_list.append(loss_pde.item())
        if epoch != 0 and epoch % 100 == 0:
            lr = 0.001 / (epoch / 100)
            optimizer = torch.optim.Adam(dnn.parameters(), lr=lr)
    return loss_list, loss_pde_list, loss_data_list


# Prediction model testing
def test(batch_size, data_np, model, model_c, lu_num):
    inputs, Ts, lushus, labels = data_trans(data_np, lu_num)
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
    width = batch_inputs[:, 2].reshape(-1, 1)
    feature = batch_inputs[:, :2]
    feature_h = model_c.get_feature_core(torch.cat([feature, width], dim=1))
    out = model(torch.cat([feature_h, T_inputs, lushu_inputs], dim=1))
    return out, batch_labels


# run
if __name__ == "__main__":
    # Number of converters
    lu_num = 3
    # read data
    features1, _, mean1, std1 = data_pretreatment.Unified_processing(
        "./data/three_ldg_data1.xlsx")
    new_features1 = data_pretreatment.get_T_num(features1)
    features2, _, mean2, std2 = data_pretreatment.Unified_processing(
        "./data/three_ldg_data2.xlsx")
    new_features2 = data_pretreatment.get_T_num(features2)
    features3, _, mean3, std3 = data_pretreatment.Unified_processing(
        "./data/three_ldg_data3.xlsx")
    new_features3 = data_pretreatment.get_T_num(features3)
    mean = (np.concatenate((new_features1, new_features2, new_features3), axis=0)).mean(axis=0)
    std = (np.concatenate((new_features1, new_features2, new_features3), axis=0)).std(axis=0)
    # According to preparation
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
    inputs, Ts, lushus, labels = data_trans(data_np, lu_num)
    # Define contrastive learning models
    width_dim_c = 56
    depth_c = 4
    state_dim_c = 3
    action_dim_c = 10
    model_c = Contrastive_model(width_dim_c, depth_c, state_dim_c, action_dim_c)
    model_c.load_state_dict(torch.load(
        "./model/contrastive_model"))
    # Define predictive models
    width_dim_p = 128
    depth_p = 5
    state_dim_p = width_dim_c + 2
    action_dim_p = 1
    dnn = MLP(width_dim_p, depth_p, state_dim_p, action_dim_p)
    # Model training and its parameters
    epoch_p = 500
    batch_size = 200
    optimizer = torch.optim.Adam(dnn.parameters(), lr=0.001)
    loss_list, loss_pde_list, loss_data_list = train(epoch_p, batch_size, data_np, dnn, model_c, optimizer, mean,
                                                     lu_num)
    # torch.save(dnn.state_dict(), f'./model/dnn_model_c_pde_{lu_num}')
