'''
Comparative learning and training
'''
import copy
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import random
import data_pretreatment
import torch.nn.functional as F

# Generate negative sample indices corresponding to different indices of three furnace data
def generate_vectors(n, threshold):
    vectors = []
    for _ in range(n):
        while True:
            vec = random.sample(range(1, threshold + 1), 3)
            if len(set(vec)) == 3:
                break
        vectors.append(vec)
    return vectors

# Comparative Learning Network Definition
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

class PJCL():
    def __init__(self, width_dim_c, depth_c, state_dim_c, action_dim_c, data_np, mean, std):
        # Definition of New and Old Models
        self.encoder_old = Contrastive_model(width_dim_c, depth_c, state_dim_c, action_dim_c)
        self.encoder_new = Contrastive_model(width_dim_c, depth_c, state_dim_c, action_dim_c)
        # Training hyperparameters
        self.learn_rate_old = 1e-3
        self.learn_rate_new = 1e-3
        self.optim_old = torch.optim.Adam(self.encoder_old.parameters(), lr=self.learn_rate_old)
        self.optim_new = torch.optim.Adam(self.encoder_new.parameters(), lr=self.learn_rate_new)
        # Transition model parameters
        self.jump_mat221 = None
        self.jump_mat321 = None
        self.jump_mat421 = None
        self.jump_mat122 = None
        self.jump_mat322 = None
        self.jump_mat422 = None
        self.jump_mat123 = None
        self.jump_mat223 = None
        self.jump_mat423 = None
        self.jump_mat124 = None
        self.jump_mat224 = None
        self.jump_mat324 = None
        # Data loading/processing
        self.data_np = data_np
        self.mean = mean
        self.std = std
        self.data_augmentation(True)

    def increasing_function(self, x, a):
        return x ** a

    # Calculate the probability of transition based on the similarity value of each element, and perform sam
    # pling to determine whether there is a transition. Output a matrix of the same size consisting of 0 and 1, where 1 represents a transition and 0 represents no transition
    def transition_probability(self, similarity_matrix, a):
        rows, cols = similarity_matrix.shape
        transition_matrix = torch.zeros_like(similarity_matrix, dtype=torch.int)
        for i in range(rows):
            for j in range(cols):
                similarity = similarity_matrix[i, j]
                prob = self.increasing_function(similarity, a)
                # 使用随机抽样决定是否跃迁
                if torch.rand(1).item() < prob:
                    transition_matrix[i, j] = 1
        return transition_matrix

    # Enhance/standardize the original data
    def data_augmentation(self, flag=False):
        for i in range(len(self.data_np)):
            for j in range(4):
                data_np = self.data_np[i][j]
                noise = np.random.uniform(-3, 3, data_np.shape)
                noisy_feature_vectors = data_np + noise
                noisy_feature_vectors = np.maximum(noisy_feature_vectors, 0)
                if flag:
                    self.data_np[i][j] = data_np/self.mean
                    noisy_feature_vectors = noisy_feature_vectors/self.mean
                self.data_np[i].append(copy.deepcopy(noisy_feature_vectors))

    # Output the data and enhanced data of three furnaces on a certain day in the same format and shape
    def data_build(self, data):
        new_data = []
        for i in range(len(data)):
            new_data.append(torch.FloatTensor(data[i]).requires_grad_(True))
        return new_data

    # Comparative loss calculation
    def get_loss(self, old_feature1, old_feature1_, old_feature2, old_feature2_, old_feature3, old_feature3_, old_feature4, old_feature4_, temper_p = 100):
        min_length12 = min(old_feature1.size(0), old_feature2.size(0))
        min_length13 = min(old_feature1.size(0), old_feature3.size(0))
        min_length14 = min(old_feature1.size(0), old_feature4.size(0))
        min_length23 = min(old_feature2.size(0), old_feature3.size(0))
        min_length24 = min(old_feature2.size(0), old_feature4.size(0))
        min_length34 = min(old_feature3.size(0), old_feature4.size(0))
        post_sim1 = F.cosine_similarity(old_feature1, old_feature1_, dim=1)
        neg_sim12 = F.cosine_similarity(old_feature1[:min_length12], old_feature2[:min_length12], dim=1)
        neg_sim13 = F.cosine_similarity(old_feature1[:min_length13], old_feature3[:min_length13], dim=1)
        neg_sim14 = F.cosine_similarity(old_feature1[:min_length14], old_feature4[:min_length14], dim=1)
        loss_old1 = -torch.log(torch.exp(torch.mean(post_sim1)) / temper_p / (
                    torch.exp(torch.mean(neg_sim12)) / temper_p + torch.exp(torch.mean(neg_sim13)) / temper_p + torch.exp(torch.mean(neg_sim14)) / temper_p))

        post_sim2 = F.cosine_similarity(old_feature2, old_feature2_, dim=1)
        neg_sim21 = F.cosine_similarity(old_feature2[:min_length12], old_feature1[:min_length12], dim=1)
        neg_sim23 = F.cosine_similarity(old_feature2[:min_length23], old_feature3[:min_length23], dim=1)
        neg_sim24 = F.cosine_similarity(old_feature2[:min_length24], old_feature4[:min_length24], dim=1)
        loss_old2 = -torch.log((torch.mean(post_sim2) / temper_p) / (
                torch.exp(torch.mean(neg_sim21)) / temper_p + torch.exp(torch.mean(neg_sim23)) / temper_p + torch.exp(torch.mean(neg_sim24)) / temper_p))

        post_sim3 = F.cosine_similarity(old_feature3, old_feature3_, dim=1)
        neg_sim31 = F.cosine_similarity(old_feature3[:min_length13], old_feature1[:min_length13], dim=1)
        neg_sim32 = F.cosine_similarity(old_feature3[:min_length23], old_feature2[:min_length23], dim=1)
        neg_sim34 = F.cosine_similarity(old_feature3[:min_length34], old_feature4[:min_length34], dim=1)
        loss_old3 = -torch.log(torch.exp(torch.mean(post_sim3)) / temper_p / (
                torch.exp(torch.mean(neg_sim31)) / temper_p + torch.exp(torch.mean(neg_sim32)) / temper_p + torch.exp(torch.mean(neg_sim34)) / temper_p))

        post_sim4 = F.cosine_similarity(old_feature4, old_feature4_, dim=1)
        neg_sim41 = F.cosine_similarity(old_feature4[:min_length14], old_feature1[:min_length14], dim=1)
        neg_sim42 = F.cosine_similarity(old_feature4[:min_length24], old_feature2[:min_length24], dim=1)
        neg_sim43 = F.cosine_similarity(old_feature4[:min_length34], old_feature3[:min_length34], dim=1)
        loss_old4 = -torch.log(torch.exp(torch.mean(post_sim4)) / temper_p / (
                torch.exp(torch.mean(neg_sim41)) / temper_p + torch.exp(torch.mean(neg_sim42)) / temper_p + torch.exp(torch.mean(neg_sim43)) / temper_p))

        loss_old = loss_old1+loss_old2+loss_old3+loss_old4
        print(f"post_sim1:{torch.mean(post_sim1)} neg_sim12:{torch.mean(neg_sim12)}")

        return loss_old, [torch.mean(post_sim1), torch.mean(neg_sim12), torch.mean(neg_sim13), torch.mean(neg_sim14)], \
               [torch.mean(post_sim2), torch.mean(neg_sim21), torch.mean(neg_sim23), torch.mean(neg_sim24)], \
               [torch.mean(post_sim3), torch.mean(neg_sim31), torch.mean(neg_sim32), torch.mean(neg_sim34)], \
               [torch.mean(post_sim4), torch.mean(neg_sim41), torch.mean(neg_sim42), torch.mean(neg_sim43)]

    # Transition processing
    def prop_jump(self, old_feature1, old_feature1_, old_feature2, old_feature2_, old_feature3, old_feature3_, old_feature4, old_feature4_,
                  old1_input, old1_input_, old2_input, old2_input_, old3_input, old3_input_, old4_input, old4_input_):
        # Calculate similarity
        a = 3 # Transition parameters
        min_length12 = min(old_feature1.size(0), old_feature2.size(0))
        min_length13 = min(old_feature1.size(0), old_feature3.size(0))
        min_length14 = min(old_feature1.size(0), old_feature4.size(0))
        min_length23 = min(old_feature2.size(0), old_feature3.size(0))
        min_length24 = min(old_feature2.size(0), old_feature4.size(0))
        min_length34 = min(old_feature3.size(0), old_feature4.size(0))

        neg_sim12 = F.cosine_similarity(old_feature1[:min_length12], old_feature2[:min_length12], dim=1).unsqueeze(1)
        neg_sim13 = F.cosine_similarity(old_feature1[:min_length13], old_feature3[:min_length13], dim=1).unsqueeze(1)
        neg_sim14 = F.cosine_similarity(old_feature1[:min_length14], old_feature4[:min_length14], dim=1).unsqueeze(1)
        self.jump_mat221 = self.transition_probability(neg_sim12, a)
        self.jump_mat321 = self.transition_probability(neg_sim13, a)
        self.jump_mat421 = self.transition_probability(neg_sim14, a)

        neg_sim21 = F.cosine_similarity(old_feature2[:min_length12], old_feature1[:min_length12], dim=1).unsqueeze(1)
        neg_sim23 = F.cosine_similarity(old_feature2[:min_length23], old_feature3[:min_length23], dim=1).unsqueeze(1)
        neg_sim24 = F.cosine_similarity(old_feature2[:min_length24], old_feature4[:min_length24], dim=1).unsqueeze(1)
        self.jump_mat122 = self.transition_probability(neg_sim21, a)
        self.jump_mat322 = self.transition_probability(neg_sim23, a)
        self.jump_mat422 = self.transition_probability(neg_sim24, a)

        neg_sim31 = F.cosine_similarity(old_feature3[:min_length13], old_feature1[:min_length13], dim=1).unsqueeze(1)
        neg_sim32 = F.cosine_similarity(old_feature3[:min_length23], old_feature2[:min_length23], dim=1).unsqueeze(1)
        neg_sim34 = F.cosine_similarity(old_feature3[:min_length34], old_feature4[:min_length34], dim=1).unsqueeze(1)
        self.jump_mat123 = self.transition_probability(neg_sim31, a)
        self.jump_mat223 = self.transition_probability(neg_sim32, a)
        self.jump_mat423 = self.transition_probability(neg_sim34, a)

        # Preparation of positive and negative samples
        post2_1 = copy.deepcopy(old1_input_.detach())
        post3_1 = copy.deepcopy(old1_input_.detach())
        post4_1 = copy.deepcopy(old1_input_.detach())
        post1_2 = copy.deepcopy(old2_input_.detach())
        post3_2 = copy.deepcopy(old2_input_.detach())
        post4_2 = copy.deepcopy(old2_input_.detach())
        post1_3 = copy.deepcopy(old3_input_.detach())
        post2_3 = copy.deepcopy(old3_input_.detach())
        post4_3 = copy.deepcopy(old3_input_.detach())
        post1_4 = copy.deepcopy(old4_input_.detach())
        post2_4 = copy.deepcopy(old4_input_.detach())
        post3_4 = copy.deepcopy(old4_input_.detach())

        neg2_1 = copy.deepcopy(old2_input[:min_length12].detach())
        neg3_1 = copy.deepcopy(old3_input[:min_length13].detach())
        neg4_1 = copy.deepcopy(old4_input[:min_length14].detach())
        neg1_2 = copy.deepcopy(old1_input[:min_length12].detach())
        neg3_2 = copy.deepcopy(old3_input[:min_length23].detach())
        neg4_2 = copy.deepcopy(old4_input[:min_length24].detach())
        neg1_3 = copy.deepcopy(old1_input[:min_length13].detach())
        neg2_3 = copy.deepcopy(old2_input[:min_length23].detach())
        neg4_3 = copy.deepcopy(old4_input[:min_length34].detach())
        neg1_4 = copy.deepcopy(old1_input[:min_length14].detach())
        neg2_4 = copy.deepcopy(old2_input[:min_length24].detach())
        neg3_4 = copy.deepcopy(old3_input[:min_length34].detach())

        # Construct positive and negative samples based on the transition situation
        for i in range(self.jump_mat221.size(0)):
            if self.jump_mat221[i, 0] == 1:
                post2_1[i] = old2_input[i]
            else:
                if i < self.jump_mat221.size(0) - 1:
                    neg2_1[i] = neg2_1[i + 1]

        for i in range(self.jump_mat321.size(0)):
            if self.jump_mat321[i, 0] == 1:
                post3_1[i] = old3_input[i]
            else:
                if i < self.jump_mat321.size(0) - 1:
                    neg3_1[i] = neg3_1[i + 1]

        for i in range(self.jump_mat421.size(0)):
            if self.jump_mat421[i, 0] == 1:
                post4_1[i] = old4_input[i]
            else:
                if i < self.jump_mat421.size(0) - 1:
                    neg4_1[i] = neg4_1[i + 1]

        for i in range(self.jump_mat122.size(0)):
            if self.jump_mat122[i, 0] == 1:
                post1_2[i] = old1_input[i]
            else:
                if i < self.jump_mat122.size(0) - 1:
                    neg1_2[i] = neg1_2[i + 1]

        for i in range(self.jump_mat322.size(0)):
            if self.jump_mat322[i, 0] == 1:
                post3_2[i] = old3_input[i]
            else:
                if i < self.jump_mat322.size(0) - 1:
                    neg3_2[i] = neg3_2[i + 1]

        for i in range(self.jump_mat422.size(0)):
            if self.jump_mat422[i, 0] == 1:
                post4_2[i] = old4_input[i]
            else:
                if i < self.jump_mat422.size(0) - 1:
                    neg4_2[i] = neg4_2[i + 1]

        for i in range(self.jump_mat123.size(0)):
            if self.jump_mat123[i, 0] == 1:
                post1_3[i] = old1_input[i]
            else:
                if i < self.jump_mat123.size(0) - 1:
                    neg1_3[i] = neg1_3[i + 1]

        for i in range(self.jump_mat223.size(0)):
            if self.jump_mat223[i, 0] == 1:
                post2_3[i] = old2_input[i]
            else:
                if i < self.jump_mat223.size(0) - 1:
                    neg2_3[i] = neg2_3[i + 1]

        for i in range(self.jump_mat423.size(0)):
            if self.jump_mat423[i, 0] == 1:
                post4_3[i] = old4_input[i]
            else:
                if i < self.jump_mat423.size(0) - 1:
                    neg4_3[i] = neg4_3[i + 1]

        # Complete the jump_mat matrix and negative sample matrix, and unify the shape of the positive and negative sample matrices
        max_length = [len(old_feature1), len(old_feature2), len(old_feature3), len(old_feature4)]

        self.jump_mat221 = F.pad(self.jump_mat221, (0, 0, 0, max_length[0] - self.jump_mat221.size(0)), "constant", 0)
        self.jump_mat321 = F.pad(self.jump_mat321, (0, 0, 0, max_length[0] - self.jump_mat321.size(0)), "constant", 0)
        self.jump_mat421 = F.pad(self.jump_mat421, (0, 0, 0, max_length[0] - self.jump_mat421.size(0)), "constant", 0)
        self.jump_mat122 = F.pad(self.jump_mat122, (0, 0, 0, max_length[1] - self.jump_mat122.size(0)), "constant", 0)
        self.jump_mat322 = F.pad(self.jump_mat322, (0, 0, 0, max_length[1] - self.jump_mat322.size(0)), "constant", 0)
        self.jump_mat422 = F.pad(self.jump_mat422, (0, 0, 0, max_length[1] - self.jump_mat422.size(0)), "constant", 0)
        self.jump_mat123 = F.pad(self.jump_mat123, (0, 0, 0, max_length[2] - self.jump_mat123.size(0)), "constant", 0)
        self.jump_mat223 = F.pad(self.jump_mat223, (0, 0, 0, max_length[2] - self.jump_mat223.size(0)), "constant", 0)
        self.jump_mat423 = F.pad(self.jump_mat423, (0, 0, 0, max_length[2] - self.jump_mat423.size(0)), "constant", 0)

        neg2_1_ = torch.cat([neg2_1, neg2_1[-1].expand(max_length[0] - neg2_1.size(0), *neg2_1.size()[1:])], dim=0)
        neg3_1_ = torch.cat([neg3_1, neg3_1[-1].expand(max_length[0] - neg3_1.size(0), *neg3_1.size()[1:])], dim=0)
        neg4_1_ = torch.cat([neg4_1, neg4_1[-1].expand(max_length[0] - neg4_1.size(0), *neg4_1.size()[1:])], dim=0)
        neg1_2_ = torch.cat([neg1_2, neg1_2[-1].expand(max_length[1] - neg1_2.size(0), *neg1_2.size()[1:])], dim=0)
        neg3_2_ = torch.cat([neg3_2, neg3_2[-1].expand(max_length[1] - neg3_2.size(0), *neg3_2.size()[1:])], dim=0)
        neg4_2_ = torch.cat([neg4_2, neg4_2[-1].expand(max_length[1] - neg4_2.size(0), *neg4_2.size()[1:])], dim=0)
        neg1_3_ = torch.cat([neg1_3, neg1_3[-1].expand(max_length[2] - neg1_3.size(0), *neg1_3.size()[1:])], dim=0)
        neg2_3_ = torch.cat([neg2_3, neg2_3[-1].expand(max_length[2] - neg2_3.size(0), *neg2_3.size()[1:])], dim=0)
        neg4_3_ = torch.cat([neg4_3, neg4_3[-1].expand(max_length[2] - neg4_3.size(0), *neg4_3.size()[1:])], dim=0)
        neg1_4_ = torch.cat([neg1_4, neg1_4[-1].expand(max_length[3] - neg1_4.size(0), *neg1_4.size()[1:])], dim=0)
        neg2_4_ = torch.cat([neg2_4, neg2_4[-1].expand(max_length[3] - neg2_4.size(0), *neg2_4.size()[1:])], dim=0)
        neg3_4_ = torch.cat([neg3_4, neg3_4[-1].expand(max_length[3] - neg3_4.size(0), *neg3_4.size()[1:])], dim=0)

        # Merge positive and negative samples
        positive_samples = [(old1_input_, post2_1, post3_1, post4_1),
                            (old2_input_, post1_2, post3_2, post4_2),
                            (old3_input_, post1_3, post2_3, post4_3),
                            (old4_input_, post1_4, post2_4, post3_4)]

        negative_samples = [(neg2_1_, neg3_1_, neg4_1_),
                            (neg1_2_, neg3_2_, neg4_2_),
                            (neg1_3_, neg2_3_, neg4_3_),
                            (neg1_4_, neg2_4_, neg3_4_)]

        if len(neg1_4) != len(neg2_4) or len(neg1_4) != len(neg3_4) or len(neg2_4) != len(neg3_4):
            a = 1

        return positive_samples, negative_samples

    # Positive and negative sample output
    def get_new_out(self, old1_input, old2_input, old3_input, old4_input, positive_samples, negative_samples):
        out1 = self.encoder_new(old1_input)
        out_post10 = self.encoder_new(positive_samples[0][0])
        out_post11 = self.encoder_new(positive_samples[0][1])
        out_post12 = self.encoder_new(positive_samples[0][2])

        out2 = self.encoder_new(old2_input)
        out_post20 = self.encoder_new(positive_samples[1][0])
        out_post21 = self.encoder_new(positive_samples[1][1])
        out_post22 = self.encoder_new(positive_samples[1][2])

        out3 = self.encoder_new(old3_input)
        out_post30 = self.encoder_new(positive_samples[2][0])
        out_post31 = self.encoder_new(positive_samples[2][1])
        out_post32 = self.encoder_new(positive_samples[2][2])

        out4 = self.encoder_new(old4_input)
        out_post40 = self.encoder_new(positive_samples[3][0])
        out_post41 = self.encoder_new(positive_samples[3][1])
        out_post42 = self.encoder_new(positive_samples[3][2])

        out_neg11 = self.encoder_new(negative_samples[0][0])
        out_neg12 = self.encoder_new(negative_samples[0][1])

        out_neg21 = self.encoder_new(negative_samples[1][0])
        out_neg22 = self.encoder_new(negative_samples[1][1])

        out_neg31 = self.encoder_new(negative_samples[2][0])
        out_neg32 = self.encoder_new(negative_samples[2][1])

        out_neg41 = self.encoder_new(negative_samples[3][0])
        out_neg42 = self.encoder_new(negative_samples[3][1])

        positive_out = [(out1, out_post10, out_post11, out_post12),
                            (out2, out_post20, out_post21, out_post22),
                            (out3, out_post30, out_post31, out_post32),
                        (out4, out_post40, out_post41, out_post42)]

        negative_out = [(out1, out_neg11, out_neg12),
                            (out2, out_neg21, out_neg22),
                            (out3, out_neg31, out_neg32),
                        (out4, out_neg41, out_neg42)]

        return positive_out, negative_out

    # Obtain the weight parameters of the first layer
    def get_first_layer_weights(self, model):
        for layer in model.net:
            if isinstance(layer, nn.Linear):
                return layer.weight.data
        return None

    # train
    def train(self, epoch=100, temper_p1 = 100, temper_p2=100):
        sim_old1_list = []
        sim_old2_list = []
        sim_old3_list = []
        sim_old4_list = []
        sim_new1_list = []
        sim_new2_list = []
        sim_new3_list = []
        sim_new4_list = []
        losses_old = []
        losses_new = []
        for i in range(epoch):
            idx = random.randint(1, len(self.data_np)-1)
            data = self.data_np[idx]
            new_data = self.data_build(data)

            old1_input = new_data[0][:, :5]
            old1_input_ = new_data[0 + 4][:, :5]
            old2_input = new_data[1][:, :5]
            old2_input_ = new_data[1 + 4][:, :5]
            old3_input = new_data[2][:, :5]
            old3_input_ = new_data[2 + 4][:, :5]
            old4_input = new_data[3][:, :5]
            old4_input_ = new_data[3 + 4][:, :5]

            old_feature1 = self.encoder_old(old1_input)
            old_feature1_ = self.encoder_old(old1_input_)
            old_feature2 = self.encoder_old(old2_input)
            old_feature2_ = self.encoder_old(old2_input_)
            old_feature3 = self.encoder_old(old3_input)
            old_feature3_ = self.encoder_old(old3_input_)
            old_feature4 = self.encoder_old(old4_input)
            old_feature4_ = self.encoder_old(old4_input_)
            #Build Loss_old and update old encoder
            loss_old, sim_old1, sim_old2, sim_old3, sim_old4 = self.get_loss(old_feature1, old_feature1_, old_feature2, old_feature2_, old_feature3, old_feature3_, old_feature4, old_feature4_, temper_p1)
            self.optim_old.zero_grad()
            loss_old.backward()
            self.optim_old.step()
            # Probability transition ->redistribution of positive and negative samples
            positive_samples, negative_samples = self.prop_jump(old_feature1, old_feature1_, old_feature2, old_feature2_, old_feature3, old_feature3_, old_feature4, old_feature4_,
                                                                old1_input, old1_input_, old2_input, old2_input_, old3_input, old3_input_, old4_input, old4_input_)
            # Encoder_new calculation
            positive_out, negative_out = self.get_new_out(old1_input, old2_input, old3_input, old4_input, positive_samples, negative_samples)
            # Building Loss_new
            post1_sim0 = torch.mean(F.cosine_similarity(positive_out[0][0], positive_out[0][1], dim=1).unsqueeze(1))
            post1_sim2 = torch.mean((F.cosine_similarity(positive_out[0][0], positive_out[0][2], dim=1).unsqueeze(1)))
            post1_sim3 = torch.mean(F.cosine_similarity(positive_out[0][0], positive_out[0][3], dim=1).unsqueeze(1))

            post2_sim0 = torch.mean(F.cosine_similarity(positive_out[1][0], positive_out[1][1], dim=1).unsqueeze(1))
            post2_sim1 = torch.mean(F.cosine_similarity(positive_out[1][0], positive_out[1][2], dim=1).unsqueeze(1))
            post2_sim3 = torch.mean(F.cosine_similarity(positive_out[1][0], positive_out[1][3], dim=1).unsqueeze(1))

            post3_sim0 = torch.mean(F.cosine_similarity(positive_out[2][0], positive_out[2][1], dim=1).unsqueeze(1))
            post3_sim1 = torch.mean(F.cosine_similarity(positive_out[2][0], positive_out[2][2], dim=1).unsqueeze(1))
            post3_sim2 = torch.mean(F.cosine_similarity(positive_out[2][0], positive_out[2][3], dim=1).unsqueeze(1))

            post4_sim0 = torch.mean(F.cosine_similarity(positive_out[3][0], positive_out[3][1], dim=1).unsqueeze(1))
            post4_sim1 = torch.mean(F.cosine_similarity(positive_out[3][0], positive_out[3][2], dim=1).unsqueeze(1))
            post4_sim2 = torch.mean(F.cosine_similarity(positive_out[3][0], positive_out[3][3], dim=1).unsqueeze(1))

            neg1_sim2 = torch.mean(F.cosine_similarity(negative_out[0][0], negative_out[0][1], dim=1).unsqueeze(1))
            neg1_sim3 = torch.mean(F.cosine_similarity(negative_out[0][0], negative_out[0][2], dim=1).unsqueeze(1))

            neg2_sim1 = torch.mean(F.cosine_similarity(negative_out[1][0], negative_out[1][1], dim=1).unsqueeze(1))
            neg2_sim3 = torch.mean(F.cosine_similarity(negative_out[1][0], negative_out[1][2], dim=1).unsqueeze(1))

            neg3_sim1 = torch.mean(F.cosine_similarity(negative_out[2][0], negative_out[2][1], dim=1).unsqueeze(1))
            neg3_sim2 = torch.mean(F.cosine_similarity(negative_out[2][0], negative_out[2][2], dim=1).unsqueeze(1))

            neg4_sim1 = torch.mean(F.cosine_similarity(negative_out[3][0], negative_out[3][1], dim=1).unsqueeze(1))
            neg4_sim2 = torch.mean(F.cosine_similarity(negative_out[3][0], negative_out[3][2], dim=1).unsqueeze(1))

            loss_new1 = -torch.log((torch.exp(post1_sim0)/temper_p2 + torch.exp(post1_sim2)/temper_p2 + torch.exp(post1_sim3)/temper_p2)
                          /(torch.exp(neg1_sim2)/temper_p2+torch.exp(neg1_sim3)/temper_p2))

            loss_new2 = -torch.log((torch.exp(post2_sim0) / temper_p2 + torch.exp(post2_sim1) / temper_p2 + torch.exp(post2_sim3) / temper_p2)
                        / (torch.exp(neg2_sim1) / temper_p2 + torch.exp(neg2_sim3) / temper_p2))

            loss_new3 = -torch.log((torch.exp(post3_sim0) / temper_p2 + torch.exp(post3_sim2) / temper_p2 + torch.exp(post3_sim1) / temper_p2)
                        / (torch.exp(neg3_sim1) / temper_p2 + torch.exp(neg3_sim2) / temper_p2))

            loss_new4 = -torch.log((torch.exp(post4_sim0) / temper_p2 + torch.exp(post4_sim2) / temper_p2 + torch.exp(post4_sim1) / temper_p2)
                        /(torch.exp(neg4_sim1) / temper_p2 + torch.exp(neg4_sim2) / temper_p2))

            #Build Total Loss Training
            loss_new = loss_new1+loss_new2+loss_new3+loss_new4
            self.optim_new.zero_grad()
            loss_new.backward()
            self.optim_new.step()
            print(f"epoch:{i} loss_old:{loss_old}, loss_new:{loss_new}, post1_sim0:{post1_sim0}, neg1_sim2:{neg1_sim2}")

            sim_new1_list.append([post1_sim0.item(), neg1_sim2.item(), neg1_sim3.item()])
            sim_new2_list.append([post2_sim0.item(), neg2_sim1.item(), neg2_sim3.item()])
            sim_new3_list.append([post3_sim0.item(), neg3_sim1.item(), neg3_sim2.item()])
            sim_new4_list.append([post4_sim0.item(), neg4_sim1.item(), neg4_sim2.item()])
            sim_old1_list.append(sim_old1)
            sim_old2_list.append(sim_old2)
            sim_old3_list.append(sim_old3)
            sim_old4_list.append(sim_old4)
            losses_old.append(loss_old.item())
            losses_new.append(loss_new.item())

            # Weight sharing
            if i%30 == 0:
                self.encoder_old = self.encoder_new

        return sim_new1_list, sim_new2_list, sim_new3_list, sim_new4_list, sim_old1_list, sim_old2_list, sim_old3_list, sim_old4_list, losses_old, losses_new

# Retrieve the starting index of each day from the data
def get_day_idx(data):
    days = []
    days_idx = []
    for i in range(len(data)):
        if i==0:
            days.append([0])
            continue
        if(data[i, 0]<data[i-1, 0]):
            days.append([1])
            days_idx.append(i)
        else:
            days.append([0])
    new_data = np.concatenate((data, np.array(days)), axis=1)
    days_idx = np.array(days_idx)
    return new_data, days_idx

def data_segmentation(data, idxs):
    data_list = []
    for i in range(len(idxs)-1):
        data_list.append((data[idxs[i]:idxs[i+1], :]).tolist())
    data_list.append((data[idxs[-1]:, :]).tolist())
    return data_list

if __name__ == "__main__":
    ##读取数据
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

    # 数据准备
    _, days_1 = get_day_idx(features1[1:, :])  # 前面的new_feature是从第2个开始的，所以这里从1开始
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
                     zip(data_list1, data_list2, data_list3, data_list4)]  # 数据整合
    combined_np = [[np.array(row1), np.array(row2), np.array(row3), np.array(row4)] for row1, row2, row3, row4 in
                   zip(data_list1, data_list2, data_list3, data_list4)]

    # Define model training parameters
    width_dim_c = 56
    depth_c = 4
    state_dim_c = 5
    action_dim_c = 10
    contrastive_mjup = PJCL(width_dim_c, depth_c, state_dim_c, action_dim_c, combined_np, mean, std)

    # train a contrastive learning model
    epoch = 300
    sim_new1_list, sim_new2_list, sim_new3_list, sim_new4_list, sim_old1_list, sim_old2_list, sim_old3_list, sim_old4_list, losses_old, losses_new = contrastive_mjup.train(epoch)
    # torch.save(contrastive_mjup.encoder_new.state_dict(), "./model/contrastive_model")






