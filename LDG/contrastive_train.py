'''
Comparative learning and training
'''
import copy
import torch
import torch.nn as nn
import numpy as np
import random
import data_pretreatment
import torch.nn.functional as F


# Comparative Learning Network Definition
class Contrastive_model(nn.Module):
    def __init__(self, width_dim, depth, state_dim, action_dim):
        super(Contrastive_model, self).__init__()
        layers = []
        layers.append(nn.Linear(state_dim, width_dim))  # Input layer
        layers.append(nn.Tanh())  # Add ReLU activation function
        for _ in range(depth):
            layers.append(nn.Linear(width_dim, width_dim))  # Hidden layer
            layers.append(nn.Tanh())  # Add ReLU activation function
        self.net = nn.Sequential(*layers)
        # Add MLP decoding network
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


class PJCL():
    '''
    The probability distribution function of similarity is obtained by inputting the similarity to obtain the probability value,
    and then randomly sampling from 0 to 1. If it is less than this value, it transitions, otherwise it remains unchanged
    '''

    def __init__(self, width_dim_c, depth_c, state_dim_c, action_dim_c, data_np, mean, std):
        # New and old models
        self.encoder_old = Contrastive_model(width_dim_c, depth_c, state_dim_c, action_dim_c)
        self.encoder_new = Contrastive_model(width_dim_c, depth_c, state_dim_c, action_dim_c)
        # optimization parameter
        self.learn_rate_old = 1e-3
        self.learn_rate_new = 1e-3
        self.optim_old = torch.optim.Adam(self.encoder_old.parameters(), lr=self.learn_rate_old)
        self.optim_new = torch.optim.Adam(self.encoder_new.parameters(), lr=self.learn_rate_new)
        # hopping model
        self.jump_mat221 = None
        self.jump_mat321 = None
        self.jump_mat122 = None
        self.jump_mat322 = None
        self.jump_mat123 = None
        self.jump_mat223 = None
        # Data loading/processing
        self.data_np = data_np
        self.mean = mean
        self.std = std
        self.data_augmentation(True)

    def increasing_function(self, x, a):
        return x ** a

    def transition_probability(self, similarity_matrix, a):
        '''
        Calculate the probability of transition based on the similarity value of each element,
        and perform sampling to determine whether there is a transition.
        Output a matrix of the same size consisting of 0 and 1, where 1 represents a transition and 0 represents no transition
        '''
        # Obtain the shape of the matrix
        rows, cols = similarity_matrix.shape
        data_list = []
        # Initialization result matrix
        transition_matrix = torch.zeros_like(similarity_matrix, dtype=torch.int)
        # Traverse each element of the similarity matrix, calculate transition probabilities, and sample
        for i in range(rows):
            for j in range(cols):
                similarity = similarity_matrix[i, j]
                prob = self.increasing_function(similarity, a)
                # Using random sampling to determine whether to transition
                if torch.rand(1).item() < prob:
                    transition_matrix[i, j] = 1
                    data_list.append([similarity.item(), prob.item(), 1])
                else:
                    data_list.append([similarity.item(), prob.item(), 0])

        return transition_matrix, data_list

    # Enhance/standardize the original data
    def data_augmentation(self, flag=False):
        for i in range(len(self.data_np)):
            for j in range(3):
                data_np = self.data_np[i][j]
                noise = np.random.uniform(-3, 3, data_np.shape)
                # Adding noise to the original feature vector
                noisy_feature_vectors = data_np + noise
                # Ensure that all elements are non negative and set negative elements to 0
                noisy_feature_vectors = np.maximum(noisy_feature_vectors, 0)
                if flag:
                    self.data_np[i][j] = data_np / self.mean
                    noisy_feature_vectors = noisy_feature_vectors / self.mean
                self.data_np[i].append(copy.deepcopy(noisy_feature_vectors))

    # Output the data and enhanced data of three furnaces on a certain day in the same format and shape
    def data_build(self, data):
        new_data = []
        for i in range(len(data)):
            new_data.append(torch.FloatTensor(data[i]).requires_grad_(True))
        return new_data

    def get_loss(self, old_feature1, old_feature1_, old_feature2, old_feature2_, old_feature3, old_feature3_,
                 temper_p=10):
        min_length12 = min(old_feature1.size(0), old_feature2.size(0))
        min_length13 = min(old_feature1.size(0), old_feature3.size(0))
        min_length23 = min(old_feature2.size(0), old_feature3.size(0))
        post_sim1 = F.cosine_similarity(old_feature1, old_feature1_, dim=1)
        neg_sim12 = F.cosine_similarity(old_feature1[:min_length12], old_feature2[:min_length12], dim=1)
        neg_sim13 = F.cosine_similarity(old_feature1[:min_length13], old_feature3[:min_length13], dim=1)
        neg_sim12_ = F.cosine_similarity(old_feature1[:min_length12], old_feature2_[:min_length12], dim=1)
        neg_sim13_ = F.cosine_similarity(old_feature1[:min_length13], old_feature3_[:min_length13], dim=1)
        loss_old1 = -torch.log(torch.exp(torch.mean(post_sim1)) / temper_p / (
                torch.exp(torch.mean(neg_sim12)) / temper_p + torch.exp(torch.mean(neg_sim13)) / temper_p))

        post_sim2 = F.cosine_similarity(old_feature2, old_feature2_, dim=1)
        neg_sim21 = F.cosine_similarity(old_feature2[:min_length12], old_feature1[:min_length12], dim=1)
        neg_sim23 = F.cosine_similarity(old_feature2[:min_length23], old_feature3[:min_length23], dim=1)
        neg_sim21_ = F.cosine_similarity(old_feature2[:min_length12], old_feature1_[:min_length12], dim=1)
        neg_sim23_ = F.cosine_similarity(old_feature2[:min_length23], old_feature3_[:min_length23], dim=1)

        loss_old2 = -torch.log(torch.exp(torch.mean(post_sim2)) / temper_p / (
                torch.exp(torch.mean(neg_sim21)) / temper_p + torch.exp(torch.mean(neg_sim23)) / temper_p))

        post_sim3 = F.cosine_similarity(old_feature3, old_feature3_, dim=1)
        neg_sim31 = F.cosine_similarity(old_feature3[:min_length13], old_feature1[:min_length13], dim=1)
        neg_sim32 = F.cosine_similarity(old_feature3[:min_length23], old_feature2[:min_length23], dim=1)
        neg_sim31_ = F.cosine_similarity(old_feature3[:min_length13], old_feature1_[:min_length13], dim=1)
        neg_sim32_ = F.cosine_similarity(old_feature3[:min_length23], old_feature2_[:min_length23], dim=1)

        loss_old3 = -torch.log(torch.exp(torch.mean(post_sim3)) / temper_p / (
                torch.exp(torch.mean(neg_sim31)) / temper_p + torch.exp(torch.mean(neg_sim32)) / temper_p))
        loss_old = loss_old1 + loss_old2 + loss_old3
        a = [torch.mean(post_sim1).item(), torch.mean(neg_sim12).item(), torch.mean(neg_sim13).item()]
        b = [torch.mean(post_sim2).item(), torch.mean(neg_sim21).item(), torch.mean(neg_sim23).item()]
        c = [torch.mean(post_sim3).item(), torch.mean(neg_sim31).item(), torch.mean(neg_sim32).item()]
        a1 = loss_old1
        b1 = loss_old2
        c1 = loss_old3
        if torch.isnan(loss_old).any():
            print("The data contains NaN")
        print(f"post_sim1:{torch.mean(post_sim1)} neg_sim12:{torch.mean(neg_sim12)}")

        return loss_old, [torch.mean(post_sim1).item(), torch.mean(neg_sim12).item(), torch.mean(neg_sim13).item()], [
            torch.mean(post_sim2).item(), torch.mean(neg_sim21).item(), torch.mean(neg_sim23).item()], [
            torch.mean(post_sim3).item(), torch.mean(neg_sim31).item(), torch.mean(neg_sim32).item()]

    def prop_jump(self, old_feature1, old_feature1_, old_feature2, old_feature2_, old_feature3, old_feature3_,
                  old1_input, old1_input_, old2_input, old2_input_, old3_input, old3_input_, statistics_flag):
        # Statistical transition situation
        post1 = len(old_feature1)
        post2 = len(old_feature2)
        post3 = len(old_feature3)
        # Calculate similarity
        a = 3  # Transition parameters
        min_length12 = min(old_feature1.size(0), old_feature2.size(0))
        min_length13 = min(old_feature1.size(0), old_feature3.size(0))
        min_length23 = min(old_feature2.size(0), old_feature3.size(0))

        neg_sim12 = F.cosine_similarity(old_feature1[:min_length12], old_feature2[:min_length12], dim=1).unsqueeze(1)
        neg_sim13 = F.cosine_similarity(old_feature1[:min_length13], old_feature3[:min_length13], dim=1).unsqueeze(1)
        self.jump_mat221, yueqian21 = self.transition_probability(neg_sim12, a)
        self.jump_mat321, yueqian31 = self.transition_probability(neg_sim13, a)
        num_21 = self.jump_mat221.sum()
        num_31 = self.jump_mat321.sum()

        neg_sim21 = F.cosine_similarity(old_feature2[:min_length12], old_feature1[:min_length12], dim=1).unsqueeze(1)
        neg_sim23 = F.cosine_similarity(old_feature2[:min_length23], old_feature3[:min_length23], dim=1).unsqueeze(1)
        self.jump_mat122, yueqian12 = self.transition_probability(neg_sim21, a)
        self.jump_mat322, yueqian32 = self.transition_probability(neg_sim23, a)
        num_12 = self.jump_mat122.sum()
        num_32 = self.jump_mat322.sum()

        neg_sim31 = F.cosine_similarity(old_feature3[:min_length13], old_feature1[:min_length13], dim=1).unsqueeze(1)
        neg_sim32 = F.cosine_similarity(old_feature3[:min_length23], old_feature2[:min_length23], dim=1).unsqueeze(1)
        self.jump_mat123, yueqian13 = self.transition_probability(neg_sim31, a)
        self.jump_mat223, yueqian23 = self.transition_probability(neg_sim32, a)
        num_13 = self.jump_mat123.sum()
        num_23 = self.jump_mat223.sum()

        prob1 = (num_13 + num_12) / (2 * post1)
        prob2 = (num_23 + num_21) / (2 * post2)
        prob3 = (num_31 + num_32) / (2 * post3)
        num1 = (num_13 + num_12)
        num2 = (num_23 + num_21)
        num3 = (num_31 + num_32)

        def get_statistics_sim_pinlv(yueqianlist):
            '''
            Statistically analyze the distribution of eligible items in the yueqianlist
            '''
            # Initialize statistical list
            statistics = [0] * 10
            # Traverse the yueqianlist and count the data that meets the criteria
            for item in yueqianlist:
                if item[2] == 1:  # Check if the third element is 1
                    index = int(item[1] * 10)  # Calculate interval index
                    if 0 <= index < 10:  # Ensure that the index is within range
                        statistics[index] += 1
            return statistics

        statistics = []
        if statistics_flag:
            statistics21 = get_statistics_sim_pinlv(yueqian21)
            statistics31 = get_statistics_sim_pinlv(yueqian31)
            statistics12 = get_statistics_sim_pinlv(yueqian12)
            statistics32 = get_statistics_sim_pinlv(yueqian32)
            statistics13 = get_statistics_sim_pinlv(yueqian13)
            statistics23 = get_statistics_sim_pinlv(yueqian23)
            statistics = [statistics21, statistics31, statistics12, statistics32, statistics13, statistics23]

        # Preparation of positive and negative samples
        post2_1 = copy.deepcopy(old1_input_.detach())
        post3_1 = copy.deepcopy(old1_input_.detach())
        post1_2 = copy.deepcopy(old2_input_.detach())
        post3_2 = copy.deepcopy(old2_input_.detach())
        post1_3 = copy.deepcopy(old3_input_.detach())
        post2_3 = copy.deepcopy(old3_input_.detach())

        neg2_1 = copy.deepcopy(old2_input[:min_length12].detach())
        neg3_1 = copy.deepcopy(old3_input[:min_length13].detach())
        neg1_2 = copy.deepcopy(old1_input[:min_length12].detach())
        neg3_2 = copy.deepcopy(old3_input[:min_length23].detach())
        neg1_3 = copy.deepcopy(old1_input[:min_length13].detach())
        neg2_3 = copy.deepcopy(old2_input[:min_length23].detach())

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

        # Complete the jump_mat matrix and negative sample matrix, unify the shape of the positive and negative sample matrices, and facilitate calculations
        max_length = [len(old_feature1), len(old_feature2), len(old_feature3)]
        self.jump_mat221 = F.pad(self.jump_mat221, (0, 0, 0, max_length[0] - self.jump_mat221.size(0)), "constant", 0)
        self.jump_mat321 = F.pad(self.jump_mat321, (0, 0, 0, max_length[0] - self.jump_mat321.size(0)), "constant", 0)
        self.jump_mat122 = F.pad(self.jump_mat122, (0, 0, 0, max_length[1] - self.jump_mat122.size(0)), "constant", 0)
        self.jump_mat322 = F.pad(self.jump_mat322, (0, 0, 0, max_length[1] - self.jump_mat322.size(0)), "constant", 0)
        self.jump_mat123 = F.pad(self.jump_mat123, (0, 0, 0, max_length[2] - self.jump_mat123.size(0)), "constant", 0)
        self.jump_mat223 = F.pad(self.jump_mat223, (0, 0, 0, max_length[2] - self.jump_mat223.size(0)), "constant", 0)

        neg2_1 = torch.cat([neg2_1, neg2_1[-1].expand(max_length[0] - neg2_1.size(0), *neg2_1.size()[1:])], dim=0)
        neg3_1 = torch.cat([neg3_1, neg3_1[-1].expand(max_length[0] - neg3_1.size(0), *neg3_1.size()[1:])], dim=0)
        neg1_2 = torch.cat([neg1_2, neg1_2[-1].expand(max_length[1] - neg1_2.size(0), *neg1_2.size()[1:])], dim=0)
        neg3_2 = torch.cat([neg3_2, neg3_2[-1].expand(max_length[1] - neg3_2.size(0), *neg3_2.size()[1:])], dim=0)
        neg1_3 = torch.cat([neg1_3, neg1_3[-1].expand(max_length[2] - neg1_3.size(0), *neg1_3.size()[1:])], dim=0)
        neg2_3 = torch.cat([neg2_3, neg2_3[-1].expand(max_length[2] - neg2_3.size(0), *neg2_3.size()[1:])], dim=0)

        # Merge positive and negative samples
        positive_samples = [(old1_input_, post2_1, post3_1),
                            (old2_input_, post1_2, post3_2),
                            (old3_input_, post1_3, post2_3)]
        negative_samples = [(neg2_1, neg3_1),
                            (neg1_2, neg3_2),
                            (neg1_3, neg2_3)]

        return positive_samples, negative_samples, [prob1.item(), prob2.item(), prob3.item()], statistics, [num1.item(),
                                                                                                        num2.item(),
                                                                                                        num3.item()]

    def get_new_out(self, old1_input, old2_input, old3_input, positive_samples, negative_samples):
        # Positive sample output
        out1 = self.encoder_new(old1_input)
        if torch.isnan(out1).any():
            a = 1
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

        # Negative sample output
        out_neg11 = self.encoder_new(negative_samples[0][0])
        out_neg12 = self.encoder_new(negative_samples[0][1])

        out_neg21 = self.encoder_new(negative_samples[1][0])
        out_neg22 = self.encoder_new(negative_samples[1][1])

        out_neg31 = self.encoder_new(negative_samples[2][0])
        out_neg32 = self.encoder_new(negative_samples[2][1])

        positive_out = [(out1, out_post10, out_post11, out_post12),
                        (out2, out_post20, out_post21, out_post22),
                        (out3, out_post30, out_post31, out_post32)]

        negative_out = [(out1, out_neg11, out_neg12),
                        (out2, out_neg21, out_neg22),
                        (out3, out_neg31, out_neg32)]

        return positive_out, negative_out

    # Define a function to obtain the weight parameters of the first layer
    def get_first_layer_weights(self, model):
        # Traverse the modules of the model, find the first linear layer and return its weight parameters
        for layer in model.net:
            if isinstance(layer, nn.Linear):
                return layer.weight.data
        return None

    def catlist(self, data_np, idx_list):
        data = copy.deepcopy(data_np[idx_list[0]])
        for i in range(1, len(idx_list)):
            data[0] = np.concatenate((data[0], data_np[i][0]), axis=0)
            data[1] = np.concatenate((data[1], data_np[i][1]), axis=0)
            data[2] = np.concatenate((data[2], data_np[i][2]), axis=0)
            data[3] = np.concatenate((data[3], data_np[i][3]), axis=0)
            data[4] = np.concatenate((data[4], data_np[i][4]), axis=0)
            data[5] = np.concatenate((data[5], data_np[i][5]), axis=0)

        return data

    # Training Part
    def train(self, epoch=100, temper_p1=5, temper_p2=5):
        sim_old1_list = []
        sim_old2_list = []
        sim_old3_list = []
        sim_new1_list = []
        sim_new2_list = []
        sim_new3_list = []
        losses_old = []
        losses_new = []
        prob_lists = []
        statistics = []
        num_list = []
        batchsize = 6
        statistics_flag = 0
        for i in range(epoch):
            # Building data for three furnaces on a certain day
            idx_list = random.sample(range(1, len(self.data_np)), batchsize)
            data = self.catlist(self.data_np, idx_list)
            new_data = self.data_build(data)
            # Encoder_old calculation
            old1_input = new_data[0][:, :3]
            old1_input_ = new_data[0 + 3][:, :3]
            old2_input = new_data[1][:, :3]
            old2_input_ = new_data[1 + 3][:, :3]
            old3_input = new_data[2][:, :3]
            old3_input_ = new_data[2 + 3][:, :3]

            old_input = torch.cat((old1_input, old2_input, old3_input))
            old_input_ = torch.cat((old1_input_, old2_input_, old3_input_))
            old_feature = self.encoder_old(old_input)
            old_feature_ = self.encoder_old(old_input_)
            old_feature1 = self.encoder_old(old1_input)
            old_feature1_ = self.encoder_old(old1_input_)
            old_feature2 = self.encoder_old(old2_input)
            old_feature2_ = self.encoder_old(old2_input_)
            old_feature3 = self.encoder_old(old3_input)
            old_feature3_ = self.encoder_old(old3_input_)
            # Build Loss_old and update old encoder
            loss_old, sim_old1, sim_old2, sim_old3 = self.get_loss(old_feature1, old_feature1_, old_feature2,
                                                                   old_feature2_, old_feature3, old_feature3_,
                                                                   temper_p1)
            self.optim_old.zero_grad()
            loss_old.backward()
            self.optim_old.step()
            # Probability transition ->redistribution of positive and negative samples
            if i == epoch - 1:
                statistics_flag = 1
            positive_samples, negative_samples, prob_list, statistics, num = self.prop_jump(old_feature1, old_feature1_,
                                                                                        old_feature2, old_feature2_,
                                                                                        old_feature3, old_feature3_,
                                                                                        old1_input, old1_input_,
                                                                                        old2_input, old2_input_,
                                                                                        old3_input, old3_input_, 1)
            # encoder_new calculation
            positive_out, negative_out = self.get_new_out(old1_input, old2_input, old3_input, positive_samples,
                                                          negative_samples)
            # bulid loss_new
            post1_sim0 = torch.mean(F.cosine_similarity(positive_out[0][0], positive_out[0][1], dim=1).unsqueeze(1))
            post1_sim2 = torch.mean((F.cosine_similarity(positive_out[0][0], positive_out[0][2], dim=1).unsqueeze(1)))
            post1_sim3 = torch.mean(F.cosine_similarity(positive_out[0][0], positive_out[0][3], dim=1).unsqueeze(1))

            post2_sim0 = torch.mean(F.cosine_similarity(positive_out[1][0], positive_out[1][1], dim=1).unsqueeze(1))
            post2_sim1 = torch.mean(F.cosine_similarity(positive_out[1][0], positive_out[1][2], dim=1).unsqueeze(1))
            post2_sim3 = torch.mean(F.cosine_similarity(positive_out[1][0], positive_out[1][3], dim=1).unsqueeze(1))

            post3_sim0 = torch.mean(F.cosine_similarity(positive_out[2][0], positive_out[2][1], dim=1).unsqueeze(1))
            post3_sim1 = torch.mean(F.cosine_similarity(positive_out[2][0], positive_out[2][2], dim=1).unsqueeze(1))
            post3_sim2 = torch.mean(F.cosine_similarity(positive_out[2][0], positive_out[2][3], dim=1).unsqueeze(1))

            neg1_sim2 = torch.mean(F.cosine_similarity(negative_out[0][0], negative_out[0][1], dim=1).unsqueeze(1))
            neg1_sim3 = torch.mean(F.cosine_similarity(negative_out[0][0], negative_out[0][2], dim=1).unsqueeze(1))

            neg2_sim1 = torch.mean(F.cosine_similarity(negative_out[1][0], negative_out[1][1], dim=1).unsqueeze(1))
            neg2_sim3 = torch.mean(F.cosine_similarity(negative_out[1][0], negative_out[1][2], dim=1).unsqueeze(1))

            neg3_sim1 = torch.mean(F.cosine_similarity(negative_out[2][0], negative_out[2][1], dim=1).unsqueeze(1))
            neg3_sim2 = torch.mean(F.cosine_similarity(negative_out[2][0], negative_out[2][2], dim=1).unsqueeze(1))

            loss_new1 = -torch.log((torch.exp(post1_sim0) / temper_p2 + torch.exp(post1_sim2) / temper_p2 + torch.exp(
                post1_sim3) / temper_p2)
                                   / (torch.exp(neg1_sim2) / temper_p2 + torch.exp(neg1_sim3) / temper_p2))

            loss_new2 = -torch.log((torch.exp(post2_sim0) / temper_p2 + torch.exp(post2_sim1) / temper_p2 + torch.exp(
                post2_sim3) / temper_p2)
                                   / (torch.exp(neg2_sim1) / temper_p2 + torch.exp(neg2_sim3) / temper_p2))

            loss_new3 = -torch.log((torch.exp(post3_sim0) / temper_p2 + torch.exp(post3_sim2) / temper_p2 + torch.exp(
                post3_sim1) / temper_p2)
                                   / (torch.exp(neg3_sim1) / temper_p2 + torch.exp(neg3_sim2) / temper_p2))

            loss_new = loss_new1 + loss_new2 + loss_new3
            self.optim_new.zero_grad()
            loss_new.backward()
            self.optim_new.step()
            print(f"epoch:{i} loss_old:{loss_old}, loss_new:{loss_new}, post1_sim0:{post1_sim0}, neg1_sim2:{neg1_sim2}")

            sim_new1_list.append([post1_sim0.item(), neg1_sim2.item(), neg1_sim3.item()])
            sim_new2_list.append([post2_sim0.item(), neg2_sim1.item(), neg2_sim3.item()])
            sim_new3_list.append([post3_sim0.item(), neg3_sim1.item(), neg3_sim2.item()])
            sim_old1_list.append(sim_old1)
            sim_old2_list.append(sim_old2)
            sim_old3_list.append(sim_old3)
            losses_old.append(loss_old.item())
            losses_new.append(loss_new.item())
            prob_lists.append(prob_list)
            num_list.append(num)

        return sim_new1_list, sim_new2_list, sim_new3_list, sim_old1_list, sim_old2_list, sim_old3_list, losses_old, losses_new, prob_lists, statistics, num_list


def calculate_errors(y_true, y_pred):
    # Ensure the input arrays are numpy arrays
    y_true = y_true.detach().numpy()
    y_pred = y_pred.detach().numpy()

    # Calculate errors
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)

    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse}


def get_day_idx(data):
    '''
    Obtain the starting index of each day in steelmaking data
    '''
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


def data_segmentation(data, idxs):
    data_list = []
    for i in range(len(idxs) - 1):
        data_list.append((data[idxs[i]:idxs[i + 1], :]).tolist())
    data_list.append((data[idxs[-1]:, :]).tolist())
    return data_list


if __name__ == "__main__":
    # read data
    features1, _, mean1, std1 = data_pretreatment.Unified_processing("./data/three_ldg_data1.xlsx")
    new_features1 = data_pretreatment.get_T_num(features1)

    features2, _, mean2, std2 = data_pretreatment.Unified_processing("./data/three_ldg_data2.xlsx")
    new_features2 = data_pretreatment.get_T_num(features2)

    features3, _, mean3, std3 = data_pretreatment.Unified_processing("./data/three_ldg_data3.xlsx")
    new_features3 = data_pretreatment.get_T_num(features3)
    mean = (np.concatenate((new_features1, new_features2, new_features3), axis=0)).mean(axis=0)
    std = (np.concatenate((new_features1, new_features2, new_features3), axis=0)).std(axis=0)

    # data prepare
    _, days_1 = get_day_idx(features1[1:, :])
    _, days_2 = get_day_idx(features2[1:, :])
    _, days_3 = get_day_idx(features3[1:, :])
    days_idx1 = [0] + list(days_1[:-1])
    days_idx2 = [0] + list(days_2[:-1])
    days_idx3 = [0] + list(days_3[:-1])

    data_list1 = data_segmentation(new_features1, days_idx1)
    data_list2 = data_segmentation(new_features2, days_idx2)
    data_list3 = data_segmentation(new_features3, days_idx3)
    combined_list = [[row1, row2, row3] for row1, row2, row3 in zip(data_list1, data_list2, data_list3)]  # 数据整合
    combined_np = [[np.array(row1), np.array(row2), np.array(row3)] for row1, row2, row3 in
                   zip(data_list1, data_list2, data_list3)]

    # Define contrastive learning models
    width_dim_c = 56
    depth_c = 4
    state_dim_c = 3
    action_dim_c = 10
    contrastive_mjup = PJCL(width_dim_c, depth_c, state_dim_c, action_dim_c, combined_np, mean, std)
    # Training contrastive learning models
    epoch = 50
    sim_new1_list, sim_new2_list, sim_new3_list, sim_old1_list, sim_old2_list, sim_old3_list, losses_old, losses_new, prob_lists, statistics, num_list = \
        contrastive_mjup.train(epoch)
