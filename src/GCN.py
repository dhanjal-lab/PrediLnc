import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import time
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.optim as optim

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):

        # Convolution operation
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        return output


class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.W_l = nn.Parameter(torch.FloatTensor(input_dim, input_dim))
        self.W_r = nn.Parameter(torch.FloatTensor(input_dim, input_dim))
        self.W_h = nn.Parameter(torch.FloatTensor(input_dim, input_dim))
        self.W_g = nn.Parameter(torch.FloatTensor(input_dim, input_dim))
        self.gamma = nn.Parameter(torch.FloatTensor([0]))  # trainable parameter γ
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_l)
        nn.init.xavier_uniform_(self.W_r)
        nn.init.xavier_uniform_(self.W_h)
        nn.init.xavier_uniform_(self.W_g)

    def forward(self, H, adj):
        # Compute attention scores
        H_l = torch.matmul(H, self.W_l)
        H_r = torch.matmul(H, self.W_r)
        S = torch.matmul(H_l, torch.transpose(H_r, 0, 1))

        # Apply softmax to normalize attention scores along the last dimension
        beta = F.softmax(S, dim=-1)

        # Weighted sum of input elements based on attention weights
        B = torch.matmul(beta, H)

        # Calculate attention feature
        O = torch.matmul(B, self.W_h)
        O = torch.matmul(O, self.W_g)

        # Interpolation step
        output = torch.matmul(adj,H) + self.gamma * O

        return output

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.relu1 = nn.ReLU()
        self.gc2 = GraphConvolution(nhid, nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.relu2 = nn.ReLU()
        self.attention = AttentionLayer(nhid)

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.gc2(x, adj)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.attention(x,adj)
        return x


def calculate_laplacian(adj):
    # Calculate the degree matrix
    degree = torch.sum(adj, dim=1)
    degree_matrix = torch.diag(degree)

    # Calculate the Laplacian matrix
    laplacian = degree_matrix - adj
    return laplacian

def adj_norm(adj):
    adj_hat = adj + torch.eye(adj.size(0), device=adj.device)

    # Compute degree matrix
    degree = torch.sum(adj_hat, dim=1)
    degree = torch.diag(degree)

    # Compute D^-0.5
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0

    # Normalize adjacency matrix
    adj_normalized = torch.mm(torch.mm(degree_inv_sqrt, adj_hat), degree_inv_sqrt)

    return adj_normalized
