import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init
import math
from util import Regularizer
import pdb


class RelationProjectionMLP(nn.Module):
    """
    define the multi-layer MLP by RelationProjectionLayer
    """
    def __init__(self):
        super().__init__()


class RelationProjectionLayer(nn.Module):
    """ 
    define three NN for w_i mu_i sigma_i
    """
    def __init__(self, nrelation, input_dim, output_dim, ngauss, projection_regularizer, bias=True):
        super().__init__()
        self.relation_num = nrelation
        self.ngauss = ngauss
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.projection_regularizer = projection_regularizer
        self.W_mlp = RelationMlp(self.relation_num, input_dim, output_dim=1, ngauss=ngauss*2)
        self.Mu_mlp = RelationMlp(self.relation_num, input_dim, output_dim, ngauss=ngauss*2)
        self.Sigma_mlp = RelationMlp(self.relation_num, input_dim, output_dim, ngauss=ngauss*2)

    def forward(self, input_embedding, project_relation):
        x = input_embedding
        W_mlp_weight, W_mlp_bias = self.W_mlp(project_relation)
        Mu_mlp_weight, Mu_mlp_bias = self.Mu_mlp(project_relation)
        Sigma_mlp_weight, Sigma_mlp_bias = self.Sigma_mlp(project_relation)
        # B, 2N, dim+1 * B, dim+1, 1 => B, 2N, 1 => B, N=>B, N, 1
        output = F.relu(torch.matmul(x, W_mlp_weight) + W_mlp_bias).squeeze(-1)
        W_output = output[:, :self.ngauss]
        W_output = F.softmax(W_output, dim=1)
        W_output = W_output.unsqueeze(dim=-1) 

        output = F.relu(torch.matmul(x, Mu_mlp_weight) + Mu_mlp_bias)
        Mu_output = output[:, :self.ngauss, :]

        output = F.relu(torch.matmul(x, Sigma_mlp_weight) + Sigma_mlp_bias)
        Sigma_output = output[:, self.ngauss:, :]

        mask_embedding = torch.zeros_like(W_output, requires_grad=True)
        output_embedding1 = torch.cat((Mu_output, W_output), dim=-1)
        output_embedding2 = torch.cat((Sigma_output, mask_embedding), dim=-1)
        output_embedding = torch.cat((output_embedding1, output_embedding2), dim=1)
        
        return output_embedding


class RelationMlp(nn.Module):
    def __init__(self, nrelation, input_dim, output_dim, ngauss, bias=True):
        super().__init__()
        self.mlp_weight = Parameter(torch.Tensor(nrelation, input_dim, output_dim), requires_grad=True)
        if bias:
            self.mlp_bias = Parameter(torch.Tensor(nrelation, ngauss, output_dim), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.mlp_weight, a=math.sqrt(5))
        if self.mlp_bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.mlp_weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.mlp_bias, -bound, bound)
    
    def forward(self, relation):
        return self.mlp_weight[relation], self.mlp_bias[relation]


class AndMLP(nn.Module):
    def __init__(self, nguass, hidden_dim, and_regularizer):
        # hidden_dim: 2(dim+1) * 2(dim+1)
        super(AndMLP, self).__init__()
        self.nguass = nguass
        self.liner_dim = (hidden_dim + 1) * 2
        self.query = nn.Linear(self.liner_dim, self.liner_dim)
        self.key = nn.Linear(self.liner_dim, self.liner_dim)
        self.value = nn.Linear(self.liner_dim, self.liner_dim)

        self.mlp = nn.Linear(self.liner_dim, self.liner_dim)
        self.and_regularizer = and_regularizer

    def forward(self, embedding1, embedding2):
        # pdb.set_trace()
        trans_embedding1 = torch.cat((embedding1[:, :self.nguass, :], embedding1[:, self.nguass:, :]), dim=-1)
        trans_embedding2 = torch.cat((embedding2[:, :self.nguass, :], embedding2[:, self.nguass:, :]), dim=-1)
        q1 = self.query(trans_embedding1)
        k1 = self.key(trans_embedding1)
        v1 = self.value(trans_embedding1)

        q2 = self.query(trans_embedding2)
        k2 = self.key(trans_embedding2)
        v2 = self.value(trans_embedding2)

        d = trans_embedding1.shape[-1]
        attention_scores_1to2 = torch.matmul(q1, k2.transpose(-2, -1))
        attention_scores_1to2 = attention_scores_1to2 / math.sqrt(d)
        attention_probs_1to2 = F.softmax(attention_scores_1to2, dim=-1)

        attention_scores_2to1 = torch.matmul(q2, k1.transpose(-2, -1))
        attention_scores_2to1 = attention_scores_2to1 / math.sqrt(d)
        attention_probs_2to1 = F.softmax(attention_scores_2to1, dim=-1)  

        out_1to2 = torch.matmul(attention_probs_1to2, v2)
        out_2to1 = torch.matmul(attention_probs_2to1, v1)

        out = out_1to2 + out_2to1
        out = F.relu(self.mlp(out))

        W_out = out[:, :, int(d/2-1)] 
        W_out = F.softmax(W_out, dim=1) 
        W_out = W_out.unsqueeze(dim=-1)

        Mu_out = out[:, :, :int(d/2-1)]
        Sigma_out = out[:, :, int(d/2): -1]

        mask_embedding = torch.zeros_like(W_out, requires_grad=True)
        embedding_temp1 = torch.cat((Mu_out, W_out), dim=-1) 
        embedding_temp2 = torch.cat((Sigma_out, mask_embedding), dim=-1)
        output_embedding = torch.cat((embedding_temp1, embedding_temp2), dim=1)

        return output_embedding
    

class OrMLP(nn.Module):
    def __init__(self, n_layers, entity_dim):
        super(OrMLP, self).__init__()
        self.n_layers = n_layers
        self.layers = []
        for i in range(1, self.n_layers + 1):
            setattr(self, "or_layer_{}".format(i), nn.Linear(2 * entity_dim, 2 * entity_dim))
        self.last_layer = nn.Linear(2 * entity_dim, entity_dim)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=-1)
        for i in range(1, self.n_layers + 1):
            x = F.relu(getattr(self, "or_layer_{}".format(i))(x))
        x = self.last_layer(x)
        return x


class NotMLP(nn.Module):
    def __init__(self, n_layers, entity_dim):
        super(NotMLP, self).__init__()
        self.n_layers = n_layers
        self.layers = []
        for i in range(1, self.n_layers + 1):
            setattr(self, "not_layer_{}".format(i), nn.Linear(entity_dim, entity_dim))
        self.last_layer = nn.Linear(entity_dim, entity_dim)

    def forward(self, x):
        for i in range(1, self.n_layers + 1):
            x = F.relu(getattr(self, "not_layer_{}".format(i))(x))
        x = self.last_layer(x)
        return x