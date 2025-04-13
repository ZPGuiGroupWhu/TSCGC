import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch


class GCNLayer(nn.Module):
    def __init__(self,in_features,out_features,dropout=0.0):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.W = Parameter(torch.ones(size=[in_features,out_features]))
        self.reset_parameters(self.W)

    def reset_parameters(self,W):
        torch.nn.init.xavier_uniform(W)

    def forward(self,input,adj):
        input = F.dropout(input,self.dropout,training=True)
        support = torch.mm(input,self.W)
        output = torch.spmm(adj,support)
        output = F.tanh(output)
        return output


