import torch.nn as nn
import math
from torch.nn import Linear
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch
from torch.nn.functional import normalize
from util import square_euclid_distance

class GCNLayer(nn.Module):
    def __init__(self,in_features,out_features,dropout=0.1):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.W = Parameter(torch.ones(size=[in_features,out_features]))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        #if self.bias is not None:
            #self.bias.data.uniform_(-stdv, stdv)

    def forward(self,input,adj):
        #input = F.dropout(input,self.dropout,training=True)
        support = torch.mm(input,self.W)
        output = torch.spmm(adj,support)
        output = torch.relu(output)
        return output

class Graph_Encoder(nn.Module):
    def __init__(self,n_input,n_dim1,n_dim2):
        super(Graph_Encoder, self).__init__()
        self.gnn1 = GCNLayer(n_input, n_dim1)
        self.gnn2 = GCNLayer(n_dim1, n_dim2)
        #self.gnn3 = GCNLayer(n_dim2, n_dim3)
    def forward(self,X,A):
        output = self.gnn1(X,A)
        output = self.gnn2(output,A)
        #output = self.gnn3(output,A)
        A_pred = torch.sigmoid(torch.matmul(output, output.t()))
        return output,A_pred
class Encoder(nn.Module):
    def __init__(self,n_input,n_dim1,n_dim2,n_dim3,n_z):
        super(Encoder, self).__init__()
        self.layer1 = Linear(n_input,n_dim1)
        self.layer2 = Linear(n_dim1, n_dim2)
        self.layer3 = Linear(n_dim2,n_dim3)
        self.layer4 = Linear(n_dim3, n_z)
        self.act = nn.Tanh()
    def forward(self,input):
        output = self.act(self.layer1(input))
        output = self.act(self.layer2(output))
        output = self.act(self.layer3(output))
        output = self.layer4(output)
        return output
class Decoder(nn.Module):
    def __init__(self,n_input,n_dim1,n_dim2,n_dim3,n_dim4):
        super(Decoder, self).__init__()
        self.layer1 = Linear(n_input,n_dim1)
        self.layer2 = Linear(n_dim1, n_dim2)
        self.layer3 = Linear(n_dim2,n_dim3)
        self.layer4 = Linear(n_dim3, n_dim4)
        self.act = nn.ReLU()
    def forward(self,input):
        output = self.act(self.layer1(input))
        output = self.act(self.layer2(output))
        output = self.act(self.layer3(output))
        output = self.layer4(output)
        return output
class GraphAE(nn.Module):
    def __init__(self,n_input,n_dim1,n_dim2,n_hid,n_out):
        super(GraphAE, self).__init__()
        self.encoder = Graph_Encoder(n_input,n_dim1,n_dim2)
        self.proj = MLP(n_dim2,n_hid,n_out)
        self.decoder = Graph_Decoder(n_dim2,n_dim1,n_input)
    def forward(self,X,A):
        hidden_emb,S = self.encoder(X,A)
        proj_emb = self.proj(hidden_emb)
        X_bar =self.decoder(proj_emb,A)
        return hidden_emb,proj_emb,S,X_bar
class GraphEncoder(nn.Module):
    def __init__(self,n_input,n_dim1,n_dim2,n_z):
        super(GraphEncoder, self).__init__()
        self.gnn1 = GCNLayer(n_input, n_dim1)
        self.gnn2 = GCNLayer(n_dim1, n_dim2)
        self.gnn3 = GCNLayer(n_dim2, n_z)
    def forward(self,X,A):
        output = self.gnn1(X,A)
        output = self.gnn2(output,A)
        output = self.gnn3(output,A)
        A_pred = torch.sigmoid(torch.matmul(output, output.t()))
        return output,A_pred
class GAE(nn.Module):
    def __init__(self,n_input,n_dim1,n_dim2,n_z):
        super(GAE, self).__init__()
        self.encoder = GraphEncoder(n_input,n_dim1,n_dim2,n_z)
        self.decoder = GraphEncoder(n_z,n_dim2,n_dim1,n_input)
        #self.decoder = GraphEncoder(n_dim2, n_dim1, n_input, n_input)
    def forward(self,X,A):
        hidden_emb,S = self.encoder(X,A)
        X_bar,_ =self.decoder(hidden_emb,A)
        return hidden_emb,S,X_bar

class Graph_Decoder(nn.Module):
    def __init__(self,n_dim2,n_dim1,n_input):
        super(Graph_Decoder, self).__init__()
        self.gcn1 = GCNLayer(n_dim2, n_dim1)
        self.gcn2 = GCNLayer(n_dim1, n_input)
        #self.gcn3 = GCNLayer(n_dim2, n_input)
    def forward(self,X,A):
        output = self.gcn1(X,A)
        output = self.gcn2(output,A)
        #output = self.gcn3(output,A)
        return output





class MLP(nn.Module):
    def __init__(self,in_dim,hid_dim,out_dim):
        super(MLP, self).__init__()
        self.mlp1 = Linear(in_dim,hid_dim)
        #self.mlp2 = Linear(hid_dim,out_dim)
        self.normlize = nn.BatchNorm1d(hid_dim)
    def forward(self,X):
        hidden = self.normlize(self.mlp1(X))
        hidden = torch.relu(hidden)
        output = torch.softmax(hidden,dim=1)
        return output


class Hyper_Graph_Contrastive_aug(nn.Module):
    def __init__(self, n_input, n_dim1, n_dim2, n_hid, n_hid2, n_out, n_clusters):
        super(Hyper_Graph_Contrastive_aug, self).__init__()
        self.Encoder_graph = GAE(n_input, n_dim1, n_dim2, n_hid)
        self.Encoder_hypergraph = GAE(n_input, n_dim1, n_dim2, n_hid)

        self.mlp = Linear(n_hid, n_out)
        self.cluster_head = MLP(n_hid, n_hid2, n_clusters)
        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, n_hid), requires_grad=True)
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.gamma_gae1 = Parameter(torch.zeros(1))
        self.gamma_hgae1 = Parameter(torch.zeros(1))

    def forward(self, x, A_norm,X2,A2, G):

        H1, S1 = self.Encoder_graph.encoder(x, A_norm)
        H2, S2 = self.Encoder_graph.encoder(X2, A2)
        H3, S3 = self.Encoder_hypergraph.encoder(x, G)
        Z1, Z2, Z3 = self.mlp(H1), self.mlp(H2), self.mlp(H3)
        #Z1, Z2 = self.mlp(H1), self.mlp(H2)
        #C1, C2, C3 = self.cluster_head(H1), self.cluster_head(H2), self.cluster_head(H3)
        # Z2, Z3 =  self.mlp(H2), self.mlp(H3)
        H12 = 0.5 * (H2 + H1)
        '''NH12, NH3 = F.normalize(H12, dim=1, p=2), F.normalize(H3, dim=1, p=2)
        s_igae = torch.mm(NH12, NH12.t())
        s_igae = torch.multiply(A_norm.to_dense(), s_igae)
        s_igae = F.softmax(s_igae, dim=1)

        s_hgae = torch.mm(NH3, NH3.t())
        s_hgae = torch.multiply(G.to_dense().t(), s_hgae)
        s_hgae = F.softmax(s_hgae, dim=1)

        z_igae_ = torch.mm((s_hgae), H12)
        z_hgae_ = torch.mm((s_igae), H3)
        z_igae = self.gamma_gae1 * z_igae_ + H12
        z_hgae = self.gamma_hgae1 * z_hgae_ + H3
        H = self.alpha * z_hgae + (1 - self.alpha) * z_igae'''
        H = self.alpha * H12 + (1 - self.alpha) * H3
        X1_, S1_ = self.Encoder_graph.decoder(H, A_norm)
        X2_, S2_ = self.Encoder_graph.decoder(H, A2)
        X3_, S3_ = self.Encoder_hypergraph.decoder(H, G)
        return H, H1,H2,H3,Z1, Z2, Z3, (S1 + S1_) / 2, (S2 + S2_) / 2, (S3 + S3_) / 2, X1_, X2_, X3_
        #return H, H1, H2, Z1, Z2, (S1 + S1_) / 2, (S2 + S2_) / 2,  X1_, X2_

    def q_distribution(self, Z):
        Q1 = 1.0 / (1.0 + torch.sum(torch.pow(Z.unsqueeze(1) - self.cluster_layer, 2), 2))
        Q1 = (Q1.t() / torch.sum(Q1, 1)).t()
        return Q1

    def get_feature_dis(self, x):
        # x :           batch_size x nhid
        # x_dis(i,j):   item means the similarity between x(i) and x(j).
        x_dis = x @ x.T
        mask = torch.eye(x_dis.shape[0]).to(x.device)
        x_sum = torch.sum(x ** 2, 1).reshape(-1, 1)
        x_sum = torch.sqrt(x_sum).reshape(-1, 1)
        x_sum = x_sum @ x_sum.T
        x_dis = x_dis * (x_sum ** (-1))
        x_dis = (1 - mask) * x_dis
        return x_dis

    def Ncontrast(self, x_dis, adj_label, tau=1):
        # compute the Ncontrast loss
        x_dis = torch.exp(tau * x_dis)
        x_dis_sum = torch.sum(x_dis, 1)
        x_dis_sum_pos = torch.sum(x_dis * adj_label, 1)
        loss = -torch.log(x_dis_sum_pos * (x_dis_sum ** (-1)) + 1e-8).mean()
        return loss

class Hyper_Graph_Contrastive_pretrain_aug(nn.Module):
    def __init__(self, n_input, n_dim1, n_dim2, n_hid, n_clusters):
        super(Hyper_Graph_Contrastive_pretrain_aug, self).__init__()
        self.Encoder_graph = GAE(n_input, n_dim1, n_dim2, n_hid)
        self.Encoder_hypergraph = GAE(n_input, n_dim1, n_dim2, n_hid)
        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, n_hid), requires_grad=True)
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.alpha = nn.Parameter(torch.tensor(0.5),requires_grad=True)
        self.gamma_gae1 = Parameter(torch.zeros(1))
        self.gamma_hgae1 = Parameter(torch.zeros(1))
    def forward(self,x,x_mask,A1,A2,G):
        H1,S1 = self.Encoder_graph.encoder(x,A1)
        H2,S2 = self.Encoder_graph.encoder(x_mask,A2)
        H3,S3 = self.Encoder_hypergraph.encoder(x,G)
        H12 = 0.5*(H1+H2)
        '''NH12, NH3 = F.normalize(H12, dim=1, p=2), F.normalize(H3, dim=1, p=2)
        s_igae = torch.mm(NH12, NH12.t())
        s_igae = torch.multiply(A1.to_dense(), s_igae)
        s_igae = F.softmax(s_igae, dim=1)

        s_hgae = torch.mm(NH3, NH3.t())
        s_hgae = torch.multiply(G.to_dense().t(), s_hgae)
        s_hgae = F.softmax(s_hgae, dim=1)

        z_igae_ = torch.mm((s_hgae), H12)
        z_hgae_ = torch.mm((s_igae), H3)
        z_igae = self.gamma_gae1 * z_igae_ + H12
        z_hgae = self.gamma_hgae1 * z_hgae_ + H3
        H = self.alpha * z_hgae + (1 - self.alpha) * z_igae'''
        H = self.alpha * H12 + (1 - self.alpha) * H3
        X1_, S1_ = self.Encoder_graph.decoder(H, A1)
        X2_,S2_ = self.Encoder_graph.decoder(H,A2)
        X3_, S3_ = self.Encoder_hypergraph.decoder(H,G)
        return H,(S1+S1_)/2,(S2+S2_)/2,(S3+S3_)/2,X1_,X2_,X3_

    def q_distribution(self, Z):
        Q1 = 1.0 / (1.0 + torch.sum(torch.pow(Z.unsqueeze(1) - self.cluster_layer, 2), 2))
        Q1 = (Q1.t() / torch.sum(Q1, 1)).t()
        return Q1









