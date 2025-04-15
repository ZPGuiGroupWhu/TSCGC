import numpy as np
import scipy.sparse as sp
import scipy.stats
import matplotlib.pyplot as plt
import torch.nn.functional as F
from evaluation import evaluation
import torch
import copy
from sklearn.cluster import KMeans
from opt import  args
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci
def load_data(data_name):
    data_path="data/"+data_name+"_feat.npy"
    label_path = "data/"+data_name+"_label.npy"
    data = np.load(data_path)
    label = np.load(label_path)
    return data,label
def load_graph(data_name,has_graph=True):
    if has_graph:
        graph_path="graph/"+data_name+"_adj.npy"
        graph = np.load(graph_path)
    else:
        graph = None
    return graph
def load_pretrain_parameter(model,path):
    pretrained_dict = torch.load(path, map_location='cpu')
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

def model_init_aug(model, path,X, y,A_norm,X2,A2,G):
    # load pre-train model
    model = load_pretrain_parameter(model,path)

    # calculate embedding similarity
    with torch.no_grad():
        #X_hat,Z_adj_hat_,A_hat,Z,Z_hat,Q,Z_hgae_adj,Z_hgae,Z_igae= model(X, A_norm, G)
        #X_hat, Z,Q=model(X)
        H, H1,H2,H3, Z1, Z2, Z3, S1, S2, S3, X1_, X2_, X3_ = model(X, A_norm,X2,A2, G)
        #H, H1, H2,  Z1, Z2,  S1, S2,  X1_, X2_ = model(X, A_norm, X2, A2, G)

    # calculate cluster centers
    acc, nmi, ari, f1,  clsuter_id ,centers= k_means(H.detach().cpu().numpy(), k=args.n_clusters,y_true=y,label=True)

    return centers, clsuter_id

def numpy_to_sparse_numpy(A):
    return sp.coo_matrix(A).astype(np.float32)
def sparse_to_tensor(sp_A):
    indices = torch.from_numpy(np.vstack((sp_A.row, sp_A.col)).astype(np.int64))
    values = torch.from_numpy(sp_A.data)
    shape = torch.Size(sp_A.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
def tensor_to_numpy(t:torch.Tensor):
    return t.numpy()
def numpy_to_tensor(array,is_sparse=False):
    if is_sparse:
        #torch.sparse.Tensor(array)
        return sparse_to_tensor(array)
    else:
        return torch.from_numpy(array)
def normalize_A(A, symmetry=True, self_loop=True):
    if self_loop:
        A = A + np.eye(A.shape[0])
    degree = np.sum(A, axis=1)
    if symmetry:
        D = np.diag(degree**(-0.5))
        return np.matmul(np.matmul(D, A), D)
    else:
        D = np.diag(degree**(-1))
        return np.matmul(D, A)
def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def mask_feat(feat, mask_rate=0.2):
    # mask feat
    masked_feat = copy.deepcopy(feat)

    # add mask
    mask = np.ones(feat.shape[0] * feat.shape[1])
    mask[:int(len(mask) * mask_rate)] = 0.0
    np.random.shuffle(mask)
    mask = mask.reshape(feat.shape[0], feat.shape[1])
    masked_feat =masked_feat* mask

    return masked_feat
def k_means(embedding,k,y_true,label=False):
    model = KMeans(n_clusters=k,n_init=50)
    cluster_id = model.fit_predict(embedding)
    center = model.cluster_centers_
    acc,nmi,ari,f1 = evaluation(y_true=y_true,y_pred=cluster_id)
    if label==False:
        return acc,nmi,ari,f1,center
    else:
        return acc, nmi, ari, f1, cluster_id,center




def square_euclid_distance(Z,Center):
    ZZ = (Z*Z).sum(-1).reshape(-1,1).repeat(1,Center.shape[0])
    CC = (Center*Center).sum(-1).reshape(1,-1).repeat(Z.shape[0],1)
    Z_C = Z@Center.T
    return ZZ + CC - 2*Z_C





def hom(A,y):
    count=0
    all=0
    counts,alls={},{}
    for i in range(A.shape[0]):
        for j in range(i,A.shape[1]):
            if i!=j and y[i]==y[j] and A[i,j]!=0:
                count+=1
                if y[i] not in counts.keys():
                    counts[y[i]]=1
                else:
                    counts[y[i]]+=1
            if i!=j and A[i,j]!=0:
                all+=1
                if y[i] not in alls.keys():
                    alls[y[i]]=1
                else:
                    alls[y[i]]+=1
    result=[]
    for i in np.unique(y):
        if i not in alls.keys():
            continue
        if i not in counts.keys():
            continue
        result.append([counts[i]/alls[i],counts[i],alls[i]])
    return result,count/all
def get_cos_similar_matrix(v1, v2):
    num = np.dot(v1, np.array(v2).T)  # 向量点乘
    denom = (np.linalg.norm(v1, axis=1).reshape(-1, 1)+1e-8) * (np.linalg.norm(v2, axis=1)+1e-8)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res
def construct_H_with_KNN(dist_mat,k_nei=5,is_proH=True):
    n_obj=dist_mat.shape[0]
    n_edge=n_obj
    H=np.zeros(shape=(n_obj,n_edge))
    W=[]
    for center_idx in range(n_obj):
        dist_mat[center_idx,center_idx]=1.0
        dis_vec=dist_mat[center_idx]
        nearest_id=np.array(np.argsort(dis_vec)).squeeze()
        avg_dis=np.average(dis_vec[nearest_id[:k_nei]])
        W.append(avg_dis)
        if not np.any(nearest_id[:k_nei]==center_idx):
            nearest_id[k_nei-1]=center_idx
        for node_idx in nearest_id[:k_nei]:
            if is_proH:
                H[node_idx,center_idx]=np.exp(-dis_vec[0,node_idx]**2/(avg_dis)**2)
            else:
                H[node_idx, center_idx]=1.0

    #H = H + np.eye(H.shape[0])
    W = W / np.sum(W)
    #W = np.ones_like(W)/H.shape[0]
    '''importance=np.sum(H+np.eye(H.shape[0]),axis=1)
    importance=importance/np.max(importance)
    W = np.multiply(W,importance)
    W = W/np.sum(W)'''
    return H,W
def generate_G_from_H(H,W_,use_W=False,variable_weight=False):
    H=np.array(H).T
    if use_W==False:
        W=np.ones(H.shape[0])/H.shape[0]
    else:
        W=W_
    DV=np.sum(H*W,axis=1)
    DE=np.sum(H,axis=0)

    invDE=np.mat(np.diag(np.power(DE,-1)))
    DV2 = np.mat(np.diag(np.power(DV,-0.5)))
    W=np.mat(np.diag(W))
    H=np.mat(H)
    HT=H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G=DV2*H*W*invDE*HT*DV2
        return G
def hom_(A,A_,y):
    rights, wrongs = {}, {}
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if i != j and y[i] == y[j] and A[i, j] == 0 and A_[i,j]!=0:
                if y[i] not in rights.keys():
                    rights[y[i]] = 1
                else:
                    rights[y[i]] += 1
            if i != j and y[i] == y[j] and A[i, j] != 0 and A_[i, j] == 0:
                if y[i] not in wrongs.keys():
                    wrongs[y[i]] = 1
                else:
                    wrongs[y[i]] += 1
            if i != j and y[i] != y[j] and A[i, j] == 0 and A_[i,j]!=0:
                if y[i] not in wrongs.keys():
                    wrongs[y[i]] = 1
                else:
                    wrongs[y[i]] += 1
            if i != j and y[i] != y[j] and A[i, j] != 0 and A_[i, j] == 0:
                if y[i] not in rights.keys():
                    rights[y[i]] = 1
                else:
                    rights[y[i]] += 1
    result1,result2 = [],[]
    for i in rights.keys():
        result1.append([i ,rights[i]])
    for i in wrongs.keys():
        result2.append([i ,wrongs[i]])
    return result1,result2


def high_confidence_pair_topological_(A,dist_mat,k):
    sim = np.matmul(A,A.T)
    sim = np.where(sim>0.0,1.0,0.0)
    sim = (sim+A+np.eye(A.shape[0]))/2
    thredhold_id = np.array(np.argsort(-dist_mat,axis=1))[:,k].reshape((-1,1))
    thredhold = dist_mat[np.arange(0,A.shape[0],1).reshape((-1,1)),thredhold_id]
    index=np.where(dist_mat<thredhold)
    sim[index]=0.0
    #sim = sim/np.sum(sim,axis=1,keepdims=True)
    return sim


def pesudo_label_generate(Y,Z,k=100):
    for i in range(Y.shape[1]):
        values,_ = torch.topk(Y[:,i],dim=0,k=int(Y.shape[0]/Y.shape[1]))
        values = values.t()
        threshold = values[-1]
        index = torch.where(Y[:,i]>=threshold)[0]
        if i==0:
            pesudo_label = [index]
            prototype = torch.mean(Z[index],dim=0)
        else:
            pesudo_label.append(index)
            prototype = torch.cat((prototype,torch.mean(Z[index],dim=0)),dim=0)
    prototype = prototype.reshape((Y.shape[1],-1))
    similarity = torch.softmax(torch.matmul(Z,prototype.t()),dim=1)

    values,_=torch.topk(similarity,k=k,dim=0)
    values = values.t()
    threshold = values[:,-1]
    index = torch.where(similarity >= threshold)
    index = torch.cat((index[0].t(),index[1].t()),dim=0).reshape((2,-1))
    index_ = index.sort(dim=1).indices
    index = index.t()[index_[1]].t()
    return index
def center_generate(C,H):
    W = torch.zeros_like(C)
    #l=[]
    for i in range(C.shape[1]):
        values, indices = torch.topk(C[:, i], dim=0, k=int(C.shape[0] / C.shape[1]/2))#0.1
        values = values.t()
        threshold = values[-1]
        #l.append(threshold)
        index = torch.where(C[:, i] >= threshold)[0]
        W[index,i] = C[index, i] / torch.sum(C[index, i])#可以试试exp或者平方
    prototype = torch.matmul(W.t(),H)
    return prototype
def cluster_graph(proto,H,tao=0.2):
    p = torch.softmax(square_euclid_distance(H, proto), dim=1)
    distance_norm, pseudo_label = torch.min(p, dim=1)
    distance_norm, pseudo_label = distance_norm.cpu(), pseudo_label.cpu()
    # value,_ = torch.topk(distance_norm,int(Z.shape[0]*(1-tao)))
    print(np.bincount(pseudo_label.numpy()))
    threshold = torch.sort(distance_norm, descending=False).values[int(len(distance_norm) * tao)]
    # index = torch.where(distance_norm<=value[-1],torch.ones_like(distance_norm),torch.zeros_like(distance_norm))
    high_confidence_idx = np.argwhere(distance_norm <= threshold)[0][:int(len(distance_norm) * tao)]

    del distance_norm
    #H = torch.cartesian_prod(high_confidence_idx, high_confidence_idx)
    #same_class = np.where(pseudo_label[H[:, 0]] == pseudo_label[H[:, 1]])
    low_confidence_idx = np.setdiff1d(np.arange(0,H.shape[0],1),high_confidence_idx)
    Q = (pseudo_label == pseudo_label.unsqueeze(1)).float()
    Q[low_confidence_idx,:]=0
    Q[:,low_confidence_idx] =0
    return Q

def neighbors(Z1,Z2,A,t=5):
    NZ1,NZ2 = F.normalize(Z1,p=2,dim=1),F.normalize(Z2,p=2,dim=1)
    sim12 = torch.exp(torch.matmul(NZ1,NZ2.t())/t)
    sim11 =torch.exp(torch.matmul(NZ1,NZ1.t())/t)
    sim = 0.5*(sim11+sim12)
    A_ =torch.where(A>0.0,1.0,0.0)
    A_ = torch.multiply(A_,sim)
    A_ = torch.softmax(A_,dim=1)
    pos12 = torch.multiply(sim12,A_)
    pos11 = torch.multiply(sim11,A_)
    pos = torch.sum(pos11+pos12,dim=1)
    loss=pos/(torch.sum(sim12,dim=1)+torch.sum(sim11,dim=1)-torch.diag(sim11))
    return -torch.log(loss).mean()
def knncontras_loss(Z1,Z2,A,t=5,k=30):#CORA K=10 cite acm dblp 50 amap 70
    NZ1, NZ2 = F.normalize(Z1, p=2, dim=1), F.normalize(Z2, p=2, dim=1)
    sim12 = torch.exp(torch.matmul(NZ1, NZ2.t()) / t)
    sim11 = torch.exp(torch.matmul(NZ1, NZ1.t()) / t)
    sim = 0.5 * (sim11 + sim12)
    sim = (sim-sim.min())/(sim.max()-sim.min())
    knn_neighbor = torch.topk(sim,dim=1,k=k+1).indices
    A[torch.arange(0,Z1.shape[0],1).reshape((-1,1)),knn_neighbor]=1.0
    positive_mask = torch.zeros_like(sim)
    positive_mask[torch.arange(0,sim.shape[0],1).reshape((-1,1)),knn_neighbor]=1.0
    pos = torch.sum(torch.multiply(sim12,positive_mask),dim=1)+torch.sum(torch.multiply(sim11,positive_mask),dim=1)-torch.diag(sim11)
    pos = pos/(2*torch.sum(positive_mask,dim=1)+1)
    neg = torch.sum(torch.multiply(sim12,1-positive_mask),dim=1) + torch.sum(torch.multiply(sim11,1-positive_mask),dim=1)
    loss = pos/(neg+pos)
    del NZ1,NZ2
    return -torch.log(loss).mean()

def infoloss(Z1,Z2,t):
    NZ1, NZ2 = F.normalize(Z1, p=2, dim=1), F.normalize(Z2, p=2, dim=1)
    sim12 = torch.exp(torch.matmul(NZ1, NZ2.t()) / t)
    sim11 = torch.exp(torch.matmul(NZ1, NZ1.t()) / t)
    loss = torch.diag(sim12)/(torch.sum(sim11,dim=1)+torch.sum(sim12,dim=1)-torch.diag(sim11))
    del NZ1,NZ2
    return -torch.log(loss).mean()

def target_distribution(Q):
    weight = Q ** 2 / Q.sum(0)
    P = (weight.t() / weight.sum(1)).t()
    return P





def ricci_curvature(adj,alpha=0.5):

    G = nx.karate_club_graph()
    orc = OllivierRicci(G, alpha=alpha, verbose="INFO")
    orc.compute_ricci_curvature()
    return orc.G.edges.data()


def load_graph_curvature(name,adj,rc_min=-0.5):
    adj_rc = np.load('data/'+name+'_rc1.npy') # rc1
    adj_rc = adj_rc[np.where(adj_rc[:,-1]<=rc_min)]
    print(adj_rc.shape[0]*2/np.sum(adj))
    mask_adj = np.ones_like(adj)
    mask_adj[adj_rc.T[0].astype(np.int),adj_rc.T[1].astype(np.int)]=0.0
    mask_adj[adj_rc.T[1].astype(np.int), adj_rc.T[0].astype(np.int)] = 0.0
    mask_adj = mask_adj*adj

    return mask_adj
