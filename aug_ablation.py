import torch
import torch.nn.functional as F
import numpy as np
from opt import args
from Model import *
from util import *

from numpy import random
import os


def train(name,t,tao,k,kk,rc,lr,la=0.1):
    args.name = name
    if args.name == 'cora':
        args.lr = lr#3e-4
        args.n_clusters = 7
        args.n_z = 10
        args.nmb_prototypes = 7
        args.temperature = 2
        args.nmb_crops = [0, 1]
        args.alpha = 0.01
        args.gamma = 1.0
        args.dim1 = 512
        args.dim2 = 256
        args.n_hid = 256
        args.n_hid2 = 128
        args.n_out = 256
        args.k_nei = 3
        args.t=0.7
        args.k=20
        args.rc = -0.25
    if args.name == 'citeseer':
        args.lr = lr
        args.n_clusters = 6
        args.n_z = 10
        args.nmb_prototypes = 6
        args.temperature = 2
        args.nmb_crops = [0, 1]
        args.alpha = 0.01
        args.gamma = 1.0
        args.dim1 = 128
        args.dim2 = 256
        args.n_hid = 256
        args.n_hid2 = 128
        args.n_out = 512
        args.k_nei = 5
        args.t=5
        args.k=80
        args.rc = -0.1
    if args.name == 'dblp':
        args.lr = lr#3e-4
        args.n_clusters = 4
        args.n_z = 10
        args.nmb_prototypes = 6
        args.temperature = 2
        args.nmb_crops = [0, 1]
        args.alpha = 0.01
        args.gamma = 1.0
        args.dim1 = 512
        args.dim2 = 256
        args.n_hid = 256
        args.n_hid2 = 128
        args.n_out = 512
        args.k_nei = 5
        args.t=2
        args.k=40
        args.rc = -0.25
    if args.name == 'acm':
        args.lr = 5e-4
        args.n_clusters = 3
        args.n_z = 10
        args.nmb_prototypes = 6
        args.temperature = 2
        args.nmb_crops = [0, 1]
        args.alpha = 0.01
        args.gamma = 1.0
        args.dim1 = 512
        args.dim2 = 256
        args.n_hid = 256
        args.n_hid2 = 128
        args.n_out = 512
        args.k_nei = 5
        args.t=2
        args.k=40
        args.rc = 0.1
    if args.name == 'amap':
        args.lr = lr#1e-3
        args.n_clusters = 8
        args.n_z = 10
        args.nmb_prototypes = 6
        args.temperature = 2
        args.nmb_crops = [0, 1]
        args.alpha = 0.01
        args.gamma = 1.0
        args.dim1 = 128
        args.dim2 = 256
        args.n_hid = 256
        args.n_hid2 =128
        args.n_out = 128
        args.k_nei = 5
        args.t= 0.7
        args.k=10
        args.rc = -0.25
    if args.name == 'amac':
        args.lr = 5e-4
        args.n_clusters = 10
        args.n_z = 10
        args.nmb_prototypes = 6
        args.temperature = 2
        args.nmb_crops = [0, 1]
        args.alpha = 0.01
        args.gamma = 1.0
        args.dim1 = 128
        args.dim2 = 128
        args.n_hid = 128
        args.n_hid2 = 128
        args.n_out = 256
        args.k_nei = 5
        args.t=0.7
        args.k=20
    args.tao = tao
    args.t = t
    args.rc = rc
    args.k = kk
    args.premodel_save_path = 'premodel_{}_{}.pkl'.format(str(args.k_nei), args.name)
    args.premodel_save_path1 = 'premodel_gae{}_{}.pkl'.format(str(args.k_nei), args.name)
    if args.name == 'amap':
        args.premodel_save_path2 = 'premodel/premodel_aug{}_{}_{}_.pkl'.format(str(args.k), str(args.rc), args.name)
    else:
        args.premodel_save_path2 = 'premodel/premodel_aug{}_{}_{}_.pkl'.format(str(args.k),str(args.rc), args.name)
    args.model_save_path = 'model_{}_{}_base.pkl'.format(str(args.k_nei), args.name)
    x, y = load_data(args.name)
    A = load_graph(args.name)
    if args.name=='amac':
        A = np.where(A > 0.0, 1.0, 0.0)
        A = A - np.diag(np.diagonal(A))
        A = A + A.T
    #A_knn, _ = construct_graph(x, k=args.k_nei, method='cosine', delta=1)
    dist_mat = get_cos_similar_matrix(x, x)


    A2 = load_graph_curvature(args.name, A + np.eye(A.shape[0]), args.rc)#dblp -0.25 cite -0.1 acm 0.1
    '''if os.path.exists('{}_{}_.npy'.format(args.name, str(args.k))):
        G = np.load('{}_{}.npy'.format(args.name, str(args.k)))
        print('hypergraph load finish')'''

    #A_second = high_confidence_pair_topological_(np.where(A2>0.0,1.0,0.0), dist_mat, k=args.k)  # cora 20 cite 80 acm 20/50 dblp 40 amap 10
    #print(hom(A_second, y))
    if os.path.exists('{}_{}_.npy'.format(args.name, str(args.k))):
        '''A_second = high_confidence_pair_topological_(A, dist_mat,
                                                     k=args.k)  # cora 20 cite 80 acm 20/50 dblp 40 amap 10
        print(hom(A_second, y))
        H, W = construct_H_with_KNN(-dist_mat, k_nei=args.k_nei, is_proH=False)
        G = generate_G_from_H(A_second, W, False)
        np.save('{}_{}.npy'.format(args.name, str(args.k)), G)'''
        G = np.load('{}_{}_.npy'.format(args.name, str(args.k)))
        print('hypergraph load finish')
    else:
        A_second = high_confidence_pair_topological_(A, dist_mat, k=args.k)  # cora 20 cite 80 acm 20/50 dblp 40 amap 10
        #A_second = A+np.eye(A.shape[0])
        print(hom(A_second, y))
        H, W = construct_H_with_KNN(-dist_mat, k_nei=args.k_nei, is_proH=False)
        G = generate_G_from_H(A_second, W, False)
        np.save('{}_{}_.npy'.format(args.name, str(args.k)), G)
    print('graph load finish')
    #np.save('{}_{}.npy'.format(args.name, str(args.k)), G)
    A_norm = normalize_A(A, symmetry=False, self_loop=True)
    A2_norm = normalize_A(A2, symmetry=False, self_loop=False)
    print('graph load finish')
    X2 = torch.tensor(mask_feat(x)).to(args.device).to(torch.float32)
    x = torch.tensor(x).to(args.device).to(torch.float32)
    A_norm = numpy_to_tensor(numpy_to_sparse_numpy(A_norm), True).to(args.device).float()
    A2_norm = numpy_to_tensor(numpy_to_sparse_numpy(A2_norm), True).to(args.device).float()

    G = numpy_to_tensor(numpy_to_sparse_numpy(G), True).to(args.device).float()
    for kk in range(10):
        seed = random.randint(0, 100000)#75265
        print(seed)
        setup_seed(seed)
        model = Hyper_Graph_Contrastive_aug(x.shape[1],n_dim1=args.dim1,n_dim2=args.dim2
                                ,n_hid=args.n_hid,n_hid2=args.n_hid2,n_out=args.n_out,n_clusters=args.n_clusters).to(args.device)

        center,_=model_init_aug(model,args.premodel_save_path2, x, y,A_norm,X2,A2_norm,G)
        model.cluster_layer.data = torch.FloatTensor(center).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)# 1e-3
        for i in range(200):
            H,  H1,H2,H3,Z1, Z2, Z3, S_adj1, S_adj2, S_hyper, X_adj1, X_adj2, X_hyper= model(x,A_norm,X2,A2_norm,G)

            G_ = torch.matmul(G.to_dense(), G.to_dense().t())
            G_ = torch.where(G_ > 0.0, 1.0, 0.0)
            rec_feat_loss =F.mse_loss(X_adj2,x)+F.mse_loss(X_adj1,x)+F.mse_loss(X_hyper,x)


            rec_adj_loss = F.mse_loss(S_adj2,A2_norm.to_dense())*1.0 + F.mse_loss(S_adj1,A_norm.to_dense()) * 1.0 + F.mse_loss(S_hyper, G_) * 1.0
            rec_loss = rec_adj_loss*0.1 + rec_feat_loss

            crossview_loss =(neighbors(Z1,Z2,G.to_dense().t(),t=args.t) + neighbors(Z2,Z1,G.to_dense().t(),t=args.t)) * 0.5#cora0.7 cite 5

            q = model.q_distribution(H)
            q1, q2 = model.q_distribution(0.5*(H1+H2)), model.q_distribution(H3)

            p = target_distribution(q.data)

            loss_kl = F.kl_div((q.log()+q1.log()+q2.log())/3, p, reduction='batchmean')

            T=50
            if i<=T:

                contrastive_loss = (infoloss(Z1, Z3, t=args.t) + infoloss(Z3, Z1, t=args.t)) * 0.5
            else:
                P = center_generate(q.data, H.data)
                Q = cluster_graph(P.data,H.data,tao=args.tao)#CORA 0.3 CITE 0.2 acm 0.4 dblp 0.8 amap0.5
                contrastive_loss = (knncontras_loss(Z1, Z3,Q,t=args.t,k=k) + knncontras_loss(Z3, Z1,Q,t=args.t,k=k)) * 0.5  # cora0.7 200


            loss = ( contrastive_loss + crossview_loss) + (rec_loss + loss_kl) * la

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("epoch: {} loss: {} contrastive_loss: {}".format(i, loss.item(),contrastive_loss.item()))
            acc, nmi, ari, f1, center = k_means(H.detach().cpu().numpy(), args.n_clusters, y)
            #model.cluster_layer.data = torch.tensor(center).to(args.device)
            print("acc: {} nmi: {} ari: {} f1: {}".format(acc, nmi, ari, f1))

    del A,G,A_norm,x,y,dist_mat,A2,A2_norm,X2





