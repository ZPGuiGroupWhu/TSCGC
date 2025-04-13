import torch
import numpy as np
import torch.nn as nn
import scipy.sparse as sp
def init_memory(model,A_F,A_T,X):
    H_T,H_F,Z_T,Z_F,P_T,P_F = model(X,A_T,A_F)
    local_memory_embeddings = torch.stack([Z_T, Z_F])
    return local_memory_embeddings
def cluster_memory(model,local_memory_embeddings,nmb_prototypes,size_dataset,proj_out,crops_for_assign, nmb_kmeans_iters=10):
    j = 0
    assignments = -100 * torch.ones(len(nmb_prototypes), size_dataset).long()
    with torch.no_grad():
        for i_K, K in enumerate(nmb_prototypes):
            # run K_means

            # init centroids with elements from memory bank of rank 0

            random_idx = torch.randperm(len(local_memory_embeddings[j]))[:K]
            assert len(random_idx) >= K, "please reduce the number of centroids"
            centroids = local_memory_embeddings[j][random_idx][:]
            for i in range(nmb_kmeans_iters+1):
                dot_products = torch.mm(local_memory_embeddings[j], centroids.t())
                _, local_assignments = dot_products.max(dim=1)
                if i == nmb_kmeans_iters:
                    break
                where_helper = get_indices_sparse(local_assignments.cpu().numpy())
                counts = torch.zeros(K).cuda().int()
                emb_sums = torch.zeros(K, proj_out).cuda()
                for k in range(len(where_helper)):
                    if len(where_helper[k][0]) > 0:
                        emb_sums[k] = torch.sum(
                            local_memory_embeddings[j][where_helper[k][0]],
                            dim=0,
                        )
                    counts[k] = len(where_helper[k][0])
                mask = counts > 0
                centroids[mask] = emb_sums[mask] / counts[mask].unsqueeze(1)

                # normalize centroids
                centroids = nn.functional.normalize(centroids, dim=1, p=2)

            getattr(model.prototypes, "prototypes" + str(i_K)).weight.copy_(centroids)
            assignments[i_K] = local_assignments
            j = (j + 1) % len(crops_for_assign)
    return assignments


def get_indices_sparse(data):
    cols = np.arange(data.size)
    #print(cols)
    #print(data.ravel())
    M = sp.csr_matrix((cols, (data.ravel(), cols)), shape=(int(data.max()) + 1, data.size))
    #print(M)
    #print(M.toarray())
    return [np.unravel_index(row.data, data.shape) for row in M]
#data = np.array([[1,1,2,2,4],[2,3,3,5,1],[1,1,3,5,4]])-1
#print(data.size)
#where_helper=get_indices_sparse(data)
#print(where_helper[3])