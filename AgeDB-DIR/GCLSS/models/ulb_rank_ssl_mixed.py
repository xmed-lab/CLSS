"""
Experiments on linear/non-linear operators learning tasks, results as mean +- standard variance over 10 runs.
"""
import torch
import numpy as np
import random


from torch import dot, argsort
from torch import sign, count_nonzero, ones, reshape, eye, dot, argsort
from torch.linalg import eig, eigh

from .compute_delta import compute_formula_torch



def rank(seq):
    return torch.argsort(torch.argsort(seq).flip(1))


def rank_normalised(seq):
    return (rank(seq) + 1).float() / seq.size()[1]


class TrueRanker(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sequence, lambda_val):
        rank = rank_normalised(sequence)
        ctx.lambda_val = lambda_val
        ctx.save_for_backward(sequence, rank)
        return rank

    @staticmethod
    def backward(ctx, grad_output):
        sequence, rank = ctx.saved_tensors
        assert grad_output.shape == rank.shape
        sequence_prime = sequence + ctx.lambda_val * grad_output
        rank_prime = rank_normalised(sequence_prime)
        gradient = -(rank - rank_prime) / (ctx.lambda_val + 1e-8)
        return gradient, None



def centering_matrix(n):
    # centering matrix, projection to the subspace orthogonal
    # to all-ones vector
    return np.eye(n) - np.ones((n, n)) / n


def get_the_subspace_basis(n, verbose=True):
    # returns the orthonormal basis of the subspace orthogonal
    # to all-ones vector
    H = centering_matrix(n)
    s, Zp = np.linalg.eigh(H)
    ind = np.argsort(-s)  # order eigenvalues descending
    s = s[ind]
    Zp = Zp[:, ind]  # second axis !!
    # if (verbose):
    #     print("...forming the Z-basis")
    #     print("check eigenvalues: ", allclose(
    #         s, concatenate((ones(n - 1), [0]), 0)))

    Z = Zp[:, :(n - 1)]
    # if (verbose):
    #     print("check ZZ'=H: ", allclose(dot(Z, Z.T), H))
    #     print("check Z'Z=I: ", allclose(dot(Z.T, Z), eye(n - 1)))
    return Z


def compute_upsets(r, C, verbose=True, which_method=""):
    n = r.shape[0]
    totmatches = count_nonzero(C) / 2
    if (len(r.shape) == 1):
        r = reshape(r, (n, 1))
    e = ones((n, 1)).cuda()
    # Chat = r.dot(e.T) - e.dot(r.T)
    Chat = torch.matmul(r, e.T) - torch.matmul(e, r.T)
    upsetsplus = count_nonzero(sign(Chat[C != 0]) != sign(C[C != 0]))
    upsetsminus = count_nonzero(sign(-Chat[C != 0]) != sign(C[C != 0]))
    winsign = 2 * (upsetsplus < upsetsminus) - 1
    # if (verbose):
    #     print(which_method + " upsets(+): %.4f" %
    #           (upsetsplus / float(2 * totmatches)))
    #     print(which_method + " upsets(-): %.4f" %
    #           (upsetsminus / float(2 * totmatches)))
    return upsetsplus / float(2 * totmatches), upsetsminus / float(2 * totmatches), winsign

def GraphLaplacian(G):
    """
    Input a simlarity graph G and return graph GraphLaplacian
    """
    D = torch.diag(G.sum(dim=1))
    L = D - G

    return L

def min_two_order_sum(M):
    n = np.shape(M)[0]
    min_value = M[0, 1] 
    min_row = 1
    min_col = 2
    #print("min_value0",min_value)

    for k in range(n-1):
        for l in range(k+1, n):
            if M[k, l] > min_value:
                min_value = M[k, l]
                #print("min_value",min_value)
                min_row = k+1
                min_col = l+1


    min_indices = [min_row, min_col]
    min_result = min_value * 2
    #print("min_result",min_result)
    return min_indices,min_result

def minimize_matrix_sum(M,m):
    n = np.shape(M)[0]  

    DP = np.full((n + 1, m + 1), -9999.0, dtype=float) 
    selected_indexes = [[[] for _ in range(m + 1)] for _ in range(n + 1)]
    
    M_index = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            M_index[i][j] = M[i - 1][j - 1]


    DP[:,:2] = 0
    #DP[0][0] = 0
    for i in range(2,n+1):
        M_submatrics = M[:i, :i]
        selected_indexes[i][2],DP[i][2] = min_two_order_sum(M_submatrics)
        #print(selected_indexes[i][2], "{:.16f}".format(DP[i][2]))
    for i in range(3, n + 1):
        for j in range(3, min(i, m) + 1):
            
            if j == 0:
                DP[i][j] = 0
                selected_indexes[i][j] = []
            else:
                DP[i][j] = DP[i - 1][j]
                #selected_indexes[i][j] = selected_indexes[i - 1][j]
                selected_indexes[i][j] = selected_indexes[i - 1][j]
                
                for k in range(0,i):
                    
                    row_sum = sum(M_index[i][idx] for idx in selected_indexes[k][j-1])*2
                    #selected_indexes[i][j] = selected_indexes[k][j - 1] + [i]
                    #I = selected_indexes[k][j - 1] + [i]
                    
                    
                    
                    #sum_M_I = sum(M_index[row_idx][col_idx] for row_idx in I for col_idx in I)
                    #print("i={}\t j = {}\t k={}\t I={}\t sum_M_I={}\t".format(i,j,k,I,sum_M_I))
                    #print("i={}\t j = {}\t k={}\t I={}\t sum_M_I={}\t".format(i,j,k,selected_indexes[k][j - 1] + [i],DP[k][j-1]+row_sum))
                    
                    #if DP[i][j] > sum_M_I:
                        #DP[i][j] = np.float16(sum_M_I)
                    if DP[i][j] < DP[k][j-1]+row_sum:
                        DP[i][j] = DP[k][j-1]+row_sum
                        selected_indexes[i][j] = selected_indexes[k][j - 1] + [i]
                        #selected_indexes[i][j] = I
                        #print(i,j,selected_indexes[i][j],DP[i][j])
            selected_indexes[i][j] = selected_indexes[i][j]
            DP[i][j] = DP[i][j]
            #print(i,j,selected_indexes[i][j],DP[i][j])
        #print("selected_indexes[i][j]",selected_indexes[i][j])
                        
    selected_indexes = selected_indexes[n][m]
    selected_indexes = [x - 1 for x in selected_indexes]

    return selected_indexes


def get_ulbps_ulbonly(simMat,yl):

    #### input is (lb, unlb) X (lb, unlb) sim matrix
    #### output is (lb + ulb_pslb_tp), ### keep simple for now, just take the closes one 
    S = simMat
    Su = S[:8,:8]     # the sim matrix of unlabeled data
    n = Su.shape[0]
    Z1 = torch.tensor(get_the_subspace_basis(n, verbose=False)).float().cuda()
    e = torch.nn.functional.normalize(torch.ones((n, 1)).cuda(),dim = 0)
    Z = torch.cat((Z1, e), dim=1)
    # print(S.shape)
    L = GraphLaplacian(S)
    Ls = GraphLaplacian(Su)
    Lu = L[:8,:8]               # the diagnal value of Lu is different from Ls.
    Lmix = L[:8,8:]
    
        
    #ztLuz = torch.matmul(torch.matmul(Z.T, Lu), Z)
    #w, v = eig(ztLuz)
    w,v = eig(Lu)
    w = torch.view_as_real(w)[:,0]
    v = torch.view_as_real(v)[...,0]

    if torch.is_complex(w):
        print("complex")
        return None

    ind = torch.argsort(w)
    w = w[ind]
    v = v[:, ind]
    V = torch.matmul(Z,v)
    v = torch.nn.functional.normalize(v,dim=1)
    r1 = torch.zeros(n,n).cuda()
    for i in range(len(v)):
        #a = reshape(torch.matmul(Z,v[:, i]), (n, 1))
        #r1 +=  torch.matmul(a,a.T)/w[i]
        if w[i] == 0:
            r1 += torch.zeros(n,n).cuda()
        else:
            r1 +=  torch.matmul(v[:, i].unsqueeze(1),v[:, i].unsqueeze(1).T)/w[i]
    r3 = v[:, 0]
    #rl = torch.nn.functional.normalize(yl.squeeze().float().cuda(),dim=0)
    rl = yl.squeeze()
    #rl = torch.nn.functional.normalize(torch.argsort(torch.argsort(yl.squeeze())).float().cuda(),dim=0)
    u = -torch.matmul(Lmix,rl.unsqueeze(1))
    r = torch.matmul(r1,u)
    #_, _, rsign = compute_upsets(r, Su, verbose=False)
    
    #r_final =  rsign * r
    ### r_final is shape [n, 1]
    #r_rank = torch.argsort(torch.argsort(r_final.reshape(-1)))
    r_rank = torch.argsort(torch.argsort(r.reshape(-1)))
    r_total = torch.cat((r.reshape(-1), rl), dim=0)

    ztLsz = torch.matmul(torch.matmul(Z1.T, Ls), Z1)
    w2, v2 = eig(ztLsz)
    w2 = torch.view_as_real(w2)[:,0]
    v2 = torch.view_as_real(v2)[...,0]
    ind2 = torch.argsort(w2)
    w2 = w2[ind2]
    v2 = v2[:, ind2]
    r2 = reshape(torch.matmul(Z1,v2[:, 0]), (n, 1))
    _, _, rsign = compute_upsets(r2, Su, verbose=False)
    r_final =  rsign * r2
    ### r_final is shape [n, 1]
    r_rank1 = torch.argsort(torch.argsort(r_final.reshape(-1)))
    r_total1 = torch.cat((r_final.reshape(-1), rl), dim=0)
    return r_rank,r



def ulb_rank(input_feat_unl, input_feat_l, yl,lambda_val = -1,subsamp = False):
    #b = torch.zeros(1,100).cuda()
    #samples_l = []

        
    #yl = (yl-yl.min())/(yl.max()-yl.min())
    
    yl = (yl-yl.mean())/yl.std()
    max_index = torch.argmax(yl).item()
    min_index = torch.argmin(yl).item()


    selected_indices = [max_index, min_index]


    remaining_indices = list(range(len(yl)))
    remaining_indices.remove(max_index)
    remaining_indices.remove(min_index)
    random_indices = random.sample(remaining_indices, 5)

    selected_indices.extend(random_indices)


    samples_l = random.sample(selected_indices, 7)
    #samples_l = random.sample(range(0, len(input_feat_l)-1), 6)  # random sample 100 features
    input_feat_ll = input_feat_l[samples_l] 
    yl = yl[samples_l]
    
    input_feat = torch.cat((input_feat_unl,input_feat_ll),dim=0)
    #print(input_feat.shape)
    p = torch.nn.functional.normalize(input_feat, dim=1) 
    
    #print(p.shape)
    feat_cosim = torch.matmul(p, p.T) 
    # print(feat_cosim.shape)
    # exit()
    feat_cosim_unl = feat_cosim[:8,:8]
    
    # return the ranking labels_ulpbs and original enginvalue
    labels_ulpbs,r= get_ulbps_ulbonly(feat_cosim,yl) 
    labels_ulbpsdornk = labels_ulpbs    

    # ktau = torch.abs(kendalrankloss(labels_ulbpsdornk, unlb_ref))
    # ktau_dist = ktau

    loss_ulb = torch.tensor(0).float().cuda()

    ps_ulb_ranked = torch.argsort(torch.argsort(labels_ulbpsdornk)) 
    #batch_unique_targets = torch.unique(ps_ulb_ranked)
    batch_unique_targets = torch.unique(r)
    if len(batch_unique_targets) < len(r):
        sampled_indices = []
        for target in batch_unique_targets:
            sampled_indices.append(random.choice((r == target).nonzero()[:,0]).item())
        feat_cosim_samp = feat_cosim_unl[:,sampled_indices]
        feat_cosim_samp = feat_cosim_samp[sampled_indices,:]        
        ps_ulb_ranked_samp = ps_ulb_ranked[sampled_indices]
    else:
        feat_cosim_samp = feat_cosim_unl
        ps_ulb_ranked_samp = ps_ulb_ranked
    
    for i in range(len(ps_ulb_ranked_samp)):
        # print("sampling i", i)
        label_ranks = rank_normalised(-torch.abs(ps_ulb_ranked_samp[i] - ps_ulb_ranked_samp).unsqueeze(-1).transpose(0,1))

        feature_ranks_ulb0ulb0 = TrueRanker.apply(feat_cosim_samp[i].unsqueeze(dim=0), lambda_val)
        # print(feature_ranks_ulb0ulb0.shape, label_ranks.shape)
        loss_ulb += torch.nn.functional.mse_loss(feature_ranks_ulb0ulb0, label_ranks)
    
    return loss_ulb, ps_ulb_ranked





def ulb_rank_prdlb(input_feat, lambda_val = -1, pred_inp=None):
    # samples = random.sample(range(0, len(input_feat)-1), 10)  # random sample 100 features
    # input_feat = input_feat[samples]
    
    # print(input_feat)
    # pred_inp = pred_inp[samples].detach()
    # print(pred_inp)
    # exit()
    
    
    p = input_feat
    
    # print(p)
    feat_cosim = -torch.abs(p - p.T)  # !!!!!!!! IMPORTANT!!!!
    ### ORDERING NEEDS TO BE CONSISTENT
    ### LARGER VALUE MUST SIGNAL MORE SIMILAR


    # print(feat_cosim.shape)
    # print(feat_cosim)
    # exit()

    labels_ulpbs = pred_inp.squeeze(-1)
    labels_ulbpsdornk = labels_ulpbs    
    # print(labels_ulbpsdornk.shape)
    # exit()
    # ktau = torch.abs(kendalrankloss(labels_ulbpsdornk, unlb_ref))
    # ktau_dist = ktau
    
    loss_ulb = torch.tensor(0).float().cuda()

    ps_ulb_ranked = torch.argsort(torch.argsort(labels_ulbpsdornk))
    batch_unique_targets = torch.unique(ps_ulb_ranked)
    if len(batch_unique_targets) < len(ps_ulb_ranked):
        sampled_indices = []
        for target in batch_unique_targets:
            sampled_indices.append(random.choice((ps_ulb_ranked == target).nonzero()[:,0]).item())
        feat_cosim_samp = feat_cosim[:,sampled_indices]
        feat_cosim_samp = feat_cosim_samp[sampled_indices,:]        
        ps_ulb_ranked_samp = ps_ulb_ranked[sampled_indices]
    else:
        feat_cosim_samp = feat_cosim
        ps_ulb_ranked_samp = ps_ulb_ranked
    

    for i in range(len(ps_ulb_ranked_samp)):
        # print("sampling i", i)
        label_ranks = rank_normalised(-torch.abs(ps_ulb_ranked_samp[i] - ps_ulb_ranked_samp).unsqueeze(-1).transpose(0,1))

        feature_ranks_ulb0ulb0 = TrueRanker.apply(feat_cosim_samp[i].unsqueeze(dim=0), lambda_val)
        # print(feature_ranks_ulb0ulb0.shape, label_ranks.shape)
        # print(feature_ranks_ulb0ulb0)
        # print(label_ranks)
        # exit()
        loss_ulb += torch.nn.functional.mse_loss(feature_ranks_ulb0ulb0, label_ranks)

    return loss_ulb

def matrix_variance(matrices):
    # Convert the list of matrices to a PyTorch tensor with float dtype
    stacked_matrices = torch.stack(matrices, dim=0)
    
    # Calculate the mean along the first dimension (dim=0)
    mean_matrix = torch.mean(stacked_matrices, dim=0)
    
    # Calculate the squared differences from the mean
    squared_diffs = (stacked_matrices - mean_matrix)**2
    
    # Calculate the mean of squared differences along the first dimension (dim=0)
    variance_matrix = torch.mean(squared_diffs, dim=0)
    
    return variance_matrix

def matrix_stddev(matrices):
    # Convert the list of matrices to a numpy array
    stacked_matrices = torch.stack(matrices, axis=0)
    
    # Calculate the mean along the first dimension (axis=0)
    mean_matrix = torch.mean(stacked_matrices, axis=0)
    
    # Calculate the squared differences from the mean
    squared_diffs = (stacked_matrices - mean_matrix)**2
    
    # Calculate the mean of squared differences along the first dimension (axis=0)
    variance_matrix = torch.mean(squared_diffs, axis=0)
    
    # Calculate the standard deviation by taking the square root of the variance
    stddev_matrix = torch.sqrt(variance_matrix)
    stddev_matrix  /= mean_matrix
    return stddev_matrix







def ulb_rank_cps(input_feat_unl_0, input_feat_unl_1, input_feat_l_0, input_feat_l_1,yl,lambda_val = -1, subsamp = False, batch_size_two = 8):


        
    
    yl = (yl-yl.mean())/yl.std()
    max_index = torch.argmax(yl).item()
    min_index = torch.argmin(yl).item()

# 将这两个索引加入到一个列表中，确保它们不会被重复采样
    selected_indices = [max_index, min_index]

# 随机采样其他4个索引，但要避免已经选择的索引
    remaining_indices = list(range(len(yl)))
    remaining_indices.remove(max_index)
    remaining_indices.remove(min_index)
    random_indices = random.sample(remaining_indices, 5)
# 将所有选定的索引合并
    selected_indices.extend(random_indices)

# 使用这些索引来从 input_feat_l 和 yl 中抽取对应的样本和标签
    samples_l = random.sample(selected_indices, 7)
    #samples_l = random.sample(range(0, len(input_feat_l)-1), 6)  # random sample 100 features
    #input_feat_unll_0 = input_feat_unl_0[samples_unl]
    #input_feat_unll_1 = input_feat_unl_1[samples_unl]
    #yu = yu[samples_unl]
    #X_train_ulb  = X_train_ulb[samples_unl]
    
    #true_feat = torch.cat((X_train_ulb,X_train_small),dim=0)
    #input_feat = torch.cat((input_feat_unl,input_feat_ll),dim=0)
    #print(input_feat.shape)
    p_unl_0 = torch.nn.functional.normalize(input_feat_unl_0, dim=1) 
    p_unl_1 = torch.nn.functional.normalize(input_feat_unl_1, dim=1) 
    #print(p.shape)
    feat_cosim_unl_0 = torch.matmul(p_unl_0, p_unl_0.T) 
    feat_cosim_unl_1 = torch.matmul(p_unl_1, p_unl_1.T) 
    #feat_cosim_unl_diff = torch.abs(feat_cosim_unl_0-feat_cosim_unl_1)
    matrices = [feat_cosim_unl_0,feat_cosim_unl_1,torch.matmul(p_unl_0, p_unl_1.T) ,torch.matmul(p_unl_1, p_unl_0.T)]

    feat_cosim_unl_diff=matrix_stddev(matrices)
    indices = minimize_matrix_sum(feat_cosim_unl_diff,8)
    #feat_cosim_unl_diff = torch.abs(feat_cosim_unl_0-feat_cosim_unl_1)
    #indices = minimize_matrix_sum(feat_cosim_unl_diff,10)
    #indices = find_top_10_min_indices(feat_cosim_unl_0,feat_cosim_unl_1).cpu()
    #print("indices",indices)
    #samples_unll = np.array(samples_unl)[indices]
    #print("samplesunll",samples_unll)
    input_feat_unll_0 = input_feat_unl_0[indices]
    #print(input_feat_unll_0.shape)
    input_feat_unll_1 = input_feat_unl_1[indices]
    input_feat_0 = torch.cat((input_feat_unll_0,input_feat_l_0),dim=0)
    input_feat_1 = torch.cat((input_feat_unll_1,input_feat_l_1),dim=0)
    
    p_0 = torch.nn.functional.normalize(input_feat_0, dim=1) 
    p_1 = torch.nn.functional.normalize(input_feat_1, dim=1) 
    # p = input_feat
    # print(p.shape)
    feat_cosim = (torch.matmul(p_0, p_0.T) + torch.matmul(p_0, p_1.T) + torch.matmul(p_1, p_0.T) + torch.matmul(p_1, p_1.T) ) / 4
    # print('feature consim is ',feat_cosim.shape)
    # exit()
    feat_cosim000 = torch.matmul(p_0, p_0.T)
    feat_cosim0001 = GraphLaplacian(feat_cosim000)
    # print('The shape of Lm is ', feat_cosim0001.shape)
    result_gpu, term1_gpu, term2_gpu = compute_formula_torch(feat_cosim0001)
    # print(f"The tolerance of DeltaS is: {result_gpu.item()}")

    feat_cosim_unl = feat_cosim[:8,:8]
    labels_ulpbs,r = get_ulbps_ulbonly(feat_cosim,yl)
    labels_ulbpsdornk = labels_ulpbs    
    # print(labels_ulbpsdornk.shape)

    # ktau = torch.abs(kendalrankloss(labels_ulbpsdornk, unlb_ref))
    # ktau_dist = ktau
    
    loss_ulb = torch.tensor(0).float().cuda()

    ps_ulb_ranked = torch.argsort(torch.argsort(labels_ulbpsdornk))
    # print(ps_ulb_ranked)
    # exit()
    #batch_unique_targets = torch.unique(ps_ulb_ranked)
    batch_unique_targets = torch.unique(r)
    if len(batch_unique_targets) < len(r):
        sampled_indices = []
        for target in batch_unique_targets:
            sampled_indices.append(random.choice((r == target).nonzero()[:,0]).item())
        feat_cosim_samp = feat_cosim_unl[:,sampled_indices]
        feat_cosim_samp = feat_cosim_samp[sampled_indices,:]        
        ps_ulb_ranked_samp = ps_ulb_ranked[sampled_indices]
    else:
        feat_cosim_samp = feat_cosim_unl
        ps_ulb_ranked_samp = ps_ulb_ranked
    
    for i in range(len(ps_ulb_ranked_samp)):
        # print("sampling i", i)
        label_ranks = rank_normalised(-torch.abs(ps_ulb_ranked_samp[i] - ps_ulb_ranked_samp).unsqueeze(-1).transpose(0,1))
        # print(-torch.abs(ps_ulb_ranked_samp[i] - ps_ulb_ranked_samp).unsqueeze(-1).transpose(0,1))
        # exit()
        feature_ranks_ulb0ulb0 = TrueRanker.apply(feat_cosim_samp[i].unsqueeze(dim=0), lambda_val)
        # print(feature_ranks_ulb0ulb0.shape, label_ranks.shape)
        loss_ulb += torch.nn.functional.mse_loss(feature_ranks_ulb0ulb0, label_ranks)

    return loss_ulb, ps_ulb_ranked, indices,r, result_gpu




def ulb_rank_prdlb_cps(input_feat_0, r,lambda_val = -1, pred_inp=None, samples = None):
    # samples = random.sample(range(0, len(input_feat)-1), 10)  # random sample 100 features
    input_feat_0 = input_feat_0[samples]
    
    # print(input_feat)
    # pred_inp = pred_inp[samples].detach()
    # print(pred_inp)
    # exit()
    
    
    p = input_feat_0
    # print(p.shape)
    
    # print(p)
    feat_cosim = -torch.abs(p - p.T)  # !!!!!!!! IMPORTANT!!!!
    ### ORDERING NEEDS TO BE CONSISTENT
    ### LARGER VALUE MUST SIGNAL MORE SIMILAR



    # print(feat_cosim.shape)
    # print(feat_cosim)
    # exit()

    labels_ulpbs = pred_inp.squeeze(-1)
    labels_ulbpsdornk = labels_ulpbs    
    # print(labels_ulbpsdornk.shape)
    # exit()
    # ktau = torch.abs(kendalrankloss(labels_ulbpsdornk, unlb_ref))
    # ktau_dist = ktau
    
    loss_ulb = torch.tensor(0).float().cuda()

    ps_ulb_ranked = torch.argsort(torch.argsort(labels_ulbpsdornk))
    #batch_unique_targets = torch.unique(ps_ulb_ranked)
    batch_unique_targets = torch.unique(r)
    if len(batch_unique_targets) < len(r):
        sampled_indices = []
        for target in batch_unique_targets:
            sampled_indices.append(random.choice((r == target).nonzero()[:,0]).item())
        feat_cosim_samp = feat_cosim[:,sampled_indices]
        feat_cosim_samp = feat_cosim_samp[sampled_indices,:]        
        ps_ulb_ranked_samp = ps_ulb_ranked[sampled_indices]
    else:
        feat_cosim_samp = feat_cosim
        ps_ulb_ranked_samp = ps_ulb_ranked
    
    feat_cosim_samp.shape
    for i in range(len(ps_ulb_ranked_samp)):
        # print("sampling i", i)
        label_ranks = rank_normalised(-torch.abs(ps_ulb_ranked_samp[i] - ps_ulb_ranked_samp).unsqueeze(-1).transpose(0,1))
        
        feature_ranks_ulb0ulb0 = TrueRanker.apply(feat_cosim_samp[i].unsqueeze(dim=0), lambda_val)
        # print(feature_ranks_ulb0ulb0.shape, label_ranks.shape)
        # print(feature_ranks_ulb0ulb0)
        # print(label_ranks)
        # exit()
        loss_ulb += torch.nn.functional.mse_loss(feature_ranks_ulb0ulb0, label_ranks)

    return loss_ulb

