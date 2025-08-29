"""
Experiments on linear/non-linear operators learning tasks, results as mean +- standard variance over 10 runs.
"""
import torch
import numpy as np
import random


from torch import dot, argsort
from torch import sign, count_nonzero, ones, reshape, eye, dot, argsort
from torch.linalg import eig, eigh



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

    # H is the hermitian matrix
    H = centering_matrix(n)
    # s and Zp are the corresponding eigenvalues and eigenvectors of H
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



def get_ulbps_ulbonly(simMat):

    ## SS algorithm
    #### input is (lb, unlb) X (lb, unlb) sim matrix
    #### output is (lb + ulb_pslb_tp), ### keep simple for now, just take the closes one 
    S = simMat

    n = S.shape[0]
    Z = torch.tensor(get_the_subspace_basis(n, verbose=False)).float().cuda()

    # print(S.shape)
    Ls = GraphLaplacian(S)
    ztLsz = torch.matmul(torch.matmul(Z.T, Ls), Z)
    w, v = eig(ztLsz)
    w = torch.view_as_real(w)[:,0]
    v = torch.view_as_real(v)[...,0]

    if torch.is_complex(w):
        print("complex")
        return None

    ind = torch.argsort(w)
    v = v[:, ind]
    r = reshape(torch.matmul(Z,v[:, 0]), (n, 1))

    _, _, rsign = compute_upsets(r, S, verbose=False)

    r_final = rsign * r
    ### r_final is shape [n, 1]
    r_rank = torch.argsort(torch.argsort(r_final.reshape(-1)))
    
    return r_rank


def mix_rank_func(input_feat_lb, input_feat_ulb, subsamp = False):
    result = 0.0
    return result

def mix_rank_prdlb_func(input_feat):
    result = 0.0
    return result

def ulb_rank(input_feat, lambda_val = -1, subsamp = False):

    if subsamp:
        samples = random.sample(range(0, len(input_feat)-1), 10)  # random sample 100 features
        input_feat = input_feat[samples]

    p = torch.nn.functional.normalize(input_feat, dim=1)
    # p = input_feat
    # print(p.shape)
    feat_cosim = torch.matmul(p, p.T)
    # print(feat_cosim)
    # exit()

    ## compute R
    labels_ulpbs = get_ulbps_ulbonly(feat_cosim)
    labels_ulbpsdornk = labels_ulpbs    
    # print(labels_ulbpsdornk.shape)

    # ktau = torch.abs(kendalrankloss(labels_ulbpsdornk, unlb_ref))
    # ktau_dist = ktau
    
    loss_ulb = torch.tensor(0).float().cuda()

    ps_ulb_ranked = torch.argsort(torch.argsort(labels_ulbpsdornk))
    # print(ps_ulb_ranked)
    # exit()
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
        # print(-torch.abs(ps_ulb_ranked_samp[i] - ps_ulb_ranked_samp).unsqueeze(-1).transpose(0,1))
        # exit()
        ## rk operator
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











def ulb_rank_cps(input_feat_0, input_feat_1, lambda_val = -1, subsamp = False, batch_size_two = 8):

    if subsamp:
        samples = random.sample(range(0, len(input_feat_0)-1), batch_size_two)  # random sample 100 features
        input_feat_0 = input_feat_0[samples]
        input_feat_1 = input_feat_1[samples]

    p_0 = torch.nn.functional.normalize(input_feat_0, dim=1)
    p_1 = torch.nn.functional.normalize(input_feat_1, dim=1)
    # p = input_feat
    # print(p.shape)
    feat_cosim = (torch.matmul(p_0, p_0.T) + torch.matmul(p_0, p_1.T) + torch.matmul(p_1, p_0.T) + torch.matmul(p_1, p_1.T) ) / 4
    # print(feat_cosim)
    # exit()

    labels_ulpbs = get_ulbps_ulbonly(feat_cosim)
    labels_ulbpsdornk = labels_ulpbs    
    # print(labels_ulbpsdornk.shape)

    # ktau = torch.abs(kendalrankloss(labels_ulbpsdornk, unlb_ref))
    # ktau_dist = ktau
    
    loss_ulb = torch.tensor(0).float().cuda()

    ps_ulb_ranked = torch.argsort(torch.argsort(labels_ulbpsdornk))
    # print(ps_ulb_ranked)
    # exit()
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
        # print(-torch.abs(ps_ulb_ranked_samp[i] - ps_ulb_ranked_samp).unsqueeze(-1).transpose(0,1))
        # exit()
        feature_ranks_ulb0ulb0 = TrueRanker.apply(feat_cosim_samp[i].unsqueeze(dim=0), lambda_val)
        # print(feature_ranks_ulb0ulb0.shape, label_ranks.shape)
        loss_ulb += torch.nn.functional.mse_loss(feature_ranks_ulb0ulb0, label_ranks)

    return loss_ulb, ps_ulb_ranked, samples




def ulb_rank_prdlb_cps(input_feat_0, lambda_val = -1, pred_inp=None, samples = None):
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

