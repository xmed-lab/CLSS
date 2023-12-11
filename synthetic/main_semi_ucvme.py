
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from OrdinalEntropy import ordinal_entropy
import scipy.io as scio
from models import MLP_unc
import time
import random


from torch import dot, argsort
from torch import sign, count_nonzero, ones, reshape, eye, dot, argsort
from torch.linalg import eig, eigh

from scipy.stats import kendalltau
from torchmetrics.regression import KendallRankCorrCoef
import argparse

import sklearn.metrics


parser = argparse.ArgumentParser(description='Hyper-parameters management')


parser.add_argument('--fraction', type=float, default=0.5, help='stride of sliding window')
parser.add_argument('--ulb_w', type=float, default=1e-3, help='stride of sliding window')
parser.add_argument('--lambda_val', type=float, default= 2, help='stride of sliding window')

parser.add_argument('--Yoe', dest='oe', action='store_true')
parser.add_argument('--Noe', dest='oe', action='store_false')
parser.set_defaults(oe=True)


parser.add_argument('--samp_ssl', type=int, default= 5, help='stride of sliding window')
parser.add_argument('--samp_fq', type=int, default= 5, help='stride of sliding window')



args = parser.parse_args()


np.random.seed(0)



kendalrankloss = KendallRankCorrCoef()

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



def get_ulbps_ulbonly(simMat):

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




def ulb_rank(input_feat, lambda_val = -1):
    samples = random.sample(range(0, len(input_feat)-1), 10)  # random sample 100 features
    input_feat = input_feat[samples]


    p = torch.nn.functional.normalize(input_feat, dim=1)
    # print(p.shape)
    feat_cosim = torch.matmul(p, p.T)
    # print(feat_cosim.shape)
    # exit()

    labels_ulpbs = get_ulbps_ulbonly(feat_cosim)
    labels_ulbpsdornk = labels_ulpbs    

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
        loss_ulb += torch.nn.functional.mse_loss(feature_ranks_ulb0ulb0, label_ranks)

    return loss_ulb



def main(oe=True, fraction = 0.25, ulb_w = 0, lambda_val = 2, samp_fq = 5, samp_ssl = 5):

    m = 240
    lr = 1e-3
    epochs = 100000
    dataset_train = "train_sde.npz"
    dataset_test = "test_sde.npz"
    Lambda_d = 1e-3
    description = 'nonlinear'

    model_0 = MLP_unc(m).cuda()
    model_1 = MLP_unc(m).cuda()

    d = np.load(dataset_train)
    X_train, y_train = (d["X_train0"], d["X_train1"]), d["y_train"]
    d = np.load(dataset_test)
    X_test, y_test = (d["X_test0"], d["X_test1"]), d["y_test"]
    X_train = np.hstack(X_train)
    X_test = np.hstack(X_test)
    X_train = Variable(torch.from_numpy(X_train), requires_grad=True).float().cuda()
    y_train = Variable(torch.from_numpy(y_train), requires_grad=True).float().cuda()
    X_test = Variable(torch.from_numpy(X_test), requires_grad=True).float().cuda()
    y_test = Variable(torch.from_numpy(y_test), requires_grad=True).float().cuda()

    mse_loss = nn.MSELoss().cuda()

    l_train = []
    l_test = []
    r2_list = []
    mae_list = []

    for times in range(10):   # run 10 times
        begin = time.time()
        model_0.init_weights()
        optimizer_0 = torch.optim.AdamW(model_0.parameters(), lr=lr)
        model_1.init_weights()
        optimizer_1 = torch.optim.AdamW(model_1.parameters(), lr=lr)
        _mse_train = 9999
        _mse_test = 9999

        _r2_val = 9999
        _mae_val = 9999

        sample_idx = np.arange(1000)
        np.random.shuffle(sample_idx)
        lab_sample_idx = sample_idx[:int(1000*fraction)]
        unlab_sample_idx = sample_idx[int(1000*fraction) : 2 * int(1000*fraction)]


        for epoch in range(epochs):
        # for epoch in range(5):

            X_train_ALL = X_train[1000 * times:1000 * (times+1)]
            y_train_ALL = y_train[1000 * times:1000 * (times+1)]

            X_train_small = X_train_ALL[lab_sample_idx]
            y_train_small = y_train_ALL[lab_sample_idx]

            X_train_ulb = X_train_ALL[unlab_sample_idx]
            y_train_ulb = y_train_ALL[unlab_sample_idx]

            model_0.train()
            model_1.train()
            optimizer_0.zero_grad()
            optimizer_1.zero_grad()


            pred_0, var_0 = model_0(X_train_small)
            mean = pred_0.view(-1)
            var = var_0.view(-1)
            # loss_0_mse = mse_loss(mean, y_train_small.view(-1))
            loss_0_mse = (mean - y_train_small.view(-1)) ** 2

            
            pred_1, var_1 = model_1(X_train_small)
            mean_1 = pred_1.view(-1)
            var_1 = var_1.view(-1)
            # loss_1_mse = mse_loss(mean_1, y_train_small.view(-1))
            loss_1_mse = (mean_1 - y_train_small.view(-1)) ** 2

            
            # print(pred_0.shape, mean.shape, y_train_small.shape)
            # print("loss_1_mse.shape, mean_1.shape, var_1.shape", loss_1_mse.shape, mean_1.shape, var_1.shape)
            
            loss1_0 = torch.mul(torch.exp(-(var + var_1) / 2), loss_0_mse)
            loss2_0 = (var + var_1) / 2
            loss_0_all = .5 * (loss1_0 + loss2_0)
            loss_0 = loss_0_all.mean()

            loss1_1 = torch.mul(torch.exp(-(var + var_1) / 2), loss_1_mse)
            loss2_1 = (var + var_1) / 2
            loss_1_all = .5 * (loss1_1 + loss2_1)
            loss_1 = loss_1_all.mean()


            loss = (loss_0 + loss_1) / 2

            pred_ulb_0, var_ulb_0 = model_0(X_train_ulb)
            pred_ulb_1, var_ulb_1 = model_1(X_train_ulb)

            mean1s_0 = []
            mean2s_0 = []
            var1s_0 = []

            mean1s_1 = []
            mean2s_1 = []
            var1s_1 = []

            with torch.no_grad():
                for samp_ssl_itr in range(samp_ssl):
                    mean1_raw_0, var1_raw_0 = model_0(X_train_ulb)
                    mean1_0 = mean1_raw_0.view(-1)
                    var1_0 = var1_raw_0.view(-1)

                    mean1s_0.append(mean1_0** 2)
                    mean2s_0.append(mean1_0)
                    var1s_0.append(var1_0)

                    mean1_raw_1, var1_raw_1 = model_1(X_train_ulb)
                    mean1_1 = mean1_raw_1.view(-1)
                    var1_1 = var1_raw_1.view(-1)

                    mean1s_1.append(mean1_1** 2)
                    mean2s_1.append(mean1_1)
                    var1s_1.append(var1_1)

            mean1s_0_ = torch.stack(mean1s_0, dim=0).mean(dim=0)
            mean2s_0_ = torch.stack(mean2s_0, dim=0).mean(dim=0)
            var1s_0_ = torch.stack(var1s_0, dim=0).mean(dim=0)

            mean1s_1_ = torch.stack(mean1s_1, dim=0).mean(dim=0)
            mean2s_1_ = torch.stack(mean2s_1, dim=0).mean(dim=0)
            var1s_1_ = torch.stack(var1s_1, dim=0).mean(dim=0)

            all_output_unlb_0_pslb = mean2s_0_
            all_output_unlb_1_pslb = mean2s_1_

            avg_mean01 = (all_output_unlb_0_pslb + all_output_unlb_1_pslb)/2
            avg_var01 = (var1s_0_ + var1s_1_)/2

            # print(pred_ulb_0.shape, avg_mean01.shape)
            # print(pred_ulb_1.shape, avg_mean01.shape)
            # print(avg_var01.shape)
            # print("***" * 10)

            # loss_mse_cps_0 = mse_loss(pred_ulb_0.view(-1), avg_mean01)
            # loss_mse_cps_1 = mse_loss(pred_ulb_1.view(-1), avg_mean01)

            # (mean_1 - y_train_small.view(-1)) ** 2
            loss_mse_cps_0 = (pred_ulb_0.view(-1) - avg_mean01) ** 2
            loss_mse_cps_1 = (pred_ulb_1.view(-1) - avg_mean01) ** 2
            # print(pred_ulb_0.shape, avg_mean01.shape, loss_mse_cps_0.shape)

            loss_cmb_cps_0 = 0.5 * (torch.mul(torch.exp(-avg_var01), loss_mse_cps_0) + avg_var01 )
            loss_cmb_cps_1 = 0.5 * (torch.mul(torch.exp(-avg_var01), loss_mse_cps_1) + avg_var01 )
            
            # print(loss_cmb_cps_1.shape)
            ulb_loss_0 = loss_cmb_cps_0.mean()
            ulb_loss_1 = loss_cmb_cps_1.mean()

            var_loss_ulb_0 = ((var_ulb_0.view(-1) - avg_var01)**2).mean()
            var_loss_ulb_1 = ((var_ulb_1.view(-1) - avg_var01)**2).mean()


            loss_ulb = (ulb_loss_0 + ulb_loss_1 + var_loss_ulb_0 + var_loss_ulb_1) * ulb_w

            # print(pred_ulb_1_pslb.shape)
            # print(y_train_small.shape)

            loss_all = loss_0 + loss_1 + loss_ulb
            loss_all.backward()

            optimizer_0.step()
            optimizer_1.step()
            
            if epoch % 1000 ==0:
            # if epoch % 2 ==0:
                model_0.eval()
                model_1.eval()

                mean1s = []
                mean2s = []
                var1s = []


                mean1s_m1 = []
                mean2s_m1 = []
                var1s_m1 = []


                with torch.no_grad():
                    for samp_itr in range(samp_fq):
                        all_ouput = model_0(X_test)
                        mean1_raw, var1_raw = all_ouput

                        mean1 = mean1_raw.view(-1)
                        var1 = var1_raw.view(-1)

                        mean1s.append(mean1** 2)
                        mean2s.append(mean1)
                        var1s.append(torch.exp(var1))


                        all_ouput_m1 = model_1(X_test)
                        mean1_raw_m1, var1_raw_m1 = all_ouput_m1

                        mean1_m1 = mean1_raw_m1.view(-1)
                        var1_m1 = var1_raw_m1.view(-1)

                        mean1s_m1.append(mean1_m1** 2)
                        mean2s_m1.append(mean1_m1)
                        var1s_m1.append(torch.exp(var1_m1))
                
                mean1s_ = torch.stack(mean1s, dim=0).mean(dim=0)
                mean2s_ = torch.stack(mean2s, dim=0).mean(dim=0)
                var1s_ = torch.stack(var1s, dim=0).mean(dim=0)

                mean1s_m1_ = torch.stack(mean1s_m1, dim=0).mean(dim=0)
                mean2s_m1_ = torch.stack(mean2s_m1, dim=0).mean(dim=0)
                var1s_m1_ = torch.stack(var1s_m1, dim=0).mean(dim=0)

                pred = (mean2s_ + mean2s_m1_) / 2
                loss_test = mse_loss(pred, y_test.view(-1))

                # print(pred.shape, y_test.shape)
                # exit()
                y_test_npy = y_test.cpu().detach().numpy()
                pred_npy = pred.cpu().detach().numpy()
                loss_mae = sklearn.metrics.mean_absolute_error(y_test_npy, pred_npy)
                loss_r2 = sklearn.metrics.r2_score(y_test_npy, pred_npy)
                print('{0}, Epoch: [{1}]\t'
                      'Loss_train: [{loss:.2e}]\t'
                      'Loss_test: [{loss_test:.2e}]\t'
                      'Loss_entropy: [{loss_e:.2e}]\t'
                      'Loss_entropy: [{loss_u:.2e}]\t'
                      .format(description, epoch, loss=loss.data, loss_test=loss_test.data, loss_e=0, loss_u = loss_ulb.data), flush = True)

                if loss_test < _mse_test:
                    _mse_test = loss_test
                    _mse_train = loss
                    _r2_val = loss_r2
                    _mae_val = loss_mae

                    print('best model, Loss_test: [{loss_test:.2e}]  [{r2_val:.2e}]  [{mae_val:.2e}] '.format(
                        loss_test=_mse_test.data, r2_val = _r2_val, mae_val = _mae_val))

        l_test.append(_mse_test.cpu().detach().numpy())
        l_train.append(_mse_train.cpu().detach().numpy())
        r2_list.append(_r2_val)
        mae_list.append(_mae_val)
        end = time.time()
        print(end-begin)

    l_train = np.array(l_train)
    l_test = np.array(l_test)
    r2_list = np.array(r2_list)
    mae_list = np.array(mae_list)
    train_dict = {}
    train_dict['train_mse'] = l_train
    train_dict['test_mse'] = l_test
    train_dict['testr2'] = r2_list
    train_dict['testmae'] = mae_list
    path = './nonlinearUCVME_{}_{}_{}_{}.mat'.format(args.oe, args.fraction, args.ulb_w, args.lambda_val)
    scio.savemat(path, train_dict)
    print('Mean: \t')
    print(np.mean(l_test))
    print('Std: \t')
    print(np.std(l_test))
    print('R2 Mean: \t')
    print(np.mean(r2_list))
    print('R2 Std: \t')
    print(np.std(r2_list))
    print('MAE Mean: \t')
    print(np.mean(mae_list))
    print('MAE Std: \t')
    print(np.std(mae_list))


if __name__ == "__main__":
    
    print("oe", args.oe, "fraction", args.fraction, "ulb_w", args.ulb_w, "lambda_val", args.lambda_val, "args.samp_fq", args.samp_fq, "args.samp_ssl", args.samp_ssl)
    main(oe = args.oe, fraction = args.fraction, ulb_w = args.ulb_w, lambda_val = args.lambda_val, samp_fq = args.samp_fq, samp_ssl = args.samp_ssl)


