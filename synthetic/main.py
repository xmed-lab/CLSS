
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from OrdinalEntropy import ordinal_entropy
import scipy.io as scio
from models import MLP
import time
import argparse

import sklearn.metrics


parser = argparse.ArgumentParser(description='Hyper-parameters management')


parser.add_argument('--fraction', type=float, default=0.5, help='stride of sliding window')
parser.add_argument('--ulb_w', type=float, default=1e-3, help='stride of sliding window')
parser.add_argument('--lambda_val', type=float, default= 2, help='stride of sliding window')

parser.add_argument('--Yoe', dest='oe', action='store_true')
parser.add_argument('--Noe', dest='oe', action='store_false')
parser.set_defaults(oe=True)


args = parser.parse_args()

def main(oe=True):

    m = 240
    lr = 1e-3
    epochs = 100000
    dataset_train = "train_sde.npz"
    dataset_test = "test_sde.npz"
    Lambda_d = 1e-3
    description = 'nonlinear'

    model = MLP(m).cuda()

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

    for times in range(10):   # run 10 times
        begin = time.time()
        model.init_weights()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        _mse_train = 9999
        _mse_test = 9999
        for epoch in range(epochs):
            X_train_small = X_train[1000 * times:1000 * (times+1)]
            y_train_small = y_train[1000 * times:1000 * (times+1)]
            model.train()
            optimizer.zero_grad()
            pred, feature = model(X_train_small)
            loss = mse_loss(pred, y_train_small)
            if oe:
                loss_oe = ordinal_entropy(feature, y_train_small) * Lambda_d
            else:
                loss_oe = loss * 0
            loss_all = loss + loss_oe
            loss_all.backward()

            optimizer.step()
            if epoch % 1000 ==0:
                model.eval()
                pred, feature = model(X_test)
                loss_test = mse_loss(pred, y_test)
                loss_mae = sklearn.metrics.mean_absolute_error(y_test, pred)
                loss_r2 = sklearn.metrics.r2_score(y_test, pred)
                print('{0}, Epoch: [{1}]\t'
                      'Loss_train: [{loss:.2e}]\t'
                      'Loss_test: [{loss_test:.2e}]\t'
                      'Loss_entropy: [{loss_e:.2e}]\t'
                      .format(description, epoch, loss=loss.data, loss_test=loss_test.data, loss_e=loss_oe.data), flush = True)

                if loss_test < _mse_test:
                    _mse_test = loss_test
                    _mse_train = loss
                    _r2_val = loss_r2
                    _mae_val = loss_mae

                    print('best model, Loss_test: [{loss_test:.2e}]  [{r2_val:.2e}]  [{mae_val:.2e}] '.format(
                        loss_test=_mse_test.data, r2_val = _r2_val.data, mae_val = _mae_val.data))

        l_test.append(_mse_test.cpu().detach().numpy())
        l_train.append(_mse_train.cpu().detach().numpy())
        r2_list.append(_r2_val.cpu().detach().numpy())
        mae_list.append(_mae_val.cpu().detach().numpy())
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

    path = './nonlinear_{}_{}_{}_{}.mat'.format(args.oe, args.fraction, args.ulb_w, args.lambda_val)
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
    
    print("oe", args.oe, "fraction", args.fraction, "ulb_w", args.ulb_w, "lambda_val", args.lambda_val)
    main(args.oe, args.fraction, args.ulb_w, args.lambda_val)


