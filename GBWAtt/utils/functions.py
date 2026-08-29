import sys
import os
import torch.nn as nn
from sklearn.metrics import roc_auc_score as ras
import numpy as np
from torch.optim import RAdam, NAdam, Adadelta, Adam, SGD
import torch
from mAtt.optimizer import MixOptimizer
sys.path.append("..")
import geoopt



def select_optimizer(model_parameters,  optimizer_name, lr=1e-3, weight_decay=0):
    optimizer_name = optimizer_name.lower()
    model_parameters = [p for p in model_parameters if p.requires_grad]
    if optimizer_name == 'adadelta':
        return Adadelta(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        return geoopt.optim.RiemannianAdam(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd' and SGD is not None:
        return geoopt.optim.RiemannianSGD(model_parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}.")





def generate_model_path(args):
    filename = (f'repeat{args.repeat}_sub{args.sub}_bs{args.bs}_epochs{args.epochs}_lr{args.lr}_wd{args.wd}_'
                f'in_size{args.in_size}_out_size{args.out_size}_power{args.power}.pt')


    return filename


def testNetwork(net, testloader):
    net.eval()
    acc, test_len = 0, 0
    for xb, yb in testloader:
        with torch.no_grad():
            test_len += yb.shape[0]
            pred = net(xb)
            acc += (torch.max(pred, 1).indices == yb).sum().item()

    return acc / test_len


def testNetwork_auc(net, testloader):
    net.eval()
    y_pred = torch.empty(0)
    y_true = torch.empty(0)
    for xb, yb in testloader:
        with torch.no_grad():
            pred = net(xb).cpu()
            y_pred = torch.cat((y_pred, pred[:, 1]), 0)
            y_true = torch.cat((y_true, yb.cpu()), 0)

    return ras(y_true.detach().numpy(), y_pred.detach().numpy())


def Network_acc(net, loader):
    net.eval()
    acc = 0
    softmax = nn.Softmax(dim=1)
    for xb, yb in loader:
        with torch.no_grad():
            pred = net(xb)
            if torch.argmax(softmax(pred)).item() == yb:
                acc += 1

    return acc / len(loader)


def Network_auc(net, loader):
    net.eval()
    y_pred = torch.empty(0)
    y_true = torch.empty(0)
    for xb, yb in loader:
        with torch.no_grad():
            pred = net(xb)
            y_pred = torch.cat((y_pred, pred[:, 1]), 0)
            y_true = torch.cat((y_true, yb), 0)

    return ras(y_true.detach().numpy(), y_pred.detach().numpy())


def save_res(res, res_path):
    res = res.numpy()
    mean = np.mean(res).item()
    std = np.std(res, axis=1, ddof=0).mean()
    mean_st = [f'{x:.4f}' for x in np.mean(res, axis=1).tolist()]
    mean_up = [f'{x:.4f}' for x in np.mean(res, axis=0).tolist()]
    std_st = [f'{x:.4f}' for x in np.std(res, axis=1, ddof=0).tolist()]
    print(f"mean:{mean:.2f}\tstd:{std:.2f}")
    header_info = 'Mean: {:.4f}\t'.format(mean) + 'Std: {:.4f}\n'.format(std) \
                  + f'St.Mean: {", ".join(mean_st)}\n' + f'St.Mean_up: {", ".join(mean_up)}\n' + f'St.Std:  {", ".join(std_st)}\n'

    np.savetxt(res_path, res, fmt='%.4f', comments='', delimiter='\t', header=header_info)

import torch.nn.utils as nn_utils
def train_network(net, train_loader, valid_loader, test_loader, args):

    CE = nn.CrossEntropyLoss()

    optimizer = select_optimizer(net.parameters(), args.optim, lr=args.lr, weight_decay=args.wd)
    bestLoss = 1e10
    test_metric = 0
    for ite in range(args.iterations):

        net.train()
        acc_val, acc_tr, tr_len, val_len, TL, train_loss = 0, 0, 0, 0, 0, 0
        for xb, yb in train_loader:
            tr_len += yb.shape[0]
            out = net(xb)
            loss = CE(out, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc_tr += (torch.max(out, 1).indices == yb).sum().item()
            train_loss += loss.item() * yb.shape[0]
        net.eval()
        for xb, yb in valid_loader:
            val_len += yb.shape[0]
            with torch.no_grad():
                out = net(xb)
                acc_val += (torch.max(out, 1).indices == yb).sum().item()
                TL += CE(out, yb).item() * yb.shape[0]

        if TL < bestLoss:
            if not os.path.exists(args.model_path):
                os.makedirs(args.model_path)
            bestLoss = TL
            final_path = generate_model_path(args)
            final_path = os.path.join(args.model_path, final_path)
            torch.save(net, final_path)
            try:
                testnet = torch.load(final_path, weights_only=False)
            except TypeError:
                testnet = torch.load(final_path)
            if args.scoring_metric == 'acc':
                test_metric = testNetwork(testnet, test_loader)
            elif args.scoring_metric == 'auc':
                test_metric = testNetwork_auc(testnet, test_loader)

        print(
            f'epoch:{ite + 1:03d}/{args.iterations} '
            f'train_loss:{train_loss / tr_len:.4f} train_acc:{acc_tr / tr_len:.4f} '
            f'val_loss:{TL / val_len:.4f} val_acc:{acc_val / val_len:.4f} '
            f'test_{args.scoring_metric}:{test_metric:.4f}')

    return test_metric

