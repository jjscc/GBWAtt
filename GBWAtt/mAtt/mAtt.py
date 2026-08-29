import torch
import torch.nn as nn
from mAtt.spd import LogEig, ReEig, TrilEmbed, cayley_map, matrix2skew, SPDTransform, SPDTangentSpace, SPDRectified
import mAtt.functionals as functions
from mAtt.functionals import *
from mAtt.functional import *
from scipy.special import beta
from typing import Any
import geoopt
from geoopt.manifolds.symmetric_positive_definite import SymmetricPositiveDefinite


def generate_sym(x):
    x_1 = (x + x.transpose(-1, -2)) / 2
    return x_1


def calcuK(S: torch.Tensor) -> torch.Tensor:
    Sr = S.unsqueeze(-2)
    Sc = S.unsqueeze(-1)
    K = Sc - Sr
    K = 1.0 / K
    K[torch.isinf(K)] = 0
    K[torch.isnan(K)] = 0
    return K


class OrthMapFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p):
        S, U = torch.linalg.eigh(x)
        S, indices = torch.sort(S, descending=True)
        U = torch.gather(U, -1, indices.unsqueeze(-2).expand_as(U))
        ctx.save_for_backward(U, S)
        return U[..., :p]

    @staticmethod
    def backward(ctx, grad_output):
        U, S = ctx.saved_tensors
        *batch_dims, h, w = grad_output.shape
        p = h - w
        pad_zero = torch.zeros(*batch_dims, h, p)
        grad_output = torch.cat((grad_output, pad_zero), -1)
        Ut = U.transpose(-1, -2)
        K = calcuK(S)
        mid_1 = K.transpose(-1, -2) * torch.matmul(Ut, grad_output)
        mid_2 = torch.matmul(U, mid_1)
        return torch.matmul(mid_2, Ut), None


class Signal2Spd(nn.Module):
    def __init__(self, power=1.0, device='cuda', tr_norm=False):
        super().__init__()
        self.device = device
        self.power = torch.tensor(power, device=self.device)
        self.tr_norm = tr_norm

    def forward(self, x):
        x = x.squeeze()
        x = x - x.mean(dim=-1, keepdim=True)
        cov = x @ x.transpose(-1, -2) / (x.shape[-1] - 1)
        if self.tr_norm:
            cov /= cov.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1, keepdim=True).unsqueeze(-1)
        cov = cov + (1e-5 * torch.eye(cov.shape[-1], device=x.device))
        return cov


def patch_len(n, epochs):
    base = n // epochs
    remainder = n % epochs
    list_len = [base + 1 if i < remainder else base for i in range(epochs)]
    return list_len


class E2R(nn.Module):
    def __init__(self, epochs, power=1.0, device='cuda', dim=-1):
        super().__init__()
        self.epochs = epochs
        self.device = device
        self.power = power
        self.signal2spd = Signal2Spd(self.power, device=device)
        self.dim = dim

    def forward(self, x):
        list_patch = patch_len(x.shape[self.dim], int(self.epochs))
        x_list = list(torch.split(x, list_patch, dim=self.dim))
        for i, item in enumerate(x_list):
            x_list[i] = self.signal2spd(item)

        x = torch.stack(x_list)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = x.permute(1, 0, 2, 3)
        return x


class E2RTimeLevel(nn.Module):
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs
        self.proj = Signal2Spd()
        self.epochs_dimension = -1

    def forward(self, x):
        x = x.squeeze()
        x_list = []
        for j in range(1, self.epochs + 1):
            list_patch = patch_len(x.shape[-1], int(j))
            x_len_list = list(torch.split(x, list_patch, dim=-1))
            for i, item in enumerate(x_len_list):
                x_list.append(self.proj(item))
        output = torch.stack(x_list)
        if output.ndim == 3:
            output = output.unsqueeze(1)
        output = output.permute(1, 0, 2, 3)
        return output




class GBWAttention(nn.Module):
    def __init__(self, in_size, out_size, device='cpu', spe_device='cpu', power=1.0):
        super(GBWAttention, self).__init__()
        self.device = device
        self.spd_device = spe_device
        self.d_in = in_size
        self.d_out = out_size
        self.dtype = torch.float
        self.alpha = nn.Parameter((torch.tensor(1).to(self.device, self.dtype)), requires_grad=True)
        self.beta = nn.Parameter((torch.tensor(1).to(self.device, self.dtype)), requires_grad=True)

        self.q_trans = SPDTransform(self.d_in, self.d_out)
        self.k_trans = SPDTransform(self.d_in, self.d_out)
        self.v_trans = SPDTransform(self.d_in, self.d_out)

        random_matrix = geoopt.ManifoldParameter(torch.eye(self.d_out, self.d_out), manifold=SymmetricPositiveDefinite()).repeat(3, 1, 1).unsqueeze(0)
        self.M = random_matrix.to(self.spd_device, self.dtype)
        self.pow = torch.tensor(float(power), dtype=self.dtype, device=self.spd_device)

   
    def tensor_log(self, t):  
        u, s, v = torch.svd(t)
        return u @ torch.diag_embed(torch.log(s)) @ v.permute(0, 1, 3, 2)

    def tensor_exp(self, t):  
        s, u = torch.linalg.eigh(t)
        return u @ torch.diag_embed(torch.exp(s)) @ u.permute(0, 1, 3, 2)

    def BuresWasserstein_distance(self, metricx_a, metricx_b):
        tra_a = tra(metricx_a)
        tra_b = tra(metricx_b)
        mid_1 = SqmEig.apply(CongrG(metricx_b, metricx_a, "pos"))
        mid_2 = tra(mid_1)
        final = tra_a + tra_b - 2 * mid_2
        final = final.sqrt()
        final = final
        return final


    def BuresWassersteinMean(self, weight, cov):
        bs = cov.shape[0]
        num_p = cov.shape[1]
        size = cov.shape[2]
        cov = SqmEig.apply(cov)

        h_1 = cov.view(bs, num_p, -1)
        h_2 = weight @ h_1 
        h_3 = h_2.view(bs, num_p, size, size)
        h_4 = PowerEig.apply(h_3, 2)
        return h_4


    def forward(self, x, shape=None):
        if len(x.shape) == 3 and shape is not None:
            x = x.view(shape[0], shape[1], self.d_in, self.d_in)

        bs = x.shape[0]
        m = x.shape[1]
        x = x.reshape(bs * m, self.d_in, self.d_in)

        Q = self.q_trans(x).view(bs, m, self.d_out, self.d_out)
        K = self.k_trans(x).view(bs, m, self.d_out, self.d_out)
        V = self.v_trans(x).view(bs, m, self.d_out, self.d_out)

        M = self.M
        Q = PowerEig.apply(Q, self.pow)
        Q = CongrG(Q, M, 'neg')
        K = PowerEig.apply(K, self.pow)
        K = CongrG(K, M, 'neg')
        V = PowerEig.apply(V, self.pow)
        V = CongrG(V, M, 'neg')

        Q_expand = Q.repeat(1, V.shape[1], 1, 1)
        K_expand = K.unsqueeze(2).repeat(1, 1, V.shape[1], 1, 1)
        K_expand = K_expand.view(K_expand.shape[0], K_expand.shape[1] * K_expand.shape[2], K_expand.shape[3],
                                 K_expand.shape[4])

        atten_energy = self.BuresWasserstein_distance(Q_expand, K_expand).view(V.shape[0], V.shape[1], V.shape[1])
        atten_prob = nn.Softmax(dim=-2)(1 / (1 + torch.log(1 + atten_energy))).permute(0, 2, 1)  # now row is c.c.

        output = self.BuresWassersteinMean(atten_prob, V)
        output = CongrG(output, M, 'pos')
        output = PowerEig.apply(output, 1 / self.pow)

        output = output.view(V.shape[0], V.shape[1], self.d_out, self.d_out)

        shape = list(output.shape[:2])
        shape.append(-1)

        output = output.contiguous().view(-1, self.d_out, self.d_out)
        return output, shape


class GBWAttBci2a(nn.Module):
    def __init__(self, args: Any):
        super().__init__()
        self.device = args.device
        self.spd_device = args.spd_device
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 22, (22, 1)).to(self.device),
            nn.BatchNorm2d(22).to(self.device),
            nn.Conv2d(22, args.in_size, (1, 12), padding=(0, 6)).to(self.device),
            nn.BatchNorm2d(args.in_size).to(self.device),
        )


        self.ract1 = E2R(epochs=args.epochs, power=1).to(self.device)
        self.ract2 = SPDRectified().to(self.spd_device)
        self.att2 = GBWAttention(args.in_size, args.out_size, self.device, self.spd_device,
                                  power=args.power)
        self.tangent = SPDTangentSpace(args.out_size).to(self.spd_device)
        self.flat = nn.Flatten().to(self.spd_device)
        self.linear = nn.Linear(args.out_size * (args.out_size + 1) // 2 * args.epochs, 4, bias=True, device='cpu')

    def forward(self, x):
        x = self.cnn(x.to(self.device))
        x = self.ract1(x)
        x, shape = self.att2(x.to(self.spd_device))
        x = self.ract2(x)
        x = self.tangent(x)
        x = x.view(shape[0], shape[1], -1)
        x = self.flat(x)
        x = self.linear(x.to(self.device))
        return x.to(self.spd_device)


class GBWAttMamem(nn.Module):
    def __init__(self, args: Any):
        super().__init__()
        self.device = args.device
        self.spd_device = args.spd_device

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 125, (8, 1)).to(self.device),
            nn.BatchNorm2d(125).to(self.device),
            nn.Conv2d(125, args.in_size, (1, 36), padding=(0, 18)).to(self.device),
            nn.BatchNorm2d(args.in_size).to(self.device)
        )

        self.ract1 = E2R(args.epochs, power=1).to(self.device)
        self.att2 = GBWAttention(args.in_size, args.out_size, self.device, self.spd_device,
                                  power=args.power)
        self.ract2 = SPDRectified().to(self.spd_device)
        self.tangent = SPDTangentSpace(args.out_size).to(self.spd_device)
        self.flat = nn.Flatten().to(self.spd_device)
        self.linear = nn.Linear(args.out_size * (args.out_size + 1) // 2 * args.epochs, 5, bias=True, device='cuda')

    def forward(self, x):
        x = self.cnn(x.to(self.device))
        x = self.ract1(x)
        x, shape = self.att2(x.to(self.spd_device))
        x = self.ract2(x)
        x = self.tangent(x)
        x = x.view(shape[0], shape[1], -1)
        x = self.flat(x)
        x = self.linear(x.to(self.device))
        return x.to(self.spd_device)


class GBWAttCha(nn.Module):
    def __init__(self, args: Any):
        super().__init__()
        self.device = args.device
        self.spd_device = args.spd_device
        self.epochs = args.epochs
        dim1 = 23
        self.cnn = nn.Sequential(
            nn.Conv2d(args.epochs, dim1 * args.epochs, (56, 1), groups=args.epochs).to(self.device),
            nn.BatchNorm2d(dim1 * args.epochs).to(self.device),
            nn.Conv2d(dim1 * args.epochs, args.in_size * args.epochs, (1, 64), padding=(0, 32), groups=args.epochs).to(
                self.device),
            nn.BatchNorm2d(args.in_size * args.epochs).to(self.device),
        )


        self.ract1 = E2R(epochs=args.epochs, power=1, device=args.device, dim=1).to(self.device)
        self.att2 = GBWAttention(args.in_size, args.out_size, self.device, self.spd_device,
                                  power=args.power)
        self.flat = nn.Flatten()
        self.linear = nn.Linear(args.out_size * (args.out_size + 1) // 2 * args.epochs, 2, bias=True).to(
            self.device)

  
        self.ract2 = SPDRectified().to(self.spd_device)
        self.tangent = SPDTangentSpace(args.out_size).to(self.spd_device)
        self.flat = nn.Flatten().to(self.spd_device)

    def forward(self, x):
        x = self.cnn(x.repeat(1, self.epochs, 1, 1).to(self.device))

        x = self.ract1(x)
        x, shape = self.att2(x.to(self.spd_device))
        x = self.ract2(x)
        x = self.tangent(x)
        x = x.view(shape[0], shape[1], -1)
        x = self.flat(x)
        x = self.linear(x.to(self.device))

        return x


