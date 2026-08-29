import numpy as np
import torch as th
import torch.nn as nn
from torch.autograd import Function as F

dtype = th.double


class StiefelParameter(nn.Parameter):
    """ Parameter constrained to the Stiefel manifold (for BiMap layers) """
    pass


def init_bimap_parameter(W):
    """ initializes a (ho,hi,ni,no) 4D-StiefelParameter"""
    ho, hi, ni, no = W.shape
    for i in range(ho):
        for j in range(hi):
            v = th.empty(ni, ni, dtype=W.dtype, device=W.device).uniform_(0., 1.)
            vv = th.svd(v.matmul(v.t()))[0][:, :no]
            W.data[i, j] = vv


def init_bimap_parameter_identity(W):
    """ initializes to identity a (ho,hi,ni,no) 4D-StiefelParameter"""
    ho, hi, ni, no = W.shape
    for i in range(ho):
        for j in range(hi):
            W.data[i, j] = th.eye(ni, no)


class SPDParameter(nn.Parameter):
    """ Parameter constrained to the SPD manifold (for ParNorm) """
    pass


def bimap(X, W):
    '''
    Bilinear mapping function
    :param X: Input matrix of shape (batch_size,n_in,n_in)
    :param W: Stiefel parameter of shape (n_in,n_out)
    :return: Bilinearly mapped matrix of shape (batch_size,n_out,n_out)
    '''
    return W.t().matmul(X).matmul(W)


def bimap_channels(X, W):
    '''
    Bilinear mapping function over multiple input and output channels
    :param X: Input matrix of shape (batch_size,channels_in,n_in,n_in)
    :param W: Stiefel parameter of shape (channels_out,channels_in,n_in,n_out)
    :return: Bilinearly mapped matrix of shape (batch_size,channels_out,n_out,n_out)
    '''
    batch_size, channels_in, n_in, _ = X.shape
    channels_out, _, _, n_out = W.shape
    P = th.zeros(batch_size, channels_out, n_out, n_out, dtype=X.dtype, device=X.device)
    for co in range(channels_out):
        P[:, co, :, :] = sum([bimap(X[:, ci, :, :], W[co, ci, :, :]) for ci in range(channels_in)])
    return P


def modeig_forward(P, op, eig_mode='svd', param=None):
    '''
    Generic forward function of non-linear eigenvalue modification
    LogEig, ReEig, etc inherit from this class
    Input P: (batch_size,channels) SPD matrices of size (n,n)
    Output X: (batch_size,channels) modified symmetric matrices of size (n,n)
    '''
    batch_size, channels, n, n = P.shape  # batch size,channel depth,dimension
    U, S = th.zeros_like(P, device=P.device), th.zeros(batch_size, channels, n, dtype=P.dtype, device=P.device)
    for i in range(batch_size):
        for j in range(channels):
            if (eig_mode == 'eig'):
                s, U[i, j] = th.linalg.eigh(P[i, j])
                S[i, j] = s[:]
            elif (eig_mode == 'svd'):
                U[i, j], S[i, j], _ = th.svd(P[i, j])
    S_fn = op.fn(S, param)
    X = U.matmul(BatchDiag(S_fn)).matmul(U.transpose(2, 3))
    return X, U, S, S_fn


def modeig_backward(dx, U, S, S_fn, op, param=None):
    '''
    Generic backward function of non-linear eigenvalue modification
    LogEig, ReEig, etc inherit from this class
    Input P: (batch_size,channels) SPD matrices of size (n,n)
    Output X: (batch_size,channels) modified symmetric matrices of size (n,n)
    '''
    S_fn_deriv = BatchDiag(op.fn_deriv(S, param))
    SS = S[..., None].repeat(1, 1, 1, S.shape[-1])
    SS_fn = S_fn[..., None].repeat(1, 1, 1, S_fn.shape[-1])
    L = (SS_fn - SS_fn.transpose(2, 3)) / (SS - SS.transpose(2, 3))
    L[L == -np.inf] = 0;
    L[L == np.inf] = 0;
    L[th.isnan(L)] = 0
    L = L + S_fn_deriv
    dp = L * (U.transpose(2, 3).matmul(dx).matmul(U))
    dp = U.matmul(dp).matmul(U.transpose(2, 3))
    return dp


class LogEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of log eigenvalues matrices of size (n,n)
    """

    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward(P, Log_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Log_op)


class ReEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of rectified eigenvalues matrices of size (n,n)
    """

    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward(P, Re_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Re_op)


class ExpEig(F):
    """
    Input P: (batch_size,h) symmetric matrices of size (n,n)
    Output X: (batch_size,h) of exponential eigenvalues matrices of size (n,n)
    """

    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward(P, Exp_op, eig_mode='eig')
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Exp_op)


class SqmEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of square root eigenvalues matrices of size (n,n)
    """

    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward(P, Sqm_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Sqm_op)


class SqminvEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of inverse square root eigenvalues matrices of size (n,n)
    """

    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward(P, Sqminv_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Sqminv_op)


class PowerEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of power eigenvalues matrices of size (n,n)
    """

    @staticmethod
    def forward(ctx, P, power):
        Power_op._power = power
        X, U, S, S_fn = modeig_forward(P, Power_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Power_op), None


class InvEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of inverse eigenvalues matrices of size (n,n)
    """

    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward(P, Inv_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Inv_op)


def cov_pool(f, reg_mode='mle'):
    """
    Input f: Temporal n-dimensionnal feature map of length T (T=1 for a unitary signal) (batch_size,n,T)
    Output ret: Covariance matrix of size (batch_size,1,n,n)
    """
    bs, n, T = f.shape
    X = f.matmul(f.transpose(-1, -2)) / (T - 1)
    if (reg_mode == 'mle'):
        ret = X
    elif (reg_mode == 'add_id'):
        ret = add_id(X, 1e-6)
    elif (reg_mode == 'adjust_eig'):
        ret = adjust_eig(X, 0.75)
    if (len(ret.shape) == 3):
        return ret[:, None, :, :]
    return ret


def cov_pool_mu(f, reg_mode):
    """
    Input f: Temporal n-dimensionnal feature map of length T (T=1 for a unitary signal) (batch_size,n,T)
    Output ret: Covariance matrix of size (batch_size,1,n,n)
    """
    alpha = 1
    bs, n, T = f.shape
    mu = f.mean(-1, True);
    f = f - mu
    X = f.matmul(f.transpose(-1, -2)) / (T - 1) + alpha * mu.matmul(mu.transpose(-1, -2))
    aug1 = th.cat((X, alpha * mu), 2)
    aug2 = th.cat((alpha * mu.transpose(1, 2), th.ones(mu.shape[0], 1, 1, dtype=mu.dtype, device=f.device)), 2)
    X = th.cat((aug1, aug2), 1)
    if (reg_mode == 'mle'):
        ret = X
    elif (reg_mode == 'add_id'):
        ret = add_id(X, 1e-6)
    elif (reg_mode == 'adjust_eig'):
        ret = adjust_eig(0.75)(X)
    if (len(ret.shape) == 3):
        return ret[:, None, :, :]
    return ret


def add_id(P, alpha):
    '''
    Input P of shape (batch_size,1,n,n)
    Add Id
    '''
    for i in range(P.shape[0]):
        P[i] = P[i] + alpha * P[i].trace() * th.eye(P[i].shape[-1], dtype=P.dtype, device=P.device)
    return P


def CongrG(P, G, mode):
    """
    Input P: (batch_size,channels) SPD matrices of size (n,n) or single matrix (n,n)
    Input G: matrix (n,n) to do the congruence by
    Output PP: (batch_size,channels) of congruence by sqm(G) or sqminv(G) or single matrix (n,n)
    """
    if (mode == 'pos'):
        GG = SqmEig.apply(G)
    elif (mode == 'neg'):
        GG = SqminvEig.apply(G)
    PP = GG.matmul(P).matmul(GG)
    return PP


def BatchDiag(P):
    """
    Input P: (batch_size,channels) vectors of size (n)
    Output Q: (batch_size,channels) diagonal matrices of size (n,n)
    """
    batch_size, channels, n = P.shape  # batch size,channel depth,dimension
    Q = th.zeros(batch_size, channels, n, n, dtype=P.dtype, device=P.device)
    for i in range(batch_size):
        for j in range(channels):
            Q[i, j] = P[i, j].diag()
    return Q


def karcher_step_weighted(x, G, alpha, weights):
    '''
    One step in the Karcher flow
    Weights is a weight vector of shape (batch_size,)
    Output is mean of shape (n,n)
    '''
    x_log = LogG(x, G)
    G_tan = x_log.mul(weights[:, None, None, None]).sum(dim=0)[None, ...]
    G = ExpG(alpha * G_tan, G)[0, 0]
    return G


def bary_geom_weighted(x, weights):
    '''
    Function which computes the weighted Riemannian barycenter for a batch of data using the Karcher flow
    Input x is a batch of SPD matrices (batch_size,1,n,n) to average
    Weights is a weight vector of shape (batch_size,)
    Output is (1,1,n,n) Riemannian mean
    '''
    k = 1
    alpha = 1
    # with th.no_grad():
    G = x.mul(weights[:, None, None, None]).sum(dim=0)[0, :, :]
    for _ in range(k):
        G = karcher_step_weighted(x, G, alpha, weights)
    return G[None, None, :, :]


class Log_op():
    """ Log function and its derivative """

    @staticmethod
    def fn(S, param=None):
        return th.log(S)

    @staticmethod
    def fn_deriv(S, param=None):
        return 1 / S


class Re_op():
    """ Log function and its derivative """
    _threshold = 1e-4

    @classmethod
    def fn(cls, S, param=None):
        return nn.Threshold(cls._threshold, cls._threshold)(S)

    @classmethod
    def fn_deriv(cls, S, param=None):
        return (S > cls._threshold).double()


class Sqm_op():
    """ Log function and its derivative """

    @staticmethod
    def fn(S, param=None):
        return th.sqrt(S)

    @staticmethod
    def fn_deriv(S, param=None):
        return 0.5 / th.sqrt(S)


class Sqminv_op():
    """ Log function and its derivative """

    @staticmethod
    def fn(S, param=None):
        return 1 / th.sqrt(S)

    @staticmethod
    def fn_deriv(S, param=None):
        return -0.5 / th.sqrt(S) ** 3


class Power_op():
    """ Power function and its derivative """
    _power = 1

    @classmethod
    def fn(cls, S, param=None):
        return S ** cls._power

    @classmethod
    def fn_deriv(cls, S, param=None):
        return (cls._power) * S ** (cls._power - 1)


class Inv_op():
    """ Inverse function and its derivative """

    @classmethod
    def fn(cls, S, param=None):
        return 1 / S

    @classmethod
    def fn_deriv(cls, S, param=None):
        return log(S)


class Exp_op():
    """ Log function and its derivative """

    @staticmethod
    def fn(S, param=None):
        return th.exp(S)

    @staticmethod
    def fn_deriv(S, param=None):
        return th.exp(S)



class Lyapunov_eig_solver(F):
    """
    Solving the Lyapunov Equation of BX+XB=C by eigen decomposition
    input (...,n,n) SPD B and symmetric C
    """

    @staticmethod
    def forward(ctx, B, C):
        X, U, L = Ly_forward(B, C)
        ctx.save_for_backward(X, U, L)
        return X

    @staticmethod
    def backward(ctx, dx):
        X, U, L, = ctx.saved_variables
        return Ly_backward(X, U, L, dx)


def Ly_forward(B, C):
    U, S, _ = th.svd(B)
    L = 1. / (S[..., :, None] + S[..., None, :])
    L[L == -np.inf] = 0;
    L[L == np.inf] = 0;
    L[th.isnan(L)] = 0
    X = first_dirivative(U, L, C)
    return X, U, L


def Ly_backward(X, U, L, dx):
    ''''
    dx should be symmetrized
    '''
    sym_dx = sym(dx)
    dc = first_dirivative(U, L, sym_dx)
    tmp = -X.matmul(dc)
    db = tmp + tmp.transpose(-1, -2)
    return db, dc


def first_dirivative(U, L, V):
    ''''
    (...,N,N) U, L ,V
    '''
    V_tmp = L * (U.transpose(-1, -2).matmul(V).matmul(U))
    V_New = U.matmul(V_tmp).matmul(U.transpose(-1, -2))
    return V_New


def sym(A):
    ''''
    Make a square matrix symmetrized, (A+A')/2
    '''
    return (A + A.transpose(-1, -2)) / 2.








class Lyapunov_eig_solver(F):
    """
    Solving the Lyapunov Equation of BX+XB=C by eigen decomposition
    input (...,n,n) SPD B and symmetric C
    """

    @staticmethod
    def forward(ctx, B, C):
        X, U, L = Ly_forward(B, C)
        ctx.save_for_backward(X, U, L)
        return X

    @staticmethod
    def backward(ctx, dx):
        X, U, L, = ctx.saved_variables
        return Ly_backward(X, U, L, dx)


def Ly_forward(B, C):
    U, S, _ = th.svd(B)
    L = 1. / (S[..., :, None] + S[..., None, :])
    L[L == -np.inf] = 0;
    L[L == np.inf] = 0;
    L[th.isnan(L)] = 0
    X = first_dirivative(U, L, C)
    return X, U, L


def Ly_backward(X, U, L, dx):
    ''''
    dx should be symmetrized
    '''
    sym_dx = sym(dx)
    dc = first_dirivative(U, L, sym_dx)
    tmp = -X.matmul(dc)
    db = tmp + tmp.transpose(-1, -2)
    return db, dc


def sym(A):
    ''''
    Make a square matrix symmetrized, (A+A')/2
    '''
    return (A + A.transpose(-1, -2)) / 2.


def first_dirivative(U, L, V):
    ''''
    (...,N,N) U, L ,V
    '''
    V_tmp = L * (U.transpose(-1, -2).matmul(V).matmul(U))
    V_New = U.matmul(V_tmp).matmul(U.transpose(-1, -2))
    return V_New


def pal1(B, x):
    N, h, n, n = x.shape
    p2, b, _ = th.svd(B)
    mid_2 = b[..., :, None] + b[..., None, :]
    mid_3 = th.pow(2 / mid_2, 0.5)
    mid_4 = mid_3 * (p2.transpose(-1, -2).matmul(x).matmul(p2))
    mid_5 = p2.matmul(mid_4).matmul(p2.transpose(-1, -2))
    mid_6 = mid_5 + th.eye(n, dtype=dtype) + (1 / 4) * (mid_5.matmul(mid_5))
    return mid_6


def pal2(A, x):
    p1, a, _ = th.svd(A)
    mid_1 = a[..., :, None] + a[..., None, :]
    mid_2 = th.pow(mid_1 / 2, 0.5)[None, None, :, :]
    mid_4 = mid_2 * (p1.transpose(-1, -2).matmul(x).matmul(p1))
    mid_5 = p1.matmul(mid_4).matmul(p1.transpose(-1, -2))
    return mid_5


def BaryGeom(x):
    with th.no_grad():
        G = th.mean(x, dim=0)[0, :, :]
        for i in range(1):
            h_1 = SqmEig.apply(CongrG(x, G, 'pos'))
            h_2 = th.sum(h_1, dim=0, keepdim=True) / x.shape[0]
            # G1 = G1[0, 0]
            h_3 = PowerEig.apply(h_2, 2)
            G1 = CongrG(h_3, G, 'neg')[0, 0]
            # print(th.norm(G1-G))
            G = G1
        return G1


def cal_var(X, po):
    with th.no_grad():
        N, h, n, n = X.shape
        dists = dist_riemann(X, th.eye(n, dtype=dtype))
        var = dists.mean() * (1 / (po ** 2))
    return var


def scale1(X, var, eps, s):
    N, h, n, n = X.shape
    factor = s / (var + eps).sqrt()
    X_1 = LogG1(X)
    X_2 = factor * X_1
    X_3 = X_2 + th.eye(n, dtype=dtype) + (1 / 4) * (X_2.matmul(X_2))
    return X_3


def LogG(x, X):
    X_1 = SqmEig.apply(X[None, None, :, :])
    X_2 = SqminvEig.apply(X[None, None, :, :])
    C_1 = X_1.matmul(SqmEig.apply(X_1.matmul(x).matmul(X_1))).matmul(X_2)
    C = C_1 + C_1.transpose(-1, -2) - 2 * X
    return C


def LogG1(x):
    N, h, n, n = x.shape
    X = 2 * (SqmEig.apply(x)) - 2 * (th.eye(n, dtype=th.double, device=x.device))
    return X


def ExpG(x, X):
    Ly = Lyapunov_eig_solver.apply(X, x)
    M = X + x + Ly.matmul(X[None, None, :, :]).matmul(Ly)
    return M


def dist_riemann(x, y):
    d_2 = tra(x) + tra(y) - 2 * tra(SqmEig.apply(CongrG(x, y, 'pos')))
    return d_2


def tra(x):
    return x.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)


def geodesic(A, B, t):
    A_1 = SqmEig.apply(A[None, None, :, :])
    A_2 = SqminvEig.apply(A[None, None, :, :])
    C_1 = A_1.matmul(SqmEig.apply(A_1.matmul(B).matmul(A_1))).matmul(A_2)
    M = ((pow(1 - t, 2)) * A + pow(t, 2) * B + t * (1 - t) * (C_1 + C_1.transpose(-1, -2)))[0, 0]
    return M


def CongrG1(P, G, mode):
    """
    Input P: (batch_size,channels) SPD matrices of size (n,n) or single matrix (n,n)
    Input G: matrix (n,n) to do the congruence by
    Output PP: (batch_size,channels) of congruence by sqm(G) or sqminv(G) or single matrix (n,n)
    """
    if (mode == 'pos'):
        GG = SqmEig.apply(G[None, None, :, :])
    elif (mode == 'neg'):
        GG = SqminvEig.apply(G[None, None, :, :])
    PP = GG.matmul(P[None, None, :, :]).matmul(GG)
    return PP[0, 0]

