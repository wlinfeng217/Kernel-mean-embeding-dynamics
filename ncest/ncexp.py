import torch
from kernel.rbf import *
from kernel.quadratic import *
from kmed import plainkmed as pk
from kmed import kakmed as ka
from kmed import wkakmed as wka
from kmed import wkmed as wk
from kmed import sir
from prior.sobolgau import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)


def nll(X):
    """
    Parameters
    ----------
    X : (d, N)

    Returns
    -------
    nll : (N,)
    """
    X = X.to(device, dtype=torch.float64)
    return 0.5 * torch.sum(X ** 2, dim=0)


def true_nc(d):
    """
    Parameters
    ----------
    d : int

    Returns
    -------
    nc : float
    """
    return (d / 2) * torch.log(torch.tensor(0.5, device=device, dtype=torch.float64)) - d / 4


def ncest_plain_kmed(d, N, alpha, N_step, X, reg = 1e-5):
    """
    Parameters
    ----------
    d : int
    N : int
    alpha : float
        Bandwidth scaling for the median heuristic.
    N_step : int
        Number of flow steps.
    X : (d, N)
        Initial particles.
    reg : float

    Returns
    -------
    nc_est : float
    """
    X_0 = X.to(device, dtype=torch.float64).clone()
    dt = 1.0 / N_step

    h_bar_record = torch.zeros(N_step + 1, device=device, dtype=torch.float64)
    h = nll(X_0)
    h_bar = torch.mean(h)
    h_bar_record[0] = h_bar

    for epoch in range(N_step):
        mu_x = X_0.mean(dim=1, keepdim=True)
        X_c = X_0 - mu_x
        P = torch.einsum("im,jm->ij", X_c, X_c) / (N - 1)
        k, grad1k = kernel(X_0, use_median=True, alpha = alpha)
        H = h - h_bar

        v = pk.velocity(H, k, grad1k, P, d, N, reg=reg)
        X_0 = X_0 + dt * v

        h = nll(X_0)
        h_bar = torch.mean(h)
        h_bar_record[epoch + 1] = h_bar

    return -torch.mean(h_bar_record[1:])


def ncest_plain_kmed_fixband(d, N, bandwidth, N_step, X, reg = 1e-5):
    """
    Parameters
    ----------
    d : int
    N : int
    bandwidth : float
        Fixed kernel bandwidth.
    N_step : int
        Number of flow steps.
    X : (d, N)
        Initial particles.
    reg : float

    Returns
    -------
    nc_est : float
    """
    X_0 = X.to(device, dtype=torch.float64).clone()
    dt = 1.0 / N_step

    h_bar_record = torch.zeros(N_step + 1, device=device, dtype=torch.float64)
    h = nll(X_0)
    h_bar = torch.mean(h)
    h_bar_record[0] = h_bar

    for epoch in range(N_step):
        mu_x = X_0.mean(dim=1, keepdim=True)
        X_c = X_0 - mu_x
        P = torch.einsum("im,jm->ij", X_c, X_c) / (N - 1)
        k, grad1k = kernel(X_0, sigma = bandwidth)
        H = h - h_bar

        v = pk.velocity(H, k, grad1k, P, d, N, reg=reg)
        X_0 = X_0 + dt * v

        h = nll(X_0)
        h_bar = torch.mean(h)
        h_bar_record[epoch + 1] = h_bar

    return -torch.mean(h_bar_record[1:])


def ncest_plain_kmed_quadratic(d, N, N_step, X, reg = 1e-5):
    """
    Parameters
    ----------
    d : int
    N : int
    N_step : int
        Number of flow steps.
    X : (d, N)
        Initial particles.
    reg : float

    Returns
    -------
    nc_est : float
    """
    X_0 = X.to(device, dtype=torch.float64).clone()
    dt = 1.0 / N_step

    h_bar_record = torch.zeros(N_step + 1, device=device, dtype=torch.float64)
    h = nll(X_0)
    h_bar = torch.mean(h)
    h_bar_record[0] = h_bar

    for epoch in range(N_step):
        mu_x = X_0.mean(dim=1, keepdim=True)
        X_c = X_0 - mu_x
        P = torch.einsum("im,jm->ij", X_c, X_c) / (N - 1)
        k, grad1k = quadratic_kernel(X_0)
        H = h - h_bar

        v = pk.velocity(H, k, grad1k, P, d, N, reg=reg)
        X_0 = X_0 + dt * v

        h = nll(X_0)
        h_bar = torch.mean(h)
        h_bar_record[epoch + 1] = h_bar

    return -torch.mean(h_bar_record[1:])


def ncest_kakmed(d, N, alpha, N_step, X, reg = 1e-5):
    """
    Parameters
    ----------
    d : int
    N : int
    alpha : float
        Bandwidth scaling for the median heuristic.
    N_step : int
        Number of flow steps.
    X : (d, N)
        Initial particles.
    reg : float

    Returns
    -------
    nc_est : float
    """
    X_0 = X.to(device, dtype=torch.float64).clone()
    dt = 1.0 / N_step

    h_bar_record = torch.zeros(N_step + 1, device=device, dtype=torch.float64)
    h = nll(X_0)
    h_bar = torch.mean(h)
    h_bar_record[0] = h_bar

    for epoch in range(N_step):
        mu_x = X_0.mean(dim=1, keepdim=True)
        X_c = X_0 - mu_x
        P = torch.einsum("im,jm->ij", X_c, X_c) / (N - 1)
        k, grad1k = kernel(X_0, use_median=True, alpha = alpha)
        H = h - h_bar

        v_0 = kb_velocity(X_0)
        v = ka.velocity(v_0, H, k, grad1k, P, d, N, reg=reg)

        X_0 = X_0 + dt * v
        h = nll(X_0)
        h_bar = torch.mean(h)
        h_bar_record[epoch + 1] = h_bar

    return -torch.mean(h_bar_record[1:])


def ncest_wkmed(d, N, alpha, N_step, X, reg = 1e-5):
    """
    Parameters
    ----------
    d : int
    N : int
    alpha : float
        Bandwidth scaling for the median heuristic.
    N_step : int
        Number of flow steps.
    X : (d, N)
        Initial particles.
    reg : float

    Returns
    -------
    nc_est : float
    """
    X_0 = X.to(device, dtype=torch.float64).clone()
    dt = 1.0 / N_step

    w_record = torch.zeros((N_step + 1, N), device=device, dtype=torch.float64)
    W_record = torch.zeros((N_step + 1, N), device=device, dtype=torch.float64)

    w_record[0, :] = torch.ones(N, device=device, dtype=torch.float64)
    W_record[0, :] = w_record[0] / torch.sum(w_record[0])

    for epoch in range(N_step):

        mu_x = X_0.mean(dim=1, keepdim=True)
        X_c = X_0 - mu_x
        P = torch.einsum("im,jm->ij", X_c, X_c) / (N - 1)
        k, grad1k, Hessk = kernel_wkmed(X_0, d=d, use_median=True, alpha = alpha)
        h = nll(X_0)
        h_bar = torch.mean(h)

        w = w_record[epoch]
        W = W_record[epoch]

        v, dlogw_dt = wk.velocity(w, W, h, h_bar, k, grad1k, Hessk, P, d, N, reg=reg)

        X_0 = X_0 + dt * v
        w_record[epoch + 1] = w * torch.exp(dt * dlogw_dt)
        W_record[epoch + 1] = w_record[epoch + 1] / torch.sum(w_record[epoch + 1])

    return torch.log(torch.mean(w_record[-1]))


def ncest_wkmed_Id(d, N, alpha, N_step, X, reg = 1e-5):
    """
    Parameters
    ----------
    d : int
    N : int
    alpha : float
        Bandwidth scaling for the median heuristic.
    N_step : int
        Number of flow steps.
    X : (d, N)
        Initial particles.
    reg : float

    Returns
    -------
    nc_est : float
    """
    C = torch.eye(d, device=device, dtype=torch.float64)
    X_0 = X.to(device, dtype=torch.float64).clone()
    dt = 1.0 / N_step

    w_record = torch.zeros((N_step + 1, N), device=device, dtype=torch.float64)
    W_record = torch.zeros((N_step + 1, N), device=device, dtype=torch.float64)

    w_record[0, :] = torch.ones(N, device=device, dtype=torch.float64)
    W_record[0, :] = w_record[0] / torch.sum(w_record[0])

    for epoch in range(N_step):

        k, grad1k, Hessk = kernel_wkmed(X_0, d=d, use_median=True, alpha = alpha)
        h = nll(X_0)
        h_bar = torch.mean(h)

        w = w_record[epoch]
        W = W_record[epoch]

        v, dlogw_dt = wk.velocity(w, W, h, h_bar, k, grad1k, Hessk, C, d, N, reg=reg)

        X_0 = X_0 + dt * v
        w_record[epoch + 1] = w * torch.exp(dt * dlogw_dt)
        W_record[epoch + 1] = w_record[epoch + 1] / torch.sum(w_record[epoch + 1])

    return torch.log(torch.mean(w_record[-1]))


def ncest_wkakmed(d, N, alpha, N_step, X, reg = 1e-5):
    """
    Parameters
    ----------
    d : int
    N : int
    alpha : float
        Bandwidth scaling for the median heuristic.
    N_step : int
        Number of flow steps.
    X : (d, N)
        Initial particles.
    reg : float

    Returns
    -------
    nc_est : float
    """
    X_0 = X.to(device, dtype=torch.float64).clone()
    dt = 1.0 / N_step

    w_record = torch.zeros((N_step + 1, N), device=device, dtype=torch.float64)
    W_record = torch.zeros((N_step + 1, N), device=device, dtype=torch.float64)

    w_record[0] = torch.ones(N, device=device, dtype=torch.float64)
    W_record[0] = w_record[0] / torch.sum(w_record[0])

    for epoch in range(N_step):
        mu_x = X_0.mean(dim=1, keepdim=True)
        X_c = X_0 - mu_x
        P = torch.einsum("im,jm->ij", X_c, X_c) / (N - 1)

        k, grad1k, Hessk = kernel_wkmed(X_0, d=d, use_median=True, alpha = alpha)
        h = nll(X_0)
        h_bar = torch.mean(h)

        w = w_record[epoch]
        W = W_record[epoch]

        v_0, div_v_0 = kb_velocity_weight(X_0)
        v, dlogw_dt = wka.velocity(v_0, div_v_0, w, W, h, h_bar, k, grad1k, Hessk, P, d, N, reg=reg)

        X_0 = X_0 + dt * v
        w_record[epoch + 1] = w * torch.exp(dt * dlogw_dt)
        W_record[epoch + 1] = w_record[epoch + 1] / torch.sum(w_record[epoch + 1])

    return torch.log(torch.mean(w_record[-1]))


def ncest_sir(d, N, N_step, X):
    """
    Parameters
    ----------
    d : int
    N : int
    N_step : int
    X : (d, N)

    Returns
    -------
    nc_est : float
    """
    C = torch.eye(d, device=device, dtype=torch.float64)
    X_0 = X.to(device, dtype=torch.float64).clone()
    dt = 1.0 / N_step

    h_bar_record = torch.zeros(N_step + 1, device=device, dtype=torch.float64)
    h = nll(X_0)
    h_bar = torch.mean(h)
    h_bar_record[0] = h_bar

    for epoch in range(N_step):
        w = torch.exp(-dt * h)
        X_0, h = sir.resampling(w, X_0, h, N)

        h_bar = torch.mean(h)
        h_bar_record[epoch + 1] = h_bar

    return -torch.mean(h_bar_record[1:])