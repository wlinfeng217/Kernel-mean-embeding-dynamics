import torch
from kernel.rbf import *
from kmed import plainkmed as pk
from kmed import kakmed as ka
from kmed import wkakmed as wka
from kmed import wkmed as wk
from prior.sobolgau import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

d = 3


def nll(X, y, R_inv):
    """
    Parameters
    ----------
    X : (d, N)
    y : (d,)
    R_inv : (d, d)

    Returns
    -------
    nll : (N,)
    """
    D = X - y[:, None]
    return 0.5 * torch.einsum("im,ij,jm->m", D, R_inv, D)


def nll_f(X, y, R_inv):
    """
    Parameters
    ----------
    X : (N_exp, d, N)
    y : (d,)
    R_inv : (d, d)

    Returns
    -------
    nll : (N_exp, N)
    """
    D = X - y[None, :, None]
    return 0.5 * torch.einsum("aim,ij,ajm->am", D, R_inv, D)


def dXdt(X, N):
    """
    Parameters
    ----------
    X : (d, N)
    N : int

    Returns
    -------
    dX : (d, N)
    """
    R = torch.zeros((d, N), device=X.device, dtype=torch.float64)
    R[0, :] = 10.0 * (X[1, :] - X[0, :])
    R[1, :] = X[0, :] * (28.0 - X[2, :]) - X[1, :]
    R[2, :] = X[0, :] * X[1, :] - (8.0 / 3.0) * X[2, :]
    return R


def dXdt_f(X, N_exp, N):
    """
    Parameters
    ----------
    X : (N_exp, d, N)
    N_exp : int
    N : int

    Returns
    -------
    dX : (N_exp, d, N)
    """
    R = torch.zeros((N_exp, d, N), device=X.device, dtype=torch.float64)
    R[:, 0, :] = 10.0 * (X[:, 1, :] - X[:, 0, :])
    R[:, 1, :] = X[:, 0, :] * (28.0 - X[:, 2, :]) - X[:, 1, :]
    R[:, 2, :] = X[:, 0, :] * X[:, 1, :] - (8.0 / 3.0) * X[:, 2, :]
    return R


def RK_Lorentz63(X, N, delta_t):
    """
    Parameters
    ----------
    X : (d, N)
    N : int
    delta_t : float

    Returns
    -------
    X_new : (d, N)
    """
    X_new = torch.zeros((d, N), device=X.device, dtype=torch.float64)
    k1 = dXdt(X, N)
    k2 = dXdt(X + 0.5 * delta_t * k1, N)
    k3 = dXdt(X + 0.5 * delta_t * k2, N)
    k4 = dXdt(X + delta_t * k3, N)
    X_new[:, :] = X + delta_t * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return X_new


def RK_Lorentz63_f(X, N_exp, N, delta_t):
    """
    Parameters
    ----------
    X : (N_exp, d, N)
    N_exp : int
    N : int
    delta_t : float

    Returns
    -------
    X_new : (N_exp, d, N)
    """
    X_new = torch.zeros((N_exp, d, N), device=X.device, dtype=torch.float64)
    k1 = dXdt_f(X, N_exp, N)
    k2 = dXdt_f(X + 0.5 * delta_t * k1, N_exp, N)
    k3 = dXdt_f(X + 0.5 * delta_t * k2, N_exp, N)
    k4 = dXdt_f(X + delta_t * k3, N_exp, N)
    X_new[:, :, :] = X + delta_t * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return X_new


def ob_generate(X_0, delta_t, N_ob, N_assimilation, sigma, device=None):
    """
    Parameters
    ----------
    X_0 : (d,)
    delta_t : float
    N_ob : int
    N_assimilation : int
    sigma : float
    device : torch.device or None

    Returns
    -------
    ob_true : (N_assimilation, d)
    ob_noisy : (N_assimilation, d)
    """
    if device is None:
        device = X_0.device
    X_0 = X_0.to(device=device, dtype=torch.float64)

    ob = torch.zeros((N_assimilation + 1, d), device=device, dtype=torch.float64)
    ob[0, :] = X_0

    for n in range(N_assimilation):
        X = ob[n, :][:, None]
        for _ in range(N_ob):
            X = RK_Lorentz63(X, 1, delta_t)
        ob[n + 1, :] = X[:, 0]

    noise = sigma * torch.randn((N_assimilation, d), device=device, dtype=torch.float64)
    return ob[1:, :], ob[1:, :] + noise


def forecast(X, N, N_ob, delta_t, sigma_f):
    """
    Parameters
    ----------
    X : (d, N)
    N : int
    N_ob : int
    delta_t : float
    sigma_f : float

    Returns
    -------
    X_f : (d, N)
    """
    X_return = X.clone()
    for _ in range(N_ob):
        X_return = RK_Lorentz63(X_return, N, delta_t)
    return X_return + sigma_f * torch.randn_like(X_return, dtype=torch.float64)


def forecast_f(X, N_exp, N, N_ob, delta_t, sigma_f):
    """
    Parameters
    ----------
    X : (N_exp, d, N)
    N_exp : int
    N : int
    N_ob : int
    delta_t : float
    sigma_f : float

    Returns
    -------
    X_f : (N_exp, d, N)
    """
    X_return = X.clone()
    for _ in range(N_ob):
        X_return = RK_Lorentz63_f(X_return, N_exp, N, delta_t)
    return X_return + sigma_f * torch.randn_like(X_return, dtype=torch.float64)


def forecast_f_mpf(X, N_exp, N, N_ob, delta_t, sigma_f):
    """
    Parameters
    ----------
    X : (N_exp, d, N)
    N_exp : int
    N : int
    N_ob : int
    delta_t : float
    sigma_f : float

    Returns
    -------
    X_f_un : (N_exp, d, N)
    X_f : (N_exp, d, N)
    """
    X_return = X.clone()
    for _ in range(N_ob):
        X_return = RK_Lorentz63_f(X_return, N_exp, N, delta_t)
    return X_return, X_return + sigma_f * torch.randn_like(X_return, dtype=torch.float64)


def enkf(N_exp, X_sobol, N, delta_t, N_ob, N_assimilation,
         ob, sigma_ob, sigma_f, device=None):
    """
    Parameters
    ----------
    N_exp : int
    X_sobol : (d, N)
    N : int
    delta_t : float
    N_ob : int
    N_assimilation : int
    ob : (N_assimilation, d)
    sigma_ob : float
    sigma_f : float
    device : torch.device or None

    Returns
    -------
    X_path : (N_exp, N_assimilation + 1, d, N)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_sobol = X_sobol.to(device, dtype=torch.float64)
    ob = ob.to(device, dtype=torch.float64)

    R = (sigma_ob ** 2) * torch.eye(d, device=device, dtype=torch.float64)

    X_path = torch.zeros((N_exp, N_assimilation + 1, d, N), device=device, dtype=torch.float64)
    X_path[:, 0, :, :] = X_sobol.unsqueeze(0).expand(N_exp, -1, -1)

    for n_assi in range(N_assimilation):

        X_f = forecast_f(X_path[:, n_assi, :, :], N_exp, N, N_ob, delta_t, sigma_f)

        mu_f = X_f.mean(dim=2, keepdim=True)
        X_b = X_f - mu_f

        P_f = torch.einsum("aim,ajm->aij", X_b, X_b) / (N - 1)

        S = P_f + R[None, :, :]
        S_inv = torch.linalg.inv(S)
        K = torch.einsum("aij,ajk->aik", P_f, S_inv)

        y = ob[n_assi, :]
        eps = sigma_ob * torch.randn(N_exp, d, N, device=device, dtype=torch.float64)
        y_ens = y[None, :, None] + eps

        innovation = y_ens - X_f
        X_a = X_f + torch.einsum("aij,ajk->aik", K, innovation)

        X_path[:, n_assi + 1, :, :] = X_a

    return X_path


def plain_kmed(N_exp, X_sobol, N, delta_t, N_ob, N_assimilation,
               ob, sigma_ob, sigma_f, Ns_kmed, band, reg=1e-3, device=None):
    """
    Parameters
    ----------
    N_exp : int
    X_sobol : (d, N)
    N : int
    delta_t : float
    N_ob : int
    N_assimilation : int
    ob : (N_assimilation, d)
    sigma_ob : float
    sigma_f : float
    Ns_kmed : int
    band : float
    reg : float
    device : torch.device or None

    Returns
    -------
    X_path : (N_exp, N_assimilation + 1, d, N)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_sobol = X_sobol.to(device)
    ob = ob.to(device)

    dt = 1.0 / Ns_kmed
    R_inv = (1.0 / sigma_ob ** 2) * torch.eye(d, device=device, dtype=torch.float64)

    X_path = torch.zeros((N_exp, N_assimilation + 1, d, N), device=device, dtype=torch.float64)
    X_path[:, 0, :, :] = X_sobol.unsqueeze(0).expand(N_exp, -1, -1)

    for n_assi in range(N_assimilation):

        X = X_path[:, n_assi, :, :]
        y = ob[n_assi, :]

        X_f = forecast_f(X, N_exp, N, N_ob, delta_t, sigma_f)

        X_a = X_f.clone()
        for _ in range(Ns_kmed):

            mu_f = X_a.mean(dim=2, keepdim=True)
            X_b = X_a - mu_f
            P_f = torch.einsum("aim,ajm->aij", X_b, X_b) / (N - 1)

            h = nll_f(X_a, y, R_inv)
            h_bar = torch.mean(h, dim=1)
            k, grad1k = kernel_f(X_a, use_median=True, alpha=band)

            H = h - h_bar[:, None]
            v = pk.velocity_f(H, k, grad1k, P_f, d, N, reg=reg)

            X_a = X_a + dt * v

        X_path[:, n_assi + 1, :, :] = X_a

    return X_path


def kakmed(N_exp, X_sobol, N, delta_t, N_ob, N_assimilation,
           ob, sigma_ob, sigma_f, Ns_kmed, band, reg=1e-3, device=None):
    """
    Parameters
    ----------
    N_exp : int
    X_sobol : (d, N)
    N : int
    delta_t : float
    N_ob : int
    N_assimilation : int
    ob : (N_assimilation, d)
    sigma_ob : float
    sigma_f : float
    Ns_kmed : int
    band : float
    reg : float
    device : torch.device or None

    Returns
    -------
    X_path : (N_exp, N_assimilation + 1, d, N)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_sobol = X_sobol.to(device)
    ob = ob.to(device)

    dt = 1.0 / Ns_kmed
    R_inv = (1.0 / sigma_ob ** 2) * torch.eye(d, device=device, dtype=torch.float64)

    X_path = torch.zeros((N_exp, N_assimilation + 1, d, N), device=device, dtype=torch.float64)
    X_path[:, 0, :, :] = X_sobol.unsqueeze(0).expand(N_exp, -1, -1)

    for n_assi in range(N_assimilation):

        X = X_path[:, n_assi, :, :]
        y = ob[n_assi, :]

        X_f = forecast_f(X, N_exp, N, N_ob, delta_t, sigma_f)

        X_a = X_f.clone()
        for _ in range(Ns_kmed):

            mu_f = X_a.mean(dim=2, keepdim=True)
            X_b = X_a - mu_f
            P_f = torch.einsum("aim,ajm->aij", X_b, X_b) / (N - 1)

            h = nll_f(X_a, y, R_inv)
            h_bar = torch.mean(h, dim=1)
            k, grad1k = kernel_f(X_a, use_median=True, alpha=band)
            H = h - h_bar[:, None]

            v_0 = kb_velocity_filter(X_a, P_f, y, R_inv)
            v = ka.velocity_f(v_0, H, k, grad1k, P_f, d, N, reg=reg)

            X_a = X_a + dt * v

        X_path[:, n_assi + 1, :, :] = X_a

    return X_path


def score_mpf(X, X_f_un, y, sigma_ob_sq, sigma_f_sq):
    """
    Parameters
    ----------
    X : (N_exp, d, N)
    X_f_un : (N_exp, d, N)
    y : (d,)
    sigma_ob_sq : float
    sigma_f_sq : float

    Returns
    -------
    score : (N_exp, d, N)
    """
    D = y[None, :, None] - X
    lh_score = D / sigma_ob_sq

    Dx = X_f_un.unsqueeze(3) - X.unsqueeze(2)
    P = torch.exp(-0.5 * (Dx ** 2).sum(1) / sigma_f_sq)

    num = torch.einsum("aij,akij->akj", P, Dx) / sigma_f_sq
    den = P.sum(1, keepdim=True)
    pr_score = num / den

    return lh_score + pr_score


def mpf(N_exp, X_sobol, N, delta_t, N_ob, N_assimilation,
        ob, sigma_ob, sigma_f, Ns_svgd, band, lr=1e-3, eps=1e-8, device=None):
    """
    Parameters
    ----------
    N_exp : int
    X_sobol : (d, N)
    N : int
    delta_t : float
    N_ob : int
    N_assimilation : int
    ob : (N_assimilation, d)
    sigma_ob : float
    sigma_f : float
    Ns_svgd : int
    band : float
    lr : float
    eps : float
    device : torch.device or None

    Returns
    -------
    X_path : (N_exp, N_assimilation + 1, d, N)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_sobol = X_sobol.to(device)
    ob = ob.to(device)

    sigma_ob_sq = sigma_ob ** 2
    sigma_f_sq = sigma_f ** 2

    X_path = torch.zeros((N_exp, N_assimilation + 1, d, N), device=device, dtype=torch.float64)
    X_path[:, 0, :, :] = X_sobol.unsqueeze(0).expand(N_exp, -1, -1)

    for n_assi in range(N_assimilation):

        X = X_path[:, n_assi, :, :]
        y = ob[n_assi, :]

        X_f_un, X_f = forecast_f_mpf(X, N_exp, N, N_ob, delta_t, sigma_f)

        X_a = X_f.clone()
        G_accum = torch.zeros_like(X_a)

        for _ in range(Ns_svgd):

            score_a = score_mpf(X_a, X_f_un, y, sigma_ob_sq, sigma_f_sq)

            k, grad1k = kernel_f(X_a, use_median=True, alpha=band)

            grad = (torch.einsum("aij,anj->ani", k, score_a) +
                    grad1k.sum(dim=2)) / N

            G_accum += grad ** 2
            adjusted_lr = lr / torch.sqrt(G_accum + eps)

            X_a += adjusted_lr * grad

        X_path[:, n_assi + 1, :, :] = X_a

    return X_path


def bootstrap(X_sobol, N, delta_t, N_ob, N_assimilation,
              ob, sigma_ob, sigma_f, device=None):
    """
    Parameters
    ----------
    X_sobol : (d, N)
    N : int
    delta_t : float
    N_ob : int
    N_assimilation : int
    ob : (N_assimilation, d)
    sigma_ob : float
    sigma_f : float
    device : torch.device or None

    Returns
    -------
    X_path : (N_assimilation + 1, d, N)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_sobol = X_sobol.to(device)
    ob = ob.to(device)

    R_inv = (1.0 / sigma_ob ** 2) * torch.eye(d, device=device, dtype=torch.float64)

    X_path = torch.zeros((N_assimilation + 1, d, N), device=device, dtype=torch.float64)
    X_path[0, :, :] = X_sobol

    for n_assi in range(N_assimilation):

        X = X_path[n_assi, :, :]
        y = ob[n_assi, :]

        X_f = forecast(X, N, N_ob, delta_t, sigma_f)
        w = torch.exp(-nll(X_f, y, R_inv))
        W = w / torch.sum(w)

        indices = torch.multinomial(W, num_samples=N, replacement=True)
        X_path[n_assi + 1, :, :] = X_f[:, indices]

    return X_path
