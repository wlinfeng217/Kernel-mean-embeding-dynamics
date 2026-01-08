import torch
from torch.quasirandom import SobolEngine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)


def sobol_multivariate_normal(N, mu, Sigma):
    """
    Parameters
    ----------
    N : int
        Number of samples.
    mu : (d,)
        Mean vector.
    Sigma : (d, d)
        Covariance matrix.

    Returns
    -------
    X : (d, N)
        Sobol-sampled multivariate normal samples.
    """
    mu = torch.as_tensor(mu, dtype=torch.float64, device=device)
    Sigma = torch.as_tensor(Sigma, dtype=torch.float64, device=device)
    d = mu.shape[0]

    sobol = SobolEngine(dimension=d, scramble=True)
    u = sobol.draw(N).to(device)

    eps = torch.finfo(torch.float64).eps
    u = u.clamp(eps, 1 - eps)

    z = torch.distributions.Normal(0.0, 1.0).icdf(u)

    try:
        L = torch.linalg.cholesky(Sigma)
    except RuntimeError:
        jitter = 1e-10
        L = torch.linalg.cholesky(Sigma + jitter * torch.eye(d, device=device))

    x = z @ L.T + mu

    return x.T


def kb_velocity(X):
    """
    Parameters
    ----------
    X : (d, N)
        Particle positions.

    Returns
    -------
    v : (d, N)
        Kalman-Bucy velocity field.
    """
    X = X.to(device, dtype=torch.float64)
    mu_X = torch.mean(X, dim=1, keepdim=True)
    C_X = torch.atleast_2d(torch.cov(X))

    return -0.5 * (C_X @ (X + mu_X))


def kb_velocity_weight(X):
    """
    Parameters
    ----------
    X : (d, N)
        Particle positions.

    Returns
    -------
    v : (d, N)
        Kalman-Bucy velocity field.
    w : float
        Scalar weight term.
    """
    X = X.to(device, dtype=torch.float64)
    mu_X = torch.mean(X, dim=1, keepdim=True)
    C_X = torch.atleast_2d(torch.cov(X))

    v = -0.5 * (C_X @ (X + mu_X))
    w = -0.5 * torch.trace(C_X)

    return v, w


def kb_velocity_filter(X, C_X, y, R_inv):
    """
    Parameters
    ----------
    X : (N_exp, d, N)
        Particle positions for each experiment.
    C_X : (N_exp, d, d)
        Covariance matrices.
    y : (d,)
        Observation vector.
    R_inv : (d, d)
        Inverse observation noise covariance.

    Returns
    -------
    v : (N_exp, d, N)
        Kalman-Bucy velocity field for filtering.
    """
    X = X.to(device, dtype=torch.float64)
    C_X = C_X.to(device, dtype=torch.float64)
    y = y.to(device, dtype=torch.float64)
    R_inv = R_inv.to(device, dtype=torch.float64)

    mu_X = torch.mean(X, dim=2, keepdim=True)
    term = X + mu_X - 2.0 * y[None, :, None]

    return -0.5 * torch.einsum('aij,jk,akn->ain', C_X, R_inv, term)
