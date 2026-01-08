import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)


def velocity(w, W, h, h_bar, k, grad1k, Hessk, C, d, N, reg=1e-3):
    """
    Parameters
    ----------
    w : (N,)
        Particle weights.
    W : (N,)
        Normalized weight vector.
    h : (N,)
        Score evaluated at particles.
    h_bar : (N,)
        Reference / mean score.
    k : (N, N)
        Kernel matrix.
    grad1k : (d, N, N)
        Gradient of kernel with respect to first argument.
    Hessk : (d, d, N, N)
        Hessian of kernel with respect to first argument.
    C : (d, d)
        Preconditioning / covariance matrix.
    d : int
        Dimension of state space.
    N : int
        Number of particles.
    reg : float, optional
        Regularization parameter.

    Returns
    -------
    v : (d, N)
        Velocity field at particles.
    dlogw_dt : (N,)
        Time derivative of log-weights.
    """
    w = w.to(device, dtype=torch.float64)
    W = W.to(device, dtype=torch.float64)
    h = h.to(device, dtype=torch.float64)
    h_bar = h_bar.to(device, dtype=torch.float64)
    k = k.to(device, dtype=torch.float64)
    grad1k = grad1k.to(device, dtype=torch.float64)
    Hessk = Hessk.to(device, dtype=torch.float64)
    C = C.to(device, dtype=torch.float64)
    reg = torch.as_tensor(reg, dtype=torch.float64, device=device)

    G = torch.einsum('mli,mk,klj->ij', grad1k, C, grad1k) / N
    G_reg = G + reg * torch.eye(N, dtype=torch.float64, device=device)

    H = (k @ (h - h_bar)) / N
    alpha = N * torch.linalg.lstsq(G_reg, H).solution

    v = -torch.einsum('il,ljk,k->ij', C, grad1k, alpha) / N

    div_v = -torch.einsum('kl,klij,j->i', C, Hessk, alpha) / N
    L_k = torch.einsum('iimn->mn', Hessk)

    psi = L_k @ W

    G_w = torch.einsum('mli,mk,klj,l->ij', grad1k, C, grad1k, W)
    G_w_reg = G_w + reg * torch.eye(N, dtype=torch.float64, device=device)

    alpha_w = N * torch.linalg.lstsq(G_w_reg, psi).solution

    P = -torch.einsum('il,ljk,k->ij', C, grad1k, alpha_w) / N

    dlogw_dt = torch.sum(P * v, dim=0) + div_v - h

    return v, dlogw_dt
