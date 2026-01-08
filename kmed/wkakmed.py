import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)


def velocity(v_0, div_v_0, w, W, h, h_bar, k, grad1k, Hessk, C, d, N, reg=1e-3):
    """
    Parameters
    ----------
    v_0 : (d, N)
        Baseline velocity field.
    div_v_0 : (N,)
        Baseline divergence term.
    w : (N,)
        Particle weights.
    W : (N,)
        Auxiliary weight vector.
    h : (N,)
        Score evaluated at particles.
    h_bar : (N,)
        Reference / mean score.
    k : (N, N)
        Kernel matrix.
    grad1k : (d, N, N)
        Gradient of kernel with respect to the first argument.
    Hessk : (d, d, N, N)
        Hessian of kernel with respect to the first argument.
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
        Updated velocity field.
    dlogw_dt : (N,)
        Time derivative of log-weights.
    """
    v_0 = v_0.to(device, dtype=torch.float64)
    div_v_0 = div_v_0.to(device, dtype=torch.float64)
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
    H_1 = torch.einsum('ijk,ij,j->k', grad1k, v_0, W)

    alpha = N * torch.linalg.solve(G_reg, H + H_1)

    v = -(torch.einsum('il,ljk,k->ij', C, grad1k, alpha) / N) + v_0

    div_v = -(torch.einsum('kl,klij,j->i', C, Hessk, alpha) / N) + div_v_0

    L_k = torch.einsum('iimn->mn', Hessk)
    psi = L_k @ W

    G_w = torch.einsum('mli,mk,klj,l->ij', grad1k, C, grad1k, W)
    G_w_reg = G_w + reg * torch.eye(N, dtype=torch.float64, device=device)

    alpha_w = N * torch.linalg.lstsq(G_w_reg, psi).solution

    P = -torch.einsum('il,ljk,k->ij', C, grad1k, alpha_w) / N

    dlogw_dt = torch.sum(P * v, dim=0) + div_v - h

    return v, dlogw_dt
