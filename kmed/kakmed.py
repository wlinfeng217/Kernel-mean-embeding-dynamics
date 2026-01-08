import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)


def velocity(v_0, h, k, grad1k, C, d, N, reg=1e-3):
    """
    Parameters
    ----------
    v_0 : (d, N)
        Baseline velocity field.
    h : (N,)
        Score evaluated at particles.
    k : (N, N)
        Kernel matrix.
    grad1k : (d, N, N)
        Gradient of kernel with respect to the first argument.
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
    """
    v_0 = v_0.to(device, dtype=torch.float64)
    h = h.to(device, dtype=torch.float64)
    k = k.to(device, dtype=torch.float64)
    grad1k = grad1k.to(device, dtype=torch.float64)
    C = C.to(device, dtype=torch.float64)
    reg = torch.as_tensor(reg, dtype=torch.float64, device=device)

    G = torch.einsum('mli,mk,klj->ij', grad1k, C, grad1k) / N
    G_reg = G + reg * torch.eye(N, dtype=torch.float64, device=device)

    H = (k @ h) / N
    H_1 = torch.einsum('ijk,ij->k', grad1k, v_0) / N

    alpha = N * torch.linalg.solve(G_reg, H + H_1)
    v = -(torch.einsum('il,ljk,k->ij', C, grad1k, alpha) / N) + v_0

    return v


def velocity_f(v_0, h, k, grad1k, C, d, N, reg=1e-3):
    """
    Parameters
    ----------
    v_0 : (N_exp, d, N)
        Baseline velocity field for each experiment.
    h : (N_exp, N)
        Score evaluated at particles.
    k : (N_exp, N, N)
        Kernel matrices.
    grad1k : (N_exp, d, N, N)
        Gradient of kernel with respect to the first argument.
    C : (N_exp, d, d)
        Preconditioning / covariance matrices.
    d : int
        Dimension of state space.
    N : int
        Number of particles.
    reg : float, optional
        Regularization parameter.

    Returns
    -------
    v : (N_exp, d, N)
        Updated velocity fields.
    """
    v_0 = v_0.to(device, dtype=torch.float64)
    h = h.to(device, dtype=torch.float64)
    k = k.to(device, dtype=torch.float64)
    grad1k = grad1k.to(device, dtype=torch.float64)
    C = C.to(device, dtype=torch.float64)
    reg = torch.as_tensor(reg, dtype=torch.float64, device=device)

    G = torch.einsum('amli,amk,aklj->aij', grad1k, C, grad1k) / N
    G_reg = G + (reg * torch.eye(N, dtype=torch.float64, device=device))[None, :, :]

    H = torch.einsum('aij,aj->ai', k, h) / N
    H_1 = torch.einsum('aijk,aij->ak', grad1k, v_0) / N

    H_expanded = (H + H_1).unsqueeze(-1)
    alpha = N * torch.linalg.lstsq(G_reg, H_expanded).solution.squeeze(-1)

    v = -(torch.einsum('ail,aljk,ak->aij', C, grad1k, alpha) / N) + v_0

    return v
