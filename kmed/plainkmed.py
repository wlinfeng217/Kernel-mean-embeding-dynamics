import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)


def velocity(h, k, grad1k, C, d, N, reg=1e-3):
    """
    h: (N,)
    k: (N, N)
    grad1k: (d, N, N)
    C: (d, d)
    reg: regularization scalar
    """
    h = h.to(device, dtype=torch.float64)
    k = k.to(device, dtype=torch.float64)
    grad1k = grad1k.to(device, dtype=torch.float64)
    C = C.to(device, dtype=torch.float64)
    reg = torch.as_tensor(reg, dtype=torch.float64, device=device)

    G = torch.einsum('mli,mk,klj->ij', grad1k, C, grad1k) / N
    G_reg = G + reg * torch.eye(N, dtype=torch.float64, device=device)

    H = (k @ h) / N
    alpha = N * torch.linalg.lstsq(G_reg, H).solution

    v = -torch.einsum('il,ljk,k->ij', C, grad1k, alpha) / N

    return v


def velocity_f(h, k, grad1k, C, d, N, reg=1e-3):
    """
    h: (N_exp, N,)
    k: (N_exp, N, N)
    grad1k: (N_exp, d, N, N)
    C: (N_exp, d, d)
    reg: regularization scalar
    """
    h = h.to(device, dtype=torch.float64)
    k = k.to(device, dtype=torch.float64)
    grad1k = grad1k.to(device, dtype=torch.float64)
    C = C.to(device, dtype=torch.float64)
    reg = torch.as_tensor(reg, dtype=torch.float64, device=device)

    G = torch.einsum('amli,amk,aklj->aij', grad1k, C, grad1k) / N
    G_reg = G + (reg * torch.eye(N, dtype=torch.float64, device=device))[None, :, :]

    H = torch.einsum('aij,aj->ai', k, h) / N
    H_expanded = H.unsqueeze(-1)

    alpha = N * torch.linalg.lstsq(G_reg, H_expanded).solution.squeeze(-1)
    v = -torch.einsum('ail,aljk,ak->aij', C, grad1k, alpha) / N

    return v
