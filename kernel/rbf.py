import torch

# Global settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)


def _median_bandwidth_from_Rsq(R_sq, alpha=1.0):
    """
    Compute sigma = alpha * median(||xi - xj||), using distances (not squared).
    R_sq: (N, N) pairwise squared distances
    """
    N = R_sq.shape[0]
    dists = torch.sqrt(R_sq + 1e-18)  # avoid sqrt(0)
    offdiag = dists[~torch.eye(N, dtype=bool, device=dists.device)]
    h = torch.median(offdiag)
    return h * alpha


# ------------------------------------------------------------
# 1) Basic kernel
# ------------------------------------------------------------
def kernel(X, sigma=None, use_median=False, alpha=1.0):
    """
    X: tensor of shape (d, N)
    sigma: float or None
    use_median: if True, compute sigma via median heuristic
    alpha: scaling factor for heuristic bandwidth

    Returns:
        K: (N, N)
        grad2: (d, N, N)
        sigma: scalar tensor used
    """
    X = X.to(device, dtype=torch.float64)
    d, N = X.shape

    R = X[:, :, None] - X[:, None, :]      # (d, N, N)
    R_sq = torch.sum(R ** 2, dim=0)        # (N, N)

    # --- Median heuristic ---
    if use_median:
        sigma = _median_bandwidth_from_Rsq(R_sq, alpha=alpha)
    else:
        sigma = torch.as_tensor(sigma, dtype=torch.float64, device=device)

    band_sq = sigma ** 2

    logK = -R_sq / (2 * band_sq)
    K = torch.exp(logK)
    grad2 = (K / band_sq)[None, :, :] * R

    return K, -grad2


# ------------------------------------------------------------
# 2) Kernel for multiple experiment batches
# ------------------------------------------------------------
def kernel_f(
    X,
    sigma=None,
    use_median=False,
    alpha=1.0
):
    """
    X: (N_exp, d, N)
    sigma: scalar or None
    use_median: use median heuristic per experiment
    alpha: scaling factor on heuristic sigma

    Returns:
        K: (N_exp, N, N)
        grad2: (N_exp, d, N, N)
        sigma: (N_exp,) tensor of bandwidths used
    """
    X = X.to(device, dtype=torch.float64)
    N_exp, d, N = X.shape

    # Pairwise differences
    R = X[:, :, :, None] - X[:, :, None, :]    # (N_exp, d, N, N)
    R_sq = torch.sum(R ** 2, dim=1)            # (N_exp, N, N)

    # ---- Median heuristic (per experiment) ----
    if use_median:
        # pairwise distances
        dists = torch.sqrt(R_sq + 1e-18)

        # extract off-diagonal entries efficiently
        mask = ~torch.eye(N, dtype=bool, device=dists.device)
        offdiag = dists[:, mask].view(N_exp, -1)   # (N_exp, N*(N-1))

        # per-experiment medians
        h = torch.median(offdiag, dim=1).values     # (N_exp,)
        sigma = h * alpha                            # scaled
    else:
        # broadcast user-supplied sigma to (N_exp,)
        sigma = torch.as_tensor(sigma, dtype=torch.float64, device=device).expand(N_exp)

    # bandwidth squared
    band_sq = sigma[:, None, None] ** 2             # (N_exp, 1, 1)

    # compute kernels
    logK = -R_sq / (2 * band_sq)
    K = torch.exp(logK)                             # (N_exp, N, N)

    # gradients
    grad2 = (K / band_sq)[:, None, :, :] * R        # (N_exp, d, N, N)

    return K, -grad2



# ------------------------------------------------------------
# 3) Kernel with Hessian
# ------------------------------------------------------------
def kernel_wkmed(X, sigma=None, d=None, use_median=False, alpha=1.0):
    """
    X: (d, N)
    sigma: float or None
    d: dimension
    Returns:
        K: (N, N)
        grad2: (d, N, N)
        Hess: (d, d, N, N)
        sigma: scalar tensor used
    """
    X = X.to(device, dtype=torch.float64)
    Id = torch.eye(d, dtype=torch.float64, device=device)

    R = X[:, :, None] - X[:, None, :]    # (d, N, N)
    R_sq = torch.sum(R ** 2, dim=0)

    # --- Median heuristic ---
    if use_median:
        sigma = _median_bandwidth_from_Rsq(R_sq, alpha=alpha)
    else:
        sigma = torch.as_tensor(sigma, dtype=torch.float64, device=device)

    band_sq = sigma ** 2

    logK = -R_sq / (2 * band_sq)
    K = torch.exp(logK)

    grad2 = (K / band_sq)[None, :, :] * R

    R_out = R[:, None, :, :] * R[None, :, :, :]  # (d, d, N, N)
    Hess = (R_out / (band_sq ** 2) - (Id / band_sq)[:, :, None, None]) * K[None, None, :, :]

    return K, -grad2, Hess
