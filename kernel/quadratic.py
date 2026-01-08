import torch

# Global settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

# ------------------------------------------------------------
# 1) Basic kernel
# ------------------------------------------------------------

def quadratic_kernel(X):
    X = X.to(device, dtype=torch.float64)

    # Compute Gram matrix
    M = torch.einsum('di,dj->ij', X, X)   # (N,N)
    A = M + 1.0
    K = A**2

    # Correct gradient wrt X (d, N, N)
    grad1 = 2.0 * A[None, :, :] * X[:, None, :]

    return K, grad1
