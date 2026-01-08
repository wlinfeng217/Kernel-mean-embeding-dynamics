import torch
from prior.sobolgau import *
import numpy as np


if __name__ == '__main__':
    d = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)
    mu_0 = torch.ones(d, device=device, dtype=torch.float64)
    Sigma_0 = torch.eye(d, device=device, dtype=torch.float64)
    N = 200
    X = np.array(sobol_multivariate_normal(N, mu_0, Sigma_0).cpu())
    np.save("Data/NC/Benchband/pr.npy", X)