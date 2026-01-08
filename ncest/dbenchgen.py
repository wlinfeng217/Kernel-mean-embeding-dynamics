import torch
from prior.sobolgau import *
import numpy as np


if __name__ == '__main__':
    d = torch.arange(1,56,1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)
    N = 100
    X = []
    for i in range(d.shape[0]):
        mu_0 = torch.ones(d[i], device=device, dtype=torch.float64)
        Sigma_0 = torch.eye(d[i], device=device, dtype=torch.float64)
        X.append(np.array(sobol_multivariate_normal(N, mu_0, Sigma_0).cpu()))

    np.savez("Data/NC/Benchd/pr.npz", *X)