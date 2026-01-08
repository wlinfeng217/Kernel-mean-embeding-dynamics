import torch
from Lorentz63oldme.experiment import *
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

def fibonacci(n, x0, x1):
    if n <= 0:
        return torch.tensor([], device=device)
    if n == 1:
        return torch.tensor([0], device=device)

    # Start with [0, 1]
    fib = torch.zeros(n, device=device)
    fib[0] = x0
    fib[1] = x1

    for i in range(2, n):
        fib[i] = fib[i-1] + fib[i-2]

    return fib.long()

if __name__ == '__main__':
    print("Job is running on",  torch.cuda.get_device_name(0))
    X_0 = torch.tensor([-0.587, -0.563, 16.870], device = device, dtype=torch.float64)
    delta_t = 0.01
    Delta_t = 0.1
    N_ob = 10
    N_assimilation = 400
    sigma_ob = np.sqrt(0.7)
    sigma_f = np.sqrt(0.14 / 10)
    ob, ob_n = ob_generate(X_0, delta_t, N_ob, N_assimilation, sigma_ob)
    mu_0 = X_0.clone()
    C_0 = 0.01 * torch.eye(d, device = device, dtype=torch.float64)

    N = fibonacci(8, 4, 6)
    X_so = []
    for i in range(N.shape[0]):
        X_so.append(np.array(sobol_multivariate_normal(N[i], mu_0, C_0).cpu()))


    ob_save = np.array(ob.cpu())
    ob_n_save = np.array(ob_n.cpu())
    np.savez("Data/Lorentz63oldme/ob.npz", ob = ob_save, ob_n = ob_n_save)
    np.savez("Data/Lorentz63oldme/pr.npz", *X_so)

