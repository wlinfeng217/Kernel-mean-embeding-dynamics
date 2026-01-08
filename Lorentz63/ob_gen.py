import torch
from Lorentz63.experiment import *
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
    Delta_t = 0.05
    N_ob = 5
    N_assimilation = 400
    sigma_ob = np.sqrt(1/15)
    sigma_f = np.sqrt(0.0838)
    ob, ob_n = ob_generate(X_0, delta_t, N_ob, N_assimilation, sigma_ob)
    mu_0 = X_0.clone()
    C_0 = 0.01 * torch.eye(d, device = device, dtype=torch.float64)
    N_true = 20000

    X_sobol_true = sobol_multivariate_normal(N_true, mu_0, C_0)

    X_path_true = bootstrap(X_sobol_true, N_true, delta_t, N_ob, N_assimilation, ob_n, sigma_ob, sigma_f)

    N = fibonacci(8, 4, 6)
    X_so = []
    for i in range(N.shape[0]):
        X_so.append(np.array(sobol_multivariate_normal(N[i], mu_0, C_0).cpu()))


    ob_save = np.array(ob.cpu())
    ob_n_save = np.array(ob_n.cpu())
    X_true = np.array(X_path_true.cpu())
    np.savez("Data/Lorentz63/ob.npz", ob = ob_save, ob_n = ob_n_save)
    np.save("Data/Lorentz63/pf.npy", X_true)
    np.savez("Data/Lorentz63/pr.npz", *X_so)

