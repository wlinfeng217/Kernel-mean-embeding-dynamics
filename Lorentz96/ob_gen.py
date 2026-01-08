import torch
from Lorentz96.experiment import *
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

if __name__ == '__main__':
    print("Job is running on",  torch.cuda.get_device_name(0))
    X_0 = torch.randn(d, device = device, dtype=torch.float64)
    delta_t = 0.001
    Delta_t = 0.05
    N_ob = 50
    N_assimilation = 400
    sigma_ob = np.sqrt(1.0)
    sigma_f = np.sqrt(0.01)
    ob, ob_n = ob_generate(X_0, delta_t, N_ob, N_assimilation, sigma_ob)
    mu_0 = X_0.clone()
    C_0 = 0.01 * torch.eye(d, device = device, dtype=torch.float64)
    N_true = 20000

    X_sobol_true = sobol_multivariate_normal(N_true, mu_0, C_0)

    X_path_true = bootstrap(X_sobol_true, N_true, delta_t, N_ob, N_assimilation, ob_n, sigma_ob, sigma_f)

    N = 8
    X_so = np.array(sobol_multivariate_normal(N, mu_0, C_0).cpu())

    ob_save = np.array(ob.cpu())
    ob_n_save = np.array(ob_n.cpu())
    X_true = np.array(X_path_true.cpu())
    np.savez("Data/Lorentz96/ob.npz", ob = ob_save, ob_n = ob_n_save)
    np.save("Data/Lorentz96/pf.npy", X_true)
    np.save("Data/Lorentz96/pr.npy", X_so)

