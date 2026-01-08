import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

delta_t = 0.001
Delta_t = 0.05
N_ob = 50
N_assimilation = 400
sigma_ob = np.sqrt(1.0)
sigma_f = np.sqrt(0.01)

inflation = torch.linspace(1.0, 1.4, 5, device = device, dtype=torch.float64)
N = 8
N_exp = 100
ob_data = np.load("Data/Lorentz96/ob.npz")
ob, ob_n = ob_data["ob"], ob_data["ob_n"]
ob = torch.tensor(ob, device = device, dtype=torch.float64)
ob_n = torch.tensor(ob_n, device = device, dtype=torch.float64)

pr = np.load("Data/Lorentz96/pr.npy")
pr = torch.tensor(pr, device = device, dtype=torch.float64)

Ns_kmed = 100
alpha = 4.0
Ns_svgd = 100