import torch
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

delta_t = 0.01
Delta_t = 0.1
N_ob = 10
N_assimilation = 100
sigma_ob = np.sqrt(0.7)
sigma_f = np.sqrt(0.14)

N = fibonacci(8, 4, 6)
N_exp = 100
ob_data = np.load("Data/Lorentz63oldle/ob.npz")
ob, ob_n = ob_data["ob"], ob_data["ob_n"]
ob = torch.tensor(ob, device = device, dtype=torch.float64)
ob_n = torch.tensor(ob_n, device = device, dtype=torch.float64)
prior_data = np.load("Data/Lorentz63oldle/pr.npz")
pr = []
for i in range(N.shape[0]):
    pr.append(torch.tensor(prior_data[("arr_"+str(i))], device = device, dtype=torch.float64))
Ns_kmed = 50
band = 5.0
Ns_svgd = 200