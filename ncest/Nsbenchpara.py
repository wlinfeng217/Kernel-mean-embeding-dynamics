import torch
import numpy as np

d = 3
N = 200
alpha = 4.0
Ns = torch.arange(5,151,1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

pr_data = np.load("Data/NC/BenchNs/pr.npy")
X = torch.tensor(pr_data, device = device, dtype=torch.float64)

