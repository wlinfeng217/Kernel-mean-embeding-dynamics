import torch
import numpy as np

d = 3
N_step = 50
band = np.linspace(0.5,4.0,200)
N = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

pr_data = np.load("Data/NC/Benchband/pr.npy")
X = torch.tensor(pr_data, device = device, dtype=torch.float64)

