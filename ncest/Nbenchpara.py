import torch
import numpy as np

d = 3
N_step = 50
alpha = 4.0
N = torch.arange(2,501,2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

pr_data = np.load("Data/NC/BenchN/pr.npz")
X = []
for i in range(N.shape[0]):
    X.append(torch.tensor(pr_data[("arr_"+str(i))], device = device, dtype=torch.float64))

