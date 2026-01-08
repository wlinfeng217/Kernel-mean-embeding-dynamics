import torch
import numpy as np

d = torch.arange(1,56,1)
N_step = 50
alpha = 4.0
N = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

pr_data = np.load("Data/NC/Benchd/pr.npz")
X = []
for i in range(d.shape[0]):
    X.append(torch.tensor(pr_data[("arr_"+str(i))], device = device, dtype=torch.float64))

