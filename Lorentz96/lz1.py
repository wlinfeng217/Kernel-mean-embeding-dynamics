import torch
from Lorentz96.experiment import *
import numpy as np
from Lorentz96.para import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

if __name__ == '__main__':
    print("Job is running on",  torch.cuda.get_device_name(0))

    po_enkf = []
    for j in range(inflation.shape[0]):
        X_po = enkf_inflation(N_exp, pr, N, delta_t, N_ob, N_assimilation, ob_n, sigma_ob, sigma_f, inflation[j])
        po_enkf.append(np.array(X_po.cpu()))

    np.savez("Data/Lorentz96/ek.npz", *po_enkf)

