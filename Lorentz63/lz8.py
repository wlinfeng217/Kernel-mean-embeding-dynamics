import torch
from Lorentz63.experiment import *
import numpy as np
from Lorentz63.para import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

if __name__ == '__main__':
    print("Job is running on",  torch.cuda.get_device_name(0))

    po_mpf = []
    for j in range(N.shape[0]):
        X_mpf = mpf(N_exp, pr[j], N[j], delta_t, N_ob, N_assimilation, ob_n, sigma_ob, sigma_f, Ns_svgd, alpha, lr = 0.05)
        po_mpf.append(np.array(X_mpf.cpu()))
        # if (j + 1) % 5 == 0:
        print(f"Ensemble size with - {N[j]} complete.")

    np.savez("Data/Lorentz63/mpf.npz", *po_mpf)

