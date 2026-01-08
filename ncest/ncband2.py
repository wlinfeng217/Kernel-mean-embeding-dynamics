import torch
from ncest.ncexp import *
import numpy as np
from ncest.bandbenchpara import *

if __name__ == '__main__':
    print("Job is running on",  torch.cuda.get_device_name(0))
    result_pk_N = np.zeros(1)
    result_pk_N[0] = ncest_plain_kmed_quadratic(d, N, N_step, X).cpu()

    np.save("Data/NC/Benchband/pk_q.npy", result_pk_N)
