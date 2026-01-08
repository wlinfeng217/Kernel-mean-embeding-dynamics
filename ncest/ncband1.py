import torch
from ncest.ncexp import *
import numpy as np
from ncest.bandbenchpara import *

if __name__ == '__main__':
    print("Job is running on",  torch.cuda.get_device_name(0))
    result_pk_N = np.zeros(band.shape[0])
    for i in range(band.shape[0]):
        result_pk_N[i] = ncest_plain_kmed_fixband(d, N, band[i], N_step, X).cpu()
        if (i+1)%100 == 0:
            print("experiment - " + str (i+1) + " has been complete!")

    np.save("Data/NC/Benchband/pk.npy", result_pk_N)
