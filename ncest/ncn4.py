import torch
from ncest.ncexp import *
import numpy as np
from ncest.Nbenchpara import *

if __name__ == '__main__':
    result_wka_N = np.zeros(N.shape[0])
    for i in range(N.shape[0]):
        result_wka_N[i] = ncest_wkakmed(d, N[i], alpha, N_step, X[i]).cpu()
        if (i+1)%100 == 0:
            print("experiment - " + str (i+1) + " has been complete!")

    np.save("Data/NC/BenchN/wka.npy", result_wka_N)