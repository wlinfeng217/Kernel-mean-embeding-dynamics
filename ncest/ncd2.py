import torch
from ncest.ncexp import *
import numpy as np
from ncest.dbenchpara import *

if __name__ == '__main__':
    result_ka_N = np.zeros(d.shape[0])
    for i in range(d.shape[0]):
        result_ka_N[i] = ncest_kakmed(d[i], N, alpha, N_step, X[i]).cpu()
        if (i+1)%100 == 0:
            print("experiment - " + str (i+1) + " has been complete!")

    np.save("Data/NC/Benchd/ka.npy", result_ka_N)