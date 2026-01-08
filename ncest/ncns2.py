import torch
from ncest.ncexp import *
import numpy as np
from ncest.Nsbenchpara import *

if __name__ == '__main__':
    result_ka_N = np.zeros(Ns.shape[0])
    for i in range(Ns.shape[0]):
        result_ka_N[i] = ncest_kakmed(d, N, alpha, Ns[i], X).cpu()
        if (i+1)%100 == 0:
            print("experiment - " + str (i+1) + " has been complete!")

    np.save("Data/NC/BenchNs/ka.npy", result_ka_N)