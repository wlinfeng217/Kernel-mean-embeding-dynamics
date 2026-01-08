import torch
from ncest.ncexp import *
import numpy as np
from ncest.Nsbenchpara import *

if __name__ == '__main__':
    result_wka_N = np.zeros(Ns.shape[0])
    for i in range(Ns.shape[0]):
        result_wka_N[i] = ncest_wkakmed(d, N, alpha, Ns[i], X).cpu()
        if (i+1)%100 == 0:
            print("experiment - " + str (i+1) + " has been complete!")

    np.save("Data/NC/BenchNs/wka.npy", result_wka_N)