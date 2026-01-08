import torch
from ncest.ncexp import *
import numpy as np
from ncest.dbenchpara import *

if __name__ == '__main__':
    result_wk_N = np.zeros(d.shape[0])
    for i in range(d.shape[0]):
        result_wk_N[i] = ncest_wkmed_Id(d[i], N, alpha, N_step, X[i], reg = 1e-3).cpu()
        if (i+1)%100 == 0:
            print("experiment - " + str (i+1) + " has been complete!")

    np.save("Data/NC/Benchd/wk_id.npy", result_wk_N)