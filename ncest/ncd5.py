import torch
from ncest.ncexp import *
import numpy as np
from ncest.dbenchpara import *

if __name__ == '__main__':
    result_sir_N = np.zeros((d.shape[0], 100))
    for i in range(d.shape[0]):
        for j in range(100):
            result_sir_N[i, j] = ncest_sir(d[i], N, N_step, X[i]).cpu()
        if (i+1)%100 == 0:
            print("experiment - " + str (i+1) + " has been complete!")

    mu_sir = np.mean(result_sir_N, axis = 1)
    P_sir = np.std(result_sir_N, axis = 1)
    np.savez("Data/NC/Benchd/sir.npz", mu = mu_sir, error = P_sir)