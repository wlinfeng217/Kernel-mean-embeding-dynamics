import torch
from torch.quasirandom import SobolEngine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)


def sobol_gaussian_mixture_1d(N):
    """
    Parameters
    ----------
    N : int
        Number of samples.

    Returns
    -------
    samples : (1, N)
        Sobol quasi-random samples drawn from
        0.5 * N(4, 1) + 0.5 * N(-4, 1).
    """
    sampler = SobolEngine(dimension=2, scramble=True)
    u = sampler.draw(N).to(device, dtype=torch.float64)

    choose = u[:, 0] < 0.5
    z = torch.distributions.Normal(0.0, 1.0).icdf(u[:, 1])

    samples = torch.where(choose, 4.0 + z, -4.0 + z)

    return samples.reshape(1, N)
