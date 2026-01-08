import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)


def resampling(w, X, h, N):
    """
    Parameters
    ----------
    w : (N,)
        Particle weights.
    X : (d, N)
        Particle positions.
    h : (N,)
        Score evaluated at particles.
    N : int
        Number of particles to resample.

    Returns
    -------
    X_resampled : (d, N)
        Resampled particle positions.
    h_resampled : (N,)
        Resampled scores.
    """
    w = w.to(device, dtype=torch.float64)
    X = X.to(device, dtype=torch.float64)
    h = h.to(device, dtype=torch.float64)

    W = w / torch.sum(w)  # Normalize weights

    # Sample indices based on normalized weights
    indices = torch.multinomial(W, num_samples=N, replacement=True)

    # Resample positions and scores
    X_resampled = X[:, indices].clone()
    h_resampled = h[indices].clone()

    return X_resampled, h_resampled
