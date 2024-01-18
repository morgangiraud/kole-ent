import torch
from torch import distributions as D


def ent_hat(X):
    """
    Kozachenko-Leonenko estimator
    More here: https://arxiv.org/abs/1602.07440
    """
    N, D = X.shape

    outer_difference_squared = (X[:, None, :] - X[None, :, :]) ** 2  # N, N, D

    dist_sq = torch.sum(outer_difference_squared, axis=-1)  # N, N

    dist_sq[torch.eye(N, dtype=torch.bool, device=X.device)] = torch.inf  # N, N

    min_values = torch.min(dist_sq, 1).values  # N

    min_dist = torch.sqrt(min_values)

    return torch.mean(torch.log((N - 1) * min_dist**D))
