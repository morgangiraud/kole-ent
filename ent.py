import torch


def compute_dist_sq(X):
    outer_difference_squared = (X[:, None, :] - X[None, :, :]) ** 2  # N, N, D

    return torch.sum(outer_difference_squared, axis=-1)  # N, N


def compute_min_dist(X):
    dist_X = compute_dist_sq(X)

    return _compute_min_dist(dist_X)


def _compute_min_dist(dist_X):
    N = dist_X.shape[0]

    dist_X[torch.eye(N, dtype=torch.bool, device=dist_X.device)] = torch.inf  # N, N
    min_values = torch.min(dist_X, 1).values  # N
    min_dist = torch.sqrt(min_values)

    return min_dist


def compute_mean_estimator(min_dist, D):
    N = min_dist.shape[0]

    return torch.mean(torch.log((N - 1) * min_dist**D))


def ent_hat(X):
    """
    Kozachenko-Leonenko estimator
    More here: https://arxiv.org/abs/1602.07440
    """
    N, D = X.shape

    min_dist = compute_min_dist(X)

    return compute_mean_estimator(min_dist, N, D)
