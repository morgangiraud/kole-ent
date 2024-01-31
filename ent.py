import torch


def kole_dist_sq(X):
    outer_diff_sq = (X[:, None, :] - X[None, :, :]) ** 2  # N, N, D

    return torch.sum(outer_diff_sq, axis=-1)  # N, N


def compute_min_dist(X):
    dist_X = kole_dist_sq(X)

    return kole_min_dist(dist_X)


def kole_min_dist(dist_sq_X):
    N = dist_sq_X.shape[0]

    dist_sq_X[torch.eye(N, dtype=torch.bool, device=dist_sq_X.device)] = torch.inf  # N, N
    min_values = torch.min(dist_sq_X, 1).values  # N
    min_dist = torch.sqrt(min_values)

    return min_dist


def kole_mean_estimator(min_dist, D):
    N = min_dist.shape[0]

    return torch.mean(torch.log((N - 1) * min_dist**D))


def kole_entropy(X):
    """
    Kozachenko-Leonenko estimator
    More here: https://arxiv.org/abs/1602.07440
    """
    D = X.shape[1]

    dist_sq_X = kole_dist_sq(X)

    min_dist = kole_min_dist(dist_sq_X)

    return kole_mean_estimator(min_dist, D)
