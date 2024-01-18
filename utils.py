import os
import random
import numpy as np
import torch
from torch import distributions as D


def build_kl_hat(p, compile=True, use_triton=True):
    """
    D_kl(p||q) = H(p, q) - H(q)
    X ~ q
    """
    if torch.cuda.is_available() and use_triton:
        from ent_triton import ent_hat_triton

        def kl_hat(X):
            neg_log_p_X = -p.log_prob(X)
            # We approximate q by the uniform distribution
            return torch.mean(neg_log_p_X) - ent_hat_triton(X)
    else:
        from ent import ent_hat

        def kl_hat(X):
            neg_log_p_X = -p.log_prob(X)
            # We approximate q by the uniform distribution
            return torch.mean(neg_log_p_X) - ent_hat(X)

    if compile is True:
        return torch.compile(kl_hat)
    else:
        return kl_hat


###
# Density
###
def mixture_normal(nb_density: int, device: str = "cpu"):
    means = (torch.rand((nb_density, 2), device=device) - 0.5) * 8  # \in [-4, 4]^2
    variances = torch.eye(2, device=device)[None, :, :].repeat((nb_density, 1, 1)) / 30
    m = D.MultivariateNormal(means, variances)

    mix = D.Categorical(torch.ones(nb_density, device=device))
    return D.MixtureSameFamily(mix, m)


###
# Random
###
def seed_everything(seed: int):
    random.seed(seed)

    npseed = random.randint(1, 1_000_000)
    np.random.seed(npseed)

    ospyseed = random.randint(1, 1_000_000)
    os.environ["PYTHONHASHSEED"] = str(ospyseed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
