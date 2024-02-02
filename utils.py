from typing import Tuple
import os
import random
import numpy as np
import torch
from torch import distributions as D

# CUDA compute capabilities
# https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
# See Table 18
GRID_MAX_DIM_X_SIZE = 2**31 - 1
GRID_MAX_DIM_Y_SIZE = 2**16 - 1
GRID_MAX_DIM_Z_SIZE = 2**16 - 1
BLOCK_MAX_NB_THREADS = 1024
BLOCK_MAX_DIM_X_SIZE = 1024  # Number of threads in each dim
BLOCK_MAX_DIM_Y_SIZE = 1024
BLOCK_MAX_DIM_Z_SIZE = 64
WARP_SIZE = 32
NB_REGISTERS_PER_SM = 64_000


def get_sm_limits(cuda_capa: Tuple[int, int]):
    match cuda_capa:
        case (5, 0) | (5, 2) | (5, 3) | (6, 0) | (6, 1) | (6, 2) | (7, 0) | (7, 2) | (8, 0) | (9, 0):
            return {
                "SM_MAX_NB_BLOCKS": 32,
                "SM_MAX_NB_WARPS": 64,
                "SM_MAX_NB_THREADS": 2048,
            }
        case (7, 5):
            return {
                "SM_MAX_NB_BLOCKS": 16,
                "SM_MAX_NB_WARPS": 32,
                "SM_MAX_NB_THREADS": 1024,
            }
        case (8, 6) | (8, 7):
            return {
                "SM_MAX_NB_BLOCKS": 16,
                "SM_MAX_NB_WARPS": 48,
                "SM_MAX_NB_THREADS": 1536,
            }
        case (8, 9):
            return {
                "SM_MAX_NB_BLOCKS": 24,
                "SM_MAX_NB_WARPS": 48,
                "SM_MAX_NB_THREADS": 1536,
            }
        case _:
            raise ValueError(f"Cuda capabilities {cuda_capa} unknown")


def build_kl_hat(p, compile=True, use_triton=True):
    """
    D_kl(p||q) = H(p, q) - H(q)
    X ~ q
    """
    if torch.cuda.is_available() and use_triton:
        from triton_ent import kole_entropy

        def kl_hat(X):
            # We approximate q by the uniform distribution
            return torch.mean(-p.log_prob(X)) - kole_entropy(X)
    else:
        from ent import kole_entropy

        def kl_hat(X):
            # We approximate q by the uniform distribution
            return torch.mean(-p.log_prob(X)) - kole_entropy(X)

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
