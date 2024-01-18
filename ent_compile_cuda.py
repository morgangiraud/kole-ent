import torch

from ent import ent_hat

opt_square = torch.compile(ent_hat)
opt_square(
    torch.randn(
        100,
        100,
    ).cuda()
)
