import os

import numpy as np
import torch
from torch import distributions as D


from utils import build_kl_hat, mixture_normal, seed_everything

RESULT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
try:
    os.mkdir(RESULT_DIR)
except FileExistsError:
    pass


nb_density = 9

nb_samples = 2**10
init_sample = D.Uniform(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0])).sample(
    (nb_samples,)
)

init_losses = []
for i in range(1000):
    seed_everything(i)

    p = mixture_normal(nb_density)
    kl_hat = build_kl_hat(p)

    init_loss = kl_hat(init_sample)
    init_losses.append(init_loss.item())
print(f"hard seed: {np.argmax(init_losses)}, easy seed: {np.argmin(init_losses)}")
