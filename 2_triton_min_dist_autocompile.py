"""
This file should be run with the following ENV variable set: TORCH_LOGS=output_code
i.e.: TORCH_LOGS=output_code python file.py
"""
import torch

from kole_entropy import compute_min_dist


opt_fn = torch.compile(compute_min_dist)

# Each time we run the function for a different input,
# triton will generate a potentially different code

# numel: 1
opt_fn(
    torch.randn(
        1,
        1,
    ).cuda()
)

# numel: 1024
opt_fn(
    torch.randn(
        32,
        32,
    ).cuda()
)


# numel: 16_384
opt_fn(
    torch.randn(
        128,
        128,
    ).cuda()
)

# numel: 262_400
opt_fn(
    torch.randn(
        256,
        1025,
    ).cuda()
)
