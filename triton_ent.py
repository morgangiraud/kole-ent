from typing import Any
import torch
from torch import empty_strided
from torch._inductor import triton_helpers
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream
import triton
import triton.language as tl

from ent import kole_entropy
from triton_dist_sq import compute_dist_sq_triton_
from triton_min_dist import _kole_min_dist_forward, _kole_min_dist_backward
from triton_mean_estimator import _kole_mean_estimator_forward, _kole_mean_estimator_backward

assert_size_stride = torch._C._dynamo.guards.assert_size_stride


class KoleEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        N, D = x.shape

        assert_size_stride(x, (N, D), (D, 1))

        dist_sq = empty_strided((N, N), (N, 1), device="cuda", dtype=torch.float32)
        xnumel = N * N
        rnumel = D

        # Meta parameters
        RBLOCK = min(triton.next_power_of_2(rnumel), 1024)
        XBLOCK = 1 + (1024 - 1) // RBLOCK
        num_warps = 2
        stream0 = get_cuda_stream(0)
        g = (1 + (xnumel - 1) // XBLOCK, 1, 1)

        compute_dist_sq_triton_[g](
            x,
            dist_sq,
            N,
            D,
            xnumel,
            rnumel,
            # Meta parameters
            XBLOCK=XBLOCK,
            RBLOCK=RBLOCK,
            num_warps=num_warps,
            stream=stream0,
        )
        del x

        # Separating the kernels into 2 function calls ensure that
        # all the blocks have finished computed their work

        min_dist = empty_strided((N,), (1,), device="cuda", dtype=torch.float32)
        xnumel = N
        rnumel = N

        # Meta parameters
        RBLOCK = min(triton.next_power_of_2(rnumel), 1024)
        XBLOCK = 1 + (1024 - 1) // RBLOCK
        num_warps = 2
        g = (1 + (xnumel - 1) // XBLOCK, 1, 1)

        # Allocate the second output
        _kole_min_dist_forward[g](
            dist_sq,
            min_dist,
            N,
            xnumel,
            N,  # Meta parameters
            XBLOCK=XBLOCK,
            RBLOCK=RBLOCK,
            num_warps=num_warps,
            stream=stream0,
        )
        del dist_sq

        entropy = empty_strided((), (), device="cuda", dtype=torch.float32)
        rnumel = N

        # Meta parameters
        RBLOCK = min(triton.next_power_of_2(rnumel), 1024)
        num_warps = 2
        g = (1 + (rnumel - 1) // RBLOCK, 1, 1)
        _kole_mean_estimator_forward[g](
            min_dist,
            entropy,
            N,
            D,
            rnumel,
            # Meta parameters
            RBLOCK=RBLOCK,
            num_warps=num_warps,
            stream=stream0,
        )
        del min_dist

        ctx.N = N
        ctx.D = D
        ctx.num_warps = num_warps

        return entropy

    @staticmethod
    def backward(ctx, entropy_grad):
        N = ctx.N
        D = ctx.D
        assert_size_stride(entropy_grad, (), ())

        pass


def kole_entropy_triton(x):
    return KoleEntropy.apply(x)


if __name__ == "__main__":
    from utils import seed_everything

    seed_everything(0)

    print("Testing triton function...")
    for i in range(2, 12):
        N = 32
        D = 2**i

        x_triton = torch.randn(N, N, device="cuda", requires_grad=True)
        x_torch = x_triton.detach().clone()
        x_torch.requires_grad = True

        loss_triton = kole_entropy_triton(x_triton)
        loss_triton.backward()

        loss_torch = kole_entropy(x_torch)
        loss_torch.backward()

        assert torch.allclose(loss_triton, loss_torch), (loss_triton, loss_torch)
        assert torch.allclose(x_triton.grad, x_torch.grad), (x_triton.grad, x_torch.grad)
        print(f"input {2**i}: good")

    print("Triton function successfully tested!")
