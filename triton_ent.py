import torch
from torch import empty_strided
from torch._inductor import triton_helpers
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream
import triton
import triton.language as tl

from ent import ent_hat
from triton_dist_sq import compute_dist_sq_triton_
from triton_min_dist import compute_min_dist_triton_

assert_size_stride = torch._C._dynamo.guards.assert_size_stride


@triton.jit
def compute_mean_estimator_triton_(in_ptr0, out_ptr0, N, D, rnumel, RBLOCK: tl.constexpr):
    acc = tl.full([RBLOCK], 0, tl.float32)
    rbase = tl.arange(0, RBLOCK)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel

        data = tl.load(in_ptr0 + rindex, rmask, other=0)

        log_data = tl.log((N - 1) * tl.math.pow(data, D))
        acc = tl.where(rmask, acc + log_data, acc)

    acc_sumed = tl.sum(acc, 0)
    mean_acc = acc_sumed / N

    tl.store(out_ptr0, mean_acc)


def compute_mean_estimator_triton(min_dist, D):
    N = min_dist.shape[0]

    assert_size_stride(min_dist, (N,), (1,))

    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)  # no-op to ensure context

        out = empty_strided((), (), device="cuda", dtype=torch.float32)
        rnumel = N

        # Meta parameters
        RBLOCK = min(triton.next_power_of_2(rnumel), 1024)
        num_warps = 2
        g = (1 + (rnumel - 1) // RBLOCK, 1, 1)

        stream0 = get_cuda_stream(0)

        compute_mean_estimator_triton_[g](
            min_dist,
            out,
            N,
            D,
            rnumel,
            # Meta parameters
            RBLOCK=RBLOCK,
            num_warps=num_warps,
            stream=stream0,
        )

        return out


def ent_hat_triton(x):
    N, D = x.shape

    assert_size_stride(x, (N, D), (D, 1))

    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)  # no-op to ensure context

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
        compute_min_dist_triton_[g](
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

        out = empty_strided((), (), device="cuda", dtype=torch.float32)
        rnumel = N

        # Meta parameters
        RBLOCK = min(triton.next_power_of_2(rnumel), 1024)
        num_warps = 2
        g = (1 + (rnumel - 1) // RBLOCK, 1, 1)
        compute_mean_estimator_triton_[g](
            min_dist,
            out,
            N,
            D,
            rnumel,
            # Meta parameters
            RBLOCK=RBLOCK,
            num_warps=num_warps,
            stream=stream0,
        )

        return out


if __name__ == "__main__":
    from utils import seed_everything

    seed_everything(0)

    print("Testing triton function...")
    for i in range(2, 12):
        x = torch.randn(129, 2**i + 1, device="cuda")

        y_triton = ent_hat_triton(x)
        y_torch = ent_hat(x)

        assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)
        print(f"input {2**i}: good")

    print("Triton function successfully tested!")
