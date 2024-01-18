import torch
from torch import empty_strided
from torch._inductor import triton_helpers
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream
import triton
import triton.language as tl

from ent import compute_min_dist
from triton_dist_sq import compute_dist_sq_triton_


assert_size_stride = torch._C._dynamo.guards.assert_size_stride


@triton.jit
def compute_min_dist_triton_(in_ptr0, out_ptr0, N, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    # Compute output values offset
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]  # XBLOCK, 1
    xmask = xindex < xnumel

    min_acc = tl.full([XBLOCK, RBLOCK], float("inf"), tl.float32)
    rbase = tl.arange(0, RBLOCK)[None, :]  # 1, RBLOCK
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel  # We guard against going out of the first dimension

        diag_mask = xindex != rindex  # outer != -> XBLOCK, RBLOCK
        guard_mask = rmask & xmask  # outer != -> XBLOCK, RBLOCK
        mask = guard_mask & diag_mask  # XBLOCK, RBLOCK

        in_ptrs = xindex * N + rindex  # outer + -> XBLOCK, RBLOCK

        data = tl.load(in_ptr0 + in_ptrs, mask, other=float("inf"))

        min_acc = triton_helpers.minimum(min_acc, data)

    min_dist_sq = triton_helpers.min2(min_acc, 1)[:, None]
    min_dist = tl.sqrt(min_dist_sq)

    tl.store(out_ptr0 + xindex, min_dist, xmask)


def _compute_min_dist_triton(dist_sq):
    N = dist_sq.shape[0]

    assert_size_stride(dist_sq, (N, N), (N, 1))

    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)  # no-op to ensure context

        out = empty_strided((N,), (1,), device="cuda", dtype=torch.float32)
        xnumel = N
        rnumel = N

        # Meta parameters
        RBLOCK = min(triton.next_power_of_2(rnumel), 1024)
        XBLOCK = 1 + (1024 - 1) // RBLOCK
        num_warps = 2
        g = (1 + (xnumel - 1) // XBLOCK, 1, 1)

        stream0 = get_cuda_stream(0)

        # Allocate the second output
        compute_min_dist_triton_[g](
            dist_sq,
            out,
            N,
            xnumel,
            rnumel,
            # Meta parameters
            XBLOCK=XBLOCK,
            RBLOCK=RBLOCK,
            num_warps=num_warps,
            stream=stream0,
        )
        del dist_sq

        return out


def compute_min_dist_triton(x):
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
        g = (1 + (xnumel - 1) // XBLOCK, 1, 1)

        # We sue a stream to make sure that are kernel calls are queued in FIFO
        stream0 = get_cuda_stream(0)

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

        out = empty_strided((N,), (1,), device="cuda", dtype=torch.float32)
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
            out,
            N,
            xnumel,
            rnumel,
            # Meta parameters
            XBLOCK=XBLOCK,
            RBLOCK=RBLOCK,
            num_warps=num_warps,
            stream=stream0,
        )
        del dist_sq

        return out


if __name__ == "__main__":
    from utils import seed_everything

    seed_everything(0)

    print("Testing triton function...")
    for i in range(2, 12):
        x = torch.randn(129, 2**i + 1, device="cuda")

        y_triton = compute_min_dist_triton(x)
        y_torch = compute_min_dist(x)

        assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)
        print(f"input {2**i}: good")

    print("Triton function successfully tested!")
