import torch
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream
import triton
import triton.language as tl

from ent import kole_min_dist


assert_size_stride = torch._C._dynamo.guards.assert_size_stride


@triton.jit
def _kole_min_dist_forward(
    dist_sq_ptr, min_dist_ptr, min_dist_sq_indices_ptr, N, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    """
    N = dist_sq_X.shape[0]
    dist_sq_X[torch.eye(N, dtype=torch.bool, device=dist_sq_X.device)] = torch.inf  # N, N
    min_values = torch.min(dist_sq_X, 1).values  # N
    min_dist = torch.sqrt(min_values)

    In our case, we also need to keep track of min dist indices for the backward pass
    """
    # Compute output values offset
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.expand_dims(xoffset + tl.arange(0, XBLOCK), 1)  # XBLOCK, 1
    xmask = xindex < xnumel

    min_acc = tl.full([XBLOCK, RBLOCK], float("inf"), tl.float32)
    min_acc_indices = tl.full([XBLOCK, RBLOCK], -1, tl.float32)
    xblock_zero = tl.zeros([XBLOCK, 1], tl.int32)
    rbase = tl.expand_dims(tl.arange(0, RBLOCK), 0)  # 1, RBLOCK
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel  # We guard against going out of the first dimension

        diag_mask = xindex != rindex  # outer != -> XBLOCK, RBLOCK
        guard_mask = rmask & xmask  # outer != -> XBLOCK, RBLOCK
        mask = guard_mask & diag_mask  # XBLOCK, RBLOCK

        in_ptrs = xindex * N + rindex  # outer + -> XBLOCK, RBLOCK

        data = tl.load(dist_sq_ptr + in_ptrs, mask, other=float("inf"))

        rindex_broadcasted = xblock_zero + rindex  # outer + -> XBLOCK, RBLOCK
        min_choice = min_acc < data
        min_acc = tl.where(min_choice, min_acc, data)
        min_acc_indices = tl.where(min_choice, min_acc_indices, rindex_broadcasted)

    min_dist_sq = tl.expand_dims(tl.min(min_acc, 1), 1)
    min_acc_indices_index = tl.expand_dims(tl.argmin(min_acc, 1), 1)
    min_dist_sq_indices = tl.expand_dims(
        tl.sum(
            tl.where(
                min_acc_indices_index == tl.expand_dims(tl.arange(0, RBLOCK), 0),
                min_acc_indices,
                tl.zeros([XBLOCK, RBLOCK], tl.int32),
            ),
            1,
        ),
        1,
    )

    min_dist = tl.sqrt(min_dist_sq)

    tl.store(min_dist_ptr + xindex, min_dist, xmask)
    tl.store(min_dist_sq_indices_ptr + xindex, min_dist_sq_indices, xmask)


@triton.jit
def _kole_min_dist_backward(
    min_dist_grad_ptr,
    dist_sq_grad_ptr,
    min_dist_ptr,
    min_dist_sq_indices_ptr,
    xnumel,
    rnumel,
    XBLOCK: tl.constexpr,
):
    """
    dist_sq[diag] = inf
    min_dist_sq = min(dist_sq, axis=1)
    min_dist = sqrt(min_dist_sq)

    min_dist_sq_grad = 1/2 * 1/min_dist * min_dist_grad
    The above value should be applied as a one hot vector row wise at the original min indices
    """

    xindex = tl.program_id(0) * XBLOCK  # ()
    xoffset = xindex + tl.arange(0, XBLOCK)  # (XBLOCK,)
    xmask = xoffset < xnumel  # (XBLOCK,)

    min_dist_grad = tl.load(min_dist_grad_ptr + xoffset, xmask, other=0)
    min_dist = tl.load(min_dist_ptr + xoffset, xmask, other=float("inf"))

    min_dist_sq_grad = 0.5 * (1 / min_dist)
    dist_sq_grad = min_dist_sq_grad * min_dist_grad

    min_dist_sq_indices = tl.load(min_dist_sq_indices_ptr + xoffset, xmask, other=0)

    out_ptrs = xoffset * rnumel + min_dist_sq_indices
    tl.store(dist_sq_grad_ptr + out_ptrs, dist_sq_grad, xmask)


class KoleMinDist(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dist_sq):
        N = dist_sq.shape[0]

        assert_size_stride(dist_sq, (N, N), (N, 1))

        # Beware that memory location for those tensors might already contains some random/past data
        min_dist = torch.empty_strided((N,), (1,), device="cuda", dtype=torch.float32)
        min_dist_indices = torch.empty_strided((N,), (1,), device="cuda", dtype=torch.int32)
        xnumel = N
        rnumel = N

        # Meta parameters
        RBLOCK = min(triton.next_power_of_2(rnumel), 1024)
        XBLOCK = min(1 + (1024 - 1) // RBLOCK, triton.next_power_of_2(xnumel))
        num_warps = 2
        g = (1 + (xnumel - 1) // XBLOCK, 1, 1)

        _kole_min_dist_forward[g](
            dist_sq,
            min_dist,
            min_dist_indices,
            N,
            xnumel,
            rnumel,
            # Meta parameters
            XBLOCK=XBLOCK,
            RBLOCK=RBLOCK,
            num_warps=num_warps,
        )

        ctx.XBLOCK = XBLOCK
        ctx.RBLOCK = RBLOCK
        ctx.num_warps = num_warps
        ctx.g = g

        ctx.save_for_backward(min_dist, min_dist_indices)

        return min_dist

    @staticmethod
    def backward(ctx, min_dist_grad):
        min_dist, min_dist_indices = ctx.saved_tensors
        N = min_dist.shape[0]

        try:
            assert_size_stride(min_dist_grad, (N,), (1,))
        except AssertionError:
            min_dist_grad = min_dist_grad.clone()
            assert_size_stride(min_dist_grad, (N,), (1,))

        assert_size_stride(min_dist, (N,), (1,))
        assert_size_stride(min_dist_indices, (N,), (1,))

        dist_sq_grad = torch.zeros((N, N), device="cuda", dtype=torch.float32)
        xnumel = N
        rnumel = N

        _kole_min_dist_backward[ctx.g](
            min_dist_grad,
            dist_sq_grad,
            min_dist,
            min_dist_indices,
            xnumel,
            rnumel,
            XBLOCK=ctx.XBLOCK,
            num_warps=ctx.num_warps,
        )

        return dist_sq_grad, None


def kole_min_dist_triton(dist_sq):
    return KoleMinDist.apply(dist_sq)


if __name__ == "__main__":
    from utils import seed_everything

    seed_everything(0)

    print("Testing triton function...")
    for i in range(2, 12):
        N = 2**i

        x_triton = torch.randn(N, N, device="cuda") ** 2
        x_triton.requires_grad = True
        x_torch = x_triton.detach().clone()
        x_torch.requires_grad = True

        # we use the identity function for the parameter of the kole_min_dist so we don't pass a leaf Variable
        loss_triton = torch.sum(kole_min_dist_triton(x_triton * 1.0))
        loss_triton.backward()

        loss_torch = torch.sum(kole_min_dist(x_torch * 1.0))
        loss_torch.backward()

        assert torch.allclose(loss_triton, loss_torch), (loss_triton, loss_torch, loss_triton - loss_torch)
        assert torch.allclose(x_triton.grad, x_torch.grad), (x_triton.grad, x_torch.grad, x_triton.grad - x_torch.grad)
        print(f"N: {N} -> good")
    print("Triton function successfully tested!")
