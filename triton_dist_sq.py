"""
This file has been auto generated by triton and then commented and edited.
"""
import torch
import triton
import triton.language as tl

from utils import BLOCK_MAX_NB_THREADS
from kole_entropy import kole_dist_sq


assert_size_stride = torch._C._dynamo.guards.assert_size_stride


@triton.jit
def _kole_dist_sq_forward(x_ptr, dist_sq_ptr, N, D, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    """
    outer_diff_sq = (X[:, None, :] - X[None, :, :]) ** 2  # N, N, D
    dist_sq = torch.sum(outer_diff_sq, axis=-1)  # N, N

    This function works from the output perspective.
    The goal is to compute every pairwise distance between each row of x_ptr
    and store this value in dist_sq_ptr for all the NxN values

    xnumel = N*N
    rnumel = D
    """
    # Compute output values offset
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.expand_dims(xoffset + tl.arange(0, XBLOCK), 1)  # XBLOCK, 1

    # We guard against going out of the output matrix number of elements
    # This guard will also be used to avoid loading already computed output values
    xmask = xindex < xnumel

    # First trick:
    # This is nice way map output values to input indexes and compute all the pairwise distances
    # While ensuring we never go beyond the number of lines of the input matrix
    pw_row1_index = D * (xindex // N)  # [0, ..., 0, 1, ...]
    pw_row2_index = D * (xindex % N)  # [0, 1, ..., N-1, 0, ...]

    # We allocate the memory to store the temporary values
    acc_add = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    # Dynamic reduction working per block
    rbase = tl.expand_dims(tl.arange(0, RBLOCK), 0)  # 1, RBLOCK
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel  # We guard against going out of the first dimension

        # Second trick
        # We use an outer AND and outer + operations to get the current reduction indexes for
        # the rows handled by the current block
        mask = rmask & xmask  # XBLOCK, RBLOCK
        in1_ptrs = pw_row1_index + rindex  # XBLOCK, RBLOCK
        in0_ptrs = pw_row2_index + rindex  # XBLOCK, RBLOCK

        data0 = tl.load(x_ptr + in1_ptrs, mask, eviction_policy="evict_last", other=0)
        data1 = tl.load(x_ptr + in0_ptrs, mask, eviction_policy="evict_last", other=0)

        # We do all the pointwise operations
        diff = data0 - data1
        diff_squared = diff * diff

        # This line is not needed because we are sure that we are working with
        # tensors of size [XBLOCK, RBLOCK] already
        # diff_squared_brodcasted = tl.broadcast_to(diff_squared, [XBLOCK, RBLOCK])

        # Those lines can be simplified because we mask our input values with the 0. value
        # and (0 - 0)**2 -> 0 so it won't interfere with the accumulation
        acc_add += diff_squared
        # # We add to our temporary buffer
        # # and make sure to only keep the values that has been updated
        # tmp6 = acc_add + diff_squared
        # acc_add = tl.where(mask, tmp6, acc_add)

    # We finally reduce to get final output values
    row_sum = tl.expand_dims(tl.sum(acc_add, 1), 1)
    # And we write back in the global memory
    tl.store(dist_sq_ptr + xindex, row_sum, xmask)


@triton.jit
def _kole_dist_sq_backward(
    grad_dist_sq_ptr,
    grad_x_ptr,
    x_ptr,
    N,
    D,
    xnumel,
    rnumel,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    """
    outer_diff_sq = (X[:, None, :] - X[None, :, :]) ** 2  # N, N, D
    dist_sq = torch.sum(outer_diff_sq, axis=-1)  # N, N

    We have:
    - dy_ij/dx_lk = d/dx_lk [sum_m((x_im - x_jm)**2)]
    - dy_ij/dx_lk = 2 * (x_lk - x_jk) if l=i
    -             = -2 * (x_lk - x_jk) if l=j
    -             = 0 otherwise

    So for an element x_lk its gradient is:
    - x_lk_grad = sum_i(dy_il/dx_lk * dy_il) + sum_j(dy_lj/dx_lk * dy_lj)
    - x_lk_grad = 2 * [sum_j((x_lk - x_jk) * dy_lj) - sum_i((x_ik - x_lk) * dy_il)]
    - x_lk_grad = 2 * [x_lk * sum(y_grad_add) - sum(x[:, k] * y_grad_add)]
        - with y_grad_add = dy_*l + dy_l* (Dim N)
    """
    xindex = tl.program_id(0) * XBLOCK
    xoffset = xindex + tl.arange(0, XBLOCK)
    xmask = xoffset < xnumel

    x_rows_index = tl.expand_dims(xoffset // D, 0)
    x_cols_index = tl.expand_dims(xoffset % D, 0)  # [1, XBLOCK]

    y_grad_add = tl.full([RBLOCK, XBLOCK], 0, tl.float32)
    x_y_grad_add = tl.full([RBLOCK, XBLOCK], 0, tl.float32)
    r_range = tl.arange(0, RBLOCK)
    for rindex in range(0, rnumel, RBLOCK):
        roffset = tl.expand_dims(rindex + r_range, 1)  # (RBLOCK, 1)
        rmask = roffset < rnumel

        mask = rmask & tl.expand_dims(xmask, 0)
        x_col = tl.load(x_ptr + (roffset * D + x_cols_index), mask, other=0)  # (RBLOCK, XBLOCK)
        grad_dist_sq_row = tl.load(grad_dist_sq_ptr + x_rows_index * N + roffset, mask, other=0)  # Get lines
        grad_dist_sq_col = tl.load(grad_dist_sq_ptr + roffset * N + x_rows_index, mask, other=0)  # Get cols

        y_grad_add += grad_dist_sq_row + grad_dist_sq_col
        x_y_grad_add += x_col * (grad_dist_sq_row + grad_dist_sq_col)

    x = tl.load(x_ptr + xoffset, xmask, other=0)  # (XBLOCK,)

    left = x * tl.sum(y_grad_add, 0)
    right = tl.sum(x_y_grad_add, 0)
    grad_x = 2 * (left - right)  # (XBLOCK,)

    tl.store(grad_x_ptr + xoffset, grad_x, xmask)


class KoleDistSQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        N, D = x.shape

        assert_size_stride(x, (N, D), (D, 1))

        dist_sq = torch.empty_strided((N, N), (N, 1), device="cuda", dtype=torch.float32)

        # Working on a 1D grid
        xnumel = N * N
        rnumel = D

        # The goal of the config is to maximize the "occupancy" of the GPU
        # Occupancy = assigned warp to SM / SM maximum supported
        # So the goal is to ensure we use the maximum number of threads possible per SM at any point in time.

        # To do so, it is useful to know
        # - Maximum number of threads per block
        # - Maximum number of warps per SM
        # - Maximum number of block per SM
        # - Number of SM

        # Secondly it is important to not go beyond other caracteristiques like:
        # - register memory limit
        # - shared memory limit
        # - maximum memory bandwidth
        # - etc.
        # but in this case we hope that triton will do the work for us
        # and it will, as long as we provide to it enough information so
        # it can do its magic

        # Meta parameters heuristics
        num_warps = 2  # I don't understand why I get worst performance when increasing this value
        RBLOCK = min(triton.next_power_of_2(D), BLOCK_MAX_NB_THREADS)  #
        XBLOCK = min(BLOCK_MAX_NB_THREADS // RBLOCK, triton.next_power_of_2(N))

        nb_blocks = 1 + (xnumel - 1) // XBLOCK
        g = (nb_blocks, 1, 1)

        _kole_dist_sq_forward[g](
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
        )

        ctx.N = N
        ctx.D = D
        ctx.num_warps = num_warps

        ctx.save_for_backward(x)

        return dist_sq

    @staticmethod
    def backward(ctx, grad_dist_sq):
        x = ctx.saved_tensors[0]
        N = ctx.N
        D = ctx.D

        # For some reason, grad_dist_sq is returned as a (N, N) tensor with strides (0, 0)
        # Thanks to the interpret, we can see this error:
        #   RuntimeError: unsupported operation:
        #       more than one element of the written-to tensor refers to a single memory location.
        #       Please clone() the tensor before performing the operation.
        # This error means that the output tensor has not been fully allocated in memory
        # Cloning it do allocate the memory
        #
        # This is efficient in the case that the tensor is sparse
        try:
            assert_size_stride(grad_dist_sq, (N, N), (N, 1))
        except AssertionError:
            grad_dist_sq = grad_dist_sq.clone()
            assert_size_stride(grad_dist_sq, (N, N), (N, 1))

        grad_x = torch.zeros(N, D, dtype=torch.float32, device="cuda")
        xnumel = N * D
        rnumel = N

        RBLOCK = min(triton.next_power_of_2(N), BLOCK_MAX_NB_THREADS)
        XBLOCK = min(BLOCK_MAX_NB_THREADS // RBLOCK, triton.next_power_of_2(N * D))
        nb_blocks = 1 + (xnumel - 1) // XBLOCK
        g = (nb_blocks, 1, 1)

        _kole_dist_sq_backward[g](
            grad_dist_sq, grad_x, x, N, D, xnumel, rnumel, XBLOCK=XBLOCK, RBLOCK=RBLOCK, num_warps=ctx.num_warps
        )

        return grad_x


def kole_dist_sq_triton(x):
    return KoleDistSQ.apply(x)


if __name__ == "__main__":
    from utils import seed_everything

    seed_everything(0)

    print("Testing triton function...")
    for i in range(2, 12):
        N = 2**i
        D = 8

        x_triton = torch.randn(N, D, device="cuda")
        x_triton.requires_grad = True
        x_torch = x_triton.detach().clone()
        x_torch.requires_grad = True

        # we use the identity function for the parameter of the kole_min_dist so we don't pass a leaf Variable
        loss_triton = torch.sum(kole_dist_sq_triton(x_triton))
        loss_triton.backward()

        loss_torch = torch.sum(kole_dist_sq(x_torch))
        loss_torch.backward()

        assert torch.allclose(loss_triton, loss_torch), (loss_triton, loss_torch, loss_triton - loss_torch)
        # TODO: investigate why we have this not so tiny divergence i
        assert torch.allclose(x_triton.grad, x_torch.grad, rtol=2e-3), (
            x_triton.grad,
            x_torch.grad,
            x_triton.grad - x_torch.grad,
        )
        print(f"N, D: {N}, {D} -> good")

    print("Triton function successfully tested!")
