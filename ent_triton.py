import torch
import triton
import triton.language as tl

from ent import ent_hat
from torch._inductor import triton_helpers


@triton.jit
def compute_dist_sq(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    This function is processing those lines:
        - dist_sq = torch.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
        - dist_sq[torch.eye(N, dtype=torch.bool, device=X.device)] = torch.inf
    """
    row_idx_1 = tl.program_id(0)
    row_idx_2 = tl.program_id(1)

    if row_idx_1 == row_idx_2:
        out_ptr = output_ptr + row_idx_1 * output_row_stride + row_idx_2
        tl.store(out_ptr, float("inf"))
    else:
        row_start_ptr_1 = input_ptr + row_idx_1 * input_row_stride
        row_start_ptr_2 = input_ptr + row_idx_2 * input_row_stride

        col_offsets = tl.arange(0, BLOCK_SIZE)

        input_ptrs_1 = row_start_ptr_1 + col_offsets
        input_ptrs_2 = row_start_ptr_2 + col_offsets

        xmask = col_offsets < input_row_stride

        row_1 = tl.load(input_ptrs_1, mask=xmask, other=0)
        row_2 = tl.load(input_ptrs_2, mask=xmask, other=0)

        diff = row_1 - row_2  # (D, )
        diff_sq = diff * diff  # (D, )
        dist_sq = tl.sum(diff_sq)  # (1,)

        tl.debug_barrier()

        out_ptr = output_ptr + row_idx_1 * output_row_stride + row_idx_2
        tl.store(out_ptr, dist_sq)


@triton.jit
def compute_raw_min(input_ptr, output_ptr, input_row_stride, N, D, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)

    input_ptrs = row_start_ptr + col_offsets

    xmask = col_offsets < input_row_stride

    row = tl.load(input_ptrs, xmask, other=float("inf"))

    row_min = triton_helpers.min2(row, 0)
    row_min_dist = tl.sqrt(row_min)

    tmp0 = (N - 1) * tl.math.pow(row_min_dist, D)  # min_dist**D
    tmp1 = tl.log(tmp0)

    out_ptr = output_ptr + row_idx
    tl.store(out_ptr, tmp1)


@triton.jit
def compute_mean(input_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    col_offsets = tl.arange(0, BLOCK_SIZE)
    xmask = col_offsets < N

    input_ptrs = input_ptr + col_offsets

    val = tl.load(input_ptrs, xmask, other=0)  # torch.min(dist_sq, 1).values
    ent_hat_val = tl.sum(val, 0) / N

    tl.store(output_ptr, ent_hat_val)


def ent_hat_triton(x):
    assert len(x.shape) == 2

    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)  # no-op to ensure context

        N, D = x.shape
        # Allocate the first output
        output_dist_sq = torch.empty((N, N), device="cuda")
        compute_dist_sq[(N, N)](
            x,
            output_dist_sq,
            x.stride(0),
            output_dist_sq.stride(0),
            BLOCK_SIZE=triton.next_power_of_2(D),
        )

        x = output_dist_sq
        del output_dist_sq

        # Allocate the second output
        output_raw_min = torch.empty((N,), device="cuda")
        compute_raw_min[(N,)](x, output_raw_min, x.stride(0), N, D, BLOCK_SIZE=triton.next_power_of_2(N))
        x = output_raw_min
        del output_raw_min

        output = torch.empty((1,), device="cuda")
        compute_mean[(1,)](x, output, N, BLOCK_SIZE=triton.next_power_of_2(N))
        return output


if __name__ == "__main__":
    from utils import seed_everything

    seed_everything(0)

    x = torch.randn(129, 65, device="cuda")
    y_triton = ent_hat_triton(x)
    y_torch = ent_hat(x)

    assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)

    print("Triton function successfully tested!")
