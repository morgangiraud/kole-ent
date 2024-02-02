import torch
from torch import empty_strided
import triton
import triton.language as tl

from utils import BLOCK_MAX_NB_THREADS
from kole_entropy import kole_entropy
from triton_dist_sq import _kole_dist_sq_forward, _kole_dist_sq_backward
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

        # Separating the kernels into 2 function calls ensure that
        # all the blocks have finished computed their work

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
        )

        ctx.N = N
        ctx.D = D
        ctx.num_warps = num_warps

        ctx.save_for_backward(min_dist, min_dist_indices, x)

        return entropy

    @staticmethod
    def backward(ctx, grad_entropy):
        min_dist, min_dist_indices, x = ctx.saved_tensors
        N = ctx.N
        D = ctx.D

        assert_size_stride(grad_entropy, (), ())
        assert_size_stride(min_dist, (N,), (1,))

        grad_min_dist = torch.empty_strided((N,), (1,), device="cuda", dtype=torch.float32)
        xnumel = N

        XBLOCK = min(triton.next_power_of_2(xnumel), BLOCK_MAX_NB_THREADS)
        g = (1 + (xnumel - 1) // XBLOCK, 1, 1)

        _kole_mean_estimator_backward[g](
            grad_entropy,
            grad_min_dist,
            min_dist,
            N,
            D,
            xnumel,
            # Meta parameters
            XBLOCK=XBLOCK,
            num_warps=ctx.num_warps,
        )
        del grad_entropy

        assert_size_stride(grad_min_dist, (N,), (1,))
        assert_size_stride(min_dist, (N,), (1,))
        assert_size_stride(min_dist_indices, (N,), (1,))

        grad_dist_sq = torch.zeros((N, N), device="cuda", dtype=torch.float32)
        xnumel = N
        rnumel = N
        XBLOCK = min(triton.next_power_of_2(xnumel), BLOCK_MAX_NB_THREADS)
        g = (1 + (xnumel - 1) // XBLOCK, 1, 1)

        _kole_min_dist_backward[g](
            grad_min_dist,
            grad_dist_sq,
            min_dist,
            min_dist_indices,
            xnumel,
            rnumel,
            XBLOCK=XBLOCK,
            num_warps=ctx.num_warps,
        )

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


def kole_entropy_triton(x):
    return KoleEntropy.apply(x)


if __name__ == "__main__":
    from utils import seed_everything

    seed_everything(0)

    print("Testing triton function...")
    for i in range(2, 12):
        N = 2**i
        D = 8

        x_triton = torch.randn(N, D, device="cuda", requires_grad=True)
        x_torch = x_triton.detach().clone()
        x_torch.requires_grad = True

        loss_triton = kole_entropy_triton(x_triton)
        loss_triton.backward()

        loss_torch = kole_entropy(x_torch)
        loss_torch.backward()

        assert torch.allclose(loss_triton, loss_torch), (loss_triton, loss_torch, loss_triton - loss_torch)
        assert torch.allclose(x_triton.grad, x_torch.grad), (x_triton.grad, x_torch.grad, x_triton.grad - x_torch.grad)
        print(f"N,D: {N},{D} -> good")

    print("Triton function successfully tested!")
