import torch
import triton
import triton.language as tl

from utils import BLOCK_MAX_NB_THREADS
from kole_entropy import kole_mean_estimator


assert_size_stride = torch._C._dynamo.guards.assert_size_stride


@triton.jit
def _kole_mean_estimator_forward(min_dist_ptr, mean_est_ptr, N, D, rnumel, RBLOCK: tl.constexpr):
    """
    x_pow = x ** D
    x_log = log( (N - 1) * x_pow)
    kl_mean_est = mean(x_log)
    """
    acc = tl.full([RBLOCK], 0, tl.float32)
    rbase = tl.arange(0, RBLOCK)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel

        data = tl.load(min_dist_ptr + rindex, rmask, other=0)

        log_data = tl.log((N - 1) * tl.math.pow(data, D))
        acc = tl.where(rmask, acc + log_data, acc)

    acc_sumed = tl.sum(acc, 0)
    mean_acc = acc_sumed / N

    tl.store(mean_est_ptr, mean_acc)


@triton.jit
def _kole_mean_estimator_backward(
    grad_entropy_ptr, grad_mind_dist_ptr, min_dist_ptr, N, D, xnumel, XBLOCK: tl.constexpr
):
    """
    x_pow = x ** D
    x_log = log( (N - 1) * x_pow)
    loss = kl_mean_est = mean(x_log)

    dloss/dx = d/dx[mean(log((N-1) * x_pow))]
    dloss/dx = d/dx_log[mean(x_log)] * d/dx_pow[log( (N - 1) * x_pow)] * d/dx[x ** D]

    dloss/dx_log = 1/N for every value in x_log
    dx_log/dx_pow = 1 / [(N-1) * x_pow)] * (N-1) = 1/x_pow
    dx_pow/dx = D * x ** (D - 1)

    dloss/dx = 1 / N * 1 / (x ** D) * D * x ** (D - 1)
    dloss/dx = (D / N) * (1 / x)
    """
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel

    min_dist = tl.load(min_dist_ptr + xindex, xmask, other=0)
    grad_entropy = tl.load(grad_entropy_ptr)
    min_dist_grad = (D / N) * (1 / min_dist) * grad_entropy

    tl.store(grad_mind_dist_ptr + xindex, min_dist_grad, xmask)


class KoleMeanEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, min_dist, D):
        N = min_dist.shape[0]

        assert_size_stride(min_dist, (N,), (1,))

        entropy = torch.empty_strided((), (), device="cuda", dtype=torch.float32)
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

        ctx.RBLOCK = RBLOCK
        ctx.num_warps = num_warps
        ctx.g = g
        ctx.D = D
        ctx.save_for_backward(min_dist)

        return entropy

    @staticmethod
    def backward(ctx, grad_entropy):
        min_dist = ctx.saved_tensors[0]
        N = min_dist.shape[0]
        D = ctx.D

        assert_size_stride(grad_entropy, (), ())
        assert_size_stride(min_dist, (N,), (1,))

        grad_mind_dist = torch.empty_strided((N,), (1,), device="cuda", dtype=torch.float32)
        xnumel = N

        XBLOCK = min(triton.next_power_of_2(xnumel), BLOCK_MAX_NB_THREADS)
        g = (1 + (xnumel - 1) // XBLOCK, 1, 1)

        _kole_mean_estimator_backward[g](
            grad_entropy,
            grad_mind_dist,
            min_dist,
            N,
            D,
            xnumel,
            # Meta parameters
            XBLOCK=XBLOCK,
            num_warps=ctx.num_warps,
        )

        return grad_mind_dist, None


def kole_mean_estimator_triton(min_dist, D):
    return KoleMeanEstimator.apply(min_dist, D)


if __name__ == "__main__":
    from utils import seed_everything

    seed_everything(0)

    print("Testing triton function...")
    for i in range(2, 12):
        N = 2**i
        D = 8
        x_triton = torch.rand(N, device="cuda", requires_grad=True)
        x_torch = x_triton.detach().clone()
        x_torch.requires_grad = True

        loss_triton = kole_mean_estimator_triton(x_triton, D)
        loss_triton.backward()

        loss_torch = kole_mean_estimator(x_torch, D)
        loss_torch.backward()

        assert torch.allclose(loss_triton, loss_torch), (loss_triton, loss_torch, loss_triton - loss_torch)
        assert torch.allclose(x_triton.grad, x_torch.grad), (x_triton.grad, x_torch.grad, x_triton.grad - x_torch.grad)
        print(f"N,D: {N},{D} -> good")
    print("Triton function successfully tested!")
