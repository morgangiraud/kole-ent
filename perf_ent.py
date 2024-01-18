import torch
import triton

from ent import ent_hat
from ent_triton import ent_hat_triton


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["D"],  # argument names to use as an x-axis for the plot
        x_vals=[2**i for i in range(2, 10)],  # different possible values for `x_name`
        line_arg="provider",  # argument name whose value corresponds to a different line in the plot
        line_vals=[
            "triton",
            "torch-eager",
            "torch-compile",
        ],  # possible values for `line_arg``
        line_names=[
            "Triton",
            "Torch (eager)",
            "Torch (compiled)",
        ],  # label name for the lines
        styles=[("blue", "-"), ("green", "-"), ("green", "--")],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="ent_hat() performance",  # name for the plot. Used also as a file name for saving the plot.
        args={"N": 256},  # values for function arguments not in `x_names` and `y_name`
    )
)
def benchmark(N, D, provider):
    x = torch.randn(N, D, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch-eager":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: ent_hat(x), quantiles=quantiles)
    if provider == "torch-compile":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.compile(ent_hat)(x), quantiles=quantiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: ent_hat_triton(x), quantiles=quantiles)

    def gbps(ms):
        return 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)

    return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark.run(show_plots=True, print_data=True, save_path=".")
