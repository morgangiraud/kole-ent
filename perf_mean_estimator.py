import torch
import triton

from ent import kole_mean_estimator
from triton_mean_estimator import kole_mean_estimator_triton


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],  # argument names to use as an x-axis for the plot
        x_vals=[2**i for i in range(2, 15)],  # different possible values for `x_name`
        line_arg="provider",  # argument name whose value corresponds to a different line in the plot
        line_vals=[
            "torch-eager",
            "torch-compile",
            "triton",
        ],  # possible values for `line_arg``
        line_names=[
            "Torch (eager)",
            "Torch (compiled)",
            "Triton",
        ],  # label name for the lines
        styles=[("blue", "-"), ("green", "-"), ("green", "--")],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="kole_mean_estimator() performance (higher the better)",  # name for the plot. Used also as a file name.
        args={"D": 8},  # values for function arguments not in `x_names` and `y_name`
    )
)
def benchmark(N, D, provider):
    x = torch.randn(N, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch-eager":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: kole_mean_estimator(x, D), quantiles=quantiles)
    if provider == "torch-compile":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.compile(kole_mean_estimator)(x, D), quantiles=quantiles
        )
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: kole_mean_estimator_triton(x, D), quantiles=quantiles)

    def gbps(ms):
        return 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)

    return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark.run(show_plots=True, print_data=True, save_path=".")
