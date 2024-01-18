import torch
import torch.utils.benchmark as benchmark
from torch import distributions as D

setup_code = """
from utils import mixture_normal, build_kl_hat
nb_density = 9
p = mixture_normal(nb_density, device=device)
kl_hat = build_kl_hat(p, compile, use_triton)
"""

label = "KL perf"
causal = False
results = []
devices = ["cpu"]
use_tritons = [False]
if torch.cuda.is_available():
    devices.append("cuda")
    use_tritons.append(True)


sizes = [4, 64, 256, 512, 1024]
for device in devices:
    for compile in [False, True]:
        for use_triton in use_tritons:
            if device == "cpu" and use_triton is True:
                continue
            for nb_samples in sizes:
                sub_label = f"{device}, compiled:{compile}, use_triton: {use_triton}, N:{nb_samples}"
                x = D.Uniform(
                    torch.tensor([0.0, 0.0], device=device),
                    torch.tensor([1.0, 1.0], device=device),
                ).sample((nb_samples,))

                print(f"{label}:{sub_label} -  device: {device}, performing benchmark!")
                ms0 = benchmark.Timer(
                    stmt="kl_hat(x)",
                    setup=setup_code,
                    globals={"x": x, "device": device, "compile": compile, "use_triton": use_triton},
                    # num_threads=num_threads,
                    label=label,
                    sub_label=sub_label,
                    description="kl_hat",
                ).blocked_autorange(min_run_time=1)
                results.append(ms0)

compare = benchmark.Compare(results)
compare.colorize()
compare.print()
