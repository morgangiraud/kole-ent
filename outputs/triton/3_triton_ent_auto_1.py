from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile

from torch import empty_strided, device
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_root/yd/cydtjw3sdgjz2enlqjobstkhsmmjaxfxozjf6zlpmsfwf2lkxpmv.py
# Source Nodes: [log, mean, min_1, mul, pow_1, pow_2, setitem, sqrt, sub, sum_1], Original ATen: [aten.index_put, aten.lift_fresh, aten.log, aten.mean, aten.min, aten.mul, aten.pow, aten.sqrt, aten.sub, aten.sum]
# log => log
# mean => mean
# min_1 => min_1
# mul => mul
# pow_1 => pow_1
# pow_2 => pow_2
# setitem => full_default_1, index_put
# sqrt => sqrt
# sub => sub
# sum_1 => sum_1
triton_poi_fused_index_put_lift_fresh_log_mean_min_mul_pow_sqrt_sub_sum_0 = async_compile.triton(
    "triton_",
    """
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[1], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_lift_fresh_log_mean_min_mul_pow_sqrt_sub_sum_0', 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tmp1 - tmp1
    tmp3 = tmp2 * tmp2
    tmp4 = tl.full([1], True, tl.int1)
    tmp5 = float("inf")
    tmp6 = tl.where(tmp4, tmp5, tmp3)
    tmp7 = tl.sqrt(tmp6)
    tmp8 = 0.0
    tmp9 = tmp7 * tmp8
    tmp10 = tl.log(tmp9)
    tmp11 = 1.0
    tmp12 = tmp10 / tmp11
    tl.store(in_out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp12, None)
""",
)

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


async_compile.wait(globals())
del async_compile


def call(args):
    (arg0_1,) = args
    args.clear()
    assert_size_stride(arg0_1, (1, 1), (1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)  # no-op to ensure context
        buf0 = empty_strided((1, 1), (1, 1), device="cuda", dtype=torch.float32)
        buf1 = reinterpret_tensor(buf0, (), ())
        del buf0  # reuse
        # Source Nodes: [log, mean, min_1, mul, pow_1, pow_2, setitem, sqrt, sub, sum_1], Original ATen: [aten.index_put, aten.lift_fresh, aten.log, aten.mean, aten.min, aten.mul, aten.pow, aten.sqrt, aten.sub, aten.sum]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_index_put_lift_fresh_log_mean_min_mul_pow_sqrt_sub_sum_0.run(
            buf1, arg0_1, 1, grid=grid(1), stream=stream0
        )
        del arg0_1
        return (buf1,)


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance

    arg0_1 = rand_strided((1, 1), (1, 1), device="cuda:0", dtype=torch.float32)
    return print_performance(lambda: call([arg0_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main

    compiled_module_main("None", benchmark_compiled_module)
