import torch
from torch import empty_strided
from torch._inductor.triton_heuristics import pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream
import triton
import triton.language as tl


assert_size_stride = torch._C._dynamo.guards.assert_size_stride
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor


@pointwise(
    size_hints=[1],
    filename=__file__,
    meta={
        "signature": {0: "*fp32", 1: "*fp32", 2: "i32"},
        "device": 0,
        "device_type": "cuda",
        "constants": {},
        "mutated_arg_names": ["in_out_ptr0"],
        "autotune_hints": set(),
        "kernel_name": "triton_poi_fused_index_put_lift_fresh_min_pow_sqrt_sub_sum_0",
        "configs": [
            instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())
        ],
    },
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
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
    tl.store(in_out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp7, None)


def call(args):
    (arg0_1,) = args
    args.clear()
    assert_size_stride(arg0_1, (1, 1), (1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)  # no-op to ensure context
        buf0 = empty_strided((1, 1), (1, 1), device="cuda", dtype=torch.float32)
        buf1 = reinterpret_tensor(buf0, (1,), (1,))
        del buf0  # reuse
        stream0 = get_cuda_stream(0)
        triton_[grid(1)](buf1, arg0_1, 1, stream=stream0)
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
