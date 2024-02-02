import torch
from torch import empty_strided
from torch._inductor import triton_helpers
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_heuristics import reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream
import triton
import triton.language as tl


assert_size_stride = torch._C._dynamo.guards.assert_size_stride


@reduction(
    size_hints=[1024, 32],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={
        "signature": {0: "*fp32", 1: "*fp32", 2: "i32", 3: "i32", 4: "i32", 5: "i32"},
        "device": 0,
        "device_type": "cuda",
        "constants": {},
        "mutated_arg_names": ["in_out_ptr0"],
        "autotune_hints": set(),
        "kernel_name": "triton_red_fused_index_put_lift_fresh_pow_sub_sum_0",
        "configs": [
            instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())
        ],
    },
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, ks0, ks1, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = xindex // ks0
    x0 = xindex % ks0
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (ks1 * x1)), rmask & xmask, eviction_policy="evict_last", other=0)
        tmp1 = tl.load(in_ptr0 + (r2 + (ks1 * x0)), rmask & xmask, eviction_policy="evict_last", other=0)
        tmp2 = tmp0 - tmp1
        tmp3 = tmp2 * tmp2
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tmp7 = x1
    tmp8 = x0
    tmp9 = tmp7 == tmp8
    tmp10 = float("inf")
    tmp11 = tl.where(tmp9, tmp10, tmp5)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp11, xmask)


@reduction(
    size_hints=[32, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={
        "signature": {0: "*fp32", 1: "*fp32", 2: "i32", 3: "i32", 4: "i32"},
        "device": 0,
        "device_type": "cuda",
        "constants": {},
        "mutated_arg_names": ["in_out_ptr0"],
        "autotune_hints": set(),
        "kernel_name": "triton_red_fused_min_sqrt_1",
        "configs": [
            instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())
        ],
    },
)
@triton.jit
def triton__min_(in_out_ptr0, in_ptr0, ks0, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], float("inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (ks0 * x0)), rmask & xmask, other=0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = triton_helpers.minimum(_tmp2, tmp1)
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = triton_helpers.min2(_tmp2, 1)[:, None]
    tmp4 = tl.sqrt(tmp2)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)


def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    assert_size_stride(arg2_1, (s0, s1), (s1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)  # no-op to ensure context
        buf0 = empty_strided((s0, s0), (s0, 1), device="cuda", dtype=torch.float32)
        buf1 = buf0
        del buf0  # reuse
        # Source Nodes: [pow_1, setitem, sub, sum_1], Original ATen: [aten.index_put, aten.lift_fresh, aten.pow, aten.sub, aten.sum]
        triton_red_fused_index_put_lift_fresh_pow_sub_sum_0_xnumel = s0 * s0
        stream0 = get_cuda_stream(0)
        triton_[grid(triton_red_fused_index_put_lift_fresh_pow_sub_sum_0_xnumel)](
            buf1,
            arg2_1,
            s0,
            s1,
            triton_red_fused_index_put_lift_fresh_pow_sub_sum_0_xnumel,
            s1,
            stream=stream0,
        )
        del arg2_1
        buf2 = empty_strided((s0,), (1,), device="cuda", dtype=torch.float32)
        buf4 = buf2
        del buf2  # reuse
        # Source Nodes: [min_1, sqrt], Original ATen: [aten.min, aten.sqrt]
        triton__min_[grid(s0)](buf4, buf1, s0, s0, s0, stream=stream0)
        return (buf4,)


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance

    arg0_1 = 32
    arg1_1 = 32
    arg2_1 = rand_strided((32, 32), (32, 1), device="cuda:0", dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main

    compiled_module_main("None", benchmark_compiled_module)
