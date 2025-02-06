# AOT ID: ['4_inference']
import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import (
    grid,
)
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool

empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_sahanp/6m/c6mvtix2k7at5bk6duga5aycitoahrnie6h4lo5zlus4ycg6vhcn.py
# Topologically Sorted Source Nodes: [x_1, x_3, x_4], Original ATen: [aten._adaptive_avg_pool2d, aten.neg, aten._softmax, aten.abs, aten.le, aten.scalar_tensor, aten.where]
# Source node to ATen node mapping:
#   x_1 => _adaptive_avg_pool2d
#   x_3 => amax, div, exp, neg, sub, sum_1
#   x_4 => abs_1, full_default, le, where
# Graph fragment:
#   %_adaptive_avg_pool2d : [num_users=1] = call_function[target=torch.ops.aten._adaptive_avg_pool2d.default](args = (%unsqueeze, [1, 10]), kwargs = {})
#   %neg : [num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%view_1,), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%neg, [1], True), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%neg, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%div,), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%abs_1, 0.5), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le, %full_default, %div), kwargs = {})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__adaptive_avg_pool2d__softmax_abs_le_neg_scalar_tensor_where_0(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 10
    R0_BLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_mask = r0_index < r0_numel
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (10*r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr0 + (1 + 10*r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr0 + (2 + 10*r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr0 + (3 + 10*r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr0 + (4 + 10*r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr0 + (5 + 10*r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr0 + (6 + 10*r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr0 + (7 + 10*r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr0 + (8 + 10*r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.load(in_ptr0 + (9 + 10*r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp16 = tmp15 + tmp14
    tmp18 = tmp17 + tmp16
    tmp19 = 0.1
    tmp20 = tmp18 * tmp19
    tmp21 = -tmp20
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, R0_BLOCK])
    tmp24 = tl.where(r0_mask, tmp22, float("-inf"))
    tmp25 = triton_helpers.max2(tmp24, 1)[:, None]
    tmp26 = tmp21 - tmp25
    tmp27 = tl_math.exp(tmp26)
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, R0_BLOCK])
    tmp30 = tl.where(r0_mask, tmp28, 0)
    tmp31 = tl.sum(tmp30, 1)[:, None]
    tmp32 = tmp27 / tmp31
    tmp33 = tl_math.abs(tmp32)
    tmp34 = 0.5
    tmp35 = tmp33 <= tmp34
    tmp36 = 0.0
    tmp37 = tl.where(tmp35, tmp36, tmp32)
    tl.store(in_out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp37, r0_mask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (1, 100), (100, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 1, 1, 10), (10, 10, 10, 1), torch.float32)
        buf3 = reinterpret_tensor(buf0, (1, 10), (10, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [x_1, x_3, x_4], Original ATen: [aten._adaptive_avg_pool2d, aten.neg, aten._softmax, aten.abs, aten.le, aten.scalar_tensor, aten.where]
        get_raw_stream(0)
        triton_per_fused__adaptive_avg_pool2d__softmax_abs_le_neg_scalar_tensor_where_0[grid(1)](buf3, arg0_1, 1, 10, XBLOCK=1, num_warps=2, num_stages=1)
        del arg0_1
    return (buf3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 100), (100, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
