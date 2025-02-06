# AOT ID: ['42_inference']
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


# kernel path: /tmp/torchinductor_sahanp/bd/cbdlifn3nsx5x3pkpee3bgjsqjfqbqppz4to2jckyivowea5jbil.py
# Topologically Sorted Source Nodes: [x_1, x_2, x_3, x_4, x_5, loss], Original ATen: [aten._adaptive_avg_pool2d, aten.relu, aten.exp, aten.mul, aten.sub, aten.mean]
# Source node to ATen node mapping:
#   loss => exp, mean, mul_32, sub_14
#   x_1 => _adaptive_avg_pool2d
#   x_2 => relu
#   x_3 => relu_1
#   x_4 => relu_2
#   x_5 => relu_3
# Graph fragment:
#   %_adaptive_avg_pool2d : [num_users=1] = call_function[target=torch.ops.aten._adaptive_avg_pool2d.default](args = (%unsqueeze, [1, 10]), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%squeeze,), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%relu,), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%relu_1,), kwargs = {})
#   %relu_3 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%relu_2,), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%relu_3,), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%relu_3, %relu_3), kwargs = {})
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%exp, %mul_32), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sub_14,), kwargs = {})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__adaptive_avg_pool2d_exp_mean_mul_relu_sub_0(in_out_ptr0, in_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    _tmp84 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_0 = (r0_index % 10)
        r0_1 = r0_index // 10
        tmp0 = tl.full([1, 1], 0, tl.int64)
        tmp1 = tl.full([1, 1], 1, tl.int64)
        tmp2 = tmp0 < tmp1
        tmp3 = (12*r0_0) // 5
        tmp4 = (33 + 24*r0_0) // 10
        tmp5 = tmp3 < tmp4
        tmp6 = tmp2 & tmp5
        tmp7 = tl.broadcast_to((-2) + ((12*r0_0) // 5), [XBLOCK, R0_BLOCK])
        tmp8 = tl.full([1, 1], 0, tl.int64)
        tmp9 = tmp7 >= tmp8
        tmp10 = tl.full([1, 1], 20, tl.int64)
        tmp11 = tmp7 < tmp10
        tmp12 = tmp9 & tmp11
        tmp13 = tmp12 & tmp6
        tmp14 = tl.load(in_ptr0 + (tl.broadcast_to((-2) + 20*r0_1 + ((12*r0_0) // 5), [XBLOCK, R0_BLOCK])), r0_mask & tmp13, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
        tmp16 = tl.where(tmp6, tmp14, tmp15)
        tmp17 = 1 + ((12*r0_0) // 5)
        tmp18 = tmp17 < tmp4
        tmp19 = tmp2 & tmp18
        tmp20 = tl.broadcast_to((-1) + ((12*r0_0) // 5), [XBLOCK, R0_BLOCK])
        tmp21 = tl.full([1, 1], 0, tl.int64)
        tmp22 = tmp20 >= tmp21
        tmp23 = tl.full([1, 1], 20, tl.int64)
        tmp24 = tmp20 < tmp23
        tmp25 = tmp22 & tmp24
        tmp26 = tmp25 & tmp19
        tmp27 = tl.load(in_ptr0 + (tl.broadcast_to((-1) + 20*r0_1 + ((12*r0_0) // 5), [XBLOCK, R0_BLOCK])), r0_mask & tmp26, eviction_policy='evict_last', other=0.0)
        tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
        tmp29 = tl.where(tmp19, tmp27, tmp28)
        tmp30 = tmp29 + tmp16
        tmp31 = 2 + ((12*r0_0) // 5)
        tmp32 = tmp31 < tmp4
        tmp33 = tmp2 & tmp32
        tmp34 = tl.broadcast_to((12*r0_0) // 5, [XBLOCK, R0_BLOCK])
        tmp35 = tl.full([1, 1], 0, tl.int64)
        tmp36 = tmp34 >= tmp35
        tmp37 = tl.full([1, 1], 20, tl.int64)
        tmp38 = tmp34 < tmp37
        tmp39 = tmp36 & tmp38
        tmp40 = tmp39 & tmp33
        tmp41 = tl.load(in_ptr0 + (tl.broadcast_to(20*r0_1 + ((12*r0_0) // 5), [XBLOCK, R0_BLOCK])), r0_mask & tmp40, eviction_policy='evict_last', other=0.0)
        tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
        tmp43 = tl.where(tmp33, tmp41, tmp42)
        tmp44 = tmp43 + tmp30
        tmp45 = 3 + ((12*r0_0) // 5)
        tmp46 = tmp45 < tmp4
        tmp47 = tmp2 & tmp46
        tmp48 = tl.broadcast_to(1 + ((12*r0_0) // 5), [XBLOCK, R0_BLOCK])
        tmp49 = tl.full([1, 1], 0, tl.int64)
        tmp50 = tmp48 >= tmp49
        tmp51 = tl.full([1, 1], 20, tl.int64)
        tmp52 = tmp48 < tmp51
        tmp53 = tmp50 & tmp52
        tmp54 = tmp53 & tmp47
        tmp55 = tl.load(in_ptr0 + (tl.broadcast_to(1 + 20*r0_1 + ((12*r0_0) // 5), [XBLOCK, R0_BLOCK])), r0_mask & tmp54, eviction_policy='evict_last', other=0.0)
        tmp56 = tl.full(tmp55.shape, 0.0, tmp55.dtype)
        tmp57 = tl.where(tmp47, tmp55, tmp56)
        tmp58 = tmp57 + tmp44
        tmp59 = 1.0
        tmp60 = tl.full(tmp59.shape, 0.0, tmp59.dtype)
        tmp61 = tl.where(tmp6, tmp59, tmp60)
        tmp62 = 1.0
        tmp63 = tl.full(tmp62.shape, 0.0, tmp62.dtype)
        tmp64 = tl.where(tmp19, tmp62, tmp63)
        tmp65 = tmp64 + tmp61
        tmp66 = 1.0
        tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
        tmp68 = tl.where(tmp33, tmp66, tmp67)
        tmp69 = tmp68 + tmp65
        tmp70 = 1.0
        tmp71 = tl.full(tmp70.shape, 0.0, tmp70.dtype)
        tmp72 = tl.where(tmp47, tmp70, tmp71)
        tmp73 = tmp72 + tmp69
        tmp74 = tmp58 / tmp73
        tmp75 = tl.full([1, 1], 0, tl.int32)
        tmp76 = triton_helpers.maximum(tmp75, tmp74)
        tmp77 = triton_helpers.maximum(tmp75, tmp76)
        tmp78 = triton_helpers.maximum(tmp75, tmp77)
        tmp79 = triton_helpers.maximum(tmp75, tmp78)
        tmp80 = tl_math.exp(tmp79)
        tmp81 = tmp79 * tmp79
        tmp82 = tmp80 - tmp81
        tmp83 = tl.broadcast_to(tmp82, [XBLOCK, R0_BLOCK])
        tmp85 = _tmp84 + tmp83
        _tmp84 = tl.where(r0_mask, tmp85, _tmp84)
    tmp84 = tl.sum(_tmp84, 1)[:, None]
    tmp86 = 10*ks0
    tmp87 = tmp86.to(tl.float32)
    tmp88 = tmp84 / tmp87
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp88, None)


def call(args):
    arg0_1, _arg1_1, arg2_1 = args
    args.clear()
    s0 = arg0_1
    assert_size_stride(arg2_1, (1, s0, 20), (20*s0, 20, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((), (), torch.float32)
        buf2 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [x_1, x_2, x_3, x_4, x_5, loss], Original ATen: [aten._adaptive_avg_pool2d, aten.relu, aten.exp, aten.mul, aten.sub, aten.mean]
        10*s0
        get_raw_stream(0)
        triton_red_fused__adaptive_avg_pool2d_exp_mean_mul_relu_sub_0[grid(1)](buf2, arg2_1, 3, 1, 30, XBLOCK=1, R0_BLOCK=32, num_warps=2, num_stages=1)
        del arg2_1
    return (buf2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 3
    arg1_1 = 20
    arg2_1 = rand_strided((1, 3, 20), (60, 20, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
