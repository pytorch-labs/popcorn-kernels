# AOT ID: ['154_inference']
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


# kernel path: /tmp/torchinductor_sahanp/uh/cuhjjeitgaevsw3o3ug73r6tpihnhhyebtarl2al4wh7qnmu26n2.py
# Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.reflection_pad2d, aten.pow, aten.avg_pool2d, aten.sign, aten.abs, aten.relu, aten.mul, aten.silu]
# Source node to ATen node mapping:
#   x => _unsafe_index, _unsafe_index_1
#   x_1 => abs_5, avg_pool2d, mul_25, mul_30, pow_1, pow_2, relu, sign
#   x_2 => mul_39, sigmoid
# Graph fragment:
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg3_1, [None, None, %sub_5, None]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index, [None, None, None, %sub_11]), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%_unsafe_index_1, 2.0), kwargs = {})
#   %avg_pool2d : [num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%pow_1, [3, 3], [2, 2]), kwargs = {})
#   %sign : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%avg_pool2d,), kwargs = {})
#   %abs_5 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%avg_pool2d,), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%abs_5,), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign, %relu), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, 9), kwargs = {})
#   %pow_2 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_30, 0.5), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%pow_2,), kwargs = {})
#   %mul_39 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_2, %sigmoid), kwargs = {})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_abs_avg_pool2d_mul_pow_reflection_pad2d_relu_sign_silu_0(in_out_ptr0, in_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks1)
    x2 = xindex // ks2
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (ks4*(tl.where((-1) + ks3 + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs((-2) + 2*x1))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs((-2) + 2*x1))) + 2*ks3, (-1) + ks3 + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs((-2) + 2*x1))))) + ks3*ks4*x2 + (tl.where((-1) + ks4 + ((-1)*tl_math.abs(1 + ((-1)*ks4) + tl_math.abs((-2) + 2*x0))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks4) + tl_math.abs((-2) + 2*x0))) + 2*ks4, (-1) + ks4 + ((-1)*tl_math.abs(1 + ((-1)*ks4) + tl_math.abs((-2) + 2*x0)))))), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (ks4*(tl.where((-1) + ks3 + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs((-2) + 2*x1))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs((-2) + 2*x1))) + 2*ks3, (-1) + ks3 + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs((-2) + 2*x1))))) + ks3*ks4*x2 + (tl.where((-1) + ks4 + ((-1)*tl_math.abs(1 + ((-1)*ks4) + tl_math.abs((-1) + 2*x0))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks4) + tl_math.abs((-1) + 2*x0))) + 2*ks4, (-1) + ks4 + ((-1)*tl_math.abs(1 + ((-1)*ks4) + tl_math.abs((-1) + 2*x0)))))), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (ks4*(tl.where((-1) + ks3 + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs((-2) + 2*x1))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs((-2) + 2*x1))) + 2*ks3, (-1) + ks3 + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs((-2) + 2*x1))))) + ks3*ks4*x2 + (tl.where((-1) + ks4 + ((-1)*tl_math.abs(1 + ((-1)*ks4) + 2*x0)) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks4) + 2*x0)) + 2*ks4, (-1) + ks4 + ((-1)*tl_math.abs(1 + ((-1)*ks4) + 2*x0))))), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (ks4*(tl.where((-1) + ks3 + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs((-1) + 2*x1))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs((-1) + 2*x1))) + 2*ks3, (-1) + ks3 + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs((-1) + 2*x1))))) + ks3*ks4*x2 + (tl.where((-1) + ks4 + ((-1)*tl_math.abs(1 + ((-1)*ks4) + tl_math.abs((-2) + 2*x0))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks4) + tl_math.abs((-2) + 2*x0))) + 2*ks4, (-1) + ks4 + ((-1)*tl_math.abs(1 + ((-1)*ks4) + tl_math.abs((-2) + 2*x0)))))), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (ks4*(tl.where((-1) + ks3 + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs((-1) + 2*x1))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs((-1) + 2*x1))) + 2*ks3, (-1) + ks3 + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs((-1) + 2*x1))))) + ks3*ks4*x2 + (tl.where((-1) + ks4 + ((-1)*tl_math.abs(1 + ((-1)*ks4) + tl_math.abs((-1) + 2*x0))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks4) + tl_math.abs((-1) + 2*x0))) + 2*ks4, (-1) + ks4 + ((-1)*tl_math.abs(1 + ((-1)*ks4) + tl_math.abs((-1) + 2*x0)))))), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (ks4*(tl.where((-1) + ks3 + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs((-1) + 2*x1))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs((-1) + 2*x1))) + 2*ks3, (-1) + ks3 + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs((-1) + 2*x1))))) + ks3*ks4*x2 + (tl.where((-1) + ks4 + ((-1)*tl_math.abs(1 + ((-1)*ks4) + 2*x0)) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks4) + 2*x0)) + 2*ks4, (-1) + ks4 + ((-1)*tl_math.abs(1 + ((-1)*ks4) + 2*x0))))), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (ks4*(tl.where((-1) + ks3 + ((-1)*tl_math.abs(1 + ((-1)*ks3) + 2*x1)) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks3) + 2*x1)) + 2*ks3, (-1) + ks3 + ((-1)*tl_math.abs(1 + ((-1)*ks3) + 2*x1)))) + ks3*ks4*x2 + (tl.where((-1) + ks4 + ((-1)*tl_math.abs(1 + ((-1)*ks4) + tl_math.abs((-2) + 2*x0))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks4) + tl_math.abs((-2) + 2*x0))) + 2*ks4, (-1) + ks4 + ((-1)*tl_math.abs(1 + ((-1)*ks4) + tl_math.abs((-2) + 2*x0)))))), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr0 + (ks4*(tl.where((-1) + ks3 + ((-1)*tl_math.abs(1 + ((-1)*ks3) + 2*x1)) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks3) + 2*x1)) + 2*ks3, (-1) + ks3 + ((-1)*tl_math.abs(1 + ((-1)*ks3) + 2*x1)))) + ks3*ks4*x2 + (tl.where((-1) + ks4 + ((-1)*tl_math.abs(1 + ((-1)*ks4) + tl_math.abs((-1) + 2*x0))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks4) + tl_math.abs((-1) + 2*x0))) + 2*ks4, (-1) + ks4 + ((-1)*tl_math.abs(1 + ((-1)*ks4) + tl_math.abs((-1) + 2*x0)))))), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (ks4*(tl.where((-1) + ks3 + ((-1)*tl_math.abs(1 + ((-1)*ks3) + 2*x1)) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks3) + 2*x1)) + 2*ks3, (-1) + ks3 + ((-1)*tl_math.abs(1 + ((-1)*ks3) + 2*x1)))) + ks3*ks4*x2 + (tl.where((-1) + ks4 + ((-1)*tl_math.abs(1 + ((-1)*ks4) + 2*x0)) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks4) + 2*x0)) + 2*ks4, (-1) + ks4 + ((-1)*tl_math.abs(1 + ((-1)*ks4) + 2*x0))))), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 * tmp0
    tmp3 = tmp2 * tmp2
    tmp4 = tmp3 + tmp1
    tmp6 = tmp5 * tmp5
    tmp7 = tmp6 + tmp4
    tmp9 = tmp8 * tmp8
    tmp10 = tmp9 + tmp7
    tmp12 = tmp11 * tmp11
    tmp13 = tmp12 + tmp10
    tmp15 = tmp14 * tmp14
    tmp16 = tmp15 + tmp13
    tmp18 = tmp17 * tmp17
    tmp19 = tmp18 + tmp16
    tmp21 = tmp20 * tmp20
    tmp22 = tmp21 + tmp19
    tmp24 = tmp23 * tmp23
    tmp25 = tmp24 + tmp22
    tmp26 = 0.1111111111111111
    tmp27 = tmp25 * tmp26
    tmp28 = tl.full([1], 0, tl.int32)
    tmp29 = tmp28 < tmp27
    tmp30 = tmp29.to(tl.int8)
    tmp31 = tmp27 < tmp28
    tmp32 = tmp31.to(tl.int8)
    tmp33 = tmp30 - tmp32
    tmp34 = tmp33.to(tmp27.dtype)
    tmp35 = tl_math.abs(tmp27)
    tmp36 = triton_helpers.maximum(tmp28, tmp35)
    tmp37 = tmp34 * tmp36
    tmp38 = 9.0
    tmp39 = tmp37 * tmp38
    tmp40 = libdevice.sqrt(tmp39)
    tmp41 = tl.sigmoid(tmp40)
    tmp42 = tmp40 * tmp41
    tl.store(in_out_ptr0 + (x3), tmp42, xmask)


def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        (3 + s2) // 2
        (3 + s1) // 2
        ((3 + s1) // 2)*((3 + s2) // 2)
        buf0 = empty_strided_cuda((1, s0, (3 + s1) // 2, (3 + s2) // 2), (s0*((3 + s1) // 2)*((3 + s2) // 2), ((3 + s1) // 2)*((3 + s2) // 2), (3 + s2) // 2, 1), torch.float32)
        buf1 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.reflection_pad2d, aten.pow, aten.avg_pool2d, aten.sign, aten.abs, aten.relu, aten.mul, aten.silu]
        triton_poi_fused_abs_avg_pool2d_mul_pow_reflection_pad2d_relu_sign_silu_0_xnumel = s0*((3 + s1) // 2)*((3 + s2) // 2)
        get_raw_stream(0)
        triton_poi_fused_abs_avg_pool2d_mul_pow_reflection_pad2d_relu_sign_silu_0[grid(triton_poi_fused_abs_avg_pool2d_mul_pow_reflection_pad2d_relu_sign_silu_0_xnumel)](buf1, arg3_1, 33, 33, 1089, 64, 64, 3267, XBLOCK=128, num_warps=4, num_stages=1)
        del arg3_1
    return (reinterpret_tensor(buf1, (1, s0, ((3 + s1) // 2)*((3 + s2) // 2)), (s0 + s0*((1 + s1) // 2) + s0*((1 + s2) // 2) + s0*((1 + s1) // 2)*((1 + s2) // 2), ((3 + s1) // 2)*((3 + s2) // 2), 1), 0), s0, 1 + ((1 + s1) // 2), 1 + ((1 + s2) // 2), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 3
    arg1_1 = 64
    arg2_1 = 64
    arg3_1 = rand_strided((1, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
