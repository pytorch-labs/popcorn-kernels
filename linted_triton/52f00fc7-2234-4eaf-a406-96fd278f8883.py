# AOT ID: ['109_inference']
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


# kernel path: /tmp/torchinductor_sahanp/x7/cx75ztmgunedv3xatqrjinqa2vut4kc2d4mvprrhranpvnxau3i4.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.constant_pad_nd, aten.avg_pool3d]
# Source node to ATen node mapping:
#   x => avg_pool3d, constant_pad_nd
# Graph fragment:
#   %constant_pad_nd : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view, [0, 0, 0, 0, 2, 2], 0.0), kwargs = {})
#   %avg_pool3d : [num_users=1] = call_function[target=torch.ops.aten.avg_pool3d.default](args = (%constant_pad_nd, [5, 1, 1], [1, 1, 1]), kwargs = {})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_avg_pool3d_constant_pad_nd_0(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // ks0
    x2 = xindex
    tmp0 = (-2) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = ks1
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x2 + ((-2)*ks2*ks3)), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp6 * tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp5, tmp7, tmp8)
    tmp10 = (-1) + x1
    tmp11 = tmp10 >= tmp1
    tmp12 = tmp10 < tmp3
    tmp13 = tmp11 & tmp12
    tmp14 = tl.load(in_ptr0 + (x2 + ((-1)*ks2*ks3)), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp14 * tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp13, tmp15, tmp16)
    tmp18 = tmp17 + tmp9
    tmp19 = x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tl.load(in_ptr0 + (x2), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp23 * tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp22, tmp24, tmp25)
    tmp27 = tmp26 + tmp18
    tmp28 = 1 + x1
    tmp29 = tmp28 >= tmp1
    tmp30 = tmp28 < tmp3
    tmp31 = tmp29 & tmp30
    tmp32 = tl.load(in_ptr0 + (ks0 + x2), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp32 * tmp32
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp31, tmp33, tmp34)
    tmp36 = tmp35 + tmp27
    tmp37 = 2 + x1
    tmp38 = tmp37 >= tmp1
    tmp39 = tmp37 < tmp3
    tmp40 = tmp38 & tmp39
    tmp41 = tl.load(in_ptr0 + (x2 + 2*ks2*ks3), tmp40 & xmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tmp41 * tmp41
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp40, tmp42, tmp43)
    tmp45 = tmp44 + tmp36
    tmp46 = 0.2
    tmp47 = tmp45 * tmp46
    tl.store(out_ptr0 + (x2), tmp47, xmask)


# kernel path: /tmp/torchinductor_sahanp/ng/cngsm4buer4nhim34jnqmcfv22rv4caltmoywhmsweh75sqagr5m.py
# Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.mul, aten.add, aten.pow, aten.div, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x => add_28, div, mul_30, pow_1
#   x_1 => constant_pad_nd_1
# Graph fragment:
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, 0.0001), kwargs = {})
#   %add_28 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_30, 1.0), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_28, 0.75), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%arg3_1, %pow_1), kwargs = {})
#   %constant_pad_nd_1 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%div, [2, 2, 2, 2], 1.0), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_constant_pad_nd_div_mul_pow_1(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // ks0) % ks1)
    x0 = (xindex % ks0)
    x2 = xindex // ks4
    x4 = xindex
    tmp0 = (-2) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = ks2
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + x0
    tmp6 = tmp5 >= tmp1
    tmp7 = ks3
    tmp8 = tmp5 < tmp7
    tmp9 = tmp2 & tmp4
    tmp10 = tmp9 & tmp6
    tmp11 = tmp10 & tmp8
    tmp12 = tl.load(in_ptr0 + ((-2) + x0 + ((-2)*ks3) + ks3*x1 + ks2*ks3*x2), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr1 + ((-2) + x0 + ((-2)*ks3) + ks3*x1 + ks2*ks3*x2), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = 0.0001
    tmp15 = tmp13 * tmp14
    tmp16 = 1.0
    tmp17 = tmp15 + tmp16
    tmp18 = 0.75
    tmp19 = libdevice.pow(tmp17, tmp18)
    tmp20 = tmp12 / tmp19
    tmp21 = tl.full(tmp20.shape, 1.0, tmp20.dtype)
    tmp22 = tl.where(tmp11, tmp20, tmp21)
    tl.store(out_ptr0 + (x4), tmp22, xmask)


def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        s1*s2
        buf0 = empty_strided_cuda((1, 1, s0, s1, s2), (s0*s1*s2, s0*s1*s2, s1*s2, s2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.constant_pad_nd, aten.avg_pool3d]
        triton_poi_fused_avg_pool3d_constant_pad_nd_0_xnumel = s0*s1*s2
        get_raw_stream(0)
        triton_poi_fused_avg_pool3d_constant_pad_nd_0[grid(triton_poi_fused_avg_pool3d_constant_pad_nd_0_xnumel)](arg3_1, buf0, 1024, 3, 32, 32, 3072, XBLOCK=128, num_warps=4, num_stages=1)
        4 + s2
        4 + s1
        16 + 4*s1 + 4*s2 + s1*s2
        buf1 = empty_strided_cuda((1, s0, 4 + s1, 4 + s2), (16*s0 + 4*s0*s1 + 4*s0*s2 + s0*s1*s2, 16 + 4*s1 + 4*s2 + s1*s2, 4 + s2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.mul, aten.add, aten.pow, aten.div, aten.constant_pad_nd]
        triton_poi_fused_add_constant_pad_nd_div_mul_pow_1_xnumel = 16*s0 + 4*s0*s1 + 4*s0*s2 + s0*s1*s2
        get_raw_stream(0)
        triton_poi_fused_add_constant_pad_nd_div_mul_pow_1[grid(triton_poi_fused_add_constant_pad_nd_div_mul_pow_1_xnumel)](arg3_1, buf0, buf1, 36, 36, 32, 32, 1296, 3888, XBLOCK=128, num_warps=4, num_stages=1)
        del arg3_1
        del buf0
    return (buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 3
    arg1_1 = 32
    arg2_1 = 32
    arg3_1 = rand_strided((1, 3, 32, 32), (3072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
