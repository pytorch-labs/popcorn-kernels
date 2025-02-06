# AOT ID: ['98_inference']
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
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
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


# kernel path: /tmp/torchinductor_sahanp/7y/c7ynqmzz6dr7vvngb67cdm4dhjrxt2beldgjqibemmjlvimhh2yi.py
# Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.mul, aten.add, aten.pow, aten.div, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x => add_28, div, mul_30, pow_1
#   x_1 => constant_pad_nd_1
# Graph fragment:
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, 0.0001), kwargs = {})
#   %add_28 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_30, 1.0), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_28, 0.75), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%arg3_1, %pow_1), kwargs = {})
#   %constant_pad_nd_1 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%div, [2, 2, 2, 2], 0.0), kwargs = {})
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
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp11, tmp20, tmp21)
    tl.store(out_ptr0 + (x4), tmp22, xmask)


# kernel path: /tmp/torchinductor_sahanp/v5/cv57xcr7yjt6awqha4pl7xrypob7gg6pc4mxo4bzbuf6z76jeolr.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.rand]
# Source node to ATen node mapping:
#   x_2 => inductor_lookup_seed_default, inductor_random_default
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, %arg0_1, 2], %inductor_lookup_seed_default, rand), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_rand_2(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (x0), tmp2, xmask)


# kernel path: /tmp/torchinductor_sahanp/bi/cbi3kzckp3xjy73l7tjk5744ss54caszxc5ibdohmoeqynfdm4vt.py
# Topologically Sorted Source Nodes: [x, x_1, x_2, x_3], Original ATen: [aten.mul, aten.add, aten.pow, aten.div, aten.constant_pad_nd, aten.fractional_max_pool2d, aten.abs, aten.le, aten.scalar_tensor, aten.where]
# Source node to ATen node mapping:
#   x => add_28, div, mul_30, pow_1
#   x_1 => constant_pad_nd_1
#   x_2 => fractional_max_pool2d
#   x_3 => abs_1, full_default, le, where
# Graph fragment:
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, 0.0001), kwargs = {})
#   %add_28 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_30, 1.0), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_28, 0.75), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%arg3_1, %pow_1), kwargs = {})
#   %constant_pad_nd_1 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%div, [2, 2, 2, 2], 0.0), kwargs = {})
#   %fractional_max_pool2d : [num_users=1] = call_function[target=torch.ops.aten.fractional_max_pool2d.default](args = (%constant_pad_nd_1, [2, 2], [14, 14], %inductor_random_default), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%getitem,), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%abs_1, 0.5), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le, %full_default, %getitem), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_abs_add_constant_pad_nd_div_fractional_max_pool2d_le_mul_pow_scalar_tensor_where_3(in_out_ptr0, in_ptr0, in_ptr1, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex // 196
    x1 = ((xindex // 14) % 14)
    x0 = (xindex % 14)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr0 + (1 + 2*x2), xmask, eviction_policy='evict_last')
    tmp1 = (2 + ks0) / 13
    tmp2 = tmp1.to(tl.float32)
    tmp3 = x1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp4 + tmp0
    tmp6 = tmp5 * tmp2
    tmp7 = libdevice.floor(tmp6)
    tmp8 = tmp0 * tmp2
    tmp9 = libdevice.floor(tmp8)
    tmp10 = tmp7 - tmp9
    tmp11 = tmp10.to(tl.int64)
    tmp12 = tl.full([1], 13, tl.int64)
    tmp13 = tmp4 < tmp12
    tmp14 = 2 + ks0
    tmp15 = tl.where(tmp13, tmp11, tmp14)
    tmp16 = ks1
    tmp17 = tmp15 + tmp16
    tmp18 = tmp15 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp15)
    tl.device_assert(((0 <= tmp19) & (tmp19 < 4 + ks0)) | ~(xmask), "index out of bounds: 0 <= tmp19 < 4 + ks0")
    tmp22 = (2 + ks2) / 13
    tmp23 = tmp22.to(tl.float32)
    tmp24 = x0
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp25 + tmp21
    tmp27 = tmp26 * tmp23
    tmp28 = libdevice.floor(tmp27)
    tmp29 = tmp21 * tmp23
    tmp30 = libdevice.floor(tmp29)
    tmp31 = tmp28 - tmp30
    tmp32 = tmp31.to(tl.int64)
    tmp33 = tmp25 < tmp12
    tmp34 = 2 + ks2
    tmp35 = tl.where(tmp33, tmp32, tmp34)
    tmp36 = ks3
    tmp37 = tmp35 + tmp36
    tmp38 = tmp35 < 0
    tmp39 = tl.where(tmp38, tmp37, tmp35)
    tl.device_assert(((0 <= tmp39) & (tmp39 < 4 + ks2)) | ~(xmask), "index out of bounds: 0 <= tmp39 < 4 + ks2")
    tmp41 = tl.load(in_ptr1 + (tmp39 + 4*tmp19 + 16*x2 + ks2*tmp19 + 4*ks0*x2 + 4*ks2*x2 + ks0*ks2*x2), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr1 + (1 + tmp39 + 4*tmp19 + 16*x2 + ks2*tmp19 + 4*ks0*x2 + 4*ks2*x2 + ks0*ks2*x2), xmask, eviction_policy='evict_last')
    tmp43 = triton_helpers.maximum(tmp42, tmp41)
    tmp44 = tl.load(in_ptr1 + (4 + ks2 + tmp39 + 4*tmp19 + 16*x2 + ks2*tmp19 + 4*ks0*x2 + 4*ks2*x2 + ks0*ks2*x2), xmask, eviction_policy='evict_last')
    tmp45 = triton_helpers.maximum(tmp44, tmp43)
    tmp46 = tl.load(in_ptr1 + (5 + ks2 + tmp39 + 4*tmp19 + 16*x2 + ks2*tmp19 + 4*ks0*x2 + 4*ks2*x2 + ks0*ks2*x2), xmask, eviction_policy='evict_last')
    tmp47 = triton_helpers.maximum(tmp46, tmp45)
    tmp48 = tl_math.abs(tmp47)
    tmp49 = 0.5
    tmp50 = tmp48 <= tmp49
    tmp51 = 0.0
    tmp52 = tl.where(tmp50, tmp51, tmp47)
    tl.store(in_out_ptr0 + (x3), tmp52, xmask)


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
        buf1 = empty_strided_cuda((1, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf1)
        4 + s2
        4 + s1
        16 + 4*s1 + 4*s2 + s1*s2
        buf3 = empty_strided_cuda((1, s0, 4 + s1, 4 + s2), (16*s0 + 4*s0*s1 + 4*s0*s2 + s0*s1*s2, 16 + 4*s1 + 4*s2 + s1*s2, 4 + s2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.mul, aten.add, aten.pow, aten.div, aten.constant_pad_nd]
        triton_poi_fused_add_constant_pad_nd_div_mul_pow_1_xnumel = 16*s0 + 4*s0*s1 + 4*s0*s2 + s0*s1*s2
        get_raw_stream(0)
        triton_poi_fused_add_constant_pad_nd_div_mul_pow_1[grid(triton_poi_fused_add_constant_pad_nd_div_mul_pow_1_xnumel)](arg3_1, buf0, buf3, 36, 36, 32, 32, 1296, 3888, XBLOCK=128, num_warps=4, num_stages=1)
        del arg3_1
        del buf0
        buf2 = empty_strided_cuda((1, s0, 2), (2*s0, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.rand]
        triton_poi_fused_rand_2_xnumel = 2*s0
        get_raw_stream(0)
        triton_poi_fused_rand_2[grid(triton_poi_fused_rand_2_xnumel)](buf1, buf2, 0, 6, XBLOCK=8, num_warps=1, num_stages=1)
        del buf1
        buf4 = empty_strided_cuda((1, s0, 14, 14), (196*s0, 196, 14, 1), torch.float32)
        buf5 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [x, x_1, x_2, x_3], Original ATen: [aten.mul, aten.add, aten.pow, aten.div, aten.constant_pad_nd, aten.fractional_max_pool2d, aten.abs, aten.le, aten.scalar_tensor, aten.where]
        triton_poi_fused_abs_add_constant_pad_nd_div_fractional_max_pool2d_le_mul_pow_scalar_tensor_where_3_xnumel = 196*s0
        get_raw_stream(0)
        triton_poi_fused_abs_add_constant_pad_nd_div_fractional_max_pool2d_le_mul_pow_scalar_tensor_where_3[grid(triton_poi_fused_abs_add_constant_pad_nd_div_fractional_max_pool2d_le_mul_pow_scalar_tensor_where_3_xnumel)](buf5, buf2, buf3, 32, 36, 32, 36, 588, XBLOCK=128, num_warps=4, num_stages=1)
        del buf2
        del buf3
    return (reinterpret_tensor(buf5, (1, 1, 1, s0, 14, 14), (196*s0, 196*s0, 196*s0, 196, 14, 1), 0), )


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
