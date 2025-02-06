# AOT ID: ['12_inference']
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


# kernel path: /tmp/torchinductor_sahanp/bk/cbkpzr5t4izc7ujrmnqer72jnyip5v5kh7gu24jcbgrwnw6psaks.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_1 => constant_pad_nd
# Graph fragment:
#   %constant_pad_nd : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view, [0, 0, 0, 0, 2, 2], 0.0), kwargs = {})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_constant_pad_nd_0(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex // ks0
    x0 = (xindex % ks2)
    x1 = ((xindex // ks2) % ks3)
    x4 = xindex
    tmp0 = (-2) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = ks1
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (2*x0 + ((-2)*ks4*ks5) + 2*ks5*x1 + ks4*ks5*x2), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr0 + (1 + 2*x0 + ((-2)*ks4*ks5) + 2*ks5*x1 + ks4*ks5*x2), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp9 = tl.load(in_ptr0 + (ks5 + 2*x0 + ((-2)*ks4*ks5) + 2*ks5*x1 + ks4*ks5*x2), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp11 = tl.load(in_ptr0 + (1 + ks5 + 2*x0 + ((-2)*ks4*ks5) + 2*ks5*x1 + ks4*ks5*x2), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp13 = tmp12 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp5, tmp13, tmp14)
    tl.store(out_ptr0 + (x4), tmp15, xmask)


# kernel path: /tmp/torchinductor_sahanp/qs/cqsfabosn3sblvb3fnw7dyys7cqhtucs3exklhr7cvexndvqwxm5.py
# Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.max_pool2d_with_indices, aten.mul, aten.add, aten.pow, aten.div]
# Source node to ATen node mapping:
#   x => _low_memory_max_pool2d_with_offsets
#   x_1 => add_36, div, mul_38, pow_1
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%arg3_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, 0.0001), kwargs = {})
#   %add_36 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_38, 1.0), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_36, 0.75), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%getitem, %pow_1), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_div_max_pool2d_with_indices_mul_pow_1(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks1)
    x2 = xindex // ks2
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 2*ks4*x1 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 2*ks4*x1 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (ks4 + 2*x0 + 2*ks4*x1 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (1 + ks4 + 2*x0 + 2*ks4*x1 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (ks2 + x3), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr1 + (x3 + 2*ks0*ks1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (x3 + 3*ks0*ks1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr1 + (x3 + 4*ks0*ks1), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp9 = tmp8 + tmp7
    tmp11 = tmp10 + tmp9
    tmp13 = tmp12 + tmp11
    tmp15 = tmp14 + tmp13
    tmp16 = 0.2
    tmp17 = tmp15 * tmp16
    tmp18 = 0.0001
    tmp19 = tmp17 * tmp18
    tmp20 = 1.0
    tmp21 = tmp19 + tmp20
    tmp22 = 0.75
    tmp23 = libdevice.pow(tmp21, tmp22)
    tmp24 = tmp6 / tmp23
    tl.store(out_ptr0 + (x3), tmp24, xmask)


# kernel path: /tmp/torchinductor_sahanp/fc/cfcft5bijcecjktgivnmsaovr4t5ui5kn4ovnvebn6b64wvvyykv.py
# Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.copy]
# Source node to ATen node mapping:
#   x_4 => copy
# Graph fragment:
#   %copy : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_5, %slice_6), kwargs = {})
#   %slice_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_tensor_1, %copy, 2, 1, 2), kwargs = {})
#   %slice_scatter_default_1 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_tensor, %slice_scatter_default, 3, 1, 6), kwargs = {})
#   %slice_scatter_default_2 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%empty, %slice_scatter_default_1, 4, 1, 6), kwargs = {})
#   %slice_scatter_default_3 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_2, %slice_15, 4, 0, 1), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_copy_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 7)
    x1 = ((xindex // 7) % 7)
    x2 = ((xindex // 49) % 3)
    x3 = xindex // 147
    x5 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = 5 + x0
    tmp4 = tl.full([1], 1, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tl.full([1], 6, tl.int64)
    tmp7 = tmp3 < tmp6
    tmp8 = tmp5 & tmp7
    tmp9 = tmp8 & tmp2
    tmp10 = x1
    tmp11 = tl.full([1], 1, tl.int64)
    tmp12 = tmp10 >= tmp11
    tmp13 = tl.full([1], 6, tl.int64)
    tmp14 = tmp10 < tmp13
    tmp15 = tmp12 & tmp14
    tmp16 = tmp15 & tmp9
    tmp17 = x2
    tmp18 = tl.full([1], 1, tl.int64)
    tmp19 = tmp17 >= tmp18
    tmp20 = tl.full([1], 2, tl.int64)
    tmp21 = tmp17 < tmp20
    tmp22 = tmp19 & tmp21
    tmp23 = tmp22 & tmp16
    tmp24 = tl.load(in_ptr0 + ((-1) + x0 + 5*x1 + 25*x3), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tl.load(in_ptr1 + (5 + x5), tmp16 & xmask, other=0.0)
    tmp26 = tl.where(tmp22, tmp24, tmp25)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp16, tmp26, tmp27)
    tmp29 = tl.load(in_ptr1 + (5 + x5), tmp9 & xmask, other=0.0)
    tmp30 = tl.where(tmp15, tmp28, tmp29)
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp9, tmp30, tmp31)
    tmp33 = float("nan")
    tmp34 = tl.where(tmp8, tmp32, tmp33)
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp2, tmp34, tmp35)
    tmp37 = tmp0 >= tmp1
    tmp38 = tl.full([1], 6, tl.int64)
    tmp39 = tmp0 < tmp38
    tmp40 = tmp37 & tmp39
    tmp41 = x1
    tmp42 = tl.full([1], 1, tl.int64)
    tmp43 = tmp41 >= tmp42
    tmp44 = tl.full([1], 6, tl.int64)
    tmp45 = tmp41 < tmp44
    tmp46 = tmp43 & tmp45
    tmp47 = tmp46 & tmp40
    tmp48 = x2
    tmp49 = tl.full([1], 1, tl.int64)
    tmp50 = tmp48 >= tmp49
    tmp51 = tl.full([1], 2, tl.int64)
    tmp52 = tmp48 < tmp51
    tmp53 = tmp50 & tmp52
    tmp54 = tmp53 & tmp47
    tmp55 = tl.load(in_ptr0 + ((-6) + x0 + 5*x1 + 25*x3), tmp54 & xmask, eviction_policy='evict_last', other=0.0)
    tmp56 = tl.load(in_ptr1 + (x5), tmp47 & xmask, other=0.0)
    tmp57 = tl.where(tmp53, tmp55, tmp56)
    tmp58 = tl.full(tmp57.shape, 0.0, tmp57.dtype)
    tmp59 = tl.where(tmp47, tmp57, tmp58)
    tmp60 = tl.load(in_ptr1 + (x5), tmp40 & xmask, other=0.0)
    tmp61 = tl.where(tmp46, tmp59, tmp60)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp40, tmp61, tmp62)
    tmp64 = float("nan")
    tmp65 = tl.where(tmp40, tmp63, tmp64)
    tmp66 = tl.where(tmp2, tmp36, tmp65)
    tl.store(out_ptr0 + (x5), tmp66, xmask)


# kernel path: /tmp/torchinductor_sahanp/t7/ct72nstz4lo34qsnlzqbf5zpdc5bcssn77meukuqos7h6jjluuuh.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %slice_scatter_default_4 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_3, %slice_20, 4, 6, 7), kwargs = {})
#   %slice_scatter_default_5 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_4, %slice_25, 3, 0, 1), kwargs = {})
#   %slice_scatter_default_6 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_5, %slice_30, 3, 6, 7), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 7) % 7)
    x0 = (xindex % 7)
    x3 = xindex // 7
    x4 = xindex
    tmp40 = tl.load(in_ptr0 + (x4), xmask)
    tmp0 = x1
    tmp1 = tl.full([1], 6, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = (-5) + x1
    tmp4 = tl.full([1], 1, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = x0
    tmp8 = tl.full([1], 6, tl.int64)
    tmp9 = tmp7 >= tmp8
    tmp10 = tmp9 & tmp6
    tmp11 = tl.load(in_ptr0 + (1 + 7*x3), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr0 + (x4), tmp6 & xmask, other=0.0)
    tmp13 = tl.where(tmp9, tmp11, tmp12)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = x0
    tmp17 = tl.full([1], 6, tl.int64)
    tmp18 = tmp16 >= tmp17
    tmp19 = tmp18 & tmp2
    tmp20 = tl.load(in_ptr0 + ((-34) + 7*x3), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.load(in_ptr0 + ((-35) + x4), tmp2 & xmask, other=0.0)
    tmp22 = tl.where(tmp18, tmp20, tmp21)
    tmp23 = tl.where(tmp5, tmp15, tmp22)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp2, tmp23, tmp24)
    tmp26 = tl.full([1], 1, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = x0
    tmp29 = tl.full([1], 6, tl.int64)
    tmp30 = tmp28 >= tmp29
    tmp31 = tmp30 & tmp27
    tmp32 = tl.load(in_ptr0 + (36 + 7*x3), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tl.load(in_ptr0 + (35 + x4), tmp27 & xmask, other=0.0)
    tmp34 = tl.where(tmp30, tmp32, tmp33)
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp27, tmp34, tmp35)
    tmp37 = x0
    tmp38 = tmp37 >= tmp1
    tmp39 = tl.load(in_ptr0 + (1 + 7*x3), tmp38 & xmask, eviction_policy='evict_last', other=0.0)
    tmp41 = tl.where(tmp38, tmp39, tmp40)
    tmp42 = tl.where(tmp27, tmp36, tmp41)
    tmp43 = tl.where(tmp2, tmp25, tmp42)
    tl.store(out_ptr0 + (x4), tmp43, xmask)


# kernel path: /tmp/torchinductor_sahanp/he/chea34a55vt76mwzovm6xlpq2ulwwvcis3whbrbkkrzy75l7a67e.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %slice_scatter_default_7 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_6, %slice_35, 2, 0, 1), kwargs = {})
#   %slice_scatter_default_8 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_7, %slice_40, 2, 2, 3), kwargs = {})
#   %squeeze_2 : [num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%slice_scatter_default_8, 2), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 49) % 3)
    x0 = (xindex % 49)
    x2 = xindex // 147
    x3 = xindex
    tmp15 = tl.load(in_ptr0 + (x3), xmask)
    tmp0 = x1
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = (-1) + x1
    tmp4 = tl.full([1], 1, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + (49 + x0 + 147*x2), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr0 + ((-49) + x3), tmp2 & xmask, other=0.0)
    tmp9 = tl.where(tmp5, tmp7, tmp8)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = tl.full([1], 1, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr0 + (49 + x0 + 147*x2), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.where(tmp13, tmp14, tmp15)
    tmp17 = tl.where(tmp2, tmp11, tmp16)
    tl.store(out_ptr0 + (x3), tmp17, xmask)


def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        (s1 // 2)*(s2 // 2)
        s2 // 2
        s1 // 2
        buf0 = empty_strided_cuda((1, 1, 4 + s0, s1 // 2, s2 // 2), (4*(s1 // 2)*(s2 // 2) + s0*(s1 // 2)*(s2 // 2), 4*(s1 // 2)*(s2 // 2) + s0*(s1 // 2)*(s2 // 2), (s1 // 2)*(s2 // 2), s2 // 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_0_xnumel = 4*(s1 // 2)*(s2 // 2) + s0*(s1 // 2)*(s2 // 2)
        get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_0[grid(triton_poi_fused_constant_pad_nd_0_xnumel)](arg3_1, buf0, 1024, 3, 32, 32, 64, 64, 7168, XBLOCK=256, num_warps=4, num_stages=1)
        buf1 = empty_strided_cuda((1, s0, s1 // 2, s2 // 2), (s0*(s1 // 2)*(s2 // 2), (s1 // 2)*(s2 // 2), s2 // 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.max_pool2d_with_indices, aten.mul, aten.add, aten.pow, aten.div]
        triton_poi_fused_add_div_max_pool2d_with_indices_mul_pow_1_xnumel = s0*(s1 // 2)*(s2 // 2)
        get_raw_stream(0)
        triton_poi_fused_add_div_max_pool2d_with_indices_mul_pow_1[grid(triton_poi_fused_add_div_max_pool2d_with_indices_mul_pow_1_xnumel)](arg3_1, buf0, buf1, 32, 32, 1024, 64, 64, 3072, XBLOCK=128, num_warps=4, num_stages=1)
        del arg3_1
        del buf0
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.adaptive_max_pool2d]
        buf2 = torch.ops.aten.adaptive_max_pool2d.default(buf1, [5, 5])
        del buf1
        buf3 = buf2[0]
        del buf2
        buf5 = empty_strided_cuda((1, s0, 3, 7, 7), (147*s0, 147, 49, 7, 1), torch.float32)
        buf6 = empty_strided_cuda((1, s0, 3, 7, 7), (147*s0, 147, 49, 7, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.copy]
        triton_poi_fused_copy_2_xnumel = 147*s0
        get_raw_stream(0)
        triton_poi_fused_copy_2[grid(triton_poi_fused_copy_2_xnumel)](buf3, buf5, buf6, 441, XBLOCK=128, num_warps=4, num_stages=1)
        del buf3
        buf7 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_3_xnumel = 147*s0
        get_raw_stream(0)
        triton_poi_fused_3[grid(triton_poi_fused_3_xnumel)](buf6, buf7, 441, XBLOCK=128, num_warps=4, num_stages=1)
        buf8 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_4_xnumel = 147*s0
        get_raw_stream(0)
        triton_poi_fused_4[grid(triton_poi_fused_4_xnumel)](buf7, buf8, 441, XBLOCK=128, num_warps=4, num_stages=1)
        del buf7
    return (buf8, )


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
