# AOT ID: ['10_inference']
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


# kernel path: /tmp/torchinductor_sahanp/nd/cndotj6z7svzqhfjhassqncmm53tra5jdrzu55sxtalfe5m7crrr.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x => _low_memory_max_pool2d_with_offsets
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%arg3_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_0(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
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
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (x3), tmp6, xmask)


# kernel path: /tmp/torchinductor_sahanp/in/cinjoarjvlviw4wjfqteyzuhfjnta6pijisrs2v45vqmof5zvzhe.py
# Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.copy]
# Source node to ATen node mapping:
#   x_4 => copy
# Graph fragment:
#   %copy : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_1, %slice_2), kwargs = {})
#   %slice_scatter_default : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%empty, %copy, 2, 1, %sub_15), kwargs = {})
#   %slice_scatter_default_1 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default, %slice_7, 2, 0, 1), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_copy_1(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0 + ks0*(ks1 // 4)*(ks2 // 4)
    tmp4 = tl.full([1], 1, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tl.broadcast_to(1 + ks0*(ks1 // 4)*(ks2 // 4), [XBLOCK])
    tmp7 = tmp3 < tmp6
    tmp8 = tmp5 & tmp7
    tmp9 = tmp8 & tmp2
    tmp10 = tl.load(in_ptr0 + (2*((((-1) + x0 + ks0*(ks1 // 4)*(ks2 // 4)) % (ks2 // 4))) + 2*ks3*(((((-1) + x0 + ks0*(ks1 // 4)*(ks2 // 4)) // (ks2 // 4)) % (ks1 // 4))) + ks3*ks4*(((((-1) + x0 + ks0*(ks1 // 4)*(ks2 // 4)) // ((ks1 // 4)*(ks2 // 4))) % ks0))), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr0 + (1 + 2*((((-1) + x0 + ks0*(ks1 // 4)*(ks2 // 4)) % (ks2 // 4))) + 2*ks3*(((((-1) + x0 + ks0*(ks1 // 4)*(ks2 // 4)) // (ks2 // 4)) % (ks1 // 4))) + ks3*ks4*(((((-1) + x0 + ks0*(ks1 // 4)*(ks2 // 4)) // ((ks1 // 4)*(ks2 // 4))) % ks0))), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp13 = tl.load(in_ptr0 + (ks3 + 2*((((-1) + x0 + ks0*(ks1 // 4)*(ks2 // 4)) % (ks2 // 4))) + 2*ks3*(((((-1) + x0 + ks0*(ks1 // 4)*(ks2 // 4)) // (ks2 // 4)) % (ks1 // 4))) + ks3*ks4*(((((-1) + x0 + ks0*(ks1 // 4)*(ks2 // 4)) // ((ks1 // 4)*(ks2 // 4))) % ks0))), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp15 = tl.load(in_ptr0 + (1 + ks3 + 2*((((-1) + x0 + ks0*(ks1 // 4)*(ks2 // 4)) % (ks2 // 4))) + 2*ks3*(((((-1) + x0 + ks0*(ks1 // 4)*(ks2 // 4)) // (ks2 // 4)) % (ks1 // 4))) + ks3*ks4*(((((-1) + x0 + ks0*(ks1 // 4)*(ks2 // 4)) // ((ks1 // 4)*(ks2 // 4))) % ks0))), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp9, tmp16, tmp17)
    tmp19 = float("nan")
    tmp20 = tl.where(tmp8, tmp18, tmp19)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp2, tmp20, tmp21)
    tmp23 = tmp0 >= tmp1
    tmp24 = 1 + ks0*(ks1 // 4)*(ks2 // 4)
    tmp25 = tmp0 < tmp24
    tmp26 = tmp23 & tmp25
    tmp27 = tl.load(in_ptr0 + (2*((((-1) + x0) % (ks2 // 4))) + 2*ks3*(((((-1) + x0) // (ks2 // 4)) % (ks1 // 4))) + ks3*ks4*(((((-1) + x0) // ((ks1 // 4)*(ks2 // 4))) % ks0))), tmp26 & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr0 + (1 + 2*((((-1) + x0) % (ks2 // 4))) + 2*ks3*(((((-1) + x0) // (ks2 // 4)) % (ks1 // 4))) + ks3*ks4*(((((-1) + x0) // ((ks1 // 4)*(ks2 // 4))) % ks0))), tmp26 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = triton_helpers.maximum(tmp28, tmp27)
    tmp30 = tl.load(in_ptr0 + (ks3 + 2*((((-1) + x0) % (ks2 // 4))) + 2*ks3*(((((-1) + x0) // (ks2 // 4)) % (ks1 // 4))) + ks3*ks4*(((((-1) + x0) // ((ks1 // 4)*(ks2 // 4))) % ks0))), tmp26 & xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tmp32 = tl.load(in_ptr0 + (1 + ks3 + 2*((((-1) + x0) % (ks2 // 4))) + 2*ks3*(((((-1) + x0) // (ks2 // 4)) % (ks1 // 4))) + ks3*ks4*(((((-1) + x0) // ((ks1 // 4)*(ks2 // 4))) % ks0))), tmp26 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = triton_helpers.maximum(tmp32, tmp31)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp26, tmp33, tmp34)
    tmp36 = float("nan")
    tmp37 = tl.where(tmp26, tmp35, tmp36)
    tmp38 = tl.where(tmp2, tmp22, tmp37)
    tl.store(out_ptr0 + (x0), tmp38, xmask)


# kernel path: /tmp/torchinductor_sahanp/qa/cqatyu6yb55skw5yfzjn5xgahfa3yq4hxsweg4fusq5tuicatn22.py
# Topologically Sorted Source Nodes: [x_5, x_10, y], Original ATen: [aten.copy, aten.linalg_vector_norm, aten.clamp_min, aten.div, aten.ones_like, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   x_10 => clamp_min, clamp_min_1, div, div_1, mul_117, pow_1, pow_2, pow_3, pow_4, sum_1, sum_2, sum_3
#   x_5 => copy_3
#   y => full
# Graph fragment:
#   %copy_3 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_14, %slice_16), kwargs = {})
#   %slice_scatter_default_3 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%empty_1, %copy_3, 2, 1, %sub_27), kwargs = {})
#   %slice_scatter_default_4 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_3, %slice_21, 2, 0, 1), kwargs = {})
#   %slice_scatter_default_5 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_4, %slice_26, 2, %sub_37, %add_38), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%view_1, 2), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1], True), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_2, 1e-08), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_1, %clamp_min), kwargs = {})
#   %full : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([1, %add_38, 1], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%full, 2), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_3, [1], True), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_2, 0.5), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_4, 1e-08), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%full, %clamp_min_1), kwargs = {})
#   %mul_117 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %div), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_117, [1]), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_clamp_min_copy_div_linalg_vector_norm_mul_ones_like_sum_2(in_out_ptr0, in_ptr0, out_ptr0, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    _tmp86 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp97 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_0 = r0_index
        tmp0 = r0_0
        tmp1 = 3 + ks0*(ks1 // 4)*(ks2 // 4)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.broadcast_to((-2) + r0_0 + ((-1)*ks0*(ks1 // 4)*(ks2 // 4)), [XBLOCK, R0_BLOCK])
        tmp4 = tl.full([1, 1], 1, tl.int64)
        tmp5 = tmp3 < tmp4
        tmp6 = tmp5 & tmp2
        tmp7 = tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])
        tmp8 = tl.full([1, 1], 1, tl.int64)
        tmp9 = tmp7 >= tmp8
        tmp10 = tl.broadcast_to(3 + ks0*(ks1 // 4)*(ks2 // 4), [XBLOCK, R0_BLOCK])
        tmp11 = tmp7 < tmp10
        tmp12 = tmp9 & tmp11
        tmp13 = tmp12 & tmp6
        tmp14 = tl.broadcast_to((-1) + r0_0, [XBLOCK, R0_BLOCK])
        tmp15 = tl.broadcast_to(1 + ks0*(ks1 // 4)*(ks2 // 4), [XBLOCK, R0_BLOCK])
        tmp16 = tmp14 >= tmp15
        tmp17 = tmp16 & tmp13
        tmp18 = tl.load(in_ptr0 + (tl.full([XBLOCK, R0_BLOCK], 1, tl.int32)), tmp17, eviction_policy='evict_last', other=0.0)
        tmp19 = tl.load(in_ptr0 + (tl.broadcast_to((-1) + r0_0, [XBLOCK, R0_BLOCK])), r0_mask & tmp13, eviction_policy='evict_last', other=0.0)
        tmp20 = tl.where(tmp16, tmp18, tmp19)
        tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
        tmp22 = tl.where(tmp13, tmp20, tmp21)
        tmp23 = float("nan")
        tmp24 = tl.where(tmp12, tmp22, tmp23)
        tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
        tmp26 = tl.where(tmp6, tmp24, tmp25)
        tmp27 = tmp3 >= tmp4
        tmp28 = tl.broadcast_to(3 + ks0*(ks1 // 4)*(ks2 // 4), [XBLOCK, R0_BLOCK])
        tmp29 = tmp3 < tmp28
        tmp30 = tmp27 & tmp29
        tmp31 = tmp30 & tmp2
        tmp32 = tl.broadcast_to((-3) + r0_0 + ((-1)*ks0*(ks1 // 4)*(ks2 // 4)), [XBLOCK, R0_BLOCK])
        tmp33 = tl.broadcast_to(1 + ks0*(ks1 // 4)*(ks2 // 4), [XBLOCK, R0_BLOCK])
        tmp34 = tmp32 >= tmp33
        tmp35 = tmp34 & tmp31
        tmp36 = tl.load(in_ptr0 + (tl.full([XBLOCK, R0_BLOCK], 1, tl.int32)), tmp35, eviction_policy='evict_last', other=0.0)
        tmp37 = tl.load(in_ptr0 + (tl.broadcast_to((-3) + r0_0 + ((-1)*ks0*(ks1 // 4)*(ks2 // 4)), [XBLOCK, R0_BLOCK])), r0_mask & tmp31, eviction_policy='evict_last', other=0.0)
        tmp38 = tl.where(tmp34, tmp36, tmp37)
        tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
        tmp40 = tl.where(tmp31, tmp38, tmp39)
        tmp41 = float("nan")
        tmp42 = tl.where(tmp30, tmp40, tmp41)
        tmp43 = tl.where(tmp5, tmp26, tmp42)
        tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
        tmp45 = tl.where(tmp2, tmp43, tmp44)
        tmp46 = tl.full([1, 1], 1, tl.int64)
        tmp47 = tmp0 < tmp46
        tmp48 = tl.broadcast_to(2 + r0_0 + ks0*(ks1 // 4)*(ks2 // 4), [XBLOCK, R0_BLOCK])
        tmp49 = tl.full([1, 1], 1, tl.int64)
        tmp50 = tmp48 >= tmp49
        tmp51 = tl.broadcast_to(3 + ks0*(ks1 // 4)*(ks2 // 4), [XBLOCK, R0_BLOCK])
        tmp52 = tmp48 < tmp51
        tmp53 = tmp50 & tmp52
        tmp54 = tmp53 & tmp47
        tmp55 = tl.broadcast_to(1 + r0_0 + ks0*(ks1 // 4)*(ks2 // 4), [XBLOCK, R0_BLOCK])
        tmp56 = tl.broadcast_to(1 + ks0*(ks1 // 4)*(ks2 // 4), [XBLOCK, R0_BLOCK])
        tmp57 = tmp55 >= tmp56
        tmp58 = tmp57 & tmp54
        tmp59 = tl.load(in_ptr0 + (tl.full([XBLOCK, R0_BLOCK], 1, tl.int32)), tmp58, eviction_policy='evict_last', other=0.0)
        tmp60 = tl.load(in_ptr0 + (tl.broadcast_to(1 + r0_0 + ks0*(ks1 // 4)*(ks2 // 4), [XBLOCK, R0_BLOCK])), r0_mask & tmp54, eviction_policy='evict_last', other=0.0)
        tmp61 = tl.where(tmp57, tmp59, tmp60)
        tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
        tmp63 = tl.where(tmp54, tmp61, tmp62)
        tmp64 = float("nan")
        tmp65 = tl.where(tmp53, tmp63, tmp64)
        tmp66 = tl.full(tmp65.shape, 0.0, tmp65.dtype)
        tmp67 = tl.where(tmp47, tmp65, tmp66)
        tmp68 = tmp0 >= tmp46
        tmp69 = tmp0 < tmp1
        tmp70 = tmp68 & tmp69
        tmp71 = tl.broadcast_to((-1) + r0_0, [XBLOCK, R0_BLOCK])
        tmp72 = tl.broadcast_to(1 + ks0*(ks1 // 4)*(ks2 // 4), [XBLOCK, R0_BLOCK])
        tmp73 = tmp71 >= tmp72
        tmp74 = tmp73 & tmp70
        tmp75 = tl.load(in_ptr0 + (tl.full([XBLOCK, R0_BLOCK], 1, tl.int32)), tmp74, eviction_policy='evict_last', other=0.0)
        tmp76 = tl.load(in_ptr0 + (tl.broadcast_to((-1) + r0_0, [XBLOCK, R0_BLOCK])), r0_mask & tmp70, eviction_policy='evict_first', other=0.0)
        tmp77 = tl.where(tmp73, tmp75, tmp76)
        tmp78 = tl.full(tmp77.shape, 0.0, tmp77.dtype)
        tmp79 = tl.where(tmp70, tmp77, tmp78)
        tmp80 = float("nan")
        tmp81 = tl.where(tmp70, tmp79, tmp80)
        tmp82 = tl.where(tmp47, tmp67, tmp81)
        tmp83 = tl.where(tmp2, tmp45, tmp82)
        tmp84 = 1.0
        tmp85 = tl.broadcast_to(tmp84, [XBLOCK, R0_BLOCK])
        tmp87 = _tmp86 + tmp85
        _tmp86 = tl.where(r0_mask, tmp87, _tmp86)
        tmp88 = 0.0
        tmp89 = tmp83 > tmp88
        tmp90 = libdevice.expm1(tmp83)
        tmp91 = tl.where(tmp89, tmp83, tmp90)
        tmp92 = tmp91 > tmp88
        tmp93 = libdevice.expm1(tmp91)
        tmp94 = tl.where(tmp92, tmp91, tmp93)
        tmp95 = tmp94 * tmp94
        tmp96 = tl.broadcast_to(tmp95, [XBLOCK, R0_BLOCK])
        tmp98 = _tmp97 + tmp96
        _tmp97 = tl.where(r0_mask, tmp98, _tmp97)
        tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp83, r0_mask)
    tmp86 = tl.sum(_tmp86, 1)[:, None]
    tmp97 = tl.sum(_tmp97, 1)[:, None]
    _tmp117 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_0 = r0_index
        tmp99 = tl.load(out_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp100 = 0.0
        tmp101 = tmp99 > tmp100
        tmp102 = libdevice.expm1(tmp99)
        tmp103 = tl.where(tmp101, tmp99, tmp102)
        tmp104 = tmp103 > tmp100
        tmp105 = libdevice.expm1(tmp103)
        tmp106 = tl.where(tmp104, tmp103, tmp105)
        tmp107 = libdevice.sqrt(tmp97)
        tmp108 = 1e-08
        tmp109 = triton_helpers.maximum(tmp107, tmp108)
        tmp110 = tmp106 / tmp109
        tmp111 = libdevice.sqrt(tmp86)
        tmp112 = triton_helpers.maximum(tmp111, tmp108)
        tmp113 = 1.0
        tmp114 = tmp113 / tmp112
        tmp115 = tmp110 * tmp114
        tmp116 = tl.broadcast_to(tmp115, [XBLOCK, R0_BLOCK])
        tmp118 = _tmp117 + tmp116
        _tmp117 = tl.where(r0_mask, tmp118, _tmp117)
    tmp117 = tl.sum(_tmp117, 1)[:, None]
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp117, None)


def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        s2 // 2
        s1 // 2
        (s1 // 2)*(s2 // 2)
        buf0 = empty_strided_cuda((1, s0, s1 // 2, s2 // 2), (s0*(s1 // 2)*(s2 // 2), (s1 // 2)*(s2 // 2), s2 // 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_0_xnumel = s0*(s1 // 2)*(s2 // 2)
        get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_0[grid(triton_poi_fused_max_pool2d_with_indices_0_xnumel)](arg3_1, buf0, 32, 32, 1024, 64, 64, 3072, XBLOCK=128, num_warps=4, num_stages=1)
        del arg3_1
        buf2 = empty_strided_cuda((1, 1, 2 + s0*(s1 // 4)*(s2 // 4)), (2 + s0*(s1 // 4)*(s2 // 4), 2 + s0*(s1 // 4)*(s2 // 4), 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.copy]
        triton_poi_fused_copy_1_xnumel = 2 + s0*(s1 // 4)*(s2 // 4)
        get_raw_stream(0)
        triton_poi_fused_copy_1[grid(triton_poi_fused_copy_1_xnumel)](buf0, buf2, 3, 64, 64, 32, 32, 770, XBLOCK=128, num_warps=4, num_stages=1)
        del buf0
        buf4 = empty_strided_cuda((1, 1, 4 + s0*(s1 // 4)*(s2 // 4)), (4 + s0*(s1 // 4)*(s2 // 4), 4 + s0*(s1 // 4)*(s2 // 4), 1), torch.float32)
        buf5 = empty_strided_cuda((1, 1, 1), (1, 1, 1), torch.float32)
        buf7 = reinterpret_tensor(buf5, (1, 1), (1, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [x_5, x_10, y], Original ATen: [aten.copy, aten.linalg_vector_norm, aten.clamp_min, aten.div, aten.ones_like, aten.mul, aten.sum]
        4 + s0*(s1 // 4)*(s2 // 4)
        get_raw_stream(0)
        triton_red_fused_clamp_min_copy_div_linalg_vector_norm_mul_ones_like_sum_2[grid(1)](buf7, buf2, buf4, 3, 64, 64, 1, 772, XBLOCK=1, R0_BLOCK=1024, num_warps=8, num_stages=1)
        del buf2
        del buf4
    return (buf7, )


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
