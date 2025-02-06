# AOT ID: ['112_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import (
    grid,
    split_scan_grid,
    grid_combo_kernels,
    start_graph,
    end_graph,
    cooperative_reduction_grid,
)
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
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


# kernel path: /tmp/torchinductor_sahanp/ka/ckamaim7h3tt6o2pt3vumd42fcizka6autx7ntlpgjwstmhqra3p.py
# Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.constant_pad_nd, aten.mean]
# Source node to ATen node mapping:
#   x => constant_pad_nd
#   x_1 => mean
# Graph fragment:
#   %constant_pad_nd : [num_users=3] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%arg4_1, [1, 1, 1, 1, 1, 1], 0.0), kwargs = {})
#   %mean : [num_users=4] = call_function[target=torch.ops.aten.mean.dim](args = (%constant_pad_nd, [2]), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_constant_pad_nd_mean_0(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x1 = ((xindex // ks1) % ks2)
    x0 = (xindex % ks1)
    x2 = xindex // ks5
    _tmp20 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x4 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        tmp0 = (-1) + r0_3
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = ks0
        tmp4 = tmp0 < tmp3
        tmp5 = (-1) + x1
        tmp6 = tmp5 >= tmp1
        tmp7 = ks3
        tmp8 = tmp5 < tmp7
        tmp9 = (-1) + x0
        tmp10 = tmp9 >= tmp1
        tmp11 = ks4
        tmp12 = tmp9 < tmp11
        tmp13 = tmp2 & tmp4
        tmp14 = tmp13 & tmp6
        tmp15 = tmp14 & tmp8
        tmp16 = tmp15 & tmp10
        tmp17 = tmp16 & tmp12
        tmp18 = tl.load(in_ptr0 + ((-1) + x0 + ((-1)*ks4) + ks4*x1 + ((-1)*ks3*ks4) + ks3*ks4*r0_3 + ks0*ks3*ks4*x2), r0_mask & tmp17 & xmask, eviction_policy='evict_last', other=0.0)
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(r0_mask & xmask, tmp21, _tmp20)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp20, xmask)




# kernel path: /tmp/torchinductor_sahanp/bc/cbc5kdcj4me3teeq6uhsvqfs7ptau73k5unngqrlbp7spovuxypj.py
# Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.constant_pad_nd, aten.mean, aten._unsafe_index, aten.sub]
# Source node to ATen node mapping:
#   x => constant_pad_nd
#   x_1 => mean
#   x_2 => _unsafe_index, _unsafe_index_1, _unsafe_index_2, _unsafe_index_3, sub_48, sub_58
# Graph fragment:
#   %constant_pad_nd : [num_users=3] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%arg4_1, [1, 1, 1, 1, 1, 1], 0.0), kwargs = {})
#   %mean : [num_users=4] = call_function[target=torch.ops.aten.mean.dim](args = (%constant_pad_nd, [2]), kwargs = {})
#   %_unsafe_index_3 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%mean, [None, None, %clamp_max, %clamp_max_1]), kwargs = {})
#   %_unsafe_index_2 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%mean, [None, None, %clamp_max, %convert_element_type_3]), kwargs = {})
#   %sub_58 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_3, %_unsafe_index_2), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%mean, [None, None, %convert_element_type_1, %clamp_max_1]), kwargs = {})
#   %_unsafe_index : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%mean, [None, None, %convert_element_type_1, %convert_element_type_3]), kwargs = {})
#   %sub_48 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_1, %_unsafe_index), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__unsafe_index_constant_pad_nd_mean_sub_1(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, ks0, ks1, ks2, ks3, ks4, ks5, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // ks1) % ks2)
    x0 = (xindex % ks1)
    x6 = xindex // ks4
    x3 = xindex
    tmp0 = 2.0
    tmp1 = ks0
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float64)
    tmp5 = tl.full([1], -1.0, tl.float64)
    tmp6 = tmp5 + tmp4
    tmp7 = tmp0 * tmp2
    tmp8 = 4.0
    tmp9 = tmp8 + tmp7
    tmp10 = tmp9.to(tl.float64)
    tmp11 = tmp5 + tmp10
    tmp12 = tmp6 / tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp14 = x1
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp15 * tmp13
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = tmp18.to(tl.int64)
    tmp20 = tl.full([1], 1, tl.int64)
    tmp21 = tmp19 + tmp20
    tmp22 = 1 + ks0
    tmp23 = triton_helpers.minimum(tmp21, tmp22)
    tmp24 = ks3
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp0 + tmp25
    tmp27 = tmp26.to(tl.float64)
    tmp28 = tmp5 + tmp27
    tmp29 = tmp0 * tmp25
    tmp30 = tmp8 + tmp29
    tmp31 = tmp30.to(tl.float64)
    tmp32 = tmp5 + tmp31
    tmp33 = tmp28 / tmp32
    tmp34 = tmp33.to(tl.float32)
    tmp35 = x0
    tmp36 = tmp35.to(tl.float32)
    tmp37 = tmp36 * tmp34
    tmp38 = triton_helpers.maximum(tmp37, tmp17)
    tmp39 = tmp38.to(tl.int64)
    tmp40 = tl.load(in_ptr0 + (tmp39 + 2*tmp23 + 4*x6 + ks3*tmp23 + 2*ks0*x6 + 2*ks3*x6 + ks0*ks3*x6), xmask, eviction_policy='evict_last')
    tmp41 = 2 + ks5
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tmp40 / tmp42
    tmp44 = tmp39 + tmp20
    tmp45 = 1 + ks3
    tmp46 = triton_helpers.minimum(tmp44, tmp45)
    tmp47 = tl.load(in_ptr0 + (tmp46 + 2*tmp23 + 4*x6 + ks3*tmp23 + 2*ks0*x6 + 2*ks3*x6 + ks0*ks3*x6), xmask, eviction_policy='evict_last')
    tmp48 = tmp47 / tmp42
    tmp49 = tmp48 - tmp43
    tmp50 = tl.load(in_ptr0 + (tmp39 + 2*tmp19 + 4*x6 + ks3*tmp19 + 2*ks0*x6 + 2*ks3*x6 + ks0*ks3*x6), xmask, eviction_policy='evict_last')
    tmp51 = tmp50 / tmp42
    tmp52 = tl.load(in_ptr0 + (tmp46 + 2*tmp19 + 4*x6 + ks3*tmp19 + 2*ks0*x6 + 2*ks3*x6 + ks0*ks3*x6), xmask, eviction_policy='evict_last')
    tmp53 = tmp52 / tmp42
    tmp54 = tmp53 - tmp51
    tl.store(out_ptr0 + (x3), tmp43, xmask)
    tl.store(out_ptr1 + (x3), tmp49, xmask)
    tl.store(out_ptr2 + (x3), tmp51, xmask)
    tl.store(out_ptr3 + (x3), tmp54, xmask)




# kernel path: /tmp/torchinductor_sahanp/wu/cwuboegqtf425tjyxaymu2h36ggxzjk5tkgwyui6d7klac3glae4.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.arange, aten._to_copy, aten.clamp, aten.view, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   x_2 => add_77, add_90, clamp_max_2, clamp_min_1, clamp_min_2, convert_element_type_2, convert_element_type_3, iota_1, mul_51, mul_64, sub_45, sub_71, view_1
# Graph fragment:
#   %iota_1 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (%floordiv_1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_1, torch.float32), kwargs = {})
#   %full_default_5 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -1.0), kwargs = {dtype: torch.float64, layout: torch.strided, device: cpu, pin_memory: False})
#   %full_default_6 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 2), kwargs = {dtype: torch.int64, layout: torch.strided, device: cpu, pin_memory: False})
#   %scalar_tensor_default_8 : [num_users=2] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (%arg3_1,), kwargs = {})
#   %add_tensor_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%full_default_6, %scalar_tensor_default_8), kwargs = {})
#   %convert_element_type_default_3 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor_4, torch.float64), kwargs = {})
#   %add_tensor_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%full_default_5, %convert_element_type_default_3), kwargs = {})
#   %full_default_7 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -1.0), kwargs = {dtype: torch.float64, layout: torch.strided, device: cpu, pin_memory: False})
#   %full_default_8 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 4), kwargs = {dtype: torch.int64, layout: torch.strided, device: cpu, pin_memory: False})
#   %full_default_9 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 2), kwargs = {dtype: torch.int64, layout: torch.strided, device: cpu, pin_memory: False})
#   %mul_tensor_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_9, %scalar_tensor_default_8), kwargs = {})
#   %add_tensor_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%full_default_8, %mul_tensor_2), kwargs = {})
#   %convert_element_type_default_4 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor_6, torch.float64), kwargs = {})
#   %add_tensor_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%full_default_7, %convert_element_type_default_4), kwargs = {})
#   %true_divide_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.true_divide.Tensor](args = (%add_tensor_5, %add_tensor_7), kwargs = {})
#   %convert_element_type_default_5 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%true_divide_tensor_1, torch.float32), kwargs = {})
#   %mul_tensor_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_2, %convert_element_type_default_5), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_tensor_3, 0.0), kwargs = {})
#   %view_1 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clamp_min_1, [%floordiv_1]), kwargs = {})
#   %convert_element_type_3 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_1, torch.int64), kwargs = {})
#   %sub_45 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_1, %convert_element_type_3), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_45, 0.0), kwargs = {})
#   %clamp_max_2 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 1.0), kwargs = {})
#   %mul_64 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_58, %clamp_max_2), kwargs = {})
#   %add_90 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_2, %mul_64), kwargs = {})
#   %mul_51 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_48, %clamp_max_2), kwargs = {})
#   %add_77 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index, %mul_51), kwargs = {})
#   %sub_71 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_90, %add_77), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_sub_view_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % ks1)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 2.0
    tmp3 = ks0
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 + tmp4
    tmp6 = tmp5.to(tl.float64)
    tmp7 = tl.full([1], -1.0, tl.float64)
    tmp8 = tmp7 + tmp6
    tmp9 = tmp2 * tmp4
    tmp10 = 4.0
    tmp11 = tmp10 + tmp9
    tmp12 = tmp11.to(tl.float64)
    tmp13 = tmp7 + tmp12
    tmp14 = tmp8 / tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = x0
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp17 * tmp15
    tmp19 = 0.0
    tmp20 = triton_helpers.maximum(tmp18, tmp19)
    tmp21 = tmp20.to(tl.int64)
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp20 - tmp22
    tmp24 = triton_helpers.maximum(tmp23, tmp19)
    tmp25 = 1.0
    tmp26 = triton_helpers.minimum(tmp24, tmp25)
    tmp27 = tmp1 * tmp26
    tmp28 = tmp0 + tmp27
    tmp31 = tmp30 * tmp26
    tmp32 = tmp29 + tmp31
    tmp33 = tmp28 - tmp32
    tl.store(in_out_ptr0 + (x2), tmp33, xmask)




# kernel path: /tmp/torchinductor_sahanp/kj/ckjfl4gfj3yshxag5eq56prqhr6n6qgut3inr6vvm2hrza3fjwcs.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten._to_copy, aten.arange, aten.clamp, aten.view, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   x_2 => add_109, add_77, clamp_max_2, clamp_max_3, clamp_min_1, clamp_min_2, clamp_min_3, convert_element_type_1, convert_element_type_2, convert_element_type_3, iota_1, mul_51, mul_79, sub_45, sub_68, view_1
# Graph fragment:
#   %convert_element_type_1 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view, torch.int64), kwargs = {})
#   %iota_1 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (%floordiv_1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_1, torch.float32), kwargs = {})
#   %full_default_5 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -1.0), kwargs = {dtype: torch.float64, layout: torch.strided, device: cpu, pin_memory: False})
#   %full_default_6 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 2), kwargs = {dtype: torch.int64, layout: torch.strided, device: cpu, pin_memory: False})
#   %scalar_tensor_default_8 : [num_users=2] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (%arg3_1,), kwargs = {})
#   %add_tensor_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%full_default_6, %scalar_tensor_default_8), kwargs = {})
#   %convert_element_type_default_3 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor_4, torch.float64), kwargs = {})
#   %add_tensor_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%full_default_5, %convert_element_type_default_3), kwargs = {})
#   %full_default_7 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -1.0), kwargs = {dtype: torch.float64, layout: torch.strided, device: cpu, pin_memory: False})
#   %full_default_8 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 4), kwargs = {dtype: torch.int64, layout: torch.strided, device: cpu, pin_memory: False})
#   %full_default_9 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 2), kwargs = {dtype: torch.int64, layout: torch.strided, device: cpu, pin_memory: False})
#   %mul_tensor_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_9, %scalar_tensor_default_8), kwargs = {})
#   %add_tensor_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%full_default_8, %mul_tensor_2), kwargs = {})
#   %convert_element_type_default_4 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor_6, torch.float64), kwargs = {})
#   %add_tensor_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%full_default_7, %convert_element_type_default_4), kwargs = {})
#   %true_divide_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.true_divide.Tensor](args = (%add_tensor_5, %add_tensor_7), kwargs = {})
#   %convert_element_type_default_5 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%true_divide_tensor_1, torch.float32), kwargs = {})
#   %mul_tensor_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_2, %convert_element_type_default_5), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_tensor_3, 0.0), kwargs = {})
#   %view_1 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clamp_min_1, [%floordiv_1]), kwargs = {})
#   %convert_element_type_3 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_1, torch.int64), kwargs = {})
#   %sub_45 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_1, %convert_element_type_3), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_45, 0.0), kwargs = {})
#   %clamp_max_2 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 1.0), kwargs = {})
#   %mul_51 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_48, %clamp_max_2), kwargs = {})
#   %add_77 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index, %mul_51), kwargs = {})
#   %sub_68 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %convert_element_type_1), kwargs = {})
#   %clamp_min_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_68, 0.0), kwargs = {})
#   %clamp_max_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_3, 1.0), kwargs = {})
#   %mul_79 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_71, %clamp_max_3), kwargs = {})
#   %add_109 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_77, %mul_79), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_sub_view_3(in_out_ptr0, in_ptr0, in_ptr1, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % ks1)
    x1 = ((xindex // ks1) % ks3)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = 2.0
    tmp3 = ks0
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 + tmp4
    tmp6 = tmp5.to(tl.float64)
    tmp7 = tl.full([1], -1.0, tl.float64)
    tmp8 = tmp7 + tmp6
    tmp9 = tmp2 * tmp4
    tmp10 = 4.0
    tmp11 = tmp10 + tmp9
    tmp12 = tmp11.to(tl.float64)
    tmp13 = tmp7 + tmp12
    tmp14 = tmp8 / tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = x0
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp17 * tmp15
    tmp19 = 0.0
    tmp20 = triton_helpers.maximum(tmp18, tmp19)
    tmp21 = tmp20.to(tl.int64)
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp20 - tmp22
    tmp24 = triton_helpers.maximum(tmp23, tmp19)
    tmp25 = 1.0
    tmp26 = triton_helpers.minimum(tmp24, tmp25)
    tmp27 = tmp1 * tmp26
    tmp28 = tmp0 + tmp27
    tmp30 = ks2
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp2 + tmp31
    tmp33 = tmp32.to(tl.float64)
    tmp34 = tmp7 + tmp33
    tmp35 = tmp2 * tmp31
    tmp36 = tmp10 + tmp35
    tmp37 = tmp36.to(tl.float64)
    tmp38 = tmp7 + tmp37
    tmp39 = tmp34 / tmp38
    tmp40 = tmp39.to(tl.float32)
    tmp41 = x1
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tmp42 * tmp40
    tmp44 = triton_helpers.maximum(tmp43, tmp19)
    tmp45 = tmp44.to(tl.int64)
    tmp46 = tmp45.to(tl.float32)
    tmp47 = tmp44 - tmp46
    tmp48 = triton_helpers.maximum(tmp47, tmp19)
    tmp49 = triton_helpers.minimum(tmp48, tmp25)
    tmp50 = tmp29 * tmp49
    tmp51 = tmp28 + tmp50
    tl.store(in_out_ptr0 + (x3), tmp51, xmask)




# kernel path: /tmp/torchinductor_sahanp/6p/c6px4sykqyng2d4bxsmywik5h4yeexwtltnxtan45ok3jdlaezwv.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.tanh, aten.sub]
# Source node to ATen node mapping:
#   x_3 => sub_84, tanh
# Graph fragment:
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%add_109,), kwargs = {})
#   %sub_84 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_109, %tanh), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_sub_tanh_4(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = libdevice.tanh(tmp0)
    tmp2 = tmp0 - tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)




# kernel path: /tmp/torchinductor_sahanp/26/c26d4usulb7squgchu7iurxoqvheqis4nhd7nqad6zdyji6jrids.py
# Topologically Sorted Source Nodes: [target], Original ATen: [aten.zeros_like]
# Source node to ATen node mapping:
#   target => full
# Graph fragment:
#   %full : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, %arg0_1, %floordiv, %floordiv_1], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_zeros_like_5(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/jj/cjjwwiyfpall65d55l2a5veukyka7wt2dhwei2uitt4cpkq2me4h.py
# Topologically Sorted Source Nodes: [ones_like], Original ATen: [aten.ones_like]
# Source node to ATen node mapping:
#   ones_like => full_1
# Graph fragment:
#   %full_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, %arg0_1, %floordiv, %floordiv_1], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_ones_like_6(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 1.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)







def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    s3 = arg3_1
    assert_size_stride(arg4_1, (1, s0, s1, s2, s3), (s0*s1*s2*s3, s1*s2*s3, s2*s3, s3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        ps0 = 2 + s3
        ps1 = 2 + s2
        ps2 = 4 + 2*s2 + 2*s3 + s2*s3
        buf0 = empty_strided_cuda((1, s0, 2 + s2, 2 + s3), (4*s0 + 2*s0*s2 + 2*s0*s3 + s0*s2*s3, 4 + 2*s2 + 2*s3 + s2*s3, 2 + s3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.constant_pad_nd, aten.mean]
        triton_red_fused_constant_pad_nd_mean_0_xnumel = 4*s0 + 2*s0*s2 + 2*s0*s3 + s0*s2*s3
        triton_red_fused_constant_pad_nd_mean_0_r0_numel = 2 + s1
        stream0 = get_raw_stream(0)
        triton_red_fused_constant_pad_nd_mean_0[grid(triton_red_fused_constant_pad_nd_mean_0_xnumel)](arg4_1, buf0, 16, 34, 34, 32, 32, 1156, 3468, 18, XBLOCK=64, R0_BLOCK=32, num_warps=16, num_stages=1)
        del arg4_1
        ps3 = 4 + 2*s3
        ps4 = 4 + 2*s2
        ps5 = 16 + 8*s2 + 8*s3 + 4*s2*s3
        buf1 = empty_strided_cuda((1, s0, 4 + 2*s2, 4 + 2*s3), (16*s0 + 8*s0*s2 + 8*s0*s3 + 4*s0*s2*s3, 16 + 8*s2 + 8*s3 + 4*s2*s3, 4 + 2*s3, 1), torch.float32)
        buf2 = empty_strided_cuda((1, s0, 4 + 2*s2, 4 + 2*s3), (16*s0 + 8*s0*s2 + 8*s0*s3 + 4*s0*s2*s3, 16 + 8*s2 + 8*s3 + 4*s2*s3, 4 + 2*s3, 1), torch.float32)
        buf3 = empty_strided_cuda((1, s0, 4 + 2*s2, 4 + 2*s3), (16*s0 + 8*s0*s2 + 8*s0*s3 + 4*s0*s2*s3, 16 + 8*s2 + 8*s3 + 4*s2*s3, 4 + 2*s3, 1), torch.float32)
        buf4 = empty_strided_cuda((1, s0, 4 + 2*s2, 4 + 2*s3), (16*s0 + 8*s0*s2 + 8*s0*s3 + 4*s0*s2*s3, 16 + 8*s2 + 8*s3 + 4*s2*s3, 4 + 2*s3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.constant_pad_nd, aten.mean, aten._unsafe_index, aten.sub]
        triton_poi_fused__unsafe_index_constant_pad_nd_mean_sub_1_xnumel = 16*s0 + 8*s0*s2 + 8*s0*s3 + 4*s0*s2*s3
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_constant_pad_nd_mean_sub_1[grid(triton_poi_fused__unsafe_index_constant_pad_nd_mean_sub_1_xnumel)](buf0, buf1, buf2, buf3, buf4, 32, 68, 68, 32, 4624, 16, 13872, XBLOCK=128, num_warps=4, num_stages=1)
        del buf0
        buf5 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.arange, aten._to_copy, aten.clamp, aten.view, aten.sub, aten.mul, aten.add]
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_view_2_xnumel = 16*s0 + 8*s0*s2 + 8*s0*s3 + 4*s0*s2*s3
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_view_2[grid(triton_poi_fused__to_copy_add_arange_clamp_mul_sub_view_2_xnumel)](buf5, buf2, buf3, buf4, 32, 68, 13872, XBLOCK=256, num_warps=4, num_stages=1)
        del buf2
        buf6 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten._to_copy, aten.arange, aten.clamp, aten.view, aten.sub, aten.mul, aten.add]
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_view_3_xnumel = 16*s0 + 8*s0*s2 + 8*s0*s3 + 4*s0*s2*s3
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_view_3[grid(triton_poi_fused__to_copy_add_arange_clamp_mul_sub_view_3_xnumel)](buf6, buf4, buf5, 32, 68, 32, 68, 13872, XBLOCK=256, num_warps=4, num_stages=1)
        buf7 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.tanh, aten.sub]
        triton_poi_fused_sub_tanh_4_xnumel = 16*s0 + 8*s0*s2 + 8*s0*s3 + 4*s0*s2*s3
        stream0 = get_raw_stream(0)
        triton_poi_fused_sub_tanh_4[grid(triton_poi_fused_sub_tanh_4_xnumel)](buf7, 13872, XBLOCK=256, num_warps=4, num_stages=1)
        buf8 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [target], Original ATen: [aten.zeros_like]
        triton_poi_fused_zeros_like_5_xnumel = 16*s0 + 8*s0*s2 + 8*s0*s3 + 4*s0*s2*s3
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_like_5[grid(triton_poi_fused_zeros_like_5_xnumel)](buf8, 13872, XBLOCK=256, num_warps=4, num_stages=1)
        buf9 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [ones_like], Original ATen: [aten.ones_like]
        triton_poi_fused_ones_like_6_xnumel = 16*s0 + 8*s0*s2 + 8*s0*s3 + 4*s0*s2*s3
        stream0 = get_raw_stream(0)
        triton_poi_fused_ones_like_6[grid(triton_poi_fused_ones_like_6_xnumel)](buf9, 13872, XBLOCK=256, num_warps=4, num_stages=1)
    return (buf7, buf8, buf9, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 3
    arg1_1 = 16
    arg2_1 = 32
    arg3_1 = 32
    arg4_1 = rand_strided((1, 3, 16, 32, 32), (49152, 16384, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
