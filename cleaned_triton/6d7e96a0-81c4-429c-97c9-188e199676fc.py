# AOT ID: ['173_inference']
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


# kernel path: /tmp/torchinductor_sahanp/mt/cmtk3d2fxlgznjepy5ycsvg2a3ouagyxmp77dihaeb7d5tosrmsf.py
# Topologically Sorted Source Nodes: [x, max_pool2d], Original ATen: [aten.constant_pad_nd, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   max_pool2d => _low_memory_max_pool2d_with_offsets
#   x => constant_pad_nd
# Graph fragment:
#   %constant_pad_nd : [num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%arg3_1, [2, 2, 2, 2], 3.0), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%constant_pad_nd, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_0(in_ptr0, out_ptr0, out_ptr1, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // ks0) % ks1)
    x0 = (xindex % ks0)
    x2 = xindex // ks4
    x4 = xindex
    tmp0 = (-2) + 2*x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = ks2
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + 2*x0
    tmp6 = tmp5 >= tmp1
    tmp7 = ks3
    tmp8 = tmp5 < tmp7
    tmp9 = tmp2 & tmp4
    tmp10 = tmp9 & tmp6
    tmp11 = tmp10 & tmp8
    tmp12 = tl.load(in_ptr0 + ((-2) + ((-2)*ks3) + 2*x0 + 2*ks3*x1 + ks2*ks3*x2), tmp11 & xmask, eviction_policy='evict_last', other=3)
    tmp13 = (-1) + 2*x0
    tmp14 = tmp13 >= tmp1
    tmp15 = tmp13 < tmp7
    tmp16 = tmp9 & tmp14
    tmp17 = tmp16 & tmp15
    tmp18 = tl.load(in_ptr0 + ((-1) + ((-2)*ks3) + 2*x0 + 2*ks3*x1 + ks2*ks3*x2), tmp17 & xmask, eviction_policy='evict_last', other=3)
    tmp19 = triton_helpers.maximum(tmp18, tmp12)
    tmp20 = (-1) + 2*x1
    tmp21 = tmp20 >= tmp1
    tmp22 = tmp20 < tmp3
    tmp23 = tmp21 & tmp22
    tmp24 = tmp23 & tmp6
    tmp25 = tmp24 & tmp8
    tmp26 = tl.load(in_ptr0 + ((-2) + ((-1)*ks3) + 2*x0 + 2*ks3*x1 + ks2*ks3*x2), tmp25 & xmask, eviction_policy='evict_last', other=3)
    tmp27 = triton_helpers.maximum(tmp26, tmp19)
    tmp28 = tmp23 & tmp14
    tmp29 = tmp28 & tmp15
    tmp30 = tl.load(in_ptr0 + ((-1) + ((-1)*ks3) + 2*x0 + 2*ks3*x1 + ks2*ks3*x2), tmp29 & xmask, eviction_policy='evict_last', other=3)
    tmp31 = triton_helpers.maximum(tmp30, tmp27)
    tmp32 = tmp18 > tmp12
    tmp33 = tl.full([1], 1, tl.int8)
    tmp34 = tl.full([1], 0, tl.int8)
    tmp35 = tl.where(tmp32, tmp33, tmp34)
    tmp36 = tmp26 > tmp19
    tmp37 = tl.full([1], 2, tl.int8)
    tmp38 = tl.where(tmp36, tmp37, tmp35)
    tmp39 = tmp30 > tmp27
    tmp40 = tl.full([1], 3, tl.int8)
    tmp41 = tl.where(tmp39, tmp40, tmp38)
    tl.store(out_ptr0 + (x4), tmp31, xmask)
    tl.store(out_ptr1 + (x4), tmp41, xmask)




# kernel path: /tmp/torchinductor_sahanp/ir/cirr34n2gt7bfvjklmgut3ttx25eq3mnswpupqqdog3z2dzr6fkx.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.max_unpool2d]
# Source node to ATen node mapping:
#   x_2 => full
# Graph fragment:
#   %full : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, %arg0_1, %sub_10, %sub_12], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_unpool2d_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int64)
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/2p/c2pj6p6ifzsn5nj46oq7khj6iyq3dlcjp7kf3h67cf3glvwg3zz2.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.max_unpool2d]
# Source node to ATen node mapping:
#   x_2 => index_put
# Graph fragment:
#   %index_put : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%view_2, [%view_1], %view_3), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_unpool2d_2(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2*(((x0 // ks0) % ks1)) + 4*(triton_helpers.div_floor_integer(x0,  4 + 2*(ks2 // 2) + 2*(ks3 // 2) + (ks2 // 2)*(ks3 // 2))) + (ks3 // 2)*(((x0 // ks0) % ks1)) + 2*(ks2 // 2)*(triton_helpers.div_floor_integer(x0,  4 + 2*(ks2 // 2) + 2*(ks3 // 2) + (ks2 // 2)*(ks3 // 2))) + 2*(ks3 // 2)*(triton_helpers.div_floor_integer(x0,  4 + 2*(ks2 // 2) + 2*(ks3 // 2) + (ks2 // 2)*(ks3 // 2))) + (ks2 // 2)*(ks3 // 2)*(triton_helpers.div_floor_integer(x0,  4 + 2*(ks2 // 2) + 2*(ks3 // 2) + (ks2 // 2)*(ks3 // 2))) + ((x0 % ks0))), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = tl.full([1], 2, tl.int32)
    tmp2 = tl.where((tmp0 < 0) != (tmp1 < 0), tl.where(tmp0 % tmp1 != 0, tmp0 // tmp1 - 1, tmp0 // tmp1), tmp0 // tmp1)
    tmp3 = tmp2 * tmp1
    tmp4 = tmp0 - tmp3
    tmp5 = 2*(((x0 // ks0) % ks1))
    tmp6 = tmp5 + tmp2
    tmp7 = 2*((x0 % ks0))
    tmp8 = tmp7 + tmp4
    tmp9 = 4 + ks3
    tmp10 = tmp6 * tmp9
    tmp11 = tmp10 + tmp8
    tmp12 = 16*(triton_helpers.div_floor_integer(x0,  4 + 2*(ks2 // 2) + 2*(ks3 // 2) + (ks2 // 2)*(ks3 // 2))) + 8*(ks2 // 2)*(triton_helpers.div_floor_integer(x0,  4 + 2*(ks2 // 2) + 2*(ks3 // 2) + (ks2 // 2)*(ks3 // 2))) + 8*(ks3 // 2)*(triton_helpers.div_floor_integer(x0,  4 + 2*(ks2 // 2) + 2*(ks3 // 2) + (ks2 // 2)*(ks3 // 2))) + 4*(ks2 // 2)*(ks3 // 2)*(triton_helpers.div_floor_integer(x0,  4 + 2*(ks2 // 2) + 2*(ks3 // 2) + (ks2 // 2)*(ks3 // 2)))
    tmp13 = tmp11 + tmp12
    tmp14 = 16*ks4 + 8*ks4*(ks2 // 2) + 8*ks4*(ks3 // 2) + 4*ks4*(ks2 // 2)*(ks3 // 2)
    tmp15 = tmp13 + tmp14
    tmp16 = tmp13 < 0
    tmp17 = tl.where(tmp16, tmp15, tmp13)
    tl.device_assert(((0 <= tmp17) & (tmp17 < 16*ks4 + 8*ks4*(ks2 // 2) + 8*ks4*(ks3 // 2) + 4*ks4*(ks2 // 2)*(ks3 // 2))) | ~(xmask), "index out of bounds: 0 <= tmp17 < 16*ks4 + 8*ks4*(ks2 // 2) + 8*ks4*(ks3 // 2) + 4*ks4*(ks2 // 2)*(ks3 // 2)")
    tl.store(out_ptr0 + (tl.broadcast_to(4*(((tmp17 // (4 + 2*(ks3 // 2))) % (4 + 2*(ks2 // 2)))) + 16*(((tmp17 // (16 + 8*(ks2 // 2) + 8*(ks3 // 2) + 4*(ks2 // 2)*(ks3 // 2))) % ks4)) + 2*(ks3 // 2)*(((tmp17 // (4 + 2*(ks3 // 2))) % (4 + 2*(ks2 // 2)))) + 8*(ks2 // 2)*(((tmp17 // (16 + 8*(ks2 // 2) + 8*(ks3 // 2) + 4*(ks2 // 2)*(ks3 // 2))) % ks4)) + 8*(ks3 // 2)*(((tmp17 // (16 + 8*(ks2 // 2) + 8*(ks3 // 2) + 4*(ks2 // 2)*(ks3 // 2))) % ks4)) + 4*(ks2 // 2)*(ks3 // 2)*(((tmp17 // (16 + 8*(ks2 // 2) + 8*(ks3 // 2) + 4*(ks2 // 2)*(ks3 // 2))) % ks4)) + ((tmp17 % (4 + 2*(ks3 // 2)))), [XBLOCK])), tmp19, xmask)




# kernel path: /tmp/torchinductor_sahanp/jy/cjy6vtwbt3siuf24zlnqsdwambqbpfqipcsvzlfcu6mtic3czcqp.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten._to_copy, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   x_3 => clamp_max_3, clamp_min_3, convert_element_type_1, sub_91
# Graph fragment:
#   %convert_element_type_1 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_5, torch.int64), kwargs = {})
#   %sub_91 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_5, %convert_element_type_1), kwargs = {})
#   %clamp_min_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_91, 0.0), kwargs = {})
#   %clamp_max_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_3, 1.0), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_clamp_sub_3(out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = ks0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 2.0
    tmp3 = tmp1 / tmp2
    tmp4 = libdevice.floor(tmp3)
    tmp5 = tmp2 * tmp4
    tmp6 = 4.0
    tmp7 = tmp6 + tmp5
    tmp8 = tmp7.to(tl.float64)
    tmp9 = tl.full([1], -1.0, tl.float64)
    tmp10 = tmp9 + tmp8
    tmp11 = tmp6 * tmp4
    tmp12 = 8.0
    tmp13 = tmp12 + tmp11
    tmp14 = tmp13.to(tl.float64)
    tmp15 = tmp9 + tmp14
    tmp16 = tmp10 / tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = x0
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp19 * tmp17
    tmp21 = 0.0
    tmp22 = triton_helpers.maximum(tmp20, tmp21)
    tmp23 = tmp22.to(tl.int64)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp22 - tmp24
    tmp26 = triton_helpers.maximum(tmp25, tmp21)
    tmp27 = 1.0
    tmp28 = triton_helpers.minimum(tmp26, tmp27)
    tl.store(out_ptr0 + (x0), tmp28, xmask)




# kernel path: /tmp/torchinductor_sahanp/4z/c4zk2jcarh4o3kkuwtqlnlywiotqcqr46c5cuifyriv2jo7wylft.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten._to_copy, aten._unsafe_index, aten.sub, aten.mul, aten.add, aten.clamp, aten.round]
# Source node to ATen node mapping:
#   x_3 => _unsafe_index, _unsafe_index_1, _unsafe_index_2, _unsafe_index_3, add_104, add_117, add_136, clamp_max_3, clamp_min_3, convert_element_type_1, convert_element_type_4, convert_element_type_5, convert_element_type_6, convert_element_type_7, convert_element_type_8, mul_106, mul_78, mul_91, round_1, sub_71, sub_81, sub_91, sub_94
# Graph fragment:
#   %convert_element_type_1 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_5, torch.int64), kwargs = {})
#   %_unsafe_index_3 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%view_4, [None, None, %clamp_max, %clamp_max_1]), kwargs = {})
#   %convert_element_type_7 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%_unsafe_index_3, torch.float32), kwargs = {})
#   %_unsafe_index_2 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%view_4, [None, None, %clamp_max, %convert_element_type_3]), kwargs = {})
#   %convert_element_type_6 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%_unsafe_index_2, torch.float32), kwargs = {})
#   %sub_81 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_7, %convert_element_type_6), kwargs = {})
#   %mul_91 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_81, %clamp_max_2), kwargs = {})
#   %add_117 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_6, %mul_91), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%view_4, [None, None, %convert_element_type_1, %clamp_max_1]), kwargs = {})
#   %convert_element_type_5 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%_unsafe_index_1, torch.float32), kwargs = {})
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%view_4, [None, None, %convert_element_type_1, %convert_element_type_3]), kwargs = {})
#   %convert_element_type_4 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%_unsafe_index, torch.float32), kwargs = {})
#   %sub_71 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_5, %convert_element_type_4), kwargs = {})
#   %mul_78 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_71, %clamp_max_2), kwargs = {})
#   %add_104 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_4, %mul_78), kwargs = {})
#   %sub_94 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_117, %add_104), kwargs = {})
#   %sub_91 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_5, %convert_element_type_1), kwargs = {})
#   %clamp_min_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_91, 0.0), kwargs = {})
#   %clamp_max_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_3, 1.0), kwargs = {})
#   %mul_106 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_94, %clamp_max_3), kwargs = {})
#   %add_136 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_104, %mul_106), kwargs = {})
#   %round_1 : [num_users=1] = call_function[target=torch.ops.aten.round.default](args = (%add_136,), kwargs = {})
#   %convert_element_type_8 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%round_1, torch.int64), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_clamp_mul_round_sub_4(in_ptr0, in_ptr1, in_ptr2, out_ptr4, ks0, ks1, ks2, ks3, ks4, ks5, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // ks1) % ks2)
    x0 = (xindex % ks1)
    x5 = xindex // ks4
    x6 = xindex
    tmp60 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp66 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = ks0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 2.0
    tmp3 = tmp1 / tmp2
    tmp4 = libdevice.floor(tmp3)
    tmp5 = tmp2 * tmp4
    tmp6 = 4.0
    tmp7 = tmp6 + tmp5
    tmp8 = tmp7.to(tl.float64)
    tmp9 = tl.full([1], -1.0, tl.float64)
    tmp10 = tmp9 + tmp8
    tmp11 = tmp6 * tmp4
    tmp12 = 8.0
    tmp13 = tmp12 + tmp11
    tmp14 = tmp13.to(tl.float64)
    tmp15 = tmp9 + tmp14
    tmp16 = tmp10 / tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = x1
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp19 * tmp17
    tmp21 = 0.0
    tmp22 = triton_helpers.maximum(tmp20, tmp21)
    tmp23 = tmp22.to(tl.int64)
    tmp24 = ks3
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp25 / tmp2
    tmp27 = libdevice.floor(tmp26)
    tmp28 = tmp2 * tmp27
    tmp29 = tmp6 + tmp28
    tmp30 = tmp29.to(tl.float64)
    tmp31 = tmp9 + tmp30
    tmp32 = tmp6 * tmp27
    tmp33 = tmp12 + tmp32
    tmp34 = tmp33.to(tl.float64)
    tmp35 = tmp9 + tmp34
    tmp36 = tmp31 / tmp35
    tmp37 = tmp36.to(tl.float32)
    tmp38 = x0
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tmp39 * tmp37
    tmp41 = triton_helpers.maximum(tmp40, tmp21)
    tmp42 = tmp41.to(tl.int64)
    tmp43 = tl.load(in_ptr0 + (4*((((tmp42 + 4*tmp23 + 16*x5 + 2*tmp23*(ks3 // 2) + 8*x5*(ks0 // 2) + 8*x5*(ks3 // 2) + 4*x5*(ks0 // 2)*(ks3 // 2)) // (4 + 2*(ks3 // 2))) % (4 + 2*(ks0 // 2)))) + 16*((((tmp42 + 4*tmp23 + 16*x5 + 2*tmp23*(ks3 // 2) + 8*x5*(ks0 // 2) + 8*x5*(ks3 // 2) + 4*x5*(ks0 // 2)*(ks3 // 2)) // (16 + 8*(ks0 // 2) + 8*(ks3 // 2) + 4*(ks0 // 2)*(ks3 // 2))) % ks5)) + 2*(ks3 // 2)*((((tmp42 + 4*tmp23 + 16*x5 + 2*tmp23*(ks3 // 2) + 8*x5*(ks0 // 2) + 8*x5*(ks3 // 2) + 4*x5*(ks0 // 2)*(ks3 // 2)) // (4 + 2*(ks3 // 2))) % (4 + 2*(ks0 // 2)))) + 8*(ks0 // 2)*((((tmp42 + 4*tmp23 + 16*x5 + 2*tmp23*(ks3 // 2) + 8*x5*(ks0 // 2) + 8*x5*(ks3 // 2) + 4*x5*(ks0 // 2)*(ks3 // 2)) // (16 + 8*(ks0 // 2) + 8*(ks3 // 2) + 4*(ks0 // 2)*(ks3 // 2))) % ks5)) + 8*(ks3 // 2)*((((tmp42 + 4*tmp23 + 16*x5 + 2*tmp23*(ks3 // 2) + 8*x5*(ks0 // 2) + 8*x5*(ks3 // 2) + 4*x5*(ks0 // 2)*(ks3 // 2)) // (16 + 8*(ks0 // 2) + 8*(ks3 // 2) + 4*(ks0 // 2)*(ks3 // 2))) % ks5)) + 4*(ks0 // 2)*(ks3 // 2)*((((tmp42 + 4*tmp23 + 16*x5 + 2*tmp23*(ks3 // 2) + 8*x5*(ks0 // 2) + 8*x5*(ks3 // 2) + 4*x5*(ks0 // 2)*(ks3 // 2)) // (16 + 8*(ks0 // 2) + 8*(ks3 // 2) + 4*(ks0 // 2)*(ks3 // 2))) % ks5)) + ((tmp42 % (4 + 2*(ks3 // 2))))), xmask, eviction_policy='evict_last')
    tmp44 = tmp43.to(tl.float32)
    tmp45 = tl.full([1], 1, tl.int64)
    tmp46 = tmp23 + tmp45
    tmp47 = 3 + 2*(ks0 // 2)
    tmp48 = triton_helpers.minimum(tmp46, tmp47)
    tmp49 = tl.load(in_ptr0 + (4*((((tmp42 + 4*tmp48 + 16*x5 + 2*tmp48*(ks3 // 2) + 8*x5*(ks0 // 2) + 8*x5*(ks3 // 2) + 4*x5*(ks0 // 2)*(ks3 // 2)) // (4 + 2*(ks3 // 2))) % (4 + 2*(ks0 // 2)))) + 16*((((tmp42 + 4*tmp48 + 16*x5 + 2*tmp48*(ks3 // 2) + 8*x5*(ks0 // 2) + 8*x5*(ks3 // 2) + 4*x5*(ks0 // 2)*(ks3 // 2)) // (16 + 8*(ks0 // 2) + 8*(ks3 // 2) + 4*(ks0 // 2)*(ks3 // 2))) % ks5)) + 2*(ks3 // 2)*((((tmp42 + 4*tmp48 + 16*x5 + 2*tmp48*(ks3 // 2) + 8*x5*(ks0 // 2) + 8*x5*(ks3 // 2) + 4*x5*(ks0 // 2)*(ks3 // 2)) // (4 + 2*(ks3 // 2))) % (4 + 2*(ks0 // 2)))) + 8*(ks0 // 2)*((((tmp42 + 4*tmp48 + 16*x5 + 2*tmp48*(ks3 // 2) + 8*x5*(ks0 // 2) + 8*x5*(ks3 // 2) + 4*x5*(ks0 // 2)*(ks3 // 2)) // (16 + 8*(ks0 // 2) + 8*(ks3 // 2) + 4*(ks0 // 2)*(ks3 // 2))) % ks5)) + 8*(ks3 // 2)*((((tmp42 + 4*tmp48 + 16*x5 + 2*tmp48*(ks3 // 2) + 8*x5*(ks0 // 2) + 8*x5*(ks3 // 2) + 4*x5*(ks0 // 2)*(ks3 // 2)) // (16 + 8*(ks0 // 2) + 8*(ks3 // 2) + 4*(ks0 // 2)*(ks3 // 2))) % ks5)) + 4*(ks0 // 2)*(ks3 // 2)*((((tmp42 + 4*tmp48 + 16*x5 + 2*tmp48*(ks3 // 2) + 8*x5*(ks0 // 2) + 8*x5*(ks3 // 2) + 4*x5*(ks0 // 2)*(ks3 // 2)) // (16 + 8*(ks0 // 2) + 8*(ks3 // 2) + 4*(ks0 // 2)*(ks3 // 2))) % ks5)) + ((tmp42 % (4 + 2*(ks3 // 2))))), xmask, eviction_policy='evict_last')
    tmp50 = tmp49.to(tl.float32)
    tmp51 = tmp42 + tmp45
    tmp52 = 3 + 2*(ks3 // 2)
    tmp53 = triton_helpers.minimum(tmp51, tmp52)
    tmp54 = tl.load(in_ptr0 + (4*((((tmp53 + 4*tmp48 + 16*x5 + 2*tmp48*(ks3 // 2) + 8*x5*(ks0 // 2) + 8*x5*(ks3 // 2) + 4*x5*(ks0 // 2)*(ks3 // 2)) // (4 + 2*(ks3 // 2))) % (4 + 2*(ks0 // 2)))) + 16*((((tmp53 + 4*tmp48 + 16*x5 + 2*tmp48*(ks3 // 2) + 8*x5*(ks0 // 2) + 8*x5*(ks3 // 2) + 4*x5*(ks0 // 2)*(ks3 // 2)) // (16 + 8*(ks0 // 2) + 8*(ks3 // 2) + 4*(ks0 // 2)*(ks3 // 2))) % ks5)) + 2*(ks3 // 2)*((((tmp53 + 4*tmp48 + 16*x5 + 2*tmp48*(ks3 // 2) + 8*x5*(ks0 // 2) + 8*x5*(ks3 // 2) + 4*x5*(ks0 // 2)*(ks3 // 2)) // (4 + 2*(ks3 // 2))) % (4 + 2*(ks0 // 2)))) + 8*(ks0 // 2)*((((tmp53 + 4*tmp48 + 16*x5 + 2*tmp48*(ks3 // 2) + 8*x5*(ks0 // 2) + 8*x5*(ks3 // 2) + 4*x5*(ks0 // 2)*(ks3 // 2)) // (16 + 8*(ks0 // 2) + 8*(ks3 // 2) + 4*(ks0 // 2)*(ks3 // 2))) % ks5)) + 8*(ks3 // 2)*((((tmp53 + 4*tmp48 + 16*x5 + 2*tmp48*(ks3 // 2) + 8*x5*(ks0 // 2) + 8*x5*(ks3 // 2) + 4*x5*(ks0 // 2)*(ks3 // 2)) // (16 + 8*(ks0 // 2) + 8*(ks3 // 2) + 4*(ks0 // 2)*(ks3 // 2))) % ks5)) + 4*(ks0 // 2)*(ks3 // 2)*((((tmp53 + 4*tmp48 + 16*x5 + 2*tmp48*(ks3 // 2) + 8*x5*(ks0 // 2) + 8*x5*(ks3 // 2) + 4*x5*(ks0 // 2)*(ks3 // 2)) // (16 + 8*(ks0 // 2) + 8*(ks3 // 2) + 4*(ks0 // 2)*(ks3 // 2))) % ks5)) + ((tmp53 % (4 + 2*(ks3 // 2))))), xmask, eviction_policy='evict_last')
    tmp55 = tmp54.to(tl.float32)
    tmp56 = tmp55 - tmp50
    tmp57 = tl.load(in_ptr0 + (4*((((tmp53 + 4*tmp23 + 16*x5 + 2*tmp23*(ks3 // 2) + 8*x5*(ks0 // 2) + 8*x5*(ks3 // 2) + 4*x5*(ks0 // 2)*(ks3 // 2)) // (4 + 2*(ks3 // 2))) % (4 + 2*(ks0 // 2)))) + 16*((((tmp53 + 4*tmp23 + 16*x5 + 2*tmp23*(ks3 // 2) + 8*x5*(ks0 // 2) + 8*x5*(ks3 // 2) + 4*x5*(ks0 // 2)*(ks3 // 2)) // (16 + 8*(ks0 // 2) + 8*(ks3 // 2) + 4*(ks0 // 2)*(ks3 // 2))) % ks5)) + 2*(ks3 // 2)*((((tmp53 + 4*tmp23 + 16*x5 + 2*tmp23*(ks3 // 2) + 8*x5*(ks0 // 2) + 8*x5*(ks3 // 2) + 4*x5*(ks0 // 2)*(ks3 // 2)) // (4 + 2*(ks3 // 2))) % (4 + 2*(ks0 // 2)))) + 8*(ks0 // 2)*((((tmp53 + 4*tmp23 + 16*x5 + 2*tmp23*(ks3 // 2) + 8*x5*(ks0 // 2) + 8*x5*(ks3 // 2) + 4*x5*(ks0 // 2)*(ks3 // 2)) // (16 + 8*(ks0 // 2) + 8*(ks3 // 2) + 4*(ks0 // 2)*(ks3 // 2))) % ks5)) + 8*(ks3 // 2)*((((tmp53 + 4*tmp23 + 16*x5 + 2*tmp23*(ks3 // 2) + 8*x5*(ks0 // 2) + 8*x5*(ks3 // 2) + 4*x5*(ks0 // 2)*(ks3 // 2)) // (16 + 8*(ks0 // 2) + 8*(ks3 // 2) + 4*(ks0 // 2)*(ks3 // 2))) % ks5)) + 4*(ks0 // 2)*(ks3 // 2)*((((tmp53 + 4*tmp23 + 16*x5 + 2*tmp23*(ks3 // 2) + 8*x5*(ks0 // 2) + 8*x5*(ks3 // 2) + 4*x5*(ks0 // 2)*(ks3 // 2)) // (16 + 8*(ks0 // 2) + 8*(ks3 // 2) + 4*(ks0 // 2)*(ks3 // 2))) % ks5)) + ((tmp53 % (4 + 2*(ks3 // 2))))), xmask, eviction_policy='evict_last')
    tmp58 = tmp57.to(tl.float32)
    tmp59 = tmp58 - tmp44
    tmp61 = tmp59 * tmp60
    tmp62 = tmp44 + tmp61
    tmp63 = tmp56 * tmp60
    tmp64 = tmp50 + tmp63
    tmp65 = tmp64 - tmp62
    tmp67 = tmp65 * tmp66
    tmp68 = tmp62 + tmp67
    tmp69 = libdevice.nearbyint(tmp68)
    tmp70 = tmp69.to(tl.int64)
    tl.store(out_ptr4 + (x6), tmp70, xmask)




# kernel path: /tmp/torchinductor_sahanp/zo/czowosoaujtuumy727wnc2ze6wkgy7aga5c4a6jijeksta2me4od.py
# Topologically Sorted Source Nodes: [x_4], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   x_4 => amax, exp, sub_110, sum_1
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%convert_element_type_8, [-3], True), kwargs = {})
#   %sub_110 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_8, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_110,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-3], True), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__softmax_5(in_ptr0, out_ptr0, out_ptr1, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_1 + 32*r0_1*(ks0 // 2) + 32*r0_1*(ks1 // 2) + 16*r0_1*(ks0 // 2)*(ks1 // 2)), r0_mask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, -9223372036854775808)
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tmp5 = tmp0 - tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl_math.exp(tmp6)
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(r0_mask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr1 + (x0), tmp11, xmask)




# kernel path: /tmp/torchinductor_sahanp/jk/cjka3ucogtdrvmk6svs4iu7wfxwbuml7umvjytbrom6wygtrafvx.py
# Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.copy]
# Source node to ATen node mapping:
#   x_6 => copy
# Graph fragment:
#   %copy : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_5, %slice_6), kwargs = {})
#   %slice_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_tensor_1, %copy, 2, 1, 2), kwargs = {})
#   %slice_scatter_default_1 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_tensor, %slice_scatter_default, 3, 1, %sub_128), kwargs = {})
#   %slice_scatter_default_2 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%empty, %slice_scatter_default_1, 4, 1, %sub_120), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_copy_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks2)
    x2 = ((xindex // ks4) % 3)
    x3 = xindex // ks5
    x4 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = 9 + 4*(ks1 // 2)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = x1
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp6 >= tmp7
    tmp9 = tl.broadcast_to(9 + 4*(ks3 // 2), [XBLOCK])
    tmp10 = tmp6 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tmp11 & tmp5
    tmp13 = x2
    tmp14 = tl.full([1], 1, tl.int64)
    tmp15 = tmp13 >= tmp14
    tmp16 = tl.full([1], 2, tl.int64)
    tmp17 = tmp13 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tmp18 & tmp12
    tmp20 = tl.load(in_ptr0 + ((-9) + x0 + ((-4)*(ks1 // 2)) + 8*x1 + 64*x3 + 4*x1*(ks1 // 2) + 32*x3*(ks1 // 2) + 32*x3*(ks3 // 2) + 16*x3*(ks1 // 2)*(ks3 // 2)), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.load(in_ptr1 + ((-9) + x0 + ((-4)*(ks1 // 2)) + 8*x1 + 4*x1*(ks1 // 2)), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 - tmp21
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tl_math.exp(tmp23)
    tmp25 = tl.load(in_ptr2 + ((-9) + x0 + ((-4)*(ks1 // 2)) + 8*x1 + 4*x1*(ks1 // 2)), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp24 / tmp25
    tmp27 = tmp26.to(tl.int64)
    tmp28 = tl.full(tmp27.shape, 0, tmp27.dtype)
    tmp29 = tl.where(tmp19, tmp27, tmp28)
    tmp30 = tl.load(in_ptr3 + (x4), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = tl.where(tmp18, tmp29, tmp30)
    tmp32 = tl.full(tmp31.shape, 0, tmp31.dtype)
    tmp33 = tl.where(tmp12, tmp31, tmp32)
    tmp34 = tl.load(in_ptr3 + (x4), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tl.where(tmp11, tmp33, tmp34)
    tmp36 = tl.full(tmp35.shape, 0, tmp35.dtype)
    tmp37 = tl.where(tmp5, tmp35, tmp36)
    tmp38 = tl.full([1], 0, tl.int64)
    tmp39 = tl.where(tmp5, tmp37, tmp38)
    tl.store(out_ptr0 + (x4), tmp39, xmask)




# kernel path: /tmp/torchinductor_sahanp/er/cer5cnkshvehn5goq4kivt6u72tm5srrmr3hqendcfauaopmdrbi.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %slice_scatter_default_3 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_2, %slice_15, 4, 0, 1), kwargs = {})
#   %slice_scatter_default_4 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_3, %slice_20, 4, %sub_164, %add_161), kwargs = {})
#   %slice_scatter_default_5 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_4, %slice_25, 3, 0, 1), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_7(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // ks0) % ks1)
    x0 = (xindex % ks0)
    x4 = xindex // ks0
    x3 = xindex
    tmp39 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0
    tmp4 = tl.broadcast_to(9 + 4*(ks2 // 2), [XBLOCK])
    tmp5 = tmp3 >= tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = (-8) + x0 + ((-4)*(ks2 // 2))
    tmp8 = tl.full([1], 1, tl.int64)
    tmp9 = tmp7 < tmp8
    tmp10 = tmp9 & tmp6
    tmp11 = tl.load(in_ptr0 + (88 + 10*x4 + 36*(ks2 // 2) + 40*(ks3 // 2) + 4*x4*(ks2 // 2) + 16*(ks2 // 2)*(ks3 // 2)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr0 + (72 + x3 + 28*(ks2 // 2) + 40*(ks3 // 2) + 16*(ks2 // 2)*(ks3 // 2)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.where(tmp9, tmp11, tmp12)
    tmp14 = tl.full(tmp13.shape, 0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.full([1], 1, tl.int64)
    tmp17 = tmp3 < tmp16
    tmp18 = tmp17 & tmp2
    tmp19 = tl.load(in_ptr0 + (88 + 10*x4 + 36*(ks2 // 2) + 40*(ks3 // 2) + 4*x4*(ks2 // 2) + 16*(ks2 // 2)*(ks3 // 2)), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.load(in_ptr0 + (80 + x3 + 32*(ks2 // 2) + 40*(ks3 // 2) + 16*(ks2 // 2)*(ks3 // 2)), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.where(tmp17, tmp19, tmp20)
    tmp22 = tl.where(tmp5, tmp15, tmp21)
    tmp23 = tl.full(tmp22.shape, 0, tmp22.dtype)
    tmp24 = tl.where(tmp2, tmp22, tmp23)
    tmp25 = x0
    tmp26 = 9 + 4*(ks2 // 2)
    tmp27 = tmp25 >= tmp26
    tmp28 = (-8) + x0 + ((-4)*(ks2 // 2))
    tmp29 = tl.full([1], 1, tl.int64)
    tmp30 = tmp28 < tmp29
    tmp31 = tmp30 & tmp27
    tmp32 = tl.load(in_ptr0 + (8 + 4*(ks2 // 2) + 10*x4 + 4*x4*(ks2 // 2)), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tl.load(in_ptr0 + ((-8) + x3 + ((-4)*(ks2 // 2))), tmp27 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.where(tmp30, tmp32, tmp33)
    tmp35 = tl.full(tmp34.shape, 0, tmp34.dtype)
    tmp36 = tl.where(tmp27, tmp34, tmp35)
    tmp37 = tmp25 < tmp1
    tmp38 = tl.load(in_ptr0 + (8 + 4*(ks2 // 2) + 10*x4 + 4*x4*(ks2 // 2)), tmp37 & xmask, eviction_policy='evict_last', other=0.0)
    tmp40 = tl.where(tmp37, tmp38, tmp39)
    tmp41 = tl.where(tmp27, tmp36, tmp40)
    tmp42 = tl.where(tmp2, tmp24, tmp41)
    tl.store(out_ptr0 + (x3), tmp42, xmask)




# kernel path: /tmp/torchinductor_sahanp/uz/cuz5vze666mhmnsasbcno6jo6dvjsp42qdlbj5mwjspur54qifc6.py
# Topologically Sorted Source Nodes: [target], Original ATen: [aten.rand_like]
# Source node to ATen node mapping:
#   target => inductor_lookup_seed_default, inductor_random_default
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, %arg0_1, 3, %add_159, %add_161], %inductor_lookup_seed_default, rand), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_rand_like_8(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (x0), tmp2, xmask)




# kernel path: /tmp/torchinductor_sahanp/wl/cwlp3mvk7emq2ztizgfxyscdns3nyrkms44ygbhisp47ykvb2mvj.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %slice_scatter_default_6 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_5, %slice_30, 3, %sub_188, %add_159), kwargs = {})
#   %slice_scatter_default_7 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_6, %slice_35, 2, 0, 1), kwargs = {})
#   %slice_scatter_default_8 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_7, %slice_40, 2, 2, 3), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_9(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // ks0) % 3)
    x6 = ((xindex // ks1) % 3)
    x1 = ((xindex // ks2) % ks3)
    x0 = (xindex % ks2)
    x9 = xindex // ks0
    x8 = xindex
    tmp41 = tl.load(in_ptr0 + (x8), xmask, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = (-1) + x6
    tmp4 = tl.full([1], 1, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = x1
    tmp8 = tl.broadcast_to(9 + 4*(ks4 // 2), [XBLOCK])
    tmp9 = tmp7 >= tmp8
    tmp10 = tmp9 & tmp6
    tmp11 = tl.load(in_ptr0 + (10 + x0 + 4*(ks5 // 2) + 100*x9 + 40*x9*(ks4 // 2) + 40*x9*(ks5 // 2) + 16*x9*(ks4 // 2)*(ks5 // 2)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr0 + (x8), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.where(tmp9, tmp11, tmp12)
    tmp14 = tl.full(tmp13.shape, 0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = x1
    tmp17 = tl.broadcast_to(9 + 4*(ks4 // 2), [XBLOCK])
    tmp18 = tmp16 >= tmp17
    tmp19 = tmp18 & tmp2
    tmp20 = tl.load(in_ptr0 + ((-90) + x0 + ((-40)*(ks4 // 2)) + ((-36)*(ks5 // 2)) + 100*x9 + ((-16)*(ks4 // 2)*(ks5 // 2)) + 40*x9*(ks4 // 2) + 40*x9*(ks5 // 2) + 16*x9*(ks4 // 2)*(ks5 // 2)), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.load(in_ptr0 + ((-100) + x8 + ((-40)*(ks4 // 2)) + ((-40)*(ks5 // 2)) + ((-16)*(ks4 // 2)*(ks5 // 2))), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.where(tmp18, tmp20, tmp21)
    tmp23 = tl.where(tmp5, tmp15, tmp22)
    tmp24 = tl.full(tmp23.shape, 0, tmp23.dtype)
    tmp25 = tl.where(tmp2, tmp23, tmp24)
    tmp26 = tl.full([1], 1, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = x1
    tmp29 = tl.broadcast_to(9 + 4*(ks4 // 2), [XBLOCK])
    tmp30 = tmp28 >= tmp29
    tmp31 = tmp30 & tmp27
    tmp32 = tl.load(in_ptr0 + (110 + x0 + 40*(ks4 // 2) + 44*(ks5 // 2) + 100*x9 + 16*(ks4 // 2)*(ks5 // 2) + 40*x9*(ks4 // 2) + 40*x9*(ks5 // 2) + 16*x9*(ks4 // 2)*(ks5 // 2)), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tl.load(in_ptr0 + (100 + x8 + 40*(ks4 // 2) + 40*(ks5 // 2) + 16*(ks4 // 2)*(ks5 // 2)), tmp27 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.where(tmp30, tmp32, tmp33)
    tmp35 = tl.full(tmp34.shape, 0, tmp34.dtype)
    tmp36 = tl.where(tmp27, tmp34, tmp35)
    tmp37 = x1
    tmp38 = 9 + 4*(ks4 // 2)
    tmp39 = tmp37 >= tmp38
    tmp40 = tl.load(in_ptr0 + (10 + x0 + 4*(ks5 // 2) + 100*x9 + 40*x9*(ks4 // 2) + 40*x9*(ks5 // 2) + 16*x9*(ks4 // 2)*(ks5 // 2)), tmp39 & xmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.where(tmp39, tmp40, tmp41)
    tmp43 = tl.where(tmp27, tmp36, tmp42)
    tmp44 = tl.where(tmp2, tmp25, tmp43)
    tl.store(out_ptr0 + (x8), tmp44, xmask)




# kernel path: /tmp/torchinductor_sahanp/bu/cbuu65wrsbzskwlnqidzucg7l74xt6bvuc6edy3avf7jbx2vcgoz.py
# Topologically Sorted Source Nodes: [target, loss], Original ATen: [aten.rand_like, aten.binary_cross_entropy]
# Source node to ATen node mapping:
#   loss => full_default_14, full_default_15, log, log1p, maximum, maximum_1, mean, mul_367, mul_368, neg, sub_235, sub_236
#   target => convert_element_type_default_6
# Graph fragment:
#   %convert_element_type_default_6 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%inductor_random_default, torch.int64), kwargs = {})
#   %sub_235 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_default_6, 1), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze_1,), kwargs = {})
#   %log1p : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%neg,), kwargs = {})
#   %full_default_14 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -100), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %maximum : [num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%log1p, %full_default_14), kwargs = {})
#   %mul_367 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_235, %maximum), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%squeeze_1,), kwargs = {})
#   %full_default_15 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -100), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %maximum_1 : [num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%log, %full_default_15), kwargs = {})
#   %mul_368 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_6, %maximum_1), kwargs = {})
#   %sub_236 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_367, %mul_368), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sub_236,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_binary_cross_entropy_rand_like_10(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 21
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp24 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = r0_1 + x0*(triton_helpers.div_floor_integer(20 + 300*ks0 + 120*ks0*(ks1 // 2) + 120*ks0*(ks2 // 2) + 48*ks0*(ks1 // 2)*(ks2 // 2),  21))
        tmp1 = 300*ks0 + 120*ks0*(ks1 // 2) + 120*ks0*(ks2 // 2) + 48*ks0*(ks1 // 2)*(ks2 // 2)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (10*((((r0_1 + x0*(triton_helpers.div_floor_integer(20 + 300*ks0 + 120*ks0*(ks1 // 2) + 120*ks0*(ks2 // 2) + 48*ks0*(ks1 // 2)*(ks2 // 2),  21))) // ks4) % ks5)) + 100*((((r0_1 + x0*(triton_helpers.div_floor_integer(20 + 300*ks0 + 120*ks0*(ks1 // 2) + 120*ks0*(ks2 // 2) + 48*ks0*(ks1 // 2)*(ks2 // 2),  21))) // ks3) % (3*ks0))) + 4*(ks2 // 2)*((((r0_1 + x0*(triton_helpers.div_floor_integer(20 + 300*ks0 + 120*ks0*(ks1 // 2) + 120*ks0*(ks2 // 2) + 48*ks0*(ks1 // 2)*(ks2 // 2),  21))) // ks4) % ks5)) + 40*(ks1 // 2)*((((r0_1 + x0*(triton_helpers.div_floor_integer(20 + 300*ks0 + 120*ks0*(ks1 // 2) + 120*ks0*(ks2 // 2) + 48*ks0*(ks1 // 2)*(ks2 // 2),  21))) // ks3) % (3*ks0))) + 40*(ks2 // 2)*((((r0_1 + x0*(triton_helpers.div_floor_integer(20 + 300*ks0 + 120*ks0*(ks1 // 2) + 120*ks0*(ks2 // 2) + 48*ks0*(ks1 // 2)*(ks2 // 2),  21))) // ks3) % (3*ks0))) + 16*(ks1 // 2)*(ks2 // 2)*((((r0_1 + x0*(triton_helpers.div_floor_integer(20 + 300*ks0 + 120*ks0*(ks1 // 2) + 120*ks0*(ks2 // 2) + 48*ks0*(ks1 // 2)*(ks2 // 2),  21))) // ks3) % (3*ks0))) + (((r0_1 + x0*(triton_helpers.div_floor_integer(20 + 300*ks0 + 120*ks0*(ks1 // 2) + 120*ks0*(ks2 // 2) + 48*ks0*(ks1 // 2)*(ks2 // 2),  21))) % ks4))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tmp3.to(tl.int64)
        tmp5 = tl.full([1, 1], 1, tl.int64)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.load(in_ptr1 + (10*((((r0_1 + x0*(triton_helpers.div_floor_integer(20 + 300*ks0 + 120*ks0*(ks1 // 2) + 120*ks0*(ks2 // 2) + 48*ks0*(ks1 // 2)*(ks2 // 2),  21))) // ks4) % ks5)) + 100*((((r0_1 + x0*(triton_helpers.div_floor_integer(20 + 300*ks0 + 120*ks0*(ks1 // 2) + 120*ks0*(ks2 // 2) + 48*ks0*(ks1 // 2)*(ks2 // 2),  21))) // ks3) % (3*ks0))) + 4*(ks2 // 2)*((((r0_1 + x0*(triton_helpers.div_floor_integer(20 + 300*ks0 + 120*ks0*(ks1 // 2) + 120*ks0*(ks2 // 2) + 48*ks0*(ks1 // 2)*(ks2 // 2),  21))) // ks4) % ks5)) + 40*(ks1 // 2)*((((r0_1 + x0*(triton_helpers.div_floor_integer(20 + 300*ks0 + 120*ks0*(ks1 // 2) + 120*ks0*(ks2 // 2) + 48*ks0*(ks1 // 2)*(ks2 // 2),  21))) // ks3) % (3*ks0))) + 40*(ks2 // 2)*((((r0_1 + x0*(triton_helpers.div_floor_integer(20 + 300*ks0 + 120*ks0*(ks1 // 2) + 120*ks0*(ks2 // 2) + 48*ks0*(ks1 // 2)*(ks2 // 2),  21))) // ks3) % (3*ks0))) + 16*(ks1 // 2)*(ks2 // 2)*((((r0_1 + x0*(triton_helpers.div_floor_integer(20 + 300*ks0 + 120*ks0*(ks1 // 2) + 120*ks0*(ks2 // 2) + 48*ks0*(ks1 // 2)*(ks2 // 2),  21))) // ks3) % (3*ks0))) + (((r0_1 + x0*(triton_helpers.div_floor_integer(20 + 300*ks0 + 120*ks0*(ks1 // 2) + 120*ks0*(ks2 // 2) + 48*ks0*(ks1 // 2)*(ks2 // 2),  21))) % ks4))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = -tmp8
        tmp10 = tmp9.to(tl.float32)
        tmp11 = libdevice.log1p(tmp10)
        tmp12 = -100.0
        tmp13 = triton_helpers.maximum(tmp11, tmp12)
        tmp14 = tmp7 * tmp13
        tmp15 = tmp4.to(tl.float32)
        tmp16 = tmp8.to(tl.float32)
        tmp17 = tl_math.log(tmp16)
        tmp18 = triton_helpers.maximum(tmp17, tmp12)
        tmp19 = tmp15 * tmp18
        tmp20 = tmp14 - tmp19
        tmp21 = tl.full(tmp20.shape, 0, tmp20.dtype)
        tmp22 = tl.where(tmp2, tmp20, tmp21)
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, R0_BLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(r0_mask & xmask, tmp25, _tmp24)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp24, xmask)




# kernel path: /tmp/torchinductor_sahanp/xd/cxdt4wec2xf7mj6hht3o6l76tcv7ahikvwdglilpmmsmondnl4fx.py
# Topologically Sorted Source Nodes: [target, loss], Original ATen: [aten.rand_like, aten.binary_cross_entropy]
# Source node to ATen node mapping:
#   loss => convert_element_type_10, full_default_14, full_default_15, log, log1p, maximum, maximum_1, mean, mul_367, mul_368, neg, sub_235, sub_236
#   target => convert_element_type_default_6
# Graph fragment:
#   %convert_element_type_default_6 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%inductor_random_default, torch.int64), kwargs = {})
#   %sub_235 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_default_6, 1), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze_1,), kwargs = {})
#   %log1p : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%neg,), kwargs = {})
#   %full_default_14 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -100), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %maximum : [num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%log1p, %full_default_14), kwargs = {})
#   %mul_367 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_235, %maximum), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%squeeze_1,), kwargs = {})
#   %full_default_15 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -100), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %maximum_1 : [num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%log, %full_default_15), kwargs = {})
#   %mul_368 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_6, %maximum_1), kwargs = {})
#   %sub_236 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_367, %mul_368), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sub_236,), kwargs = {})
#   %convert_element_type_10 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mean, torch.int64), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_binary_cross_entropy_rand_like_11(in_ptr0, out_ptr1, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 21
    R0_BLOCK: tl.constexpr = 32
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 300*ks0 + 120*ks0*(ks1 // 2) + 120*ks0*(ks2 // 2) + 48*ks0*(ks1 // 2)*(ks2 // 2)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp4 / tmp6
    tmp8 = tmp7.to(tl.int64)
    tl.store(out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp8, None)







def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        ps0 = 2 + (s2 // 2)
        ps1 = 2 + (s1 // 2)
        ps2 = 4 + 2*(s1 // 2) + 2*(s2 // 2) + (s1 // 2)*(s2 // 2)
        buf0 = empty_strided_cuda((1, s0, 2 + (s1 // 2), 2 + (s2 // 2)), (4*s0 + 2*s0*(s1 // 2) + 2*s0*(s2 // 2) + s0*(s1 // 2)*(s2 // 2), 4 + 2*(s1 // 2) + 2*(s2 // 2) + (s1 // 2)*(s2 // 2), 2 + (s2 // 2), 1), torch.int64)
        buf1 = empty_strided_cuda((1, s0, 2 + (s1 // 2), 2 + (s2 // 2)), (4*s0 + 2*s0*(s1 // 2) + 2*s0*(s2 // 2) + s0*(s1 // 2)*(s2 // 2), 4 + 2*(s1 // 2) + 2*(s2 // 2) + (s1 // 2)*(s2 // 2), 2 + (s2 // 2), 1), torch.int8)
        # Topologically Sorted Source Nodes: [x, max_pool2d], Original ATen: [aten.constant_pad_nd, aten.max_pool2d_with_indices]
        triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_0_xnumel = 4*s0 + 2*s0*(s1 // 2) + 2*s0*(s2 // 2) + s0*(s1 // 2)*(s2 // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_0[grid(triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_0_xnumel)](arg3_1, buf0, buf1, 34, 34, 64, 64, 1156, 3468, XBLOCK=128, num_warps=4, num_stages=1)
        del arg3_1
        buf2 = empty_strided_cuda((1, s0, 4 + 2*(s1 // 2), 4 + 2*(s2 // 2)), (16*s0 + 8*s0*(s1 // 2) + 8*s0*(s2 // 2) + 4*s0*(s1 // 2)*(s2 // 2), 16 + 8*(s1 // 2) + 8*(s2 // 2) + 4*(s1 // 2)*(s2 // 2), 4 + 2*(s2 // 2), 1), torch.int64)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.max_unpool2d]
        triton_poi_fused_max_unpool2d_1_xnumel = 16*s0 + 8*s0*(s1 // 2) + 8*s0*(s2 // 2) + 4*s0*(s1 // 2)*(s2 // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_unpool2d_1[grid(triton_poi_fused_max_unpool2d_1_xnumel)](buf2, 13872, XBLOCK=128, num_warps=4, num_stages=1)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.max_unpool2d]
        triton_poi_fused_max_unpool2d_2_xnumel = 4*s0 + 2*s0*(s1 // 2) + 2*s0*(s2 // 2) + s0*(s1 // 2)*(s2 // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_unpool2d_2[grid(triton_poi_fused_max_unpool2d_2_xnumel)](buf1, buf0, buf2, 34, 34, 64, 64, 3, 3468, XBLOCK=128, num_warps=4, num_stages=1)
        del buf0
        del buf1
        buf4 = empty_strided_cuda((1, s0, 3, 10 + 4*(s1 // 2), 10 + 4*(s2 // 2)), (300*s0 + 120*s0*(s1 // 2) + 120*s0*(s2 // 2) + 48*s0*(s1 // 2)*(s2 // 2), 300 + 120*(s1 // 2) + 120*(s2 // 2) + 48*(s1 // 2)*(s2 // 2), 100 + 40*(s1 // 2) + 40*(s2 // 2) + 16*(s1 // 2)*(s2 // 2), 10 + 4*(s2 // 2), 1), torch.int64)
        buf10 = empty_strided_cuda((8 + 4*(s1 // 2), 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten._to_copy, aten.sub, aten.clamp]
        triton_poi_fused__to_copy_clamp_sub_3_xnumel = 8 + 4*(s1 // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_clamp_sub_3[grid(triton_poi_fused__to_copy_clamp_sub_3_xnumel)](buf10, 64, 136, XBLOCK=256, num_warps=4, num_stages=1)
        buf7 = empty_strided_cuda((8 + 4*(s2 // 2), ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.arange, aten._to_copy, aten.clamp, aten.view, aten.sub]
        triton_poi_fused__to_copy_clamp_sub_3_xnumel = 8 + 4*(s2 // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_clamp_sub_3[grid(triton_poi_fused__to_copy_clamp_sub_3_xnumel)](buf7, 64, 136, XBLOCK=256, num_warps=4, num_stages=1)
        ps3 = 8 + 4*(s2 // 2)
        ps4 = 8 + 4*(s1 // 2)
        ps5 = 64 + 32*(s1 // 2) + 32*(s2 // 2) + 16*(s1 // 2)*(s2 // 2)
        buf11 = empty_strided_cuda((1, s0, 8 + 4*(s1 // 2), 8 + 4*(s2 // 2)), (64*s0 + 32*s0*(s1 // 2) + 32*s0*(s2 // 2) + 16*s0*(s1 // 2)*(s2 // 2), 64 + 32*(s1 // 2) + 32*(s2 // 2) + 16*(s1 // 2)*(s2 // 2), 8 + 4*(s2 // 2), 1), torch.int64)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten._to_copy, aten._unsafe_index, aten.sub, aten.mul, aten.add, aten.clamp, aten.round]
        triton_poi_fused__to_copy__unsafe_index_add_clamp_mul_round_sub_4_xnumel = 64*s0 + 32*s0*(s1 // 2) + 32*s0*(s2 // 2) + 16*s0*(s1 // 2)*(s2 // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_clamp_mul_round_sub_4[grid(triton_poi_fused__to_copy__unsafe_index_add_clamp_mul_round_sub_4_xnumel)](buf2, buf7, buf10, buf11, 64, 136, 136, 64, 18496, 3, 55488, XBLOCK=256, num_warps=4, num_stages=1)
        del buf10
        del buf2
        del buf7
        buf12 = empty_strided_cuda((1, 1, 8 + 4*(s1 // 2), 8 + 4*(s2 // 2)), (64 + 32*(s1 // 2) + 32*(s2 // 2) + 16*(s1 // 2)*(s2 // 2), 64 + 32*(s1 // 2) + 32*(s2 // 2) + 16*(s1 // 2)*(s2 // 2), 8 + 4*(s2 // 2), 1), torch.int64)
        buf13 = empty_strided_cuda((1, 1, 8 + 4*(s1 // 2), 8 + 4*(s2 // 2)), (64 + 32*(s1 // 2) + 32*(s2 // 2) + 16*(s1 // 2)*(s2 // 2), 64 + 32*(s1 // 2) + 32*(s2 // 2) + 16*(s1 // 2)*(s2 // 2), 8 + 4*(s2 // 2), 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten._softmax]
        triton_per_fused__softmax_5_xnumel = 64 + 32*(s1 // 2) + 32*(s2 // 2) + 16*(s1 // 2)*(s2 // 2)
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_5[grid(triton_per_fused__softmax_5_xnumel)](buf11, buf12, buf13, 64, 64, 18496, 3, XBLOCK=8, num_warps=2, num_stages=1)
        ps6 = 10 + 4*(s2 // 2)
        ps7 = 10 + 4*(s1 // 2)
        ps8 = 100 + 40*(s1 // 2) + 40*(s2 // 2) + 16*(s1 // 2)*(s2 // 2)
        ps9 = 300 + 120*(s1 // 2) + 120*(s2 // 2) + 48*(s1 // 2)*(s2 // 2)
        buf14 = empty_strided_cuda((1, s0, 3, 10 + 4*(s1 // 2), 10 + 4*(s2 // 2)), (300*s0 + 120*s0*(s1 // 2) + 120*s0*(s2 // 2) + 48*s0*(s1 // 2)*(s2 // 2), 300 + 120*(s1 // 2) + 120*(s2 // 2) + 48*(s1 // 2)*(s2 // 2), 100 + 40*(s1 // 2) + 40*(s2 // 2) + 16*(s1 // 2)*(s2 // 2), 10 + 4*(s2 // 2), 1), torch.int64)
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.copy]
        triton_poi_fused_copy_6_xnumel = 300*s0 + 120*s0*(s1 // 2) + 120*s0*(s2 // 2) + 48*s0*(s1 // 2)*(s2 // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_copy_6[grid(triton_poi_fused_copy_6_xnumel)](buf11, buf12, buf13, buf4, buf14, 138, 64, 138, 64, 19044, 57132, 171396, XBLOCK=512, num_warps=8, num_stages=1)
        del buf11
        del buf12
        del buf13
        buf15 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_7_xnumel = 300*s0 + 120*s0*(s1 // 2) + 120*s0*(s2 // 2) + 48*s0*(s1 // 2)*(s2 // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_7[grid(triton_poi_fused_7_xnumel)](buf14, buf15, 138, 138, 64, 64, 171396, XBLOCK=512, num_warps=8, num_stages=1)
        buf16 = empty_strided_cuda((1, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf16)
        buf17 = empty_strided_cuda((1, s0, 3, 10 + 4*(s1 // 2), 10 + 4*(s2 // 2)), (300*s0 + 120*s0*(s1 // 2) + 120*s0*(s2 // 2) + 48*s0*(s1 // 2)*(s2 // 2), 300 + 120*(s1 // 2) + 120*(s2 // 2) + 48*(s1 // 2)*(s2 // 2), 100 + 40*(s1 // 2) + 40*(s2 // 2) + 16*(s1 // 2)*(s2 // 2), 10 + 4*(s2 // 2), 1), torch.float32)
        # Topologically Sorted Source Nodes: [target], Original ATen: [aten.rand_like]
        triton_poi_fused_rand_like_8_xnumel = 300*s0 + 120*s0*(s1 // 2) + 120*s0*(s2 // 2) + 48*s0*(s1 // 2)*(s2 // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_rand_like_8[grid(triton_poi_fused_rand_like_8_xnumel)](buf16, buf17, 0, 171396, XBLOCK=512, num_warps=8, num_stages=1)
        ps10 = 100 + 40*(s1 // 2) + 40*(s2 // 2) + 16*(s1 // 2)*(s2 // 2)
        buf18 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_9_xnumel = 300*s0 + 120*s0*(s1 // 2) + 120*s0*(s2 // 2) + 48*s0*(s1 // 2)*(s2 // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_9[grid(triton_poi_fused_9_xnumel)](buf15, buf18, 19044, 19044, 138, 138, 64, 64, 171396, XBLOCK=512, num_warps=8, num_stages=1)
        del buf15
        buf19 = empty_strided_cuda((21, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [target, loss], Original ATen: [aten.rand_like, aten.binary_cross_entropy]
        triton_red_fused_binary_cross_entropy_rand_like_10_r0_numel = (20 + 300*s0 + 120*s0*(s1 // 2) + 120*s0*(s2 // 2) + 48*s0*(s1 // 2)*(s2 // 2)) // 21
        stream0 = get_raw_stream(0)
        triton_red_fused_binary_cross_entropy_rand_like_10[grid(21)](buf17, buf18, buf19, 3, 64, 64, 19044, 138, 138, 21, 8162, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del buf17
        del buf18
        buf21 = reinterpret_tensor(buf16, (), (), 0); del buf16  # reuse
        # Topologically Sorted Source Nodes: [target, loss], Original ATen: [aten.rand_like, aten.binary_cross_entropy]
        stream0 = get_raw_stream(0)
        triton_per_fused_binary_cross_entropy_rand_like_11[grid(1)](buf19, buf21, 3, 64, 64, 1, 21, XBLOCK=1, num_warps=2, num_stages=1)
        del buf19
    return (buf21, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 3
    arg1_1 = 64
    arg2_1 = 64
    arg3_1 = rand_strided((1, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
