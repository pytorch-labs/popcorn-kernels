# AOT ID: ['50_inference']
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


# kernel path: /tmp/torchinductor_sahanp/io/cioiswspoyuhngbnu5vhbsgjfyx7jwobn327qzfwtscrfsnfnw6k.py
# Topologically Sorted Source Nodes: [max_pool1d], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   max_pool1d => _low_memory_max_pool2d_with_offsets
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%unsqueeze_1, [1, 2], [1, 2], [0, 0], [1, 1], False), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_0(in_ptr0, out_ptr0, out_ptr1, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % ks2)
    x1 = xindex // ks2
    tmp0 = 2.0
    tmp1 = ks0
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float64)
    tmp5 = tl.full([1], 2.0, tl.float64)
    tmp6 = tmp5 * tmp4
    tmp7 = tmp4 / tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = ((x2 // (2 + ks1)) % (4 + 2*ks0))
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp10 * tmp8
    tmp12 = tmp11.to(tl.int64)
    tmp13 = 2 + ks0
    tmp14 = tmp12 + tmp13
    tmp15 = tmp12 < 0
    tmp16 = tl.where(tmp15, tmp14, tmp12)
    tmp17 = ks1
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp0 + tmp18
    tmp20 = tmp19.to(tl.float64)
    tmp21 = tmp5 * tmp20
    tmp22 = tmp20 / tmp21
    tmp23 = tmp22.to(tl.float32)
    tmp24 = 2*x0
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp25 * tmp23
    tmp27 = tmp26.to(tl.int64)
    tmp28 = ks2
    tmp29 = tmp27 + tmp28
    tmp30 = tmp27 < 0
    tmp31 = tl.where(tmp30, tmp29, tmp27)
    tmp32 = (-1) + tmp16
    tmp33 = tmp32.to(tl.int32)
    tmp34 = tl.full([1], 0, tl.int64)
    tmp35 = tmp33 >= tmp34
    tmp36 = tmp33 < tmp1
    tmp37 = (-1) + tmp31
    tmp38 = tmp37.to(tl.int32)
    tmp39 = tmp38 >= tmp34
    tmp40 = tmp38 < tmp17
    tmp41 = tmp35 & tmp36
    tmp42 = tmp41 & tmp39
    tmp43 = tmp42 & tmp40
    tmp44 = tl.load(in_ptr0 + ((-1) + tmp31 + ((-1)*ks1) + ks1*tmp16 + ks0*ks1*(x1 // (4 + 2*ks0))), tmp43 & xmask, eviction_policy='evict_last', other=0.5)
    tmp45 = ((x2 // ks2) % (4 + 2*ks0))
    tmp46 = tmp45.to(tl.float32)
    tmp47 = tmp46 * tmp8
    tmp48 = tmp47.to(tl.int64)
    tmp49 = tmp48 + tmp13
    tmp50 = tmp48 < 0
    tmp51 = tl.where(tmp50, tmp49, tmp48)
    tmp52 = 1 + 2*x0
    tmp53 = tmp52.to(tl.float32)
    tmp54 = tmp53 * tmp23
    tmp55 = tmp54.to(tl.int64)
    tmp56 = tmp55 + tmp28
    tmp57 = tmp55 < 0
    tmp58 = tl.where(tmp57, tmp56, tmp55)
    tmp59 = (-1) + tmp51
    tmp60 = tmp59.to(tl.int32)
    tmp61 = tmp60 >= tmp34
    tmp62 = tmp60 < tmp1
    tmp63 = (-1) + tmp58
    tmp64 = tmp63.to(tl.int32)
    tmp65 = tmp64 >= tmp34
    tmp66 = tmp64 < tmp17
    tmp67 = tmp61 & tmp62
    tmp68 = tmp67 & tmp65
    tmp69 = tmp68 & tmp66
    tmp70 = tl.load(in_ptr0 + ((-1) + tmp58 + ((-1)*ks1) + ks1*tmp51 + ks0*ks1*(x1 // (4 + 2*ks0))), tmp69 & xmask, eviction_policy='evict_last', other=0.5)
    tmp71 = triton_helpers.maximum(tmp70, tmp44)
    tmp72 = tmp67 & tmp39
    tmp73 = tmp72 & tmp40
    tmp74 = tl.load(in_ptr0 + ((-1) + tmp31 + ((-1)*ks1) + ks1*tmp51 + ks0*ks1*(x1 // (4 + 2*ks0))), tmp73 & xmask, eviction_policy='evict_last', other=0.5)
    tmp75 = tmp70 > tmp74
    tmp76 = tl.full([1], 1, tl.int8)
    tmp77 = tl.full([1], 0, tl.int8)
    tmp78 = tl.where(tmp75, tmp76, tmp77)
    tmp79 = triton_helpers.maximum(tmp70, tmp74)
    tl.store(out_ptr0 + (x2), tmp71, xmask)
    tl.store(out_ptr1 + (x2), tmp78, xmask)




# kernel path: /tmp/torchinductor_sahanp/ed/ceds4yuzbugem5hwhns5nxw7i7vfi5mkzucj7dng6sicmv6uri44.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.max_unpool2d]
# Source node to ATen node mapping:
#   x_3 => full
# Graph fragment:
#   %full : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, %mul_32, %sym_sum_1, 1], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
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
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/xa/cxaktuzhqmbuxttadczfbsgt5zbxkc2r2hrin5i3zaoywjxpluoy.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.max_unpool2d]
# Source node to ATen node mapping:
#   x_3 => index_put
# Graph fragment:
#   %index_put : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%view_3, [%view_2], %view_4), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_unpool2d_2(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2*(x0 // ks0) + ks1*(x0 // ks0) + ((x0 % ks0))), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = tl.full([1], 2, tl.int32)
    tmp2 = tl.where((tmp0 < 0) != (tmp1 < 0), tl.where(tmp0 % tmp1 != 0, tmp0 // tmp1 - 1, tmp0 // tmp1), tmp0 // tmp1)
    tmp3 = tmp2 * tmp1
    tmp4 = tmp0 - tmp3
    tmp5 = tl.full([1], 0, tl.int64)
    tmp6 = tmp5 + tmp2
    tmp7 = 2*((x0 % ks0))
    tmp8 = tmp7 + tmp4
    tmp9 = 4 + 2*ks1
    tmp10 = tmp6 * tmp9
    tmp11 = tmp10 + tmp8
    tmp12 = 4*(x0 // ks0) + 2*ks1*(x0 // ks0)
    tmp13 = tmp11 + tmp12
    tmp14 = 16*ks2 + 8*ks1*ks2 + 8*ks2*ks3 + 4*ks1*ks2*ks3
    tmp15 = tmp13 + tmp14
    tmp16 = tmp13 < 0
    tmp17 = tl.where(tmp16, tmp15, tmp13)
    tl.device_assert(((0 <= tmp17) & (tmp17 < 16*ks2 + 8*ks1*ks2 + 8*ks2*ks3 + 4*ks1*ks2*ks3)) | ~(xmask), "index out of bounds: 0 <= tmp17 < 16*ks2 + 8*ks1*ks2 + 8*ks2*ks3 + 4*ks1*ks2*ks3")
    tl.store(out_ptr0 + (tl.broadcast_to(4*(((tmp17 // (4 + 2*ks1)) % (4*ks2 + 2*ks2*ks3))) + 2*ks1*(((tmp17 // (4 + 2*ks1)) % (4*ks2 + 2*ks2*ks3))) + ((tmp17 % (4 + 2*ks1))), [XBLOCK])), tmp19, xmask)




# kernel path: /tmp/torchinductor_sahanp/x4/cx4dth5td7lw2ucun3uonpjowvexcs4vlwhbozajvjvqnq55n7il.py
# Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_5 => add_76, erf, mul_73, mul_74, mul_75
# Graph fragment:
#   %mul_73 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_6, 0.5), kwargs = {})
#   %mul_74 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_6, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_74,), kwargs = {})
#   %add_76 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_75 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_73, %add_76), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_gelu_3(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks1)
    x2 = xindex // ks2
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4*(((x1 + 4*x2 + 2*ks4*x2) % (4*ks3 + 2*ks3*ks4))) + 2*ks5*(((x1 + 4*x2 + 2*ks4*x2) % (4*ks3 + 2*ks3*ks4)))), xmask, eviction_policy='evict_last')
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = libdevice.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tl.store(out_ptr0 + (x3), tmp8, xmask)




# kernel path: /tmp/torchinductor_sahanp/4z/c4z7o6iuqmvqervmzgfezq5ybnm2dtzzmhqmofwq2fpoaigqpa25.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten.ne, aten.fill, aten.sub, aten.clamp_min, aten.zeros_like, aten.where, aten.add, aten.mean]
# Source node to ATen node mapping:
#   loss => add_141, clamp_min, full_default_4, full_default_5, full_default_6, full_default_7, mean, sub_70, where, where_1
# Graph fragment:
#   %full_default_6 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, %arg0_1, %sym_sum, %sym_sum_1], False), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_5 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, %arg0_1, %sym_sum, %sym_sum_1], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sub_70 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default_5, %mul_75), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_70, 0), kwargs = {})
#   %full_default_4 : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([1, %arg0_1, %sym_sum, %sym_sum_1], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%full_default_6, %clamp_min, %full_default_4), kwargs = {})
#   %full_default_7 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, %arg0_1, %sym_sum, %sym_sum_1], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%full_default_7, %mul_75, %full_default_4), kwargs = {})
#   %add_141 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where, %where_1), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%add_141,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_clamp_min_fill_mean_ne_sub_where_zeros_like_4(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp11 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (4*((((r0_1 + 8*ks2*x0 + 4*ks2*ks3*x0 + 4*ks2*ks4*x0 + 2*ks2*ks3*ks4*x0) // ks0) % ks1)) + 16*((((r0_1 + 8*ks2*x0 + 4*ks2*ks3*x0 + 4*ks2*ks4*x0 + 2*ks2*ks3*ks4*x0) // (16 + 8*ks3 + 8*ks4 + 4*ks3*ks4)) % ks2)) + 2*ks4*((((r0_1 + 8*ks2*x0 + 4*ks2*ks3*x0 + 4*ks2*ks4*x0 + 2*ks2*ks3*ks4*x0) // ks0) % ks1)) + 8*ks3*((((r0_1 + 8*ks2*x0 + 4*ks2*ks3*x0 + 4*ks2*ks4*x0 + 2*ks2*ks3*ks4*x0) // (16 + 8*ks3 + 8*ks4 + 4*ks3*ks4)) % ks2)) + 8*ks4*((((r0_1 + 8*ks2*x0 + 4*ks2*ks3*x0 + 4*ks2*ks4*x0 + 2*ks2*ks3*ks4*x0) // (16 + 8*ks3 + 8*ks4 + 4*ks3*ks4)) % ks2)) + 4*ks3*ks4*((((r0_1 + 8*ks2*x0 + 4*ks2*ks3*x0 + 4*ks2*ks4*x0 + 2*ks2*ks3*ks4*x0) // (16 + 8*ks3 + 8*ks4 + 4*ks3*ks4)) % ks2)) + ((r0_1 % ks0))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 1.0
        tmp2 = tmp1 - tmp0
        tmp3 = 0.0
        tmp4 = triton_helpers.maximum(tmp2, tmp3)
        tmp5 = tl.full([1, 1], False, tl.int1)
        tmp6 = tl.where(tmp5, tmp4, tmp3)
        tmp7 = tl.full([1, 1], True, tl.int1)
        tmp8 = tl.where(tmp7, tmp0, tmp3)
        tmp9 = tmp6 + tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(r0_mask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp11, xmask)




# kernel path: /tmp/torchinductor_sahanp/vq/cvqzgtvpha2gavaf4ce622ora4cf736v5qg4zo35a3ihvomicen4.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten.ne, aten.fill, aten.sub, aten.clamp_min, aten.zeros_like, aten.where, aten.add, aten.mean]
# Source node to ATen node mapping:
#   loss => add_141, clamp_min, full_default_4, full_default_5, full_default_6, full_default_7, mean, sub_70, where, where_1
# Graph fragment:
#   %full_default_6 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, %arg0_1, %sym_sum, %sym_sum_1], False), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_5 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, %arg0_1, %sym_sum, %sym_sum_1], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sub_70 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default_5, %mul_75), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_70, 0), kwargs = {})
#   %full_default_4 : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([1, %arg0_1, %sym_sum, %sym_sum_1], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%full_default_6, %clamp_min, %full_default_4), kwargs = {})
#   %full_default_7 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, %arg0_1, %sym_sum, %sym_sum_1], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%full_default_7, %mul_75, %full_default_4), kwargs = {})
#   %add_141 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where, %where_1), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%add_141,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_clamp_min_fill_mean_ne_sub_where_zeros_like_5(in_out_ptr0, in_ptr0, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 2
    R0_BLOCK: tl.constexpr = 2
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None]
    tmp4 = 16*ks0 + 8*ks0*ks1 + 8*ks0*ks2 + 4*ks0*ks1*ks2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp6, None)







def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        ps0 = 2 + s2
        buf0 = empty_strided_cuda((1, 4*s0 + 2*s0*s1, 1, 2 + s2), (8*s0 + 4*s0*s1 + 4*s0*s2 + 2*s0*s1*s2, 2 + s2, 2 + s2, 1), torch.float32)
        buf1 = empty_strided_cuda((1, 4*s0 + 2*s0*s1, 1, 2 + s2), (8*s0 + 4*s0*s1 + 4*s0*s2 + 2*s0*s1*s2, 2 + s2, 8*s0 + 4*s0*s1 + 4*s0*s2 + 2*s0*s1*s2, 1), torch.int8)
        # Topologically Sorted Source Nodes: [max_pool1d], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_0_xnumel = 8*s0 + 4*s0*s1 + 4*s0*s2 + 2*s0*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_0[grid(triton_poi_fused_max_pool2d_with_indices_0_xnumel)](arg3_1, buf0, buf1, 32, 32, 34, 6936, XBLOCK=128, num_warps=4, num_stages=1)
        del arg3_1
        buf2 = empty_strided_cuda((1, 4*s0 + 2*s0*s1, 4 + 2*s2, 1), (16*s0 + 8*s0*s1 + 8*s0*s2 + 4*s0*s1*s2, 4 + 2*s2, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.max_unpool2d]
        triton_poi_fused_max_unpool2d_1_xnumel = 16*s0 + 8*s0*s1 + 8*s0*s2 + 4*s0*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_unpool2d_1[grid(triton_poi_fused_max_unpool2d_1_xnumel)](buf2, 13872, XBLOCK=256, num_warps=4, num_stages=1)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.max_unpool2d]
        triton_poi_fused_max_unpool2d_2_xnumel = 8*s0 + 4*s0*s1 + 4*s0*s2 + 2*s0*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_unpool2d_2[grid(triton_poi_fused_max_unpool2d_2_xnumel)](buf1, buf0, buf2, 34, 32, 3, 32, 6936, XBLOCK=128, num_warps=4, num_stages=1)
        del buf0
        del buf1
        ps1 = 4 + 2*s2
        ps2 = 4 + 2*s1
        ps3 = 16 + 8*s1 + 8*s2 + 4*s1*s2
        buf4 = empty_strided_cuda((1, s0, 4 + 2*s1, 4 + 2*s2), (16*s0 + 8*s0*s1 + 8*s0*s2 + 4*s0*s1*s2, 16 + 8*s1 + 8*s2 + 4*s1*s2, 4 + 2*s2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3_xnumel = 16*s0 + 8*s0*s1 + 8*s0*s2 + 4*s0*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_3[grid(triton_poi_fused_gelu_3_xnumel)](buf2, buf4, 68, 68, 4624, 3, 32, 32, 13872, XBLOCK=256, num_warps=4, num_stages=1)
        del buf2
        buf5 = empty_strided_cuda((2, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [loss], Original ATen: [aten.ne, aten.fill, aten.sub, aten.clamp_min, aten.zeros_like, aten.where, aten.add, aten.mean]
        triton_red_fused_add_clamp_min_fill_mean_ne_sub_where_zeros_like_4_r0_numel = 8*s0 + 4*s0*s1 + 4*s0*s2 + 2*s0*s1*s2
        stream0 = get_raw_stream(0)
        triton_red_fused_add_clamp_min_fill_mean_ne_sub_where_zeros_like_4[grid(2)](buf4, buf5, 68, 68, 3, 32, 32, 2, 6936, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        buf6 = empty_strided_cuda((), (), torch.float32)
        buf7 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [loss], Original ATen: [aten.ne, aten.fill, aten.sub, aten.clamp_min, aten.zeros_like, aten.where, aten.add, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clamp_min_fill_mean_ne_sub_where_zeros_like_5[grid(1)](buf7, buf5, 3, 32, 32, 1, 2, XBLOCK=1, num_warps=2, num_stages=1)
        del buf5
    return (buf4, buf7, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 3
    arg1_1 = 32
    arg2_1 = 32
    arg3_1 = rand_strided((1, 3, 32, 32), (3072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
