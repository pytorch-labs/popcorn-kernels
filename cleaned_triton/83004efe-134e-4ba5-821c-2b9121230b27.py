# AOT ID: ['13_inference']
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


# kernel path: /tmp/torchinductor_sahanp/7u/c7ute72rlckfur2ehipuxufogbv2yt4rqr5z7wcsgntifcnabqqi.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   x_1 => add_6, mul_9, rsqrt, sub_2, var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_1, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_1, %getitem_1), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_6,), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_0(in_ptr0, out_ptr2, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + ks0*ks1*ks2*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(r0_mask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(r0_mask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(r0_mask & xmask, tmp2_weight_next, tmp2_weight)
    tmp5, tmp6, tmp7 = triton_helpers.welford(tmp2_mean, tmp2_m2, tmp2_weight, 1)
    tmp2 = tmp5[:, None]
    tmp3 = tmp6[:, None]
    tmp4 = tmp7[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp8 = tl.load(in_ptr0 + (r0_1 + ks0*ks1*ks2*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tmp8 - tmp2
        tmp10 = ks0*ks1*ks2
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tmp3 / tmp11
        tmp13 = 1e-05
        tmp14 = tmp12 + tmp13
        tmp15 = libdevice.rsqrt(tmp14)
        tmp16 = tmp9 * tmp15
        tl.store(out_ptr2 + (r0_1 + ks0*ks1*ks2*x0), tmp16, r0_mask & xmask)




# kernel path: /tmp/torchinductor_sahanp/uq/cuqrikhfertit4m4ssvbxvwzqueed65bz3hn7gpm55xrzfaqqxo4.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.avg_pool3d]
# Source node to ATen node mapping:
#   x_3 => avg_pool3d
# Graph fragment:
#   %avg_pool3d : [num_users=1] = call_function[target=torch.ops.aten.avg_pool3d.default](args = (%view_3, [2, 2, 2], [2, 2, 2]), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_avg_pool3d_1(in_ptr0, out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 4)
    x1 = ((xindex // 4) % 4)
    x2 = ((xindex // 16) % 4)
    x3 = xindex // 64
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 16*x1 + 128*x2 + ks0*ks1*ks2*x3), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 16*x1 + 128*x2 + ks0*ks1*ks2*x3), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (8 + 2*x0 + 16*x1 + 128*x2 + ks0*ks1*ks2*x3), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (9 + 2*x0 + 16*x1 + 128*x2 + ks0*ks1*ks2*x3), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (64 + 2*x0 + 16*x1 + 128*x2 + ks0*ks1*ks2*x3), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (65 + 2*x0 + 16*x1 + 128*x2 + ks0*ks1*ks2*x3), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (72 + 2*x0 + 16*x1 + 128*x2 + ks0*ks1*ks2*x3), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (73 + 2*x0 + 16*x1 + 128*x2 + ks0*ks1*ks2*x3), None, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp15 = 0.125
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (x4), tmp16, None)




# kernel path: /tmp/torchinductor_sahanp/ol/coluoqxa64bo4zyouqcuwfarpqqgv4urvauoynde5taizvwpakar.py
# Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.arange, aten._to_copy, aten.clamp, aten.view, aten.add]
# Source node to ATen node mapping:
#   x_5 => add_41, clamp_max_1, clamp_min_1, convert_element_type_2, convert_element_type_3, iota_1, view_6
# Graph fragment:
#   %iota_1 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (%floordiv,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_1, torch.float32), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -1.0), kwargs = {dtype: torch.float64, layout: torch.strided, device: cpu, pin_memory: False})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 4), kwargs = {dtype: torch.int64, layout: torch.strided, device: cpu, pin_memory: False})
#   %scalar_tensor_default_2 : [num_users=1] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (%arg0_1,), kwargs = {})
#   %scalar_tensor_default_3 : [num_users=1] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (%arg1_1,), kwargs = {})
#   %mul_tensor : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%scalar_tensor_default_2, %scalar_tensor_default_3), kwargs = {})
#   %scalar_tensor_default_4 : [num_users=1] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (%arg2_1,), kwargs = {})
#   %mul_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_tensor, %scalar_tensor_default_4), kwargs = {})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 128), kwargs = {dtype: torch.int64, layout: torch.strided, device: cpu, pin_memory: False})
#   %div_tensor_mode : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor_mode](args = (%mul_tensor_1, %full_default_2), kwargs = {rounding_mode: floor})
#   %mul_tensor_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_1, %div_tensor_mode), kwargs = {})
#   %convert_element_type_default : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_2, torch.float64), kwargs = {})
#   %add_tensor : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%full_default, %convert_element_type_default), kwargs = {})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -1.0), kwargs = {dtype: torch.float64, layout: torch.strided, device: cpu, pin_memory: False})
#   %full_default_4 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 8), kwargs = {dtype: torch.int64, layout: torch.strided, device: cpu, pin_memory: False})
#   %mul_tensor_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_4, %div_tensor_mode), kwargs = {})
#   %convert_element_type_default_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_3, torch.float64), kwargs = {})
#   %add_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%full_default_3, %convert_element_type_default_1), kwargs = {})
#   %true_divide_tensor : [num_users=1] = call_function[target=torch.ops.aten.true_divide.Tensor](args = (%add_tensor, %add_tensor_1), kwargs = {})
#   %convert_element_type_default_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%true_divide_tensor, torch.float32), kwargs = {})
#   %mul_tensor_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_2, %convert_element_type_default_2), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_tensor_4, 0.0), kwargs = {})
#   %view_6 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clamp_min_1, [%floordiv]), kwargs = {})
#   %convert_element_type_3 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_6, torch.int64), kwargs = {})
#   %add_41 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_3, 1), kwargs = {})
#   %clamp_max_1 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_41, %sub_19), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_view_2(out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = ks0*ks1*ks2
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 128.0
    tmp3 = tmp1 / tmp2
    tmp4 = libdevice.floor(tmp3)
    tmp5 = 4.0
    tmp6 = tmp5 * tmp4
    tmp7 = tmp6.to(tl.float64)
    tmp8 = tl.full([1], -1.0, tl.float64)
    tmp9 = tmp8 + tmp7
    tmp10 = 8.0
    tmp11 = tmp10 * tmp4
    tmp12 = tmp11.to(tl.float64)
    tmp13 = tmp8 + tmp12
    tmp14 = tmp9 / tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = x0
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp17 * tmp15
    tmp19 = 0.0
    tmp20 = triton_helpers.maximum(tmp18, tmp19)
    tmp21 = tmp20.to(tl.int64)
    tmp22 = tl.full([1], 1, tl.int64)
    tmp23 = tmp21 + tmp22
    tmp24 = (-1) + 4*((ks0*ks1*ks2) // 128)
    tmp25 = triton_helpers.minimum(tmp23, tmp24)
    tl.store(out_ptr0 + (x0), tmp25, xmask)




# kernel path: /tmp/torchinductor_sahanp/47/c47ee5ympnsvr3yxo35oxhiwekd5uqtuzac5dteoj4tg6xfzppgb.py
# Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.arange, aten._to_copy, aten.clamp, aten.view, aten.sub]
# Source node to ATen node mapping:
#   x_5 => clamp_max_2, clamp_min_1, clamp_min_2, convert_element_type_2, convert_element_type_3, iota_1, sub_25, view_6
# Graph fragment:
#   %iota_1 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (%floordiv,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_1, torch.float32), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -1.0), kwargs = {dtype: torch.float64, layout: torch.strided, device: cpu, pin_memory: False})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 4), kwargs = {dtype: torch.int64, layout: torch.strided, device: cpu, pin_memory: False})
#   %scalar_tensor_default_2 : [num_users=1] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (%arg0_1,), kwargs = {})
#   %scalar_tensor_default_3 : [num_users=1] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (%arg1_1,), kwargs = {})
#   %mul_tensor : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%scalar_tensor_default_2, %scalar_tensor_default_3), kwargs = {})
#   %scalar_tensor_default_4 : [num_users=1] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (%arg2_1,), kwargs = {})
#   %mul_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_tensor, %scalar_tensor_default_4), kwargs = {})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 128), kwargs = {dtype: torch.int64, layout: torch.strided, device: cpu, pin_memory: False})
#   %div_tensor_mode : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor_mode](args = (%mul_tensor_1, %full_default_2), kwargs = {rounding_mode: floor})
#   %mul_tensor_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_1, %div_tensor_mode), kwargs = {})
#   %convert_element_type_default : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_2, torch.float64), kwargs = {})
#   %add_tensor : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%full_default, %convert_element_type_default), kwargs = {})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -1.0), kwargs = {dtype: torch.float64, layout: torch.strided, device: cpu, pin_memory: False})
#   %full_default_4 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 8), kwargs = {dtype: torch.int64, layout: torch.strided, device: cpu, pin_memory: False})
#   %mul_tensor_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_4, %div_tensor_mode), kwargs = {})
#   %convert_element_type_default_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_3, torch.float64), kwargs = {})
#   %add_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%full_default_3, %convert_element_type_default_1), kwargs = {})
#   %true_divide_tensor : [num_users=1] = call_function[target=torch.ops.aten.true_divide.Tensor](args = (%add_tensor, %add_tensor_1), kwargs = {})
#   %convert_element_type_default_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%true_divide_tensor, torch.float32), kwargs = {})
#   %mul_tensor_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_2, %convert_element_type_default_2), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_tensor_4, 0.0), kwargs = {})
#   %view_6 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clamp_min_1, [%floordiv]), kwargs = {})
#   %convert_element_type_3 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_6, torch.int64), kwargs = {})
#   %sub_25 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_6, %convert_element_type_3), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_25, 0.0), kwargs = {})
#   %clamp_max_2 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 1.0), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_arange_clamp_sub_view_3(out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = ks0*ks1*ks2
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 128.0
    tmp3 = tmp1 / tmp2
    tmp4 = libdevice.floor(tmp3)
    tmp5 = 4.0
    tmp6 = tmp5 * tmp4
    tmp7 = tmp6.to(tl.float64)
    tmp8 = tl.full([1], -1.0, tl.float64)
    tmp9 = tmp8 + tmp7
    tmp10 = 8.0
    tmp11 = tmp10 * tmp4
    tmp12 = tmp11.to(tl.float64)
    tmp13 = tmp8 + tmp12
    tmp14 = tmp9 / tmp13
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
    tl.store(out_ptr0 + (x0), tmp26, xmask)




# kernel path: /tmp/torchinductor_sahanp/a6/ca6moj7i7yqdgx24v3ttnrdeehd347ea5775v5wbuv2mzlczxgdz.py
# Topologically Sorted Source Nodes: [x_5, x_6], Original ATen: [aten._to_copy, aten._unsafe_index, aten.sub, aten.mul, aten.add, aten.clamp, aten.gelu]
# Source node to ATen node mapping:
#   x_5 => _unsafe_index, _unsafe_index_1, _unsafe_index_2, _unsafe_index_3, add_100, add_74, add_87, clamp_max_3, clamp_min_3, convert_element_type_1, mul_63, mul_76, mul_89, sub_28, sub_32, sub_36, sub_37
#   x_6 => add_105, erf, mul_102, mul_103, mul_104
# Graph fragment:
#   %convert_element_type_1 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_5, torch.int64), kwargs = {})
#   %_unsafe_index_3 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%view_4, [None, None, %clamp_max, %clamp_max_1]), kwargs = {})
#   %_unsafe_index_2 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%view_4, [None, None, %clamp_max, %convert_element_type_3]), kwargs = {})
#   %sub_32 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_3, %_unsafe_index_2), kwargs = {})
#   %mul_76 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_32, %clamp_max_2), kwargs = {})
#   %add_87 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_2, %mul_76), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%view_4, [None, None, %convert_element_type_1, %clamp_max_1]), kwargs = {})
#   %_unsafe_index : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%view_4, [None, None, %convert_element_type_1, %convert_element_type_3]), kwargs = {})
#   %sub_28 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_1, %_unsafe_index), kwargs = {})
#   %mul_63 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_28, %clamp_max_2), kwargs = {})
#   %add_74 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index, %mul_63), kwargs = {})
#   %sub_37 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_87, %add_74), kwargs = {})
#   %sub_36 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_5, %convert_element_type_1), kwargs = {})
#   %clamp_min_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_36, 0.0), kwargs = {})
#   %clamp_max_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_3, 1.0), kwargs = {})
#   %mul_89 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_37, %clamp_max_3), kwargs = {})
#   %add_100 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_74, %mul_89), kwargs = {})
#   %mul_102 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_100, 0.5), kwargs = {})
#   %mul_103 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_100, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_103,), kwargs = {})
#   %add_105 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_104 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_102, %add_105), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_clamp_gelu_mul_sub_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // ks0) % 8)
    x0 = (xindex % ks0)
    x2 = xindex // ks4
    x3 = xindex
    tmp34 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.42857142857142855
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full([1], 3, tl.int64)
    tmp10 = triton_helpers.minimum(tmp8, tmp9)
    tmp11 = ks1*ks2*ks3
    tmp12 = tmp11.to(tl.float32)
    tmp13 = 128.0
    tmp14 = tmp12 / tmp13
    tmp15 = libdevice.floor(tmp14)
    tmp16 = 4.0
    tmp17 = tmp16 * tmp15
    tmp18 = tmp17.to(tl.float64)
    tmp19 = tl.full([1], -1.0, tl.float64)
    tmp20 = tmp19 + tmp18
    tmp21 = 8.0
    tmp22 = tmp21 * tmp15
    tmp23 = tmp22.to(tl.float64)
    tmp24 = tmp19 + tmp23
    tmp25 = tmp20 / tmp24
    tmp26 = tmp25.to(tl.float32)
    tmp27 = x0
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp28 * tmp26
    tmp30 = triton_helpers.maximum(tmp29, tmp4)
    tmp31 = tmp30.to(tl.int64)
    tmp32 = tl.load(in_ptr0 + (16*tmp10 + 64*x2 + ((tmp31 % 16))), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr0 + (16*tmp6 + 64*x2 + ((tmp31 % 16))), None, eviction_policy='evict_last')
    tmp35 = tl.full([XBLOCK], 16, tl.int32)
    tmp36 = tmp34 + tmp35
    tmp37 = tmp34 < 0
    tmp38 = tl.where(tmp37, tmp36, tmp34)
    tmp39 = tl.load(in_ptr0 + (16*tmp6 + 64*x2 + ((tmp38 % 16))), None, eviction_policy='evict_last')
    tmp40 = tmp39 - tmp33
    tmp42 = tmp40 * tmp41
    tmp43 = tmp33 + tmp42
    tmp44 = tl.load(in_ptr0 + (16*tmp10 + 64*x2 + ((tmp38 % 16))), None, eviction_policy='evict_last')
    tmp45 = tmp44 - tmp32
    tmp46 = tmp45 * tmp41
    tmp47 = tmp32 + tmp46
    tmp48 = tmp47 - tmp43
    tmp49 = tmp6.to(tl.float32)
    tmp50 = tmp5 - tmp49
    tmp51 = triton_helpers.maximum(tmp50, tmp4)
    tmp52 = 1.0
    tmp53 = triton_helpers.minimum(tmp51, tmp52)
    tmp54 = tmp48 * tmp53
    tmp55 = tmp43 + tmp54
    tmp56 = 0.5
    tmp57 = tmp55 * tmp56
    tmp58 = 0.7071067811865476
    tmp59 = tmp55 * tmp58
    tmp60 = libdevice.erf(tmp59)
    tmp61 = tmp60 + tmp52
    tmp62 = tmp57 * tmp61
    tl.store(in_out_ptr0 + (x3), tmp62, None)







def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s1 = arg0_1
    s2 = arg1_1
    s3 = arg2_1
    assert_size_stride(arg3_1, (1, 64, s1, s2, s3), (64*s1*s2*s3, s1*s2*s3, s2*s3, s3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf3 = empty_strided_cuda((1, 64, s1*s2*s3), (64*s1*s2*s3, s1*s2*s3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit]
        triton_red_fused__native_batch_norm_legit_0_r0_numel = s1*s2*s3
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_0[grid(64)](arg3_1, buf3, 8, 8, 8, 64, 512, XBLOCK=1, R0_BLOCK=512, num_warps=4, num_stages=1)
        del arg3_1
        buf4 = empty_strided_cuda((1, 64, 4, 4, 4), (4096, 64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.avg_pool3d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool3d_1[grid(4096)](buf3, buf4, 8, 8, 8, 4096, XBLOCK=128, num_warps=4, num_stages=1)
        del buf3
        buf5 = empty_strided_cuda((8*((s1*s2*s3) // 128), ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.arange, aten._to_copy, aten.clamp, aten.view, aten.add]
        triton_poi_fused__to_copy_add_arange_clamp_view_2_xnumel = 8*((s1*s2*s3) // 128)
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_view_2[grid(triton_poi_fused__to_copy_add_arange_clamp_view_2_xnumel)](buf5, 8, 8, 8, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf7 = empty_strided_cuda((8*((s1*s2*s3) // 128), ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.arange, aten._to_copy, aten.clamp, aten.view, aten.sub]
        triton_poi_fused__to_copy_arange_clamp_sub_view_3_xnumel = 8*((s1*s2*s3) // 128)
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_sub_view_3[grid(triton_poi_fused__to_copy_arange_clamp_sub_view_3_xnumel)](buf7, 8, 8, 8, 32, XBLOCK=32, num_warps=1, num_stages=1)
        ps0 = 8*((s1*s2*s3) // 128)
        ps1 = 64*((s1*s2*s3) // 128)
        buf8 = empty_strided_cuda((1, 64, 8, 8*((s1*s2*s3) // 128)), (4096*((s1*s2*s3) // 128), 64*((s1*s2*s3) // 128), 8*((s1*s2*s3) // 128), 1), torch.float32)
        buf9 = buf8; del buf8  # reuse
        buf10 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [x_5, x_6], Original ATen: [aten._to_copy, aten._unsafe_index, aten.sub, aten.mul, aten.add, aten.clamp, aten.gelu]
        triton_poi_fused__to_copy__unsafe_index_add_clamp_gelu_mul_sub_4_xnumel = 4096*((s1*s2*s3) // 128)
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_clamp_gelu_mul_sub_4[grid(triton_poi_fused__to_copy__unsafe_index_add_clamp_gelu_mul_sub_4_xnumel)](buf10, buf4, buf5, buf7, 32, 8, 8, 8, 256, 16384, XBLOCK=256, num_warps=4, num_stages=1)
        del buf4
        del buf5
        del buf7
    return (reinterpret_tensor(buf10, (1, 64, 64*((s1*s2*s3) // 128)), (4096*((s1*s2*s3) // 128), 64*((s1*s2*s3) // 128), 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 8
    arg1_1 = 8
    arg2_1 = 8
    arg3_1 = rand_strided((1, 64, 8, 8, 8), (32768, 512, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
