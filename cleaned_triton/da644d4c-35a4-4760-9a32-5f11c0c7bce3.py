# AOT ID: ['169_forward']
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


# kernel path: /tmp/torchinductor_sahanp/wm/cwmndgpb2ietajjlqut2wjvavuafffznsx45cigaytcbcerlixx7.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.copy]
# Source node to ATen node mapping:
#   x => copy
# Graph fragment:
#   %copy : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_3, %slice_4), kwargs = {})
#   %slice_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_tensor, %copy, 2, 2, %add), kwargs = {})
#   %slice_scatter_default_1 : [num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%empty, %slice_scatter_default, 3, 2, %add_2), kwargs = {})
#   %slice_scatter_default_2 : [num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_1, %slice_11, 3, 0, 2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_copy_0(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks2)
    x2 = xindex // ks4
    x4 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = ks1 + x0
    tmp4 = tl.full([1], 2, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tl.broadcast_to(2 + ks1, [XBLOCK])
    tmp7 = tmp3 < tmp6
    tmp8 = tmp5 & tmp7
    tmp9 = tmp8 & tmp2
    tmp10 = x1
    tmp11 = tl.full([1], 2, tl.int64)
    tmp12 = tmp10 >= tmp11
    tmp13 = tl.broadcast_to(2 + ks3, [XBLOCK])
    tmp14 = tmp10 < tmp13
    tmp15 = tmp12 & tmp14
    tmp16 = tmp15 & tmp9
    tmp17 = tl.load(in_ptr0 + ((-2) + x0 + ((-1)*ks1) + ks1*x1 + ks1*ks3*x2), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.load(in_ptr1 + (ks1 + x4), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.where(tmp15, tmp17, tmp18)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp9, tmp19, tmp20)
    tmp22 = float("nan")
    tmp23 = tl.where(tmp8, tmp21, tmp22)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp2, tmp23, tmp24)
    tmp26 = tmp0 >= tmp1
    tmp27 = 2 + ks1
    tmp28 = tmp0 < tmp27
    tmp29 = tmp26 & tmp28
    tmp30 = x1
    tmp31 = tl.full([1], 2, tl.int64)
    tmp32 = tmp30 >= tmp31
    tmp33 = tl.broadcast_to(2 + ks3, [XBLOCK])
    tmp34 = tmp30 < tmp33
    tmp35 = tmp32 & tmp34
    tmp36 = tmp35 & tmp29
    tmp37 = tl.load(in_ptr0 + ((-2) + x0 + ((-2)*ks1) + ks1*x1 + ks1*ks3*x2), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.load(in_ptr1 + (x4), tmp29 & xmask, eviction_policy='evict_last', other=0.0)
    tmp39 = tl.where(tmp35, tmp37, tmp38)
    tmp40 = tl.full(tmp39.shape, 0.0, tmp39.dtype)
    tmp41 = tl.where(tmp29, tmp39, tmp40)
    tmp42 = float("nan")
    tmp43 = tl.where(tmp29, tmp41, tmp42)
    tmp44 = tl.where(tmp2, tmp25, tmp43)
    tl.store(out_ptr0 + (x4), tmp44, xmask)




# kernel path: /tmp/torchinductor_sahanp/p6/cp6auu4s52xcwgoicy23b3oeiuxuzh53qsphd5z4bzbujgyubm5z.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %slice_scatter_default_3 : [num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_2, %slice_16, 3, %add_2, %add_3), kwargs = {})
#   %slice_scatter_default_4 : [num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_3, %slice_21, 2, 0, 2), kwargs = {})
#   %slice_scatter_default_5 : [num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_4, %slice_26, 2, %add, %add_1), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // ks0) % ks1)
    x0 = (xindex % ks0)
    x3 = xindex
    tmp41 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = 2 + ks2
    tmp2 = tmp0 >= tmp1
    tmp3 = x1 + ((-1)*ks2)
    tmp4 = tl.full([1], 2, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = x0
    tmp8 = tl.broadcast_to(2 + ks3, [XBLOCK])
    tmp9 = tmp7 >= tmp8
    tmp10 = tmp9 & tmp6
    tmp11 = tl.load(in_ptr0 + (x3 + ((-1)*ks3)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr0 + (x3), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.where(tmp9, tmp11, tmp12)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = x0
    tmp17 = tl.broadcast_to(2 + ks3, [XBLOCK])
    tmp18 = tmp16 >= tmp17
    tmp19 = tmp18 & tmp2
    tmp20 = tl.load(in_ptr0 + (x3 + ((-1)*ks3) + ((-4)*ks2) + ((-1)*ks2*ks3)), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.load(in_ptr0 + (x3 + ((-4)*ks2) + ((-1)*ks2*ks3)), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.where(tmp18, tmp20, tmp21)
    tmp23 = tl.where(tmp5, tmp15, tmp22)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp2, tmp23, tmp24)
    tmp26 = tl.full([1], 2, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = x0
    tmp29 = tl.broadcast_to(2 + ks3, [XBLOCK])
    tmp30 = tmp28 >= tmp29
    tmp31 = tmp30 & tmp27
    tmp32 = tl.load(in_ptr0 + (x3 + ((-1)*ks3) + 4*ks2 + ks2*ks3), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tl.load(in_ptr0 + (x3 + 4*ks2 + ks2*ks3), tmp27 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.where(tmp30, tmp32, tmp33)
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp27, tmp34, tmp35)
    tmp37 = x0
    tmp38 = 2 + ks3
    tmp39 = tmp37 >= tmp38
    tmp40 = tl.load(in_ptr0 + (x3 + ((-1)*ks3)), tmp39 & xmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.where(tmp39, tmp40, tmp41)
    tmp43 = tl.where(tmp27, tmp36, tmp42)
    tmp44 = tl.where(tmp2, tmp25, tmp43)
    tl.store(out_ptr0 + (x3), tmp44, xmask)




# kernel path: /tmp/torchinductor_sahanp/pv/cpvivrcdv2ucqcwkvzg7dcc5jpj44q2dj2tbefd5hkc5thnpqax7.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.hardswish]
# Source node to ATen node mapping:
#   x_1 => add_116, clamp_max, clamp_min, div, mul_114
# Graph fragment:
#   %add_116 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default_5, 3), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_116, 0), kwargs = {})
#   %clamp_max : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 6), kwargs = {})
#   %mul_114 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_scatter_default_5, %clamp_max), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_114, 6), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_hardswish_2(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = 0.16666666666666666
    tmp9 = tmp7 * tmp8
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)




# kernel path: /tmp/torchinductor_sahanp/xi/cxi3zc5mbgyccnf7ji7nytvxr3u3bpmpecckdwfkds3h4xf3h3xv.py
# Topologically Sorted Source Nodes: [x_1, x_2, x_3], Original ATen: [aten.hardswish, aten.view, aten.native_group_norm]
# Source node to ATen node mapping:
#   x_1 => add_116, clamp_max, clamp_min, div, mul_114
#   x_2 => view
#   x_3 => add_124, rsqrt, var_mean
# Graph fragment:
#   %add_116 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default_5, 3), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_116, 0), kwargs = {})
#   %clamp_max : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 6), kwargs = {})
#   %mul_114 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_scatter_default_5, %clamp_max), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_114, 6), kwargs = {})
#   %view : [num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%div, [1, 10, -1]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_1, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_124 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_124,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_hardswish_native_group_norm_view_3(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, ks0, ks1, ks2, ks3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2
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
        r0_1 = (r0_index % ks0)
        r0_2 = r0_index // ks0
        r0_3 = r0_index
        tmp0 = tl.load(in_ptr0 + (4*(r0_1 // ks1) + 16*r0_2 + 80*x0 + ks3*(r0_1 // ks1) + 4*ks2*r0_2 + 4*ks3*r0_2 + 20*ks2*x0 + 20*ks3*x0 + ks2*ks3*r0_2 + 5*ks2*ks3*x0 + ((r0_1 % ks1))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(r0_mask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(r0_mask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(r0_mask & xmask, tmp2_weight_next, tmp2_weight)
        tl.store(out_ptr0 + (r0_3 + 80*x0 + 20*ks2*x0 + 20*ks3*x0 + 5*ks2*ks3*x0), tmp0, r0_mask & xmask)
    tmp5, tmp6, tmp7 = triton_helpers.welford(tmp2_mean, tmp2_m2, tmp2_weight, 1)
    tmp2 = tmp5[:, None]
    tmp3 = tmp6[:, None]
    tmp4 = tmp7[:, None]
    tl.store(out_ptr1 + (x0), tmp2, xmask)
    tl.store(out_ptr2 + (x0), tmp3, xmask)
    tmp8 = 80 + 20*ks2 + 20*ks3 + 5*ks2*ks3
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp3 / tmp9
    tmp11 = 1e-05
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.rsqrt(tmp12)
    tl.store(out_ptr3 + (x0), tmp13, xmask)




# kernel path: /tmp/torchinductor_sahanp/dh/cdhexgw25jfel3czmachwx3xixydcrgbkri4osye376ly35krgx3.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   x_3 => add_125, mul_126
# Graph fragment:
#   %mul_126 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_2, %unsqueeze_3), kwargs = {})
#   %add_125 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_126, %unsqueeze_1), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_group_norm_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // ks0
    tmp0 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x1 // 5), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1 // 5), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 80 + 20*ks1 + 20*ks2 + 5*ks1*ks2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tl.store(out_ptr0 + (x2), tmp14, xmask)




# kernel path: /tmp/torchinductor_sahanp/tl/ctlxouravqng2u6tnnk26svddx573n2f3lgzbsbjxo2h2xgvraub.py
# Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.unsqueeze]
# Source node to ATen node mapping:
#   x_6 => unsqueeze_4
# Graph fragment:
#   %unsqueeze_4 : [num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_4, -2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_unsqueeze_5(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (4*(x0 // ks1) + 16*x1 + ks3*(x0 // ks1) + 4*ks2*x1 + 4*ks3*x1 + ks2*ks3*x1 + ((x0 % ks1))), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/gr/cgrdentawde47owkc4abogu7oxfcqsnjgz67mtka5xzrwk3hmhqz.py
# Topologically Sorted Source Nodes: [x_8], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   x_8 => amax, exp, sub_60, sum_1
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_5, [1], True), kwargs = {})
#   %sub_60 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_5, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_60,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__softmax_6(in_ptr0, out_ptr0, out_ptr1, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 10
    R0_BLOCK: tl.constexpr = 16
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
    tmp0 = tl.load(in_ptr0 + (2*x0 + 16*r0_1 + 4*ks0*r0_1 + 4*ks1*r0_1 + ks0*ks1*r0_1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 16*r0_1 + 4*ks0*r0_1 + 4*ks1*r0_1 + ks0*ks1*r0_1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1 + tmp0
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp5, float("-inf"))
    tmp8 = triton_helpers.max2(tmp7, 1)[:, None]
    tmp9 = tmp4 - tmp8
    tmp10 = tl_math.exp(tmp9)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp13 = tl.where(r0_mask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)




# kernel path: /tmp/torchinductor_sahanp/sq/csq2eiv5ktgro2bzhp47ziloelay2g2zcuano2p3gej7r7lpiill.py
# Topologically Sorted Source Nodes: [x_8], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   x_8 => div_1, exp, sub_60
# Graph fragment:
#   %sub_60 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_5, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_60,), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__softmax_7(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 16*x1 + 4*ks1*x1 + 4*ks2*x1 + ks1*ks2*x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 16*x1 + 4*ks1*x1 + 4*ks2*x1 + ks1*ks2*x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 - tmp5
    tmp7 = tl_math.exp(tmp6)
    tmp9 = tmp7 / tmp8
    tl.store(out_ptr0 + (x2), tmp9, xmask)







def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    s1 = primals_1
    s2 = primals_2
    assert_size_stride(primals_3, (1, 10, s1, s2), (10*s1*s2, s1*s2, s2, 1))
    assert_size_stride(primals_4, (10, ), (1, ))
    assert_size_stride(primals_5, (10, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 10, 4 + s1, 4 + s2), (160 + 40*s1 + 40*s2 + 10*s1*s2, 16 + 4*s1 + 4*s2 + s1*s2, 4 + s2, 1), torch.float32)
        ps0 = 4 + s2
        ps1 = 4 + s1
        ps2 = 16 + 4*s1 + 4*s2 + s1*s2
        buf1 = empty_strided_cuda((1, 10, 4 + s1, 4 + s2), (160 + 40*s1 + 40*s2 + 10*s1*s2, 16 + 4*s1 + 4*s2 + s1*s2, 4 + s2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.copy]
        triton_poi_fused_copy_0_xnumel = 160 + 40*s1 + 40*s2 + 10*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused_copy_0[grid(triton_poi_fused_copy_0_xnumel)](primals_3, buf0, buf1, 36, 32, 36, 32, 1296, 12960, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_3
        buf2 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1_xnumel = 160 + 40*s1 + 40*s2 + 10*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused_1[grid(triton_poi_fused_1_xnumel)](buf1, buf2, 36, 36, 32, 32, 12960, XBLOCK=128, num_warps=4, num_stages=1)
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.hardswish]
        triton_poi_fused_hardswish_2_xnumel = 160 + 40*s1 + 40*s2 + 10*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused_hardswish_2[grid(triton_poi_fused_hardswish_2_xnumel)](buf3, 12960, XBLOCK=128, num_warps=4, num_stages=1)
        ps3 = 16 + 4*s1 + 4*s2 + s1*s2
        buf4 = reinterpret_tensor(buf1, (1, 10, 16 + 4*s1 + 4*s2 + s1*s2), (160 + 40*s1 + 40*s2 + 10*s1*s2, 16 + 4*s1 + 4*s2 + s1*s2, 1), 0); del buf1  # reuse
        buf5 = empty_strided_cuda((1, 2, 1, 1), (2, 1, 2, 2), torch.float32)
        buf6 = empty_strided_cuda((1, 2, 1, 1), (2, 1, 2, 2), torch.float32)
        buf13 = empty_strided_cuda((1, 2, 1, 1), (2, 1, 2, 2), torch.float32)
        # Topologically Sorted Source Nodes: [x_1, x_2, x_3], Original ATen: [aten.hardswish, aten.view, aten.native_group_norm]
        triton_red_fused_hardswish_native_group_norm_view_3_r0_numel = 80 + 20*s1 + 20*s2 + 5*s1*s2
        stream0 = get_raw_stream(0)
        triton_red_fused_hardswish_native_group_norm_view_3[grid(2)](buf3, buf4, buf5, buf6, buf13, 1296, 36, 32, 32, 2, 6480, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        buf8 = reinterpret_tensor(buf3, (1, 10, 16 + 4*s1 + 4*s2 + s1*s2), (160 + 40*s1 + 40*s2 + 10*s1*s2, 16 + 4*s1 + 4*s2 + s1*s2, 1), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.native_group_norm]
        triton_poi_fused_native_group_norm_4_xnumel = 160 + 40*s1 + 40*s2 + 10*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_4[grid(triton_poi_fused_native_group_norm_4_xnumel)](buf4, buf5, buf6, primals_4, primals_5, buf8, 1296, 32, 32, 12960, XBLOCK=256, num_warps=4, num_stages=1)
        del buf6
        del primals_4
        del primals_5
        buf9 = empty_strided_cuda((1, 10, 1, 16 + 4*s1 + 4*s2 + s1*s2), (160 + 40*s1 + 40*s2 + 10*s1*s2, 16 + 4*s1 + 4*s2 + s1*s2, 16 + 4*s1 + 4*s2 + s1*s2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.unsqueeze]
        triton_poi_fused_unsqueeze_5_xnumel = 160 + 40*s1 + 40*s2 + 10*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused_unsqueeze_5[grid(triton_poi_fused_unsqueeze_5_xnumel)](buf8, buf9, 1296, 36, 32, 32, 12960, XBLOCK=256, num_warps=4, num_stages=1)
        del buf8
        buf10 = empty_strided_cuda((1, 1, 8 + 2*s1 + 2*s2 + ((s1*s2) // 2)), (8 + 2*s1 + 2*s2 + ((s1*s2) // 2), 8 + 2*s1 + 2*s2 + ((s1*s2) // 2), 1), torch.float32)
        buf11 = empty_strided_cuda((1, 1, 8 + 2*s1 + 2*s2 + ((s1*s2) // 2)), (8 + 2*s1 + 2*s2 + ((s1*s2) // 2), 8 + 2*s1 + 2*s2 + ((s1*s2) // 2), 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten._softmax]
        triton_per_fused__softmax_6_xnumel = 8 + 2*s1 + 2*s2 + ((s1*s2) // 2)
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_6[grid(triton_per_fused__softmax_6_xnumel)](buf9, buf10, buf11, 32, 32, 648, 10, XBLOCK=128, num_warps=8, num_stages=1)
        ps4 = 8 + 2*s1 + 2*s2 + ((s1*s2) // 2)
        buf12 = empty_strided_cuda((1, 10, 8 + 2*s1 + 2*s2 + ((s1*s2) // 2)), (80 + 10*((s1*s2) // 2) + 20*s1 + 20*s2, 8 + 2*s1 + 2*s2 + ((s1*s2) // 2), 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_7_xnumel = 80 + 10*((s1*s2) // 2) + 20*s1 + 20*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_7[grid(triton_poi_fused__softmax_7_xnumel)](buf9, buf10, buf11, buf12, 648, 32, 32, 6480, XBLOCK=256, num_warps=4, num_stages=1)
        del buf10
        del buf11
    return (buf12, buf4, buf9, buf12, reinterpret_tensor(buf5, (1, 2, 1), (2, 1, 1), 0), reinterpret_tensor(buf13, (1, 2, 1), (2, 1, 1), 0), s1, s2, 4 + s1, 4 + s2, 16 + 4*s1 + 4*s2 + s1*s2, 8 + 2*s1 + 2*s2 + ((s1*s2) // 2), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = 32
    primals_2 = 32
    primals_3 = rand_strided((1, 10, 32, 32), (10240, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
