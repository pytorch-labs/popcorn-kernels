# AOT ID: ['7_forward']
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


# kernel path: /tmp/torchinductor_sahanp/sp/cspkwcvxpywe4gcwzje5m64y7ynvj6ey7j4vip6xltxrlin65vnc.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   multi_head_attention_forward => clone
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 10)
    x2 = xindex // 640
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x2 + 1280*x1), xmask)
    tl.store(out_ptr0 + (x3), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/2p/c2pxdynzsiv25yofq2wmxfxk5t6bam7hqd2raffra4tu3r35o5gf.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   multi_head_attention_forward => clone_1
# Graph fragment:
#   %clone_1 : [num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze,), kwargs = {memory_format: torch.contiguous_format})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 38400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 200)
    x2 = xindex // 12800
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x2 + 192*x1), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, xmask)




# kernel path: /tmp/torchinductor_sahanp/b5/cb5lkbycp2fch3ypjnnf7k2joab7vzquxyesfd3zl3la6peckext.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.view]
# Source node to ATen node mapping:
#   multi_head_attention_forward => view_6
# Graph fragment:
#   %view_6 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_3, [10, 8, 20, 8]), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_view_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x4), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/65/c65cwarnlrmi2yn32ykm5cjvkqucb5rlvckuey5n7g4jw4gfa4ni.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.view]
# Source node to ATen node mapping:
#   multi_head_attention_forward => view_7
# Graph fragment:
#   %view_7 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_4, [10, 8, 20, 8]), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_view_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (12800 + x4), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/bs/cbsn7sgeh7ja63cntdfyu547u73l2dd6hhhwy2doyj2r46owc4ad.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.view]
# Source node to ATen node mapping:
#   multi_head_attention_forward => view_8
# Graph fragment:
#   %view_8 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_5, [10, 8, 20, 8]), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_view_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (25600 + x4), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/ya/cyaju4javdawemwfes43mbun3553dkb6zukzxvsjcdz3xymfjbez.py
# Topologically Sorted Source Nodes: [dropout, add, x_1, dropout_9, add_6, x_10], Original ATen: [aten.native_dropout, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   add => add_1
#   add_6 => add_22
#   dropout => gt, inductor_lookup_seed_default, inductor_random_default_21, mul, mul_1
#   dropout_9 => gt_9, inductor_lookup_seed_default_9, inductor_random_default_12, mul_32, mul_33
#   x_1 => add_2, add_3, clone_3, mul_2, mul_3, rsqrt, sub, var_mean
#   x_10 => add_23, add_24, clone_11, mul_34, mul_35, rsqrt_7, sub_7, var_mean_7
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default_21 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([20, 10, 64], %inductor_lookup_seed_default, rand), kwargs = {})
#   %gt : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%inductor_random_default_21, 0.1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gt, %view_10), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, 1.1111111111111112), kwargs = {})
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute, %mul_1), kwargs = {})
#   %clone_3 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_1,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_3, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-05), kwargs = {})
#   %rsqrt : [num_users=3] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_3, %getitem_5), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %primals_6), kwargs = {})
#   %add_3 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %primals_7), kwargs = {})
#   %inductor_lookup_seed_default_9 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 9), kwargs = {})
#   %inductor_random_default_12 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([20, 10, 64], %inductor_lookup_seed_default_9, rand), kwargs = {})
#   %gt_9 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%inductor_random_default_12, 0.1), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gt_9, %view_55), kwargs = {})
#   %mul_33 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_32, 1.1111111111111112), kwargs = {})
#   %add_22 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute, %mul_33), kwargs = {})
#   %clone_11 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_22,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_7 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_11, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_23 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_30, 1e-05), kwargs = {})
#   %rsqrt_7 : [num_users=3] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_23,), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_11, %getitem_31), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %rsqrt_7), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_34, %primals_44), kwargs = {})
#   %add_24 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_35, %primals_45), kwargs = {})
#   %sub_45 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_22, %getitem_31), kwargs = {})
#   %mul_169 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_45, %rsqrt_7), kwargs = {})
#   %div_10 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_7, 64), kwargs = {})
#   %sub_66 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %getitem_5), kwargs = {})
#   %mul_236 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_66, %rsqrt), kwargs = {})
#   %div_17 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt, 64), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr1, out_ptr3, out_ptr6, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, load_seed_offset, load_seed_offset1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 200
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    x2 = (xindex % 10)
    x3 = xindex // 10
    tmp8 = tl.load(in_ptr1 + (r0_1 + 64*x3 + 1280*x2), xmask, other=0.0)
    tmp10 = tl.load(in_ptr2 + (r0_1 + 64*x0), xmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr4 + (r0_1), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr5 + (r0_1), None, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr6 + (r0_1 + 64*x0), xmask, other=0.0)
    tmp46 = tl.load(in_ptr7 + (r0_1), None, eviction_policy='evict_last')
    tmp70 = tl.load(in_ptr8 + (r0_1), None, eviction_policy='evict_last')
    tmp72 = tl.load(in_ptr9 + (r0_1), None, eviction_policy='evict_last')
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = r0_1 + 64*x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tl.load(in_ptr0 + load_seed_offset1)
    tmp6 = tl.rand(tmp5, (tmp1).to(tl.uint32))
    tmp7 = tmp6 > tmp3
    tmp9 = tmp7.to(tl.float32)
    tmp12 = tmp10 + tmp11
    tmp13 = tmp9 * tmp12
    tmp14 = 1.1111111111111112
    tmp15 = tmp13 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(xmask, tmp17, 0)
    tmp20 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
    tmp22 = tl.where(xmask, tmp20, 0)
    tmp23 = tl.sum(tmp22, 1)[:, None]
    tmp24 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp17 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, R0_BLOCK])
    tmp31 = tl.where(xmask, tmp29, 0)
    tmp32 = tl.sum(tmp31, 1)[:, None]
    tmp33 = tmp16 - tmp26
    tmp34 = 64.0
    tmp35 = tmp32 / tmp34
    tmp36 = 1e-05
    tmp37 = tmp35 + tmp36
    tmp38 = libdevice.rsqrt(tmp37)
    tmp39 = tmp33 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tmp4.to(tl.float32)
    tmp47 = tmp45 + tmp46
    tmp48 = tmp44 * tmp47
    tmp49 = tmp48 * tmp14
    tmp50 = tmp8 + tmp49
    tmp51 = tl.broadcast_to(tmp50, [XBLOCK, R0_BLOCK])
    tmp53 = tl.where(xmask, tmp51, 0)
    tmp54 = tl.broadcast_to(tmp51, [XBLOCK, R0_BLOCK])
    tmp56 = tl.where(xmask, tmp54, 0)
    tmp57 = tl.sum(tmp56, 1)[:, None]
    tmp58 = tmp57 / tmp25
    tmp59 = tmp51 - tmp58
    tmp60 = tmp59 * tmp59
    tmp61 = tl.broadcast_to(tmp60, [XBLOCK, R0_BLOCK])
    tmp63 = tl.where(xmask, tmp61, 0)
    tmp64 = tl.sum(tmp63, 1)[:, None]
    tmp65 = tmp50 - tmp58
    tmp66 = tmp64 / tmp34
    tmp67 = tmp66 + tmp36
    tmp68 = libdevice.rsqrt(tmp67)
    tmp69 = tmp65 * tmp68
    tmp71 = tmp69 * tmp70
    tmp73 = tmp71 + tmp72
    tmp74 = 0.015625
    tmp75 = tmp68 * tmp74
    tmp76 = tmp38 * tmp74
    tl.store(out_ptr1 + (r0_1 + 64*x0), tmp4, xmask)
    tl.store(out_ptr3 + (r0_1 + 64*x0), tmp7, xmask)
    tl.store(out_ptr6 + (r0_1 + 64*x0), tmp43, xmask)
    tl.store(out_ptr9 + (r0_1 + 64*x0), tmp73, xmask)
    tl.store(out_ptr10 + (r0_1 + 64*x3 + 1280*x2), tmp69, xmask)
    tl.store(out_ptr11 + (r0_1 + 64*x3 + 1280*x2), tmp39, xmask)
    tl.store(out_ptr12 + (x0), tmp75, xmask)
    tl.store(out_ptr13 + (x0), tmp76, xmask)




# kernel path: /tmp/torchinductor_sahanp/r4/cr4meiscklae6r26huuvvsxoxahcswcp2edn7qkeuw5yray6vlvz.py
# Topologically Sorted Source Nodes: [relu], Original ATen: [aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   relu => relu
# Graph fragment:
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%view_12,), kwargs = {})
#   %le_5 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu, 0), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_relu_threshold_backward_6(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 409600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 2048)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(out_ptr0 + (x2), tmp6, None)




# kernel path: /tmp/torchinductor_sahanp/lq/clq7ucy27c7golovrzxf3ueulu4gs35spi2r7z2w7apckq7fx4q6.py
# Topologically Sorted Source Nodes: [relu, dropout_1], Original ATen: [aten.relu, aten.native_dropout]
# Source node to ATen node mapping:
#   dropout_1 => gt_1, inductor_lookup_seed_default_1, inductor_random_default_20, mul_4, mul_5
#   relu => relu
# Graph fragment:
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%view_12,), kwargs = {})
#   %inductor_lookup_seed_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 1), kwargs = {})
#   %inductor_random_default_20 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([20, 10, 2048], %inductor_lookup_seed_default_1, rand), kwargs = {})
#   %gt_1 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%inductor_random_default_20, 0.1), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gt_1, %relu), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, 1.1111111111111112), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_dropout_relu_7(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 409600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    x1 = (xindex % 2048)
    tmp6 = tl.load(in_out_ptr0 + (x0), None)
    tmp7 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full([1], 0, tl.int32)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp11 = tmp5 * tmp10
    tmp12 = 1.1111111111111112
    tmp13 = tmp11 * tmp12
    tl.store(out_ptr1 + (x0), tmp4, None)
    tl.store(in_out_ptr0 + (x0), tmp13, None)




# kernel path: /tmp/torchinductor_sahanp/zh/czh7axkryhuncj4ru5o7xahubpgt7h7fskck2j7rvf7cfzvkhxaq.py
# Topologically Sorted Source Nodes: [dropout_2, add_1, x_3], Original ATen: [aten.native_dropout, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   add_1 => add_4
#   dropout_2 => gt_2, inductor_lookup_seed_default_2, inductor_random_default_19, mul_6, mul_7
#   x_3 => add_5, add_6, mul_8, mul_9, rsqrt_1, sub_1, var_mean_1
# Graph fragment:
#   %inductor_lookup_seed_default_2 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 2), kwargs = {})
#   %inductor_random_default_19 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([20, 10, 64], %inductor_lookup_seed_default_2, rand), kwargs = {})
#   %gt_2 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%inductor_random_default_19, 0.1), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gt_2, %view_14), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, 1.1111111111111112), kwargs = {})
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %mul_7), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_4, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_5,), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_4, %getitem_7), kwargs = {})
#   %mul_8 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, %primals_12), kwargs = {})
#   %add_6 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %primals_13), kwargs = {})
#   %div_16 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_1, 64), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr4, out_ptr5, load_seed_offset, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 200
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp5 = tl.load(in_ptr1 + (r0_1 + 64*x0), xmask, other=0.0)
    tmp7 = tl.load(in_out_ptr0 + (r0_1 + 64*x0), xmask, other=0.0)
    tmp8 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr4 + (r0_1), None, eviction_policy='evict_last')
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = r0_1 + 64*x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp6 = tmp4.to(tl.float32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp5 + tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.where(xmask, tmp14, 0)
    tmp17 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp20 / tmp22
    tmp24 = tmp14 - tmp23
    tmp25 = tmp24 * tmp24
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
    tmp28 = tl.where(xmask, tmp26, 0)
    tmp29 = tl.sum(tmp28, 1)[:, None]
    tmp30 = tmp13 - tmp23
    tmp31 = 64.0
    tmp32 = tmp29 / tmp31
    tmp33 = 1e-05
    tmp34 = tmp32 + tmp33
    tmp35 = libdevice.rsqrt(tmp34)
    tmp36 = tmp30 * tmp35
    tmp38 = tmp36 * tmp37
    tmp40 = tmp38 + tmp39
    tmp41 = 0.015625
    tmp42 = tmp35 * tmp41
    tl.store(out_ptr1 + (r0_1 + 64*x0), tmp4, xmask)
    tl.store(in_out_ptr0 + (r0_1 + 64*x0), tmp36, xmask)
    tl.store(out_ptr4 + (r0_1 + 64*x0), tmp40, xmask)
    tl.store(out_ptr5 + (x0), tmp42, xmask)




# kernel path: /tmp/torchinductor_sahanp/fp/cfpbaqn6xxlqfmqellmmssbnzw64t32rph7nkrk2koo7gojyaxpt.py
# Topologically Sorted Source Nodes: [relu_1, dropout_4], Original ATen: [aten.relu, aten.native_dropout]
# Source node to ATen node mapping:
#   dropout_4 => gt_4, inductor_lookup_seed_default_4, inductor_random_default_17, mul_14, mul_15
#   relu_1 => relu_1
# Graph fragment:
#   %relu_1 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%view_27,), kwargs = {})
#   %inductor_lookup_seed_default_4 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 4), kwargs = {})
#   %inductor_random_default_17 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([20, 10, 2048], %inductor_lookup_seed_default_4, rand), kwargs = {})
#   %gt_4 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%inductor_random_default_17, 0.1), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gt_4, %relu_1), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_14, 1.1111111111111112), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_dropout_relu_9(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 409600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    x1 = (xindex % 2048)
    tmp6 = tl.load(in_out_ptr0 + (x0), None)
    tmp7 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full([1], 0, tl.int32)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp11 = tmp5 * tmp10
    tmp12 = 1.1111111111111112
    tmp13 = tmp11 * tmp12
    tl.store(out_ptr1 + (x0), tmp4, None)
    tl.store(in_out_ptr0 + (x0), tmp13, None)




# kernel path: /tmp/torchinductor_sahanp/vu/cvujon3bahpmf3unt34ehba36xtpi4bf2df6ske2xdqdgwkvacii.py
# Topologically Sorted Source Nodes: [dropout_8, add_5, x_9, output], Original ATen: [aten.native_dropout, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   add_5 => add_16
#   dropout_8 => gt_8, inductor_lookup_seed_default_8, inductor_random_default_13, mul_26, mul_27
#   output => add_19, add_20, mul_30, mul_31, rsqrt_6, sub_6, var_mean_6
#   x_9 => add_17, add_18, mul_28, mul_29, rsqrt_5, sub_5, var_mean_5
# Graph fragment:
#   %inductor_lookup_seed_default_8 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 8), kwargs = {})
#   %inductor_random_default_13 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([20, 10, 64], %inductor_lookup_seed_default_8, rand), kwargs = {})
#   %gt_8 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%inductor_random_default_13, 0.1), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gt_8, %view_44), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_26, 1.1111111111111112), kwargs = {})
#   %add_16 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_15, %mul_27), kwargs = {})
#   %var_mean_5 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_16, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_22, 1e-05), kwargs = {})
#   %rsqrt_5 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_17,), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_16, %getitem_23), kwargs = {})
#   %mul_28 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %rsqrt_5), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, %primals_36), kwargs = {})
#   %add_18 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_29, %primals_37), kwargs = {})
#   %var_mean_6 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_18, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_24, 1e-05), kwargs = {})
#   %rsqrt_6 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_19,), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_18, %getitem_25), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %rsqrt_6), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_30, %primals_38), kwargs = {})
#   %add_20 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_31, %primals_39), kwargs = {})
#   %div_12 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_5, 64), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_10(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, out_ptr4, out_ptr5, out_ptr6, load_seed_offset, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 200
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp5 = tl.load(in_ptr1 + (r0_1 + 64*x0), xmask, other=0.0)
    tmp7 = tl.load(in_out_ptr0 + (r0_1 + 64*x0), xmask, other=0.0)
    tmp8 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr4 + (r0_1), None, eviction_policy='evict_last')
    tmp62 = tl.load(in_ptr5 + (r0_1), None, eviction_policy='evict_last')
    tmp64 = tl.load(in_ptr6 + (r0_1), None, eviction_policy='evict_last')
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = r0_1 + 64*x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp6 = tmp4.to(tl.float32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp5 + tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.where(xmask, tmp14, 0)
    tmp17 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp20 / tmp22
    tmp24 = tmp14 - tmp23
    tmp25 = tmp24 * tmp24
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
    tmp28 = tl.where(xmask, tmp26, 0)
    tmp29 = tl.sum(tmp28, 1)[:, None]
    tmp30 = tmp13 - tmp23
    tmp31 = 64.0
    tmp32 = tmp29 / tmp31
    tmp33 = 1e-05
    tmp34 = tmp32 + tmp33
    tmp35 = libdevice.rsqrt(tmp34)
    tmp36 = tmp30 * tmp35
    tmp38 = tmp36 * tmp37
    tmp40 = tmp38 + tmp39
    tmp41 = tl.broadcast_to(tmp40, [XBLOCK, R0_BLOCK])
    tmp43 = tl.where(xmask, tmp41, 0)
    tmp44 = tl.broadcast_to(tmp41, [XBLOCK, R0_BLOCK])
    tmp46 = tl.where(xmask, tmp44, 0)
    tmp47 = tl.sum(tmp46, 1)[:, None]
    tmp48 = tmp47 / tmp22
    tmp49 = tmp41 - tmp48
    tmp50 = tmp49 * tmp49
    tmp51 = tl.broadcast_to(tmp50, [XBLOCK, R0_BLOCK])
    tmp53 = tl.where(xmask, tmp51, 0)
    tmp54 = tl.sum(tmp53, 1)[:, None]
    tmp55 = tmp54 / tmp31
    tmp56 = tmp55 + tmp33
    tmp57 = libdevice.rsqrt(tmp56)
    tmp58 = 0.015625
    tmp59 = tmp35 * tmp58
    tmp60 = tmp40 - tmp48
    tmp61 = tmp60 * tmp57
    tmp63 = tmp61 * tmp62
    tmp65 = tmp63 + tmp64
    tl.store(out_ptr1 + (r0_1 + 64*x0), tmp4, xmask)
    tl.store(in_out_ptr0 + (r0_1 + 64*x0), tmp36, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp57, xmask)
    tl.store(out_ptr5 + (x0), tmp59, xmask)
    tl.store(out_ptr6 + (r0_1 + 64*x0), tmp65, xmask)
    tl.store(out_ptr4 + (x0), tmp48, xmask)




# kernel path: /tmp/torchinductor_sahanp/ks/cksncykiwxzin6ulffbdnk3oth53onlv56wxdefdrlthg3k44swk.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   multi_head_attention_forward_4 => clone_12
# Graph fragment:
#   %clone_12 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze_4,), kwargs = {memory_format: torch.contiguous_format})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_11(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 200)
    x2 = xindex // 12800
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x2 + 128*x1), xmask)
    tmp1 = tl.load(in_ptr1 + (64 + x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, xmask)




# kernel path: /tmp/torchinductor_sahanp/lw/clwghj4bkzbiy64tzi46lxd6hmaqsucr6tyiyq2wmn724qd4vw5v.py
# Topologically Sorted Source Nodes: [relu_5, dropout_19], Original ATen: [aten.relu, aten.native_dropout, aten.threshold_backward]
# Source node to ATen node mapping:
#   dropout_19 => gt_19, inductor_lookup_seed_default_19, inductor_random_default_2, mul_68, mul_69
#   relu_5 => relu_5
# Graph fragment:
#   %relu_5 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%view_126,), kwargs = {})
#   %inductor_lookup_seed_default_19 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 19), kwargs = {})
#   %inductor_random_default_2 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([20, 10, 2048], %inductor_lookup_seed_default_19, rand), kwargs = {})
#   %gt_19 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%inductor_random_default_2, 0.1), kwargs = {})
#   %mul_68 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gt_19, %relu_5), kwargs = {})
#   %mul_69 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_68, 1.1111111111111112), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_5, 0), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_dropout_relu_threshold_backward_12(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 409600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    x1 = (xindex % 2048)
    tmp6 = tl.load(in_ptr1 + (x0), None)
    tmp7 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full([1], 0, tl.int32)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp11 = tmp5 * tmp10
    tmp12 = 1.1111111111111112
    tmp13 = tmp11 * tmp12
    tmp14 = 0.0
    tmp15 = tmp10 <= tmp14
    tl.store(out_ptr1 + (x0), tmp4, None)
    tl.store(out_ptr2 + (x0), tmp13, None)
    tl.store(out_ptr3 + (x0), tmp15, None)




# kernel path: /tmp/torchinductor_sahanp/sp/cspodb6qm6fbqa2uv2adrto6ihan27n53d6p3q74todctwlpgypo.py
# Topologically Sorted Source Nodes: [dropout_20, add_14, x_21, output_1, x_22], Original ATen: [aten.native_dropout, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   add_14 => add_46
#   dropout_20 => gt_20, inductor_lookup_seed_default_20, inductor_random_default_1, mul_70, mul_71
#   output_1 => add_49, add_50, mul_74, mul_75, rsqrt_16, sub_16, var_mean_16
#   x_21 => add_47, add_48, mul_72, mul_73, rsqrt_15, sub_15, var_mean_15
#   x_22 => gt_21, inductor_lookup_seed_default_21, inductor_random_default, mul_76, mul_77
# Graph fragment:
#   %inductor_lookup_seed_default_20 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 20), kwargs = {})
#   %inductor_random_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([20, 10, 64], %inductor_lookup_seed_default_20, rand), kwargs = {})
#   %gt_20 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%inductor_random_default_1, 0.1), kwargs = {})
#   %mul_70 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gt_20, %view_128), kwargs = {})
#   %mul_71 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_70, 1.1111111111111112), kwargs = {})
#   %add_46 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_45, %mul_71), kwargs = {})
#   %var_mean_15 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_46, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_47 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_78, 1e-05), kwargs = {})
#   %rsqrt_15 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_47,), kwargs = {})
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_46, %getitem_79), kwargs = {})
#   %mul_72 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_15, %rsqrt_15), kwargs = {})
#   %mul_73 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_72, %primals_92), kwargs = {})
#   %add_48 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_73, %primals_93), kwargs = {})
#   %var_mean_16 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_48, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_49 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_80, 1e-05), kwargs = {})
#   %rsqrt_16 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_49,), kwargs = {})
#   %sub_16 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_48, %getitem_81), kwargs = {})
#   %mul_74 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_16, %rsqrt_16), kwargs = {})
#   %mul_75 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_74, %primals_94), kwargs = {})
#   %add_50 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_75, %primals_95), kwargs = {})
#   %inductor_lookup_seed_default_21 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 21), kwargs = {})
#   %inductor_random_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([20, 10, 64], %inductor_lookup_seed_default_21, rand), kwargs = {})
#   %gt_21 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%inductor_random_default, 0.5), kwargs = {})
#   %mul_76 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gt_21, %add_50), kwargs = {})
#   %mul_77 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_76, 2.0), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_15, 64), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_13(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, out_ptr4, out_ptr5, out_ptr7, out_ptr8, load_seed_offset, load_seed_offset1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 200
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp5 = tl.load(in_ptr1 + (r0_1 + 64*x0), xmask, other=0.0)
    tmp7 = tl.load(in_out_ptr0 + (r0_1 + 64*x0), xmask, other=0.0)
    tmp8 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr4 + (r0_1), None, eviction_policy='evict_last')
    tmp67 = tl.load(in_ptr5 + (r0_1), None, eviction_policy='evict_last')
    tmp69 = tl.load(in_ptr6 + (r0_1), None, eviction_policy='evict_last')
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = r0_1 + 64*x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp6 = tmp4.to(tl.float32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp5 + tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.where(xmask, tmp14, 0)
    tmp17 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp20 / tmp22
    tmp24 = tmp14 - tmp23
    tmp25 = tmp24 * tmp24
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
    tmp28 = tl.where(xmask, tmp26, 0)
    tmp29 = tl.sum(tmp28, 1)[:, None]
    tmp30 = tmp13 - tmp23
    tmp31 = 64.0
    tmp32 = tmp29 / tmp31
    tmp33 = 1e-05
    tmp34 = tmp32 + tmp33
    tmp35 = libdevice.rsqrt(tmp34)
    tmp36 = tmp30 * tmp35
    tmp38 = tmp36 * tmp37
    tmp40 = tmp38 + tmp39
    tmp41 = tl.broadcast_to(tmp40, [XBLOCK, R0_BLOCK])
    tmp43 = tl.where(xmask, tmp41, 0)
    tmp44 = tl.broadcast_to(tmp41, [XBLOCK, R0_BLOCK])
    tmp46 = tl.where(xmask, tmp44, 0)
    tmp47 = tl.sum(tmp46, 1)[:, None]
    tmp48 = tmp47 / tmp22
    tmp49 = tmp41 - tmp48
    tmp50 = tmp49 * tmp49
    tmp51 = tl.broadcast_to(tmp50, [XBLOCK, R0_BLOCK])
    tmp53 = tl.where(xmask, tmp51, 0)
    tmp54 = tl.sum(tmp53, 1)[:, None]
    tmp55 = tmp54 / tmp31
    tmp56 = tmp55 + tmp33
    tmp57 = libdevice.rsqrt(tmp56)
    tmp58 = 0.015625
    tmp59 = tmp35 * tmp58
    tmp60 = tl.load(in_ptr0 + load_seed_offset1)
    tmp61 = tl.rand(tmp60, (tmp1).to(tl.uint32))
    tmp62 = 0.5
    tmp63 = tmp61 > tmp62
    tmp64 = tmp63.to(tl.float32)
    tmp65 = tmp40 - tmp48
    tmp66 = tmp65 * tmp57
    tmp68 = tmp66 * tmp67
    tmp70 = tmp68 + tmp69
    tmp71 = tmp64 * tmp70
    tmp72 = 2.0
    tmp73 = tmp71 * tmp72
    tl.store(out_ptr1 + (r0_1 + 64*x0), tmp4, xmask)
    tl.store(in_out_ptr0 + (r0_1 + 64*x0), tmp36, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp57, xmask)
    tl.store(out_ptr5 + (x0), tmp59, xmask)
    tl.store(out_ptr7 + (r0_1 + 64*x0), tmp63, xmask)
    tl.store(out_ptr8 + (r0_1 + 64*x0), tmp73, xmask)
    tl.store(out_ptr4 + (x0), tmp48, xmask)




# kernel path: /tmp/torchinductor_sahanp/32/c32nzbkgip4weuujulbbusuva3x2t3h7bhphfqrn6eqzplraogwf.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten.mul, aten.exp, aten.sub, aten.mean]
# Source node to ATen node mapping:
#   loss => exp, mean, mul_78, sub_17
# Graph fragment:
#   %mul_78 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_79, %permute_79), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%permute_79,), kwargs = {})
#   %sub_17 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%exp, %mul_78), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sub_17,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_exp_mean_mul_sub_14(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 200
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 10)
    x1 = xindex // 10
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_2 + 64*((r0_2 + 64*x0 + 640*x1) // 1280) + 640*(((x0 + 10*x1) % 20))), xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl_math.exp(tmp0)
    tmp2 = tmp0 * tmp0
    tmp3 = tmp1 - tmp2
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)




# kernel path: /tmp/torchinductor_sahanp/2h/c2hl42jo3vl6dpakpgfviuwx6aiker2xbdc4ffjz5g2jcc6dg2rc.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten.mul, aten.exp, aten.sub, aten.mean]
# Source node to ATen node mapping:
#   loss => exp, mean, mul_78, sub_17
# Graph fragment:
#   %mul_78 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_79, %permute_79), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%permute_79,), kwargs = {})
#   %sub_17 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%exp, %mul_78), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sub_17,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_exp_mean_mul_sub_15(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 200
    R0_BLOCK: tl.constexpr = 256
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
    tmp5 = 12800.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp6, None)







def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95 = args
    args.clear()
    assert_size_stride(primals_1, (10, 20, 64), (1280, 64, 1))
    assert_size_stride(primals_2, (192, ), (1, ))
    assert_size_stride(primals_3, (192, 64), (64, 1))
    assert_size_stride(primals_4, (64, 64), (64, 1))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_7, (64, ), (1, ))
    assert_size_stride(primals_8, (2048, 64), (64, 1))
    assert_size_stride(primals_9, (2048, ), (1, ))
    assert_size_stride(primals_10, (64, 2048), (2048, 1))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_12, (64, ), (1, ))
    assert_size_stride(primals_13, (64, ), (1, ))
    assert_size_stride(primals_14, (192, ), (1, ))
    assert_size_stride(primals_15, (192, 64), (64, 1))
    assert_size_stride(primals_16, (64, 64), (64, 1))
    assert_size_stride(primals_17, (64, ), (1, ))
    assert_size_stride(primals_18, (64, ), (1, ))
    assert_size_stride(primals_19, (64, ), (1, ))
    assert_size_stride(primals_20, (2048, 64), (64, 1))
    assert_size_stride(primals_21, (2048, ), (1, ))
    assert_size_stride(primals_22, (64, 2048), (2048, 1))
    assert_size_stride(primals_23, (64, ), (1, ))
    assert_size_stride(primals_24, (64, ), (1, ))
    assert_size_stride(primals_25, (64, ), (1, ))
    assert_size_stride(primals_26, (192, ), (1, ))
    assert_size_stride(primals_27, (192, 64), (64, 1))
    assert_size_stride(primals_28, (64, 64), (64, 1))
    assert_size_stride(primals_29, (64, ), (1, ))
    assert_size_stride(primals_30, (64, ), (1, ))
    assert_size_stride(primals_31, (64, ), (1, ))
    assert_size_stride(primals_32, (2048, 64), (64, 1))
    assert_size_stride(primals_33, (2048, ), (1, ))
    assert_size_stride(primals_34, (64, 2048), (2048, 1))
    assert_size_stride(primals_35, (64, ), (1, ))
    assert_size_stride(primals_36, (64, ), (1, ))
    assert_size_stride(primals_37, (64, ), (1, ))
    assert_size_stride(primals_38, (64, ), (1, ))
    assert_size_stride(primals_39, (64, ), (1, ))
    assert_size_stride(primals_40, (192, ), (1, ))
    assert_size_stride(primals_41, (192, 64), (64, 1))
    assert_size_stride(primals_42, (64, 64), (64, 1))
    assert_size_stride(primals_43, (64, ), (1, ))
    assert_size_stride(primals_44, (64, ), (1, ))
    assert_size_stride(primals_45, (64, ), (1, ))
    assert_size_stride(primals_46, (192, 64), (64, 1))
    assert_size_stride(primals_47, (192, ), (1, ))
    assert_size_stride(primals_48, (64, 64), (64, 1))
    assert_size_stride(primals_49, (64, ), (1, ))
    assert_size_stride(primals_50, (64, ), (1, ))
    assert_size_stride(primals_51, (64, ), (1, ))
    assert_size_stride(primals_52, (2048, 64), (64, 1))
    assert_size_stride(primals_53, (2048, ), (1, ))
    assert_size_stride(primals_54, (64, 2048), (2048, 1))
    assert_size_stride(primals_55, (64, ), (1, ))
    assert_size_stride(primals_56, (64, ), (1, ))
    assert_size_stride(primals_57, (64, ), (1, ))
    assert_size_stride(primals_58, (192, ), (1, ))
    assert_size_stride(primals_59, (192, 64), (64, 1))
    assert_size_stride(primals_60, (64, 64), (64, 1))
    assert_size_stride(primals_61, (64, ), (1, ))
    assert_size_stride(primals_62, (64, ), (1, ))
    assert_size_stride(primals_63, (64, ), (1, ))
    assert_size_stride(primals_64, (192, 64), (64, 1))
    assert_size_stride(primals_65, (192, ), (1, ))
    assert_size_stride(primals_66, (64, 64), (64, 1))
    assert_size_stride(primals_67, (64, ), (1, ))
    assert_size_stride(primals_68, (64, ), (1, ))
    assert_size_stride(primals_69, (64, ), (1, ))
    assert_size_stride(primals_70, (2048, 64), (64, 1))
    assert_size_stride(primals_71, (2048, ), (1, ))
    assert_size_stride(primals_72, (64, 2048), (2048, 1))
    assert_size_stride(primals_73, (64, ), (1, ))
    assert_size_stride(primals_74, (64, ), (1, ))
    assert_size_stride(primals_75, (64, ), (1, ))
    assert_size_stride(primals_76, (192, ), (1, ))
    assert_size_stride(primals_77, (192, 64), (64, 1))
    assert_size_stride(primals_78, (64, 64), (64, 1))
    assert_size_stride(primals_79, (64, ), (1, ))
    assert_size_stride(primals_80, (64, ), (1, ))
    assert_size_stride(primals_81, (64, ), (1, ))
    assert_size_stride(primals_82, (192, 64), (64, 1))
    assert_size_stride(primals_83, (192, ), (1, ))
    assert_size_stride(primals_84, (64, 64), (64, 1))
    assert_size_stride(primals_85, (64, ), (1, ))
    assert_size_stride(primals_86, (64, ), (1, ))
    assert_size_stride(primals_87, (64, ), (1, ))
    assert_size_stride(primals_88, (2048, 64), (64, 1))
    assert_size_stride(primals_89, (2048, ), (1, ))
    assert_size_stride(primals_90, (64, 2048), (2048, 1))
    assert_size_stride(primals_91, (64, ), (1, ))
    assert_size_stride(primals_92, (64, ), (1, ))
    assert_size_stride(primals_93, (64, ), (1, ))
    assert_size_stride(primals_94, (64, ), (1, ))
    assert_size_stride(primals_95, (64, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf13 = empty_strided_cuda((22, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [22], out=buf13)
        buf0 = empty_strided_cuda((20, 10, 64), (640, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_0[grid(12800)](primals_1, buf0, 12800, XBLOCK=256, num_warps=4, num_stages=1)
        buf1 = empty_strided_cuda((200, 192), (192, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf0, (200, 64), (64, 1), 0), reinterpret_tensor(primals_3, (64, 192), (1, 64), 0), out=buf1)
        del primals_3
        buf2 = empty_strided_cuda((3, 20, 10, 64), (12800, 640, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1[grid(38400)](buf1, primals_2, buf2, 38400, XBLOCK=512, num_warps=4, num_stages=1)
        del primals_2
        buf3 = empty_strided_cuda((10, 8, 20, 8), (64, 8, 640, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_2[grid(12800)](buf2, buf3, 12800, XBLOCK=128, num_warps=4, num_stages=1)
        buf4 = empty_strided_cuda((10, 8, 20, 8), (64, 8, 640, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3[grid(12800)](buf2, buf4, 12800, XBLOCK=128, num_warps=4, num_stages=1)
        buf5 = empty_strided_cuda((10, 8, 20, 8), (64, 8, 640, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_4[grid(12800)](buf2, buf5, 12800, XBLOCK=256, num_warps=4, num_stages=1)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf3, buf4, buf5, None, True, 0.1)
        buf7 = buf6[0]
        buf8 = buf6[1]
        buf9 = buf6[2]
        buf10 = buf6[3]
        del buf6
        buf11 = empty_strided_cuda((20, 10, 8, 8), (640, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_0[grid(12800)](buf7, buf11, 12800, XBLOCK=256, num_warps=4, num_stages=1)
        buf12 = empty_strided_cuda((200, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf11, (200, 64), (64, 1), 0), reinterpret_tensor(primals_4, (64, 64), (1, 64), 0), out=buf12)
        buf97 = reinterpret_tensor(buf2, (200, 192), (192, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf0, (200, 64), (64, 1), 0), reinterpret_tensor(primals_41, (64, 192), (1, 64), 0), out=buf97)
        del primals_41
        buf98 = reinterpret_tensor(buf1, (3, 20, 10, 64), (12800, 640, 64, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1[grid(38400)](buf97, primals_40, buf98, 38400, XBLOCK=512, num_warps=4, num_stages=1)
        del primals_40
        buf100 = empty_strided_cuda((10, 8, 20, 8), (64, 8, 640, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3[grid(12800)](buf98, buf100, 12800, XBLOCK=128, num_warps=4, num_stages=1)
        buf101 = empty_strided_cuda((10, 8, 20, 8), (64, 8, 640, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_4[grid(12800)](buf98, buf101, 12800, XBLOCK=256, num_warps=4, num_stages=1)
        buf99 = empty_strided_cuda((10, 8, 20, 8), (64, 8, 640, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_2[grid(12800)](buf98, buf99, 12800, XBLOCK=128, num_warps=4, num_stages=1)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf102 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf99, buf100, buf101, None, True, 0.1)
        buf103 = buf102[0]
        buf104 = buf102[1]
        buf105 = buf102[2]
        buf106 = buf102[3]
        del buf102
        buf107 = empty_strided_cuda((20, 10, 8, 8), (640, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_0[grid(12800)](buf103, buf107, 12800, XBLOCK=256, num_warps=4, num_stages=1)
        buf108 = empty_strided_cuda((200, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf107, (200, 64), (64, 1), 0), reinterpret_tensor(primals_42, (64, 64), (1, 64), 0), out=buf108)
        buf110 = empty_strided_cuda((20, 10, 64), (640, 64, 1), torch.bool)
        buf15 = empty_strided_cuda((20, 10, 64), (640, 64, 1), torch.bool)
        buf19 = empty_strided_cuda((20, 10, 64), (640, 64, 1), torch.float32)
        buf114 = empty_strided_cuda((20, 10, 64), (640, 64, 1), torch.float32)
        buf266 = empty_strided_cuda((20, 10, 64), (64, 1280, 1), torch.float32)
        buf276 = empty_strided_cuda((20, 10, 64), (64, 1280, 1), torch.float32)
        buf267 = empty_strided_cuda((20, 10, 1), (10, 1, 1), torch.float32)
        buf277 = empty_strided_cuda((20, 10, 1), (10, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [dropout, add, x_1, dropout_9, add_6, x_10], Original ATen: [aten.native_dropout, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_5[grid(200)](buf13, primals_1, buf12, primals_5, primals_6, primals_7, buf108, primals_43, primals_44, primals_45, buf110, buf15, buf19, buf114, buf266, buf276, buf267, buf277, 9, 0, 200, 64, XBLOCK=1, num_warps=2, num_stages=1)
        del primals_1
        del primals_43
        del primals_45
        del primals_5
        del primals_7
        buf115 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(reinterpret_tensor(primals_47, (64, ), (1, ), 0), reinterpret_tensor(buf114, (200, 64), (64, 1), 0), reinterpret_tensor(primals_46, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf115)
        buf20 = empty_strided_cuda((200, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf19, (200, 64), (64, 1), 0), reinterpret_tensor(primals_8, (64, 2048), (1, 64), 0), out=buf20)
        buf275 = empty_strided_cuda((20, 10, 2048), (20480, 2048, 1), torch.bool)
        # Topologically Sorted Source Nodes: [relu], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_6[grid(409600)](buf20, primals_9, buf275, 409600, XBLOCK=1024, num_warps=4, num_stages=1)
        buf22 = empty_strided_cuda((20, 10, 2048), (20480, 2048, 1), torch.bool)
        buf23 = reinterpret_tensor(buf20, (20, 10, 2048), (20480, 2048, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [relu, dropout_1], Original ATen: [aten.relu, aten.native_dropout]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_dropout_relu_7[grid(409600)](buf23, buf13, primals_9, buf22, 1, 409600, XBLOCK=512, num_warps=8, num_stages=1)
        del primals_9
        buf24 = buf108; del buf108  # reuse
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf23, (200, 2048), (2048, 1), 0), reinterpret_tensor(primals_10, (2048, 64), (1, 2048), 0), out=buf24)
        buf26 = empty_strided_cuda((20, 10, 64), (640, 64, 1), torch.bool)
        buf30 = reinterpret_tensor(buf24, (20, 10, 64), (640, 64, 1), 0); del buf24  # reuse
        buf31 = empty_strided_cuda((20, 10, 64), (640, 64, 1), torch.float32)
        buf274 = empty_strided_cuda((20, 10, 1), (10, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [dropout_2, add_1, x_3], Original ATen: [aten.native_dropout, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_8[grid(200)](buf30, buf13, buf19, primals_11, primals_12, primals_13, buf26, buf31, buf274, 18, 200, 64, XBLOCK=1, num_warps=2, num_stages=1)
        del primals_11
        del primals_13
        buf32 = reinterpret_tensor(buf98, (200, 192), (192, 1), 0); del buf98  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf31, (200, 64), (64, 1), 0), reinterpret_tensor(primals_15, (64, 192), (1, 64), 0), out=buf32)
        buf33 = reinterpret_tensor(buf97, (3, 20, 10, 64), (12800, 640, 64, 1), 0); del buf97  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1[grid(38400)](buf32, primals_14, buf33, 38400, XBLOCK=512, num_warps=4, num_stages=1)
        del primals_14
        buf34 = empty_strided_cuda((10, 8, 20, 8), (64, 8, 640, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_2[grid(12800)](buf33, buf34, 12800, XBLOCK=128, num_warps=4, num_stages=1)
        buf35 = empty_strided_cuda((10, 8, 20, 8), (64, 8, 640, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3[grid(12800)](buf33, buf35, 12800, XBLOCK=128, num_warps=4, num_stages=1)
        buf36 = empty_strided_cuda((10, 8, 20, 8), (64, 8, 640, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_4[grid(12800)](buf33, buf36, 12800, XBLOCK=256, num_warps=4, num_stages=1)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf37 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf34, buf35, buf36, None, True, 0.1)
        buf38 = buf37[0]
        buf39 = buf37[1]
        buf40 = buf37[2]
        buf41 = buf37[3]
        del buf37
        buf42 = empty_strided_cuda((20, 10, 8, 8), (640, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_0[grid(12800)](buf38, buf42, 12800, XBLOCK=256, num_warps=4, num_stages=1)
        buf43 = empty_strided_cuda((200, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf42, (200, 64), (64, 1), 0), reinterpret_tensor(primals_16, (64, 64), (1, 64), 0), out=buf43)
        buf45 = empty_strided_cuda((20, 10, 64), (640, 64, 1), torch.bool)
        buf49 = reinterpret_tensor(buf43, (20, 10, 64), (640, 64, 1), 0); del buf43  # reuse
        buf50 = empty_strided_cuda((20, 10, 64), (640, 64, 1), torch.float32)
        buf273 = empty_strided_cuda((20, 10, 1), (10, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [dropout_3, add_2, x_4], Original ATen: [aten.native_dropout, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_8[grid(200)](buf49, buf13, buf31, primals_17, primals_18, primals_19, buf45, buf50, buf273, 18, 200, 64, XBLOCK=1, num_warps=2, num_stages=1)
        del primals_17
        del primals_19
        buf51 = empty_strided_cuda((200, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf50, (200, 64), (64, 1), 0), reinterpret_tensor(primals_20, (64, 2048), (1, 64), 0), out=buf51)
        buf272 = empty_strided_cuda((20, 10, 2048), (20480, 2048, 1), torch.bool)
        # Topologically Sorted Source Nodes: [relu_1], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_6[grid(409600)](buf51, primals_21, buf272, 409600, XBLOCK=1024, num_warps=4, num_stages=1)
        buf53 = empty_strided_cuda((20, 10, 2048), (20480, 2048, 1), torch.bool)
        buf54 = reinterpret_tensor(buf51, (20, 10, 2048), (20480, 2048, 1), 0); del buf51  # reuse
        # Topologically Sorted Source Nodes: [relu_1, dropout_4], Original ATen: [aten.relu, aten.native_dropout]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_dropout_relu_9[grid(409600)](buf54, buf13, primals_21, buf53, 15, 409600, XBLOCK=512, num_warps=8, num_stages=1)
        del primals_21
        buf55 = empty_strided_cuda((200, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf54, (200, 2048), (2048, 1), 0), reinterpret_tensor(primals_22, (2048, 64), (1, 2048), 0), out=buf55)
        buf57 = empty_strided_cuda((20, 10, 64), (640, 64, 1), torch.bool)
        buf61 = reinterpret_tensor(buf55, (20, 10, 64), (640, 64, 1), 0); del buf55  # reuse
        buf62 = empty_strided_cuda((20, 10, 64), (640, 64, 1), torch.float32)
        buf271 = empty_strided_cuda((20, 10, 1), (10, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [dropout_5, add_3, x_6], Original ATen: [aten.native_dropout, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_8[grid(200)](buf61, buf13, buf50, primals_23, primals_24, primals_25, buf57, buf62, buf271, 18, 200, 64, XBLOCK=1, num_warps=2, num_stages=1)
        del primals_23
        del primals_25
        buf63 = reinterpret_tensor(buf33, (200, 192), (192, 1), 0); del buf33  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf62, (200, 64), (64, 1), 0), reinterpret_tensor(primals_27, (64, 192), (1, 64), 0), out=buf63)
        buf64 = reinterpret_tensor(buf32, (3, 20, 10, 64), (12800, 640, 64, 1), 0); del buf32  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1[grid(38400)](buf63, primals_26, buf64, 38400, XBLOCK=512, num_warps=4, num_stages=1)
        del primals_26
        buf65 = empty_strided_cuda((10, 8, 20, 8), (64, 8, 640, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_2[grid(12800)](buf64, buf65, 12800, XBLOCK=128, num_warps=4, num_stages=1)
        buf66 = empty_strided_cuda((10, 8, 20, 8), (64, 8, 640, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3[grid(12800)](buf64, buf66, 12800, XBLOCK=128, num_warps=4, num_stages=1)
        buf67 = empty_strided_cuda((10, 8, 20, 8), (64, 8, 640, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_4[grid(12800)](buf64, buf67, 12800, XBLOCK=256, num_warps=4, num_stages=1)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf68 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf65, buf66, buf67, None, True, 0.1)
        buf69 = buf68[0]
        buf70 = buf68[1]
        buf71 = buf68[2]
        buf72 = buf68[3]
        del buf68
        buf73 = empty_strided_cuda((20, 10, 8, 8), (640, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_0[grid(12800)](buf69, buf73, 12800, XBLOCK=256, num_warps=4, num_stages=1)
        buf74 = empty_strided_cuda((200, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf73, (200, 64), (64, 1), 0), reinterpret_tensor(primals_28, (64, 64), (1, 64), 0), out=buf74)
        buf76 = empty_strided_cuda((20, 10, 64), (640, 64, 1), torch.bool)
        buf80 = reinterpret_tensor(buf74, (20, 10, 64), (640, 64, 1), 0); del buf74  # reuse
        buf81 = empty_strided_cuda((20, 10, 64), (640, 64, 1), torch.float32)
        buf270 = empty_strided_cuda((20, 10, 1), (10, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [dropout_6, add_4, x_7], Original ATen: [aten.native_dropout, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_8[grid(200)](buf80, buf13, buf62, primals_29, primals_30, primals_31, buf76, buf81, buf270, 18, 200, 64, XBLOCK=1, num_warps=2, num_stages=1)
        del primals_29
        del primals_31
        buf82 = empty_strided_cuda((200, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf81, (200, 64), (64, 1), 0), reinterpret_tensor(primals_32, (64, 2048), (1, 64), 0), out=buf82)
        buf269 = empty_strided_cuda((20, 10, 2048), (20480, 2048, 1), torch.bool)
        # Topologically Sorted Source Nodes: [relu_2], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_6[grid(409600)](buf82, primals_33, buf269, 409600, XBLOCK=1024, num_warps=4, num_stages=1)
        buf84 = empty_strided_cuda((20, 10, 2048), (20480, 2048, 1), torch.bool)
        buf85 = reinterpret_tensor(buf82, (20, 10, 2048), (20480, 2048, 1), 0); del buf82  # reuse
        # Topologically Sorted Source Nodes: [relu_2, dropout_7], Original ATen: [aten.relu, aten.native_dropout]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_dropout_relu_9[grid(409600)](buf85, buf13, primals_33, buf84, 15, 409600, XBLOCK=512, num_warps=8, num_stages=1)
        del primals_33
        buf86 = empty_strided_cuda((200, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf85, (200, 2048), (2048, 1), 0), reinterpret_tensor(primals_34, (2048, 64), (1, 2048), 0), out=buf86)
        buf88 = empty_strided_cuda((20, 10, 64), (640, 64, 1), torch.bool)
        buf92 = reinterpret_tensor(buf86, (20, 10, 64), (640, 64, 1), 0); del buf86  # reuse
        buf93 = empty_strided_cuda((20, 10, 1), (10, 1, 1), torch.float32)
        buf94 = empty_strided_cuda((20, 10, 1), (10, 1, 200), torch.float32)
        buf96 = reinterpret_tensor(buf94, (20, 10, 1), (10, 1, 1), 0); del buf94  # reuse
        buf268 = empty_strided_cuda((20, 10, 1), (10, 1, 1), torch.float32)
        buf116 = empty_strided_cuda((20, 10, 64), (640, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [dropout_8, add_5, x_9, output], Original ATen: [aten.native_dropout, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_10[grid(200)](buf92, buf96, buf13, buf81, primals_35, primals_36, primals_37, primals_38, primals_39, buf88, buf93, buf268, buf116, 8, 200, 64, XBLOCK=1, num_warps=2, num_stages=1)
        del primals_35
        del primals_39
        buf117 = empty_strided_cuda((200, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf116, (200, 64), (64, 1), 0), reinterpret_tensor(primals_46, (64, 128), (1, 64), 4096), out=buf117)
        buf118 = empty_strided_cuda((2, 20, 10, 64), (12800, 640, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_11[grid(25600)](buf117, primals_47, buf118, 25600, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_47
        buf119 = empty_strided_cuda((10, 8, 20, 8), (64, 8, 640, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_2[grid(12800)](buf118, buf119, 12800, XBLOCK=128, num_warps=4, num_stages=1)
        buf120 = empty_strided_cuda((10, 8, 20, 8), (64, 8, 640, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3[grid(12800)](buf118, buf120, 12800, XBLOCK=128, num_warps=4, num_stages=1)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf121 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf115, (10, 8, 20, 8), (64, 8, 640, 1), 0), buf119, buf120, None, True, 0.1)
        buf122 = buf121[0]
        buf123 = buf121[1]
        buf124 = buf121[2]
        buf125 = buf121[3]
        del buf121
        buf126 = empty_strided_cuda((20, 10, 8, 8), (640, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_0[grid(12800)](buf122, buf126, 12800, XBLOCK=256, num_warps=4, num_stages=1)
        buf127 = empty_strided_cuda((200, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf126, (200, 64), (64, 1), 0), reinterpret_tensor(primals_48, (64, 64), (1, 64), 0), out=buf127)
        buf129 = empty_strided_cuda((20, 10, 64), (640, 64, 1), torch.bool)
        buf133 = reinterpret_tensor(buf127, (20, 10, 64), (640, 64, 1), 0); del buf127  # reuse
        buf134 = empty_strided_cuda((20, 10, 64), (640, 64, 1), torch.float32)
        buf265 = empty_strided_cuda((20, 10, 1), (10, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [dropout_10, add_7, x_11], Original ATen: [aten.native_dropout, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_8[grid(200)](buf133, buf13, buf114, primals_49, primals_50, primals_51, buf129, buf134, buf265, 18, 200, 64, XBLOCK=1, num_warps=2, num_stages=1)
        del primals_49
        del primals_51
        buf167 = reinterpret_tensor(buf118, (200, 128), (128, 1), 0); del buf118  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_6], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf116, (200, 64), (64, 1), 0), reinterpret_tensor(primals_64, (64, 128), (1, 64), 4096), out=buf167)
        buf168 = reinterpret_tensor(buf117, (2, 20, 10, 64), (12800, 640, 64, 1), 0); del buf117  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_6], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_11[grid(25600)](buf167, primals_65, buf168, 25600, XBLOCK=128, num_warps=4, num_stages=1)
        buf169 = empty_strided_cuda((10, 8, 20, 8), (64, 8, 640, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_6], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_2[grid(12800)](buf168, buf169, 12800, XBLOCK=128, num_warps=4, num_stages=1)
        buf170 = empty_strided_cuda((10, 8, 20, 8), (64, 8, 640, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_6], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3[grid(12800)](buf168, buf170, 12800, XBLOCK=128, num_warps=4, num_stages=1)
        buf217 = reinterpret_tensor(buf168, (200, 128), (128, 1), 0); del buf168  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_8], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf116, (200, 64), (64, 1), 0), reinterpret_tensor(primals_82, (64, 128), (1, 64), 4096), out=buf217)
        buf218 = reinterpret_tensor(buf167, (2, 20, 10, 64), (12800, 640, 64, 1), 0); del buf167  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_8], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_11[grid(25600)](buf217, primals_83, buf218, 25600, XBLOCK=128, num_warps=4, num_stages=1)
        del buf217
        buf219 = empty_strided_cuda((10, 8, 20, 8), (64, 8, 640, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_8], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_2[grid(12800)](buf218, buf219, 12800, XBLOCK=128, num_warps=4, num_stages=1)
        buf220 = empty_strided_cuda((10, 8, 20, 8), (64, 8, 640, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_8], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3[grid(12800)](buf218, buf220, 12800, XBLOCK=128, num_warps=4, num_stages=1)
        del buf218
        buf135 = empty_strided_cuda((200, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf134, (200, 64), (64, 1), 0), reinterpret_tensor(primals_52, (64, 2048), (1, 64), 0), out=buf135)
        buf264 = empty_strided_cuda((20, 10, 2048), (20480, 2048, 1), torch.bool)
        # Topologically Sorted Source Nodes: [relu_3], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_6[grid(409600)](buf135, primals_53, buf264, 409600, XBLOCK=1024, num_warps=4, num_stages=1)
        buf137 = empty_strided_cuda((20, 10, 2048), (20480, 2048, 1), torch.bool)
        buf138 = reinterpret_tensor(buf135, (20, 10, 2048), (20480, 2048, 1), 0); del buf135  # reuse
        # Topologically Sorted Source Nodes: [relu_3, dropout_11], Original ATen: [aten.relu, aten.native_dropout]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_dropout_relu_9[grid(409600)](buf138, buf13, primals_53, buf137, 15, 409600, XBLOCK=512, num_warps=8, num_stages=1)
        del primals_53
        buf139 = empty_strided_cuda((200, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf138, (200, 2048), (2048, 1), 0), reinterpret_tensor(primals_54, (2048, 64), (1, 2048), 0), out=buf139)
        buf141 = empty_strided_cuda((20, 10, 64), (640, 64, 1), torch.bool)
        buf145 = reinterpret_tensor(buf139, (20, 10, 64), (640, 64, 1), 0); del buf139  # reuse
        buf146 = empty_strided_cuda((20, 10, 64), (640, 64, 1), torch.float32)
        buf263 = empty_strided_cuda((20, 10, 1), (10, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [dropout_12, add_8, x_13], Original ATen: [aten.native_dropout, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_8[grid(200)](buf145, buf13, buf134, primals_55, primals_56, primals_57, buf141, buf146, buf263, 18, 200, 64, XBLOCK=1, num_warps=2, num_stages=1)
        del primals_55
        del primals_57
        buf147 = reinterpret_tensor(buf64, (200, 192), (192, 1), 0); del buf64  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf146, (200, 64), (64, 1), 0), reinterpret_tensor(primals_59, (64, 192), (1, 64), 0), out=buf147)
        buf148 = reinterpret_tensor(buf63, (3, 20, 10, 64), (12800, 640, 64, 1), 0); del buf63  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1[grid(38400)](buf147, primals_58, buf148, 38400, XBLOCK=512, num_warps=4, num_stages=1)
        del primals_58
        buf149 = empty_strided_cuda((10, 8, 20, 8), (64, 8, 640, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_2[grid(12800)](buf148, buf149, 12800, XBLOCK=128, num_warps=4, num_stages=1)
        buf150 = empty_strided_cuda((10, 8, 20, 8), (64, 8, 640, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3[grid(12800)](buf148, buf150, 12800, XBLOCK=128, num_warps=4, num_stages=1)
        buf151 = empty_strided_cuda((10, 8, 20, 8), (64, 8, 640, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_4[grid(12800)](buf148, buf151, 12800, XBLOCK=256, num_warps=4, num_stages=1)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf152 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf149, buf150, buf151, None, True, 0.1)
        buf153 = buf152[0]
        buf154 = buf152[1]
        buf155 = buf152[2]
        buf156 = buf152[3]
        del buf152
        buf157 = empty_strided_cuda((20, 10, 8, 8), (640, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_0[grid(12800)](buf153, buf157, 12800, XBLOCK=256, num_warps=4, num_stages=1)
        buf158 = empty_strided_cuda((200, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf157, (200, 64), (64, 1), 0), reinterpret_tensor(primals_60, (64, 64), (1, 64), 0), out=buf158)
        buf160 = empty_strided_cuda((20, 10, 64), (640, 64, 1), torch.bool)
        buf164 = reinterpret_tensor(buf158, (20, 10, 64), (640, 64, 1), 0); del buf158  # reuse
        buf165 = empty_strided_cuda((20, 10, 64), (640, 64, 1), torch.float32)
        buf262 = empty_strided_cuda((20, 10, 1), (10, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [dropout_13, add_9, x_14], Original ATen: [aten.native_dropout, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_8[grid(200)](buf164, buf13, buf146, primals_61, primals_62, primals_63, buf160, buf165, buf262, 18, 200, 64, XBLOCK=1, num_warps=2, num_stages=1)
        del primals_61
        del primals_63
        buf166 = empty_strided_cuda((200, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_6], Original ATen: [aten.addmm]
        extern_kernels.addmm(reinterpret_tensor(primals_65, (64, ), (1, ), 0), reinterpret_tensor(buf165, (200, 64), (64, 1), 0), reinterpret_tensor(primals_64, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf166)
        del primals_65
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_6], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf171 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf166, (10, 8, 20, 8), (64, 8, 640, 1), 0), buf169, buf170, None, True, 0.1)
        buf172 = buf171[0]
        buf173 = buf171[1]
        buf174 = buf171[2]
        buf175 = buf171[3]
        del buf171
        buf176 = empty_strided_cuda((20, 10, 8, 8), (640, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_6], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_0[grid(12800)](buf172, buf176, 12800, XBLOCK=256, num_warps=4, num_stages=1)
        buf177 = empty_strided_cuda((200, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_6], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf176, (200, 64), (64, 1), 0), reinterpret_tensor(primals_66, (64, 64), (1, 64), 0), out=buf177)
        buf179 = empty_strided_cuda((20, 10, 64), (640, 64, 1), torch.bool)
        buf183 = reinterpret_tensor(buf177, (20, 10, 64), (640, 64, 1), 0); del buf177  # reuse
        buf184 = empty_strided_cuda((20, 10, 64), (640, 64, 1), torch.float32)
        buf261 = empty_strided_cuda((20, 10, 1), (10, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [dropout_14, add_10, x_15], Original ATen: [aten.native_dropout, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_8[grid(200)](buf183, buf13, buf165, primals_67, primals_68, primals_69, buf179, buf184, buf261, 18, 200, 64, XBLOCK=1, num_warps=2, num_stages=1)
        del primals_67
        del primals_69
        buf185 = empty_strided_cuda((200, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf184, (200, 64), (64, 1), 0), reinterpret_tensor(primals_70, (64, 2048), (1, 64), 0), out=buf185)
        buf260 = empty_strided_cuda((20, 10, 2048), (20480, 2048, 1), torch.bool)
        # Topologically Sorted Source Nodes: [relu_4], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_6[grid(409600)](buf185, primals_71, buf260, 409600, XBLOCK=1024, num_warps=4, num_stages=1)
        buf187 = empty_strided_cuda((20, 10, 2048), (20480, 2048, 1), torch.bool)
        buf188 = reinterpret_tensor(buf185, (20, 10, 2048), (20480, 2048, 1), 0); del buf185  # reuse
        # Topologically Sorted Source Nodes: [relu_4, dropout_15], Original ATen: [aten.relu, aten.native_dropout]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_dropout_relu_9[grid(409600)](buf188, buf13, primals_71, buf187, 15, 409600, XBLOCK=512, num_warps=8, num_stages=1)
        del primals_71
        buf189 = empty_strided_cuda((200, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_16], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf188, (200, 2048), (2048, 1), 0), reinterpret_tensor(primals_72, (2048, 64), (1, 2048), 0), out=buf189)
        buf191 = empty_strided_cuda((20, 10, 64), (640, 64, 1), torch.bool)
        buf195 = reinterpret_tensor(buf189, (20, 10, 64), (640, 64, 1), 0); del buf189  # reuse
        buf196 = empty_strided_cuda((20, 10, 64), (640, 64, 1), torch.float32)
        buf259 = empty_strided_cuda((20, 10, 1), (10, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [dropout_16, add_11, x_17], Original ATen: [aten.native_dropout, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_8[grid(200)](buf195, buf13, buf184, primals_73, primals_74, primals_75, buf191, buf196, buf259, 18, 200, 64, XBLOCK=1, num_warps=2, num_stages=1)
        del primals_73
        del primals_75
        buf197 = reinterpret_tensor(buf148, (200, 192), (192, 1), 0); del buf148  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_7], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf196, (200, 64), (64, 1), 0), reinterpret_tensor(primals_77, (64, 192), (1, 64), 0), out=buf197)
        buf198 = reinterpret_tensor(buf147, (3, 20, 10, 64), (12800, 640, 64, 1), 0); del buf147  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_7], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1[grid(38400)](buf197, primals_76, buf198, 38400, XBLOCK=512, num_warps=4, num_stages=1)
        del buf197
        del primals_76
        buf199 = empty_strided_cuda((10, 8, 20, 8), (64, 8, 640, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_7], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_2[grid(12800)](buf198, buf199, 12800, XBLOCK=128, num_warps=4, num_stages=1)
        buf200 = empty_strided_cuda((10, 8, 20, 8), (64, 8, 640, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_7], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3[grid(12800)](buf198, buf200, 12800, XBLOCK=128, num_warps=4, num_stages=1)
        buf201 = empty_strided_cuda((10, 8, 20, 8), (64, 8, 640, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_7], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_4[grid(12800)](buf198, buf201, 12800, XBLOCK=256, num_warps=4, num_stages=1)
        del buf198
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_7], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf202 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf199, buf200, buf201, None, True, 0.1)
        buf203 = buf202[0]
        buf204 = buf202[1]
        buf205 = buf202[2]
        buf206 = buf202[3]
        del buf202
        buf207 = empty_strided_cuda((20, 10, 8, 8), (640, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_7], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_0[grid(12800)](buf203, buf207, 12800, XBLOCK=256, num_warps=4, num_stages=1)
        buf208 = empty_strided_cuda((200, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_7], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf207, (200, 64), (64, 1), 0), reinterpret_tensor(primals_78, (64, 64), (1, 64), 0), out=buf208)
        buf210 = empty_strided_cuda((20, 10, 64), (640, 64, 1), torch.bool)
        buf214 = reinterpret_tensor(buf208, (20, 10, 64), (640, 64, 1), 0); del buf208  # reuse
        buf215 = empty_strided_cuda((20, 10, 64), (640, 64, 1), torch.float32)
        buf258 = empty_strided_cuda((20, 10, 1), (10, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [dropout_17, add_12, x_18], Original ATen: [aten.native_dropout, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_8[grid(200)](buf214, buf13, buf196, primals_79, primals_80, primals_81, buf210, buf215, buf258, 18, 200, 64, XBLOCK=1, num_warps=2, num_stages=1)
        del primals_79
        del primals_81
        buf216 = empty_strided_cuda((200, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(reinterpret_tensor(primals_83, (64, ), (1, ), 0), reinterpret_tensor(buf215, (200, 64), (64, 1), 0), reinterpret_tensor(primals_82, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf216)
        del primals_83
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_8], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf221 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf216, (10, 8, 20, 8), (64, 8, 640, 1), 0), buf219, buf220, None, True, 0.1)
        buf222 = buf221[0]
        buf223 = buf221[1]
        buf224 = buf221[2]
        buf225 = buf221[3]
        del buf221
        buf226 = empty_strided_cuda((20, 10, 8, 8), (640, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_8], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_0[grid(12800)](buf222, buf226, 12800, XBLOCK=256, num_warps=4, num_stages=1)
        buf227 = empty_strided_cuda((200, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_8], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf226, (200, 64), (64, 1), 0), reinterpret_tensor(primals_84, (64, 64), (1, 64), 0), out=buf227)
        buf229 = empty_strided_cuda((20, 10, 64), (640, 64, 1), torch.bool)
        buf233 = reinterpret_tensor(buf227, (20, 10, 64), (640, 64, 1), 0); del buf227  # reuse
        buf234 = empty_strided_cuda((20, 10, 64), (640, 64, 1), torch.float32)
        buf257 = empty_strided_cuda((20, 10, 1), (10, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [dropout_18, add_13, x_19], Original ATen: [aten.native_dropout, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_8[grid(200)](buf233, buf13, buf215, primals_85, primals_86, primals_87, buf229, buf234, buf257, 18, 200, 64, XBLOCK=1, num_warps=2, num_stages=1)
        del primals_85
        del primals_87
        buf235 = empty_strided_cuda((200, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_10], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf234, (200, 64), (64, 1), 0), reinterpret_tensor(primals_88, (64, 2048), (1, 64), 0), out=buf235)
        buf237 = empty_strided_cuda((20, 10, 2048), (20480, 2048, 1), torch.bool)
        buf238 = empty_strided_cuda((20, 10, 2048), (20480, 2048, 1), torch.float32)
        buf256 = empty_strided_cuda((20, 10, 2048), (20480, 2048, 1), torch.bool)
        # Topologically Sorted Source Nodes: [relu_5, dropout_19], Original ATen: [aten.relu, aten.native_dropout, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_dropout_relu_threshold_backward_12[grid(409600)](buf13, buf235, primals_89, buf237, buf238, buf256, 19, 409600, XBLOCK=512, num_warps=8, num_stages=1)
        del buf235
        del primals_89
        buf239 = empty_strided_cuda((200, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_20], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf238, (200, 2048), (2048, 1), 0), reinterpret_tensor(primals_90, (2048, 64), (1, 2048), 0), out=buf239)
        buf241 = empty_strided_cuda((20, 10, 64), (640, 64, 1), torch.bool)
        buf245 = reinterpret_tensor(buf239, (20, 10, 64), (640, 64, 1), 0); del buf239  # reuse
        buf246 = empty_strided_cuda((20, 10, 1), (10, 1, 1), torch.float32)
        buf247 = empty_strided_cuda((20, 10, 1), (10, 1, 200), torch.float32)
        buf249 = reinterpret_tensor(buf247, (20, 10, 1), (10, 1, 1), 0); del buf247  # reuse
        buf255 = empty_strided_cuda((20, 10, 1), (10, 1, 1), torch.float32)
        buf251 = empty_strided_cuda((20, 10, 64), (640, 64, 1), torch.bool)
        buf252 = empty_strided_cuda((20, 10, 64), (640, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [dropout_20, add_14, x_21, output_1, x_22], Original ATen: [aten.native_dropout, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_13[grid(200)](buf245, buf249, buf13, buf234, primals_91, primals_92, primals_93, primals_94, primals_95, buf241, buf246, buf255, buf251, buf252, 20, 21, 200, 64, XBLOCK=1, num_warps=2, num_stages=1)
        del buf13
        del primals_91
        buf253 = empty_strided_cuda((20, 10, 1), (10, 1, 200), torch.float32)
        # Topologically Sorted Source Nodes: [loss], Original ATen: [aten.mul, aten.exp, aten.sub, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_exp_mean_mul_sub_14[grid(200)](buf252, buf253, 200, 64, XBLOCK=32, num_warps=8, num_stages=1)
        del buf252
        buf254 = empty_strided_cuda((), (), torch.float32)
        buf278 = buf254; del buf254  # reuse
        # Topologically Sorted Source Nodes: [loss], Original ATen: [aten.mul, aten.exp, aten.sub, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_exp_mean_mul_sub_15[grid(1)](buf278, buf253, 1, 200, XBLOCK=1, num_warps=2, num_stages=1)
        del buf253
    return (buf278, primals_6, primals_12, primals_18, primals_24, primals_30, primals_36, primals_37, primals_38, primals_44, primals_50, primals_56, primals_62, primals_68, primals_74, primals_80, primals_86, primals_92, primals_93, primals_94, primals_95, reinterpret_tensor(buf0, (200, 64), (64, 1), 0), buf3, buf4, buf5, buf7, buf8, buf9, buf10, reinterpret_tensor(buf11, (200, 64), (64, 1), 0), buf15, reinterpret_tensor(buf19, (200, 64), (64, 1), 0), buf22, reinterpret_tensor(buf23, (200, 2048), (2048, 1), 0), buf26, buf30, reinterpret_tensor(buf31, (200, 64), (64, 1), 0), buf34, buf35, buf36, buf38, buf39, buf40, buf41, reinterpret_tensor(buf42, (200, 64), (64, 1), 0), buf45, buf49, reinterpret_tensor(buf50, (200, 64), (64, 1), 0), buf53, reinterpret_tensor(buf54, (200, 2048), (2048, 1), 0), buf57, buf61, reinterpret_tensor(buf62, (200, 64), (64, 1), 0), buf65, buf66, buf67, buf69, buf70, buf71, buf72, reinterpret_tensor(buf73, (200, 64), (64, 1), 0), buf76, buf80, reinterpret_tensor(buf81, (200, 64), (64, 1), 0), buf84, reinterpret_tensor(buf85, (200, 2048), (2048, 1), 0), buf88, buf92, buf93, buf96, buf99, buf100, buf101, buf103, buf104, buf105, buf106, reinterpret_tensor(buf107, (200, 64), (64, 1), 0), buf110, reinterpret_tensor(buf114, (200, 64), (64, 1), 0), reinterpret_tensor(buf116, (200, 64), (64, 1), 0), reinterpret_tensor(buf115, (10, 8, 20, 8), (64, 8, 640, 1), 0), buf119, buf120, buf122, buf123, buf124, buf125, reinterpret_tensor(buf126, (200, 64), (64, 1), 0), buf129, buf133, reinterpret_tensor(buf134, (200, 64), (64, 1), 0), buf137, reinterpret_tensor(buf138, (200, 2048), (2048, 1), 0), buf141, buf145, reinterpret_tensor(buf146, (200, 64), (64, 1), 0), buf149, buf150, buf151, buf153, buf154, buf155, buf156, reinterpret_tensor(buf157, (200, 64), (64, 1), 0), buf160, buf164, reinterpret_tensor(buf165, (200, 64), (64, 1), 0), reinterpret_tensor(buf166, (10, 8, 20, 8), (64, 8, 640, 1), 0), buf169, buf170, buf172, buf173, buf174, buf175, reinterpret_tensor(buf176, (200, 64), (64, 1), 0), buf179, buf183, reinterpret_tensor(buf184, (200, 64), (64, 1), 0), buf187, reinterpret_tensor(buf188, (200, 2048), (2048, 1), 0), buf191, buf195, reinterpret_tensor(buf196, (200, 64), (64, 1), 0), buf199, buf200, buf201, buf203, buf204, buf205, buf206, reinterpret_tensor(buf207, (200, 64), (64, 1), 0), buf210, buf214, reinterpret_tensor(buf215, (200, 64), (64, 1), 0), reinterpret_tensor(buf216, (10, 8, 20, 8), (64, 8, 640, 1), 0), buf219, buf220, buf222, buf223, buf224, buf225, reinterpret_tensor(buf226, (200, 64), (64, 1), 0), buf229, buf233, reinterpret_tensor(buf234, (200, 64), (64, 1), 0), buf237, reinterpret_tensor(buf238, (200, 2048), (2048, 1), 0), buf241, buf245, buf246, buf249, buf251, buf255, primals_90, buf256, primals_88, buf257, primals_84, reinterpret_tensor(primals_82, (128, 64), (64, 1), 4096), reinterpret_tensor(primals_82, (64, 64), (64, 1), 0), buf258, primals_78, primals_77, buf259, primals_72, buf260, primals_70, buf261, primals_66, reinterpret_tensor(primals_64, (128, 64), (64, 1), 4096), reinterpret_tensor(primals_64, (64, 64), (64, 1), 0), buf262, primals_60, primals_59, buf263, primals_54, buf264, primals_52, buf265, primals_48, reinterpret_tensor(primals_46, (128, 64), (64, 1), 4096), reinterpret_tensor(primals_46, (64, 64), (64, 1), 0), buf266, buf267, primals_42, buf268, primals_34, buf269, primals_32, buf270, primals_28, primals_27, buf271, primals_22, buf272, primals_20, buf273, primals_16, primals_15, buf274, primals_10, buf275, primals_8, buf276, buf277, primals_4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((10, 20, 64), (1280, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((192, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((2048, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((192, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((2048, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((64, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((192, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((2048, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((64, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((192, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((192, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((2048, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((64, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((192, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((192, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((2048, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((64, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((192, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((192, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((2048, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((64, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
