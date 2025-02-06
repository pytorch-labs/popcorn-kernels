# AOT ID: ['100_inference']
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


# kernel path: /tmp/torchinductor_sahanp/x3/cx3zh2agc75ltp7hfhb7dy2dwcsh55fzc3a3yclzruofj4xhxcfb.py
# Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x => constant_pad_nd
#   x_1 => constant_pad_nd_1
# Graph fragment:
#   %constant_pad_nd : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%arg4_1, [1, 1, 1, 1, 1, 1], 0.5), kwargs = {})
#   %constant_pad_nd_1 : [num_users=5] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%constant_pad_nd, [2, 2, 2, 2, 2, 2], 0.25), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_constant_pad_nd_0(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, ks6, ks7, ks8, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x6 = ((xindex // ks0) % ks1)
    x1 = ((xindex // ks3) % ks4)
    x0 = (xindex % ks3)
    x2 = ((xindex // ks7) % ks1)
    x3 = xindex // ks8
    x8 = xindex
    tmp0 = (-2) + x6
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = 2 + ks2
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = 2 + ks5
    tmp8 = tmp5 < tmp7
    tmp9 = (-2) + x0
    tmp10 = tmp9 >= tmp1
    tmp11 = 2 + ks6
    tmp12 = tmp9 < tmp11
    tmp13 = tmp2 & tmp4
    tmp14 = tmp13 & tmp6
    tmp15 = tmp14 & tmp8
    tmp16 = tmp15 & tmp10
    tmp17 = tmp16 & tmp12
    tmp18 = (-3) + x6
    tmp19 = tl.full([1], 0, tl.int64)
    tmp20 = tmp18 >= tmp19
    tmp21 = tl.broadcast_to(ks2, [XBLOCK])
    tmp22 = tmp18 < tmp21
    tmp23 = (-3) + x1
    tmp24 = tmp23 >= tmp19
    tmp25 = tl.broadcast_to(ks5, [XBLOCK])
    tmp26 = tmp23 < tmp25
    tmp27 = (-3) + x0
    tmp28 = tmp27 >= tmp19
    tmp29 = tl.broadcast_to(ks6, [XBLOCK])
    tmp30 = tmp27 < tmp29
    tmp31 = tmp20 & tmp22
    tmp32 = tmp31 & tmp24
    tmp33 = tmp32 & tmp26
    tmp34 = tmp33 & tmp28
    tmp35 = tmp34 & tmp30
    tmp36 = tmp35 & tmp17
    tmp37 = tl.load(in_ptr0 + ((-3) + x0 + ((-3)*ks6) + ks6*x1 + ((-3)*ks5*ks6) + ks5*ks6*x2 + ks2*ks5*ks6*x3), tmp36 & xmask, eviction_policy='evict_last', other=0.5)
    tmp38 = tl.full(tmp37.shape, 0.25, tmp37.dtype)
    tmp39 = tl.where(tmp17, tmp37, tmp38)
    tl.store(out_ptr0 + (x8), tmp39, xmask)




# kernel path: /tmp/torchinductor_sahanp/xb/cxbor62mht4eobrekl4q6n2mvlskglr3p54wbmenu4tr63nlunxx.py
# Topologically Sorted Source Nodes: [x_2, x_3, target, loss], Original ATen: [aten.tanh, aten.sub, aten.zeros_like, aten.smooth_l1_loss]
# Source node to ATen node mapping:
#   loss => abs_1, div, lt_7, mean, mul_37, pow_1, sub_30, sub_31, where
#   target => full
#   x_2 => sub_12, tanh
#   x_3 => sub_21, tanh_1
# Graph fragment:
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%constant_pad_nd_1,), kwargs = {})
#   %sub_12 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%constant_pad_nd_1, %tanh), kwargs = {})
#   %tanh_1 : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%sub_12,), kwargs = {})
#   %sub_21 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_12, %tanh_1), kwargs = {})
#   %full : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, %arg0_1, %sym_size_int_4, %sym_size_int_5, %sym_size_int_6], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sub_30 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_21, %full), kwargs = {})
#   %abs_1 : [num_users=3] = call_function[target=torch.ops.aten.abs.default](args = (%sub_30,), kwargs = {})
#   %lt_7 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%abs_1, 1.0), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%abs_1, 2), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_1, 0.5), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_37, 1.0), kwargs = {})
#   %sub_31 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%abs_1, 0.5), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%lt_7, %div, %sub_31), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%where,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_smooth_l1_loss_sub_tanh_zeros_like_1(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, ks6, ks7, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 21
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp22 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = r0_1 + x0*((20 + 216*ks0 + 36*ks0*ks1 + 36*ks0*ks2 + 36*ks0*ks3 + 6*ks0*ks1*ks2 + 6*ks0*ks1*ks3 + 6*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 21)
        tmp1 = 216*ks0 + 36*ks0*ks1 + 36*ks0*ks2 + 36*ks0*ks3 + 6*ks0*ks1*ks2 + 6*ks0*ks1*ks3 + 6*ks0*ks2*ks3 + ks0*ks1*ks2*ks3
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (6*((((r0_1 + x0*((20 + 216*ks0 + 36*ks0*ks1 + 36*ks0*ks2 + 36*ks0*ks3 + 6*ks0*ks1*ks2 + 6*ks0*ks1*ks3 + 6*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 21)) // ks6) % ks7)) + 36*((((r0_1 + x0*((20 + 216*ks0 + 36*ks0*ks1 + 36*ks0*ks2 + 36*ks0*ks3 + 6*ks0*ks1*ks2 + 6*ks0*ks1*ks3 + 6*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 21)) // ks4) % ks5)) + 216*((((r0_1 + x0*((20 + 216*ks0 + 36*ks0*ks1 + 36*ks0*ks2 + 36*ks0*ks3 + 6*ks0*ks1*ks2 + 6*ks0*ks1*ks3 + 6*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 21)) // (216 + 36*ks1 + 36*ks2 + 36*ks3 + 6*ks1*ks2 + 6*ks1*ks3 + 6*ks2*ks3 + ks1*ks2*ks3)) % ks0)) + ks3*((((r0_1 + x0*((20 + 216*ks0 + 36*ks0*ks1 + 36*ks0*ks2 + 36*ks0*ks3 + 6*ks0*ks1*ks2 + 6*ks0*ks1*ks3 + 6*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 21)) // ks6) % ks7)) + 6*ks2*((((r0_1 + x0*((20 + 216*ks0 + 36*ks0*ks1 + 36*ks0*ks2 + 36*ks0*ks3 + 6*ks0*ks1*ks2 + 6*ks0*ks1*ks3 + 6*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 21)) // ks4) % ks5)) + 6*ks3*((((r0_1 + x0*((20 + 216*ks0 + 36*ks0*ks1 + 36*ks0*ks2 + 36*ks0*ks3 + 6*ks0*ks1*ks2 + 6*ks0*ks1*ks3 + 6*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 21)) // ks4) % ks5)) + 36*ks1*((((r0_1 + x0*((20 + 216*ks0 + 36*ks0*ks1 + 36*ks0*ks2 + 36*ks0*ks3 + 6*ks0*ks1*ks2 + 6*ks0*ks1*ks3 + 6*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 21)) // (216 + 36*ks1 + 36*ks2 + 36*ks3 + 6*ks1*ks2 + 6*ks1*ks3 + 6*ks2*ks3 + ks1*ks2*ks3)) % ks0)) + 36*ks2*((((r0_1 + x0*((20 + 216*ks0 + 36*ks0*ks1 + 36*ks0*ks2 + 36*ks0*ks3 + 6*ks0*ks1*ks2 + 6*ks0*ks1*ks3 + 6*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 21)) // (216 + 36*ks1 + 36*ks2 + 36*ks3 + 6*ks1*ks2 + 6*ks1*ks3 + 6*ks2*ks3 + ks1*ks2*ks3)) % ks0)) + 36*ks3*((((r0_1 + x0*((20 + 216*ks0 + 36*ks0*ks1 + 36*ks0*ks2 + 36*ks0*ks3 + 6*ks0*ks1*ks2 + 6*ks0*ks1*ks3 + 6*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 21)) // (216 + 36*ks1 + 36*ks2 + 36*ks3 + 6*ks1*ks2 + 6*ks1*ks3 + 6*ks2*ks3 + ks1*ks2*ks3)) % ks0)) + ks2*ks3*((((r0_1 + x0*((20 + 216*ks0 + 36*ks0*ks1 + 36*ks0*ks2 + 36*ks0*ks3 + 6*ks0*ks1*ks2 + 6*ks0*ks1*ks3 + 6*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 21)) // ks4) % ks5)) + 6*ks1*ks2*((((r0_1 + x0*((20 + 216*ks0 + 36*ks0*ks1 + 36*ks0*ks2 + 36*ks0*ks3 + 6*ks0*ks1*ks2 + 6*ks0*ks1*ks3 + 6*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 21)) // (216 + 36*ks1 + 36*ks2 + 36*ks3 + 6*ks1*ks2 + 6*ks1*ks3 + 6*ks2*ks3 + ks1*ks2*ks3)) % ks0)) + 6*ks1*ks3*((((r0_1 + x0*((20 + 216*ks0 + 36*ks0*ks1 + 36*ks0*ks2 + 36*ks0*ks3 + 6*ks0*ks1*ks2 + 6*ks0*ks1*ks3 + 6*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 21)) // (216 + 36*ks1 + 36*ks2 + 36*ks3 + 6*ks1*ks2 + 6*ks1*ks3 + 6*ks2*ks3 + ks1*ks2*ks3)) % ks0)) + 6*ks2*ks3*((((r0_1 + x0*((20 + 216*ks0 + 36*ks0*ks1 + 36*ks0*ks2 + 36*ks0*ks3 + 6*ks0*ks1*ks2 + 6*ks0*ks1*ks3 + 6*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 21)) // (216 + 36*ks1 + 36*ks2 + 36*ks3 + 6*ks1*ks2 + 6*ks1*ks3 + 6*ks2*ks3 + ks1*ks2*ks3)) % ks0)) + ks1*ks2*ks3*((((r0_1 + x0*((20 + 216*ks0 + 36*ks0*ks1 + 36*ks0*ks2 + 36*ks0*ks3 + 6*ks0*ks1*ks2 + 6*ks0*ks1*ks3 + 6*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 21)) // (216 + 36*ks1 + 36*ks2 + 36*ks3 + 6*ks1*ks2 + 6*ks1*ks3 + 6*ks2*ks3 + ks1*ks2*ks3)) % ks0)) + (((r0_1 + x0*((20 + 216*ks0 + 36*ks0*ks1 + 36*ks0*ks2 + 36*ks0*ks3 + 6*ks0*ks1*ks2 + 6*ks0*ks1*ks3 + 6*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 21)) % ks6))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = libdevice.tanh(tmp3)
        tmp5 = tmp3 - tmp4
        tmp6 = libdevice.tanh(tmp5)
        tmp7 = tmp5 - tmp6
        tmp8 = 0.0
        tmp9 = tmp7 - tmp8
        tmp10 = tl_math.abs(tmp9)
        tmp11 = 1.0
        tmp12 = tmp10 < tmp11
        tmp13 = tmp10 * tmp10
        tmp14 = 0.5
        tmp15 = tmp13 * tmp14
        tmp16 = tmp15 * tmp11
        tmp17 = tmp10 - tmp14
        tmp18 = tl.where(tmp12, tmp16, tmp17)
        tmp19 = tl.full(tmp18.shape, 0, tmp18.dtype)
        tmp20 = tl.where(tmp2, tmp18, tmp19)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(r0_mask & xmask, tmp23, _tmp22)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp22, xmask)




# kernel path: /tmp/torchinductor_sahanp/cq/ccqbdzw6p74cgt6l7baximv36edrwp66piyvsl6r3f2anq62zibw.py
# Topologically Sorted Source Nodes: [x_2, x_3, target, loss], Original ATen: [aten.tanh, aten.sub, aten.zeros_like, aten.smooth_l1_loss]
# Source node to ATen node mapping:
#   loss => abs_1, div, lt_7, mean, mul_37, pow_1, sub_30, sub_31, where
#   target => full
#   x_2 => sub_12, tanh
#   x_3 => sub_21, tanh_1
# Graph fragment:
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%constant_pad_nd_1,), kwargs = {})
#   %sub_12 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%constant_pad_nd_1, %tanh), kwargs = {})
#   %tanh_1 : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%sub_12,), kwargs = {})
#   %sub_21 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_12, %tanh_1), kwargs = {})
#   %full : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, %arg0_1, %sym_size_int_4, %sym_size_int_5, %sym_size_int_6], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sub_30 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_21, %full), kwargs = {})
#   %abs_1 : [num_users=3] = call_function[target=torch.ops.aten.abs.default](args = (%sub_30,), kwargs = {})
#   %lt_7 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%abs_1, 1.0), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%abs_1, 2), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_1, 0.5), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_37, 1.0), kwargs = {})
#   %sub_31 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%abs_1, 0.5), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%lt_7, %div, %sub_31), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%where,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_smooth_l1_loss_sub_tanh_zeros_like_2(in_out_ptr0, in_ptr0, ks0, ks1, ks2, ks3, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp5 = 216*ks0 + 36*ks0*ks1 + 36*ks0*ks2 + 36*ks0*ks3 + 6*ks0*ks1*ks2 + 6*ks0*ks1*ks3 + 6*ks0*ks2*ks3 + ks0*ks1*ks2*ks3
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp4 / tmp6
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp7, None)







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
        ps0 = 36 + 6*s2 + 6*s3 + s2*s3
        ps1 = 6 + s1
        ps2 = 6 + s3
        ps3 = 6 + s2
        ps4 = 36 + 6*s2 + 6*s3 + s2*s3
        ps5 = 216 + 36*s1 + 36*s2 + 36*s3 + 6*s1*s2 + 6*s1*s3 + 6*s2*s3 + s1*s2*s3
        buf0 = empty_strided_cuda((1, s0, 6 + s1, 6 + s2, 6 + s3), (216*s0 + 36*s0*s1 + 36*s0*s2 + 36*s0*s3 + 6*s0*s1*s2 + 6*s0*s1*s3 + 6*s0*s2*s3 + s0*s1*s2*s3, 216 + 36*s1 + 36*s2 + 36*s3 + 6*s1*s2 + 6*s1*s3 + 6*s2*s3 + s1*s2*s3, 36 + 6*s2 + 6*s3 + s2*s3, 6 + s3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_0_xnumel = 216*s0 + 36*s0*s1 + 36*s0*s2 + 36*s0*s3 + 6*s0*s1*s2 + 6*s0*s1*s3 + 6*s0*s2*s3 + s0*s1*s2*s3
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_0[grid(triton_poi_fused_constant_pad_nd_0_xnumel)](arg4_1, buf0, 1444, 38, 32, 38, 38, 32, 32, 1444, 54872, 164616, XBLOCK=512, num_warps=8, num_stages=1)
        del arg4_1
        buf1 = empty_strided_cuda((21, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [x_2, x_3, target, loss], Original ATen: [aten.tanh, aten.sub, aten.zeros_like, aten.smooth_l1_loss]
        triton_red_fused_smooth_l1_loss_sub_tanh_zeros_like_1_r0_numel = (20 + 216*s0 + 36*s0*s1 + 36*s0*s2 + 36*s0*s3 + 6*s0*s1*s2 + 6*s0*s1*s3 + 6*s0*s2*s3 + s0*s1*s2*s3) // 21
        stream0 = get_raw_stream(0)
        triton_red_fused_smooth_l1_loss_sub_tanh_zeros_like_1[grid(21)](buf0, buf1, 3, 32, 32, 32, 1444, 38, 38, 38, 21, 7839, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del buf0
        buf2 = empty_strided_cuda((), (), torch.float32)
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [x_2, x_3, target, loss], Original ATen: [aten.tanh, aten.sub, aten.zeros_like, aten.smooth_l1_loss]
        stream0 = get_raw_stream(0)
        triton_per_fused_smooth_l1_loss_sub_tanh_zeros_like_2[grid(1)](buf3, buf1, 3, 32, 32, 32, 1, 21, XBLOCK=1, num_warps=2, num_stages=1)
        del buf1
    return (buf3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 3
    arg1_1 = 32
    arg2_1 = 32
    arg3_1 = 32
    arg4_1 = rand_strided((1, 3, 32, 32, 32), (98304, 32768, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
