# AOT ID: ['66_inference']
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


# kernel path: /tmp/torchinductor_sahanp/jo/cjoc4r25cie6s5pqzn5aewv34hmzxjdpewdjaf3tkrk7jykpvliw.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4], Original ATen: [aten.max_pool2d_with_indices, aten.constant_pad_nd, aten.hardtanh, aten.tanh]
# Source node to ATen node mapping:
#   input_1 => _low_memory_max_pool2d_with_offsets
#   input_2 => constant_pad_nd
#   input_3 => clamp_max, clamp_min
#   input_4 => tanh
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%arg3_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %constant_pad_nd : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%getitem, [1, 1, 1, 1, 1, 1], 0.0), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%constant_pad_nd, -1), kwargs = {})
#   %clamp_max : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 1), kwargs = {})
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%clamp_max,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_constant_pad_nd_hardtanh_max_pool2d_with_indices_tanh_0(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, ks6, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x5 = xindex // ks0
    x1 = ((xindex // ks2) % ks3)
    x0 = (xindex % ks2)
    x2 = xindex // ks6
    x6 = xindex
    tmp0 = (-1) + x5
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = ks1
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = ks4 // 2
    tmp8 = tmp5 < tmp7
    tmp9 = (-1) + x0
    tmp10 = tmp9 >= tmp1
    tmp11 = ks5 // 2
    tmp12 = tmp9 < tmp11
    tmp13 = tmp2 & tmp4
    tmp14 = tmp13 & tmp6
    tmp15 = tmp14 & tmp8
    tmp16 = tmp15 & tmp10
    tmp17 = tmp16 & tmp12
    tmp18 = tl.load(in_ptr0 + ((-2) + ((-2)*ks5) + 2*x0 + ((-1)*ks4*ks5) + 2*ks5*x1 + ks4*ks5*x2), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.load(in_ptr0 + ((-1) + ((-2)*ks5) + 2*x0 + ((-1)*ks4*ks5) + 2*ks5*x1 + ks4*ks5*x2), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = triton_helpers.maximum(tmp19, tmp18)
    tmp21 = tl.load(in_ptr0 + ((-2) + ((-1)*ks5) + 2*x0 + ((-1)*ks4*ks5) + 2*ks5*x1 + ks4*ks5*x2), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = triton_helpers.maximum(tmp21, tmp20)
    tmp23 = tl.load(in_ptr0 + ((-1) + ((-1)*ks5) + 2*x0 + ((-1)*ks4*ks5) + 2*ks5*x1 + ks4*ks5*x2), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = triton_helpers.maximum(tmp23, tmp22)
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp17, tmp24, tmp25)
    tmp27 = -1.0
    tmp28 = triton_helpers.maximum(tmp26, tmp27)
    tmp29 = 1.0
    tmp30 = triton_helpers.minimum(tmp28, tmp29)
    tmp31 = libdevice.tanh(tmp30)
    tl.store(out_ptr0 + (x6), tmp31, xmask)




# kernel path: /tmp/torchinductor_sahanp/wo/cwoedzysywa2j7qvbn5lysyh6up7nvcecvbr23xs5olg5t3tb6mq.py
# Topologically Sorted Source Nodes: [input_5, input_6, input_7, input_8], Original ATen: [aten.max_pool2d_with_indices, aten.constant_pad_nd, aten.hardtanh, aten.tanh]
# Source node to ATen node mapping:
#   input_5 => _low_memory_max_pool2d_with_offsets_1
#   input_6 => constant_pad_nd_1
#   input_7 => clamp_max_1, clamp_min_1
#   input_8 => tanh_1
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%tanh, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %constant_pad_nd_1 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%getitem_2, [1, 1, 1, 1, 1, 1], 0.0), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%constant_pad_nd_1, -1), kwargs = {})
#   %clamp_max_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_1, 1), kwargs = {})
#   %tanh_1 : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%clamp_max_1,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_constant_pad_nd_hardtanh_max_pool2d_with_indices_tanh_1(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, ks6, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x5 = xindex // ks0
    x1 = ((xindex // ks2) % ks3)
    x0 = (xindex % ks2)
    x2 = xindex // ks6
    x6 = xindex
    tmp0 = (-1) + x5
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = 2 + ks1
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = 1 + (ks4 // 4)
    tmp8 = tmp5 < tmp7
    tmp9 = (-1) + x0
    tmp10 = tmp9 >= tmp1
    tmp11 = 1 + (ks5 // 4)
    tmp12 = tmp9 < tmp11
    tmp13 = tmp2 & tmp4
    tmp14 = tmp13 & tmp6
    tmp15 = tmp14 & tmp8
    tmp16 = tmp15 & tmp10
    tmp17 = tmp16 & tmp12
    tmp18 = tl.load(in_ptr0 + ((-10) + ((-4)*(ks5 // 2)) + ((-2)*(ks4 // 2)) + 2*x0 + 4*x1 + 4*x2 + ((-1)*(ks4 // 2)*(ks5 // 2)) + 2*x1*(ks5 // 2) + 2*x2*(ks4 // 2) + 2*x2*(ks5 // 2) + x2*(ks4 // 2)*(ks5 // 2)), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.load(in_ptr0 + ((-9) + ((-4)*(ks5 // 2)) + ((-2)*(ks4 // 2)) + 2*x0 + 4*x1 + 4*x2 + ((-1)*(ks4 // 2)*(ks5 // 2)) + 2*x1*(ks5 // 2) + 2*x2*(ks4 // 2) + 2*x2*(ks5 // 2) + x2*(ks4 // 2)*(ks5 // 2)), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = triton_helpers.maximum(tmp19, tmp18)
    tmp21 = tl.load(in_ptr0 + ((-8) + ((-3)*(ks5 // 2)) + ((-2)*(ks4 // 2)) + 2*x0 + 4*x1 + 4*x2 + ((-1)*(ks4 // 2)*(ks5 // 2)) + 2*x1*(ks5 // 2) + 2*x2*(ks4 // 2) + 2*x2*(ks5 // 2) + x2*(ks4 // 2)*(ks5 // 2)), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = triton_helpers.maximum(tmp21, tmp20)
    tmp23 = tl.load(in_ptr0 + ((-7) + ((-3)*(ks5 // 2)) + ((-2)*(ks4 // 2)) + 2*x0 + 4*x1 + 4*x2 + ((-1)*(ks4 // 2)*(ks5 // 2)) + 2*x1*(ks5 // 2) + 2*x2*(ks4 // 2) + 2*x2*(ks5 // 2) + x2*(ks4 // 2)*(ks5 // 2)), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = triton_helpers.maximum(tmp23, tmp22)
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp17, tmp24, tmp25)
    tmp27 = -1.0
    tmp28 = triton_helpers.maximum(tmp26, tmp27)
    tmp29 = 1.0
    tmp30 = triton_helpers.minimum(tmp28, tmp29)
    tmp31 = libdevice.tanh(tmp30)
    tl.store(out_ptr0 + (x6), tmp31, xmask)




# kernel path: /tmp/torchinductor_sahanp/nz/cnz3qyvq76ktsuwbec6occw45ykqcscrelp5wrylo4pc2vyqpgd7.py
# Topologically Sorted Source Nodes: [dist_pos, add, dist_neg, sub, loss, triplet_loss, poisson_loss, ones_like, add_1, zeros_like, huber_loss, add_2, softmargin_loss, ones_like_1, add_3, truediv], Original ATen: [aten.norm, aten.add, aten.sub, aten.clamp_min, aten.mean, aten.exp, aten.ones_like, aten.mul, aten.zeros_like, aten.huber_loss, aten.soft_margin_loss, aten.div]
# Source node to ATen node mapping:
#   add => add_46
#   add_1 => add_59
#   add_2 => add_60
#   add_3 => add_61
#   dist_neg => pow_4
#   dist_pos => pow_2
#   huber_loss => abs_1, lt_1, mean_2, mul_55, mul_56, mul_57, sub_41, sub_42, where
#   loss => clamp_min_2
#   ones_like => full
#   ones_like_1 => full_2
#   poisson_loss => exp, mean_1, mul_46, sub_38
#   softmargin_loss => exp_1, log1p, mean_3, mul_60, neg
#   sub => sub_34
#   triplet_loss => mean
#   truediv => div
#   zeros_like => full_1
# Graph fragment:
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %add_46 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_2, 1.0), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_2, 0.5), kwargs = {})
#   %sub_34 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_46, %pow_4), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_34, 0), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%clamp_min_2,), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%view,), kwargs = {})
#   %full : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, %sym_size_int_14], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full, %view), kwargs = {})
#   %sub_38 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%exp, %mul_46), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sub_38,), kwargs = {})
#   %add_59 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, %mean_1), kwargs = {})
#   %full_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, %sym_size_int_14], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sub_41 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %full_1), kwargs = {})
#   %abs_1 : [num_users=4] = call_function[target=torch.ops.aten.abs.default](args = (%sub_41,), kwargs = {})
#   %lt_1 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%abs_1, 1.0), kwargs = {})
#   %mul_55 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%abs_1, 0.5), kwargs = {})
#   %mul_56 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_55, %abs_1), kwargs = {})
#   %sub_42 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%abs_1, 0.5), kwargs = {})
#   %mul_57 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_42, 1.0), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%lt_1, %mul_56, %mul_57), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%where,), kwargs = {})
#   %add_60 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_59, %mean_2), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%view,), kwargs = {})
#   %full_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, %sym_size_int_14], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_60 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg, %full_2), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_60,), kwargs = {})
#   %log1p : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_1,), kwargs = {})
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%log1p,), kwargs = {})
#   %add_61 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_60, %mean_3), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_61, 4), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_clamp_min_div_exp_huber_loss_mean_mul_norm_ones_like_soft_margin_loss_sub_zeros_like_2(in_out_ptr0, in_ptr0, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp6 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp19 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp26 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl_math.exp(tmp0)
        tmp2 = 1.0
        tmp3 = tmp2 * tmp0
        tmp4 = tmp1 - tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask, tmp7, _tmp6)
        tmp8 = 0.0
        tmp9 = tmp0 - tmp8
        tmp10 = tl_math.abs(tmp9)
        tmp11 = tmp10 < tmp2
        tmp12 = 0.5
        tmp13 = tmp10 * tmp12
        tmp14 = tmp13 * tmp10
        tmp15 = tmp10 - tmp12
        tmp16 = tmp15 * tmp2
        tmp17 = tl.where(tmp11, tmp14, tmp16)
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(r0_mask, tmp20, _tmp19)
        tmp21 = -tmp0
        tmp22 = tmp21 * tmp2
        tmp23 = tl_math.exp(tmp22)
        tmp24 = libdevice.log1p(tmp23)
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(r0_mask, tmp27, _tmp26)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tmp28 = 0.0
    tmp29 = tmp28 / tmp28
    tmp30 = 36 + 9*ks0 + 12*(ks1 // 4) + 12*(ks2 // 4) + 3*ks0*(ks1 // 4) + 3*ks0*(ks2 // 4) + 4*(ks1 // 4)*(ks2 // 4) + ks0*(ks1 // 4)*(ks2 // 4)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp6 / tmp31
    tmp33 = tmp29 + tmp32
    tmp34 = tmp19 / tmp31
    tmp35 = tmp33 + tmp34
    tmp36 = tmp26 / tmp31
    tmp37 = tmp35 + tmp36
    tmp38 = 0.25
    tmp39 = tmp37 * tmp38
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp39, None)







def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        ps0 = 4 + 2*(s1 // 2) + 2*(s2 // 2) + (s1 // 2)*(s2 // 2)
        ps1 = 2 + (s2 // 2)
        ps2 = 2 + (s1 // 2)
        ps3 = 4 + 2*(s1 // 2) + 2*(s2 // 2) + (s1 // 2)*(s2 // 2)
        buf0 = empty_strided_cuda((1, 2 + s0, 2 + (s1 // 2), 2 + (s2 // 2)), (8 + 4*s0 + 4*(s1 // 2) + 4*(s2 // 2) + 2*s0*(s1 // 2) + 2*s0*(s2 // 2) + 2*(s1 // 2)*(s2 // 2) + s0*(s1 // 2)*(s2 // 2), 4 + 2*(s1 // 2) + 2*(s2 // 2) + (s1 // 2)*(s2 // 2), 2 + (s2 // 2), 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4], Original ATen: [aten.max_pool2d_with_indices, aten.constant_pad_nd, aten.hardtanh, aten.tanh]
        triton_poi_fused_constant_pad_nd_hardtanh_max_pool2d_with_indices_tanh_0_xnumel = 8 + 4*s0 + 4*(s1 // 2) + 4*(s2 // 2) + 2*s0*(s1 // 2) + 2*s0*(s2 // 2) + 2*(s1 // 2)*(s2 // 2) + s0*(s1 // 2)*(s2 // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_hardtanh_max_pool2d_with_indices_tanh_0[grid(triton_poi_fused_constant_pad_nd_hardtanh_max_pool2d_with_indices_tanh_0_xnumel)](arg3_1, buf0, 1156, 3, 34, 34, 64, 64, 1156, 5780, XBLOCK=256, num_warps=4, num_stages=1)
        del arg3_1
        ps4 = 9 + 3*(s1 // 4) + 3*(s2 // 4) + (s1 // 4)*(s2 // 4)
        ps5 = 3 + (s2 // 4)
        ps6 = 3 + (s1 // 4)
        ps7 = 9 + 3*(s1 // 4) + 3*(s2 // 4) + (s1 // 4)*(s2 // 4)
        buf1 = empty_strided_cuda((1, 4 + s0, 3 + (s1 // 4), 3 + (s2 // 4)), (36 + 9*s0 + 12*(s1 // 4) + 12*(s2 // 4) + 3*s0*(s1 // 4) + 3*s0*(s2 // 4) + 4*(s1 // 4)*(s2 // 4) + s0*(s1 // 4)*(s2 // 4), 9 + 3*(s1 // 4) + 3*(s2 // 4) + (s1 // 4)*(s2 // 4), 3 + (s2 // 4), 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_5, input_6, input_7, input_8], Original ATen: [aten.max_pool2d_with_indices, aten.constant_pad_nd, aten.hardtanh, aten.tanh]
        triton_poi_fused_constant_pad_nd_hardtanh_max_pool2d_with_indices_tanh_1_xnumel = 36 + 9*s0 + 12*(s1 // 4) + 12*(s2 // 4) + 3*s0*(s1 // 4) + 3*s0*(s2 // 4) + 4*(s1 // 4)*(s2 // 4) + s0*(s1 // 4)*(s2 // 4)
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_hardtanh_max_pool2d_with_indices_tanh_1[grid(triton_poi_fused_constant_pad_nd_hardtanh_max_pool2d_with_indices_tanh_1_xnumel)](buf0, buf1, 361, 3, 19, 19, 64, 64, 361, 2527, XBLOCK=128, num_warps=4, num_stages=1)
        del buf0
        buf4 = empty_strided_cuda((), (), torch.float32)
        buf7 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [dist_pos, add, dist_neg, sub, loss, triplet_loss, poisson_loss, ones_like, add_1, zeros_like, huber_loss, add_2, softmargin_loss, ones_like_1, add_3, truediv], Original ATen: [aten.norm, aten.add, aten.sub, aten.clamp_min, aten.mean, aten.exp, aten.ones_like, aten.mul, aten.zeros_like, aten.huber_loss, aten.soft_margin_loss, aten.div]
        triton_red_fused_add_clamp_min_div_exp_huber_loss_mean_mul_norm_ones_like_soft_margin_loss_sub_zeros_like_2_r0_numel = 36 + 9*s0 + 12*(s1 // 4) + 12*(s2 // 4) + 3*s0*(s1 // 4) + 3*s0*(s2 // 4) + 4*(s1 // 4)*(s2 // 4) + s0*(s1 // 4)*(s2 // 4)
        stream0 = get_raw_stream(0)
        triton_red_fused_add_clamp_min_div_exp_huber_loss_mean_mul_norm_ones_like_soft_margin_loss_sub_zeros_like_2[grid(1)](buf7, buf1, 3, 64, 64, 1, 2527, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del buf1
    return (buf7, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 3
    arg1_1 = 64
    arg2_1 = 64
    arg3_1 = rand_strided((1, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
