# AOT ID: ['161_inference']
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


# kernel path: /tmp/torchinductor_sahanp/rs/crskuld6jenluj4rizsahgnikbtaiscfdqjwcdyjgnrg7gnendwh.py
# Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.constant_pad_nd, aten.view, aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   x => constant_pad_nd
#   x_1 => add_8, mul_15, rsqrt, sub_4, var_mean, view, view_1
#   x_2 => var_mean_1, view_2
# Graph fragment:
#   %constant_pad_nd : [num_users=3] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%arg2_1, [2, 2, 2, 2], 3.0), kwargs = {})
#   %view : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%constant_pad_nd, [1, 3, %sym_size_int, %sym_size_int_1]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %getitem_1), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_8,), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt), kwargs = {})
#   %view_1 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_15, [1, 3, %sym_size_int, %sym_size_int_1]), kwargs = {})
#   %view_2 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_1, [1, 3, %sym_size_int, %sym_size_int_1]), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_2, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_constant_pad_nd_view_0(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 3
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp14_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp14_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp14_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index // ks0
        r0_1 = (r0_index % ks0)
        tmp0 = (-2) + r0_2
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = ks1
        tmp4 = tmp0 < tmp3
        tmp5 = (-2) + r0_1
        tmp6 = tmp5 >= tmp1
        tmp7 = ks2
        tmp8 = tmp5 < tmp7
        tmp9 = tmp2 & tmp4
        tmp10 = tmp9 & tmp6
        tmp11 = tmp10 & tmp8
        tmp12 = tl.load(in_ptr0 + ((-2) + r0_1 + ((-2)*ks2) + ks2*r0_2 + ks1*ks2*x0), r0_mask & tmp11 & xmask, eviction_policy='evict_last', other=3.0)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
        tmp14_mean_next, tmp14_m2_next, tmp14_weight_next = triton_helpers.welford_reduce(
            tmp13, tmp14_mean, tmp14_m2, tmp14_weight, roffset == 0
        )
        tmp14_mean = tl.where(r0_mask & xmask, tmp14_mean_next, tmp14_mean)
        tmp14_m2 = tl.where(r0_mask & xmask, tmp14_m2_next, tmp14_m2)
        tmp14_weight = tl.where(r0_mask & xmask, tmp14_weight_next, tmp14_weight)
    tmp17, tmp18, tmp19 = triton_helpers.welford(tmp14_mean, tmp14_m2, tmp14_weight, 1)
    tmp14 = tmp17[:, None]
    tmp15 = tmp18[:, None]
    tmp16 = tmp19[:, None]
    tl.store(out_ptr0 + (x0), tmp14, xmask)
    tl.store(out_ptr1 + (x0), tmp15, xmask)
    tmp42_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp42_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp42_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index // ks0
        r0_1 = (r0_index % ks0)
        tmp20 = (-2) + r0_2
        tmp21 = tl.full([1, 1], 0, tl.int64)
        tmp22 = tmp20 >= tmp21
        tmp23 = ks1
        tmp24 = tmp20 < tmp23
        tmp25 = (-2) + r0_1
        tmp26 = tmp25 >= tmp21
        tmp27 = ks2
        tmp28 = tmp25 < tmp27
        tmp29 = tmp22 & tmp24
        tmp30 = tmp29 & tmp26
        tmp31 = tmp30 & tmp28
        tmp32 = tl.load(in_ptr0 + ((-2) + r0_1 + ((-2)*ks2) + ks2*r0_2 + ks1*ks2*x0), r0_mask & tmp31 & xmask, eviction_policy='evict_last', other=3.0)
        tmp33 = tmp32 - tmp14
        tmp34 = 16 + 4*ks1 + 4*ks2 + ks1*ks2
        tmp35 = tmp34.to(tl.float32)
        tmp36 = tmp15 / tmp35
        tmp37 = 1e-05
        tmp38 = tmp36 + tmp37
        tmp39 = libdevice.rsqrt(tmp38)
        tmp40 = tmp33 * tmp39
        tmp41 = tl.broadcast_to(tmp40, [XBLOCK, R0_BLOCK])
        tmp42_mean_next, tmp42_m2_next, tmp42_weight_next = triton_helpers.welford_reduce(
            tmp41, tmp42_mean, tmp42_m2, tmp42_weight, roffset == 0
        )
        tmp42_mean = tl.where(r0_mask & xmask, tmp42_mean_next, tmp42_mean)
        tmp42_m2 = tl.where(r0_mask & xmask, tmp42_m2_next, tmp42_m2)
        tmp42_weight = tl.where(r0_mask & xmask, tmp42_weight_next, tmp42_weight)
    tmp45, tmp46, tmp47 = triton_helpers.welford(tmp42_mean, tmp42_m2, tmp42_weight, 1)
    tmp42 = tmp45[:, None]
    tmp43 = tmp46[:, None]
    tmp44 = tmp47[:, None]
    tl.store(out_ptr2 + (x0), tmp42, xmask)
    tl.store(out_ptr3 + (x0), tmp43, xmask)




# kernel path: /tmp/torchinductor_sahanp/wx/cwxceqdppnx66achky5hkgynjh3esjr7od2ollxwu3afb6e6gdue.py
# Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.constant_pad_nd, aten.view, aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   x => constant_pad_nd
#   x_1 => add_8, mul_15, rsqrt, sub_4, var_mean, view, view_1
#   x_2 => add_21, mul_35, rsqrt_1, sub_11, var_mean_1, view_2
# Graph fragment:
#   %constant_pad_nd : [num_users=3] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%arg2_1, [2, 2, 2, 2], 3.0), kwargs = {})
#   %view : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%constant_pad_nd, [1, 3, %sym_size_int, %sym_size_int_1]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %getitem_1), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_8,), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt), kwargs = {})
#   %view_1 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_15, [1, 3, %sym_size_int, %sym_size_int_1]), kwargs = {})
#   %view_2 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_1, [1, 3, %sym_size_int, %sym_size_int_1]), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_2, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_2, %getitem_3), kwargs = {})
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_21,), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %rsqrt_1), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_constant_pad_nd_view_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // ks0) % ks1)
    x0 = (xindex % ks0)
    x2 = xindex // ks4
    x4 = xindex
    tmp13 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp0 = (-2) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = ks2
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + x0
    tmp6 = tmp5 >= tmp1
    tmp7 = ks3
    tmp8 = tmp5 < tmp7
    tmp9 = tmp2 & tmp4
    tmp10 = tmp9 & tmp6
    tmp11 = tmp10 & tmp8
    tmp12 = tl.load(in_ptr0 + ((-2) + x0 + ((-2)*ks3) + ks3*x1 + ks2*ks3*x2), tmp11 & xmask, eviction_policy='evict_last', other=3.0)
    tmp14 = tmp12 - tmp13
    tmp16 = 16 + 4*ks2 + 4*ks3 + ks2*ks3
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tmp22 = tmp14 * tmp21
    tmp24 = tmp22 - tmp23
    tmp26 = tmp25 / tmp17
    tmp27 = tmp26 + tmp19
    tmp28 = libdevice.rsqrt(tmp27)
    tmp29 = tmp24 * tmp28
    tl.store(out_ptr0 + (x4), tmp29, xmask)




# kernel path: /tmp/torchinductor_sahanp/qd/cqdjvudwpqbdc5452vspwou6jdvswqcqsqhmr3iusrhbfgcf675k.py
# Topologically Sorted Source Nodes: [mean, var], Original ATen: [aten.mean, aten.var]
# Source node to ATen node mapping:
#   mean => mean
#   var => var
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_3, [2, 3], True), kwargs = {})
#   %var : [num_users=1] = call_function[target=torch.ops.aten.var.correction](args = (%view_3, [2, 3]), kwargs = {correction: 1, keepdim: True})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_mean_var_2(in_out_ptr0, in_out_ptr1, in_ptr0, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 3
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    tmp4_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 16*x0 + 4*ks0*x0 + 4*ks1*x0 + ks0*ks1*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp4_mean, tmp4_m2, tmp4_weight, roffset == 0
        )
        tmp4_mean = tl.where(r0_mask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(r0_mask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(r0_mask & xmask, tmp4_weight_next, tmp4_weight)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp7, tmp8, tmp9 = triton_helpers.welford(tmp4_mean, tmp4_m2, tmp4_weight, 1)
    tmp4 = tmp7[:, None]
    tmp5 = tmp8[:, None]
    tmp6 = tmp9[:, None]
    tmp10 = 16 + 4*ks0 + 4*ks1 + ks0*ks1
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp2 / tmp11
    tmp13 = 1.0
    tmp14 = tmp11 - tmp13
    tmp15 = 0.0
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tmp17 = tmp5 / tmp16
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp12, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp17, xmask)




# kernel path: /tmp/torchinductor_sahanp/th/cthizqnhk7qckq4qoloorp7dymijkxlg4szaetvka4sb6o3gmef2.py
# Topologically Sorted Source Nodes: [target], Original ATen: [aten.zeros_like]
# Source node to ATen node mapping:
#   target => full_default
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 3], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_zeros_like_3(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)







def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    s1 = arg0_1
    s2 = arg1_1
    assert_size_stride(arg2_1, (1, 3, s1, s2), (3*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        ps0 = 4 + s2
        buf0 = empty_strided_cuda((1, 3, 1, 1), (3, 1, 3, 3), torch.float32)
        buf1 = empty_strided_cuda((1, 3, 1, 1), (3, 1, 3, 3), torch.float32)
        buf3 = empty_strided_cuda((1, 3, 1, 1), (3, 1, 3, 3), torch.float32)
        buf4 = empty_strided_cuda((1, 3, 1, 1), (3, 1, 3, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.constant_pad_nd, aten.view, aten._native_batch_norm_legit]
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_view_0_r0_numel = 16 + 4*s1 + 4*s2 + s1*s2
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_view_0[grid(3)](arg2_1, buf0, buf1, buf3, buf4, 68, 64, 64, 3, 4624, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        ps1 = 4 + s1
        ps2 = 16 + 4*s1 + 4*s2 + s1*s2
        buf6 = empty_strided_cuda((1, 3, 4 + s1, 4 + s2), (48 + 12*s1 + 12*s2 + 3*s1*s2, 16 + 4*s1 + 4*s2 + s1*s2, 4 + s2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.constant_pad_nd, aten.view, aten._native_batch_norm_legit]
        triton_poi_fused__native_batch_norm_legit_constant_pad_nd_view_1_xnumel = 48 + 12*s1 + 12*s2 + 3*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_constant_pad_nd_view_1[grid(triton_poi_fused__native_batch_norm_legit_constant_pad_nd_view_1_xnumel)](arg2_1, buf0, buf1, buf3, buf4, buf6, 68, 68, 64, 64, 4624, 13872, XBLOCK=256, num_warps=4, num_stages=1)
        del arg2_1
        del buf0
        buf7 = buf4; del buf4  # reuse
        buf11 = buf3; del buf3  # reuse
        buf8 = buf7; del buf7  # reuse
        buf13 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [mean, var], Original ATen: [aten.mean, aten.var]
        triton_red_fused_mean_var_2_r0_numel = 16 + 4*s1 + 4*s2 + s1*s2
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_var_2[grid(3)](buf8, buf13, buf6, 64, 64, 3, 4624, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del buf6
        buf9 = reinterpret_tensor(buf1, (1, 3), (3, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [target], Original ATen: [aten.zeros_like]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_like_3[grid(3)](buf9, 3, XBLOCK=4, num_warps=1, num_stages=1)
    return (reinterpret_tensor(buf8, (1, 3), (3, 1), 0), buf9, reinterpret_tensor(buf13, (1, 3), (3, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 64
    arg1_1 = 64
    arg2_1 = rand_strided((1, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
