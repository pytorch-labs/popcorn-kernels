# AOT ID: ['9_forward']
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


# kernel path: /tmp/torchinductor_sahanp/kx/ckxm2cv4tujsd7rhe53hi6b3zialdmztweiakzmizqnhcbhy66vo.py
# Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.convolution, aten.mish]
# Source node to ATen node mapping:
#   x => convolution
#   x_1 => exp, gt, log1p, mul, tanh, where
# Graph fragment:
#   %convolution : [num_users=5] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_1, %primals_2, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%convolution,), kwargs = {})
#   %log1p : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp,), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution, 20), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %convolution, %log1p), kwargs = {})
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%where,), kwargs = {})
#   %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution, %tanh), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_mish_0(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 4096
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 20.0
    tmp4 = tmp2 > tmp3
    tmp5 = tl_math.exp(tmp2)
    tmp6 = libdevice.log1p(tmp5)
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp8 = libdevice.tanh(tmp7)
    tmp9 = tmp2 * tmp8
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp9, None)




# kernel path: /tmp/torchinductor_sahanp/xy/cxyoympeif2tb6vd2ssn4ha5e7gqvl7vju5rwvs6zkhhro4eek5t.py
# Topologically Sorted Source Nodes: [x_2, x_3], Original ATen: [aten.adaptive_max_pool2d, aten._native_batch_norm_legit_functional, aten.mean, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   x_2 => _low_memory_max_pool2d_with_offsets, getitem_1
#   x_3 => add, add_3, mean, mean_1, mul_1, mul_7, rsqrt, sub, var_mean
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%mul, [4, 4], [4, 4], [0, 0], [1, 1], False), kwargs = {})
#   %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%getitem, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%getitem, %getitem_3), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_1), kwargs = {})
#   %add_3 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %unsqueeze_3), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_2, [0]), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_4, [0]), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%getitem, %unsqueeze_6), kwargs = {})
#   %copy_ : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_4, %mean), kwargs = {})
#   %copy__1 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_5, %mean_1), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_adaptive_max_pool2d_mean_native_batch_norm_backward_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr4, out_ptr5, out_ptr6, out_ptr8, out_ptr10, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 16
    r0_numel = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp78_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp78_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp78_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = (r0_index % 16)
        r0_2 = r0_index // 16
        r0_3 = r0_index
        tmp0 = tl.load(in_ptr0 + (4*r0_1 + 256*r0_2 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr0 + (1 + 4*r0_1 + 256*r0_2 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr0 + (2 + 4*r0_1 + 256*r0_2 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr0 + (3 + 4*r0_1 + 256*r0_2 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr0 + (64 + 4*r0_1 + 256*r0_2 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr0 + (65 + 4*r0_1 + 256*r0_2 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.load(in_ptr0 + (66 + 4*r0_1 + 256*r0_2 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr0 + (67 + 4*r0_1 + 256*r0_2 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.load(in_ptr0 + (128 + 4*r0_1 + 256*r0_2 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr0 + (129 + 4*r0_1 + 256*r0_2 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp19 = tl.load(in_ptr0 + (130 + 4*r0_1 + 256*r0_2 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tl.load(in_ptr0 + (131 + 4*r0_1 + 256*r0_2 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.load(in_ptr0 + (192 + 4*r0_1 + 256*r0_2 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp25 = tl.load(in_ptr0 + (193 + 4*r0_1 + 256*r0_2 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp27 = tl.load(in_ptr0 + (194 + 4*r0_1 + 256*r0_2 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp29 = tl.load(in_ptr0 + (195 + 4*r0_1 + 256*r0_2 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = triton_helpers.maximum(tmp1, tmp0)
        tmp4 = triton_helpers.maximum(tmp3, tmp2)
        tmp6 = triton_helpers.maximum(tmp5, tmp4)
        tmp8 = triton_helpers.maximum(tmp7, tmp6)
        tmp10 = triton_helpers.maximum(tmp9, tmp8)
        tmp12 = triton_helpers.maximum(tmp11, tmp10)
        tmp14 = triton_helpers.maximum(tmp13, tmp12)
        tmp16 = triton_helpers.maximum(tmp15, tmp14)
        tmp18 = triton_helpers.maximum(tmp17, tmp16)
        tmp20 = triton_helpers.maximum(tmp19, tmp18)
        tmp22 = triton_helpers.maximum(tmp21, tmp20)
        tmp24 = triton_helpers.maximum(tmp23, tmp22)
        tmp26 = triton_helpers.maximum(tmp25, tmp24)
        tmp28 = triton_helpers.maximum(tmp27, tmp26)
        tmp30 = triton_helpers.maximum(tmp29, tmp28)
        tmp31 = tmp1 > tmp0
        tmp32 = tl.full([1, 1], 1, tl.int8)
        tmp33 = tl.full([1, 1], 0, tl.int8)
        tmp34 = tl.where(tmp31, tmp32, tmp33)
        tmp35 = tmp3 > tmp2
        tmp36 = tl.full([1, 1], 2, tl.int8)
        tmp37 = tl.where(tmp35, tmp36, tmp34)
        tmp38 = tmp5 > tmp4
        tmp39 = tl.full([1, 1], 3, tl.int8)
        tmp40 = tl.where(tmp38, tmp39, tmp37)
        tmp41 = tmp7 > tmp6
        tmp42 = tl.full([1, 1], 4, tl.int8)
        tmp43 = tl.where(tmp41, tmp42, tmp40)
        tmp44 = tmp9 > tmp8
        tmp45 = tl.full([1, 1], 5, tl.int8)
        tmp46 = tl.where(tmp44, tmp45, tmp43)
        tmp47 = tmp11 > tmp10
        tmp48 = tl.full([1, 1], 6, tl.int8)
        tmp49 = tl.where(tmp47, tmp48, tmp46)
        tmp50 = tmp13 > tmp12
        tmp51 = tl.full([1, 1], 7, tl.int8)
        tmp52 = tl.where(tmp50, tmp51, tmp49)
        tmp53 = tmp15 > tmp14
        tmp54 = tl.full([1, 1], 8, tl.int8)
        tmp55 = tl.where(tmp53, tmp54, tmp52)
        tmp56 = tmp17 > tmp16
        tmp57 = tl.full([1, 1], 9, tl.int8)
        tmp58 = tl.where(tmp56, tmp57, tmp55)
        tmp59 = tmp19 > tmp18
        tmp60 = tl.full([1, 1], 10, tl.int8)
        tmp61 = tl.where(tmp59, tmp60, tmp58)
        tmp62 = tmp21 > tmp20
        tmp63 = tl.full([1, 1], 11, tl.int8)
        tmp64 = tl.where(tmp62, tmp63, tmp61)
        tmp65 = tmp23 > tmp22
        tmp66 = tl.full([1, 1], 12, tl.int8)
        tmp67 = tl.where(tmp65, tmp66, tmp64)
        tmp68 = tmp25 > tmp24
        tmp69 = tl.full([1, 1], 13, tl.int8)
        tmp70 = tl.where(tmp68, tmp69, tmp67)
        tmp71 = tmp27 > tmp26
        tmp72 = tl.full([1, 1], 14, tl.int8)
        tmp73 = tl.where(tmp71, tmp72, tmp70)
        tmp74 = tmp29 > tmp28
        tmp75 = tl.full([1, 1], 15, tl.int8)
        tmp76 = tl.where(tmp74, tmp75, tmp73)
        tmp77 = tl.broadcast_to(tmp30, [XBLOCK, R0_BLOCK])
        tmp78_mean_next, tmp78_m2_next, tmp78_weight_next = triton_helpers.welford_reduce(
            tmp77, tmp78_mean, tmp78_m2, tmp78_weight, roffset == 0
        )
        tmp78_mean = tl.where(r0_mask & xmask, tmp78_mean_next, tmp78_mean)
        tmp78_m2 = tl.where(r0_mask & xmask, tmp78_m2_next, tmp78_m2)
        tmp78_weight = tl.where(r0_mask & xmask, tmp78_weight_next, tmp78_weight)
        tl.store(out_ptr0 + (r0_3 + 256*x0), tmp30, r0_mask & xmask)
        tl.store(out_ptr1 + (r0_3 + 256*x0), tmp76, r0_mask & xmask)
    tmp81, tmp82, tmp83 = triton_helpers.welford(tmp78_mean, tmp78_m2, tmp78_weight, 1)
    tmp78 = tmp81[:, None]
    tmp79 = tmp82[:, None]
    tmp80 = tmp83[:, None]
    tmp92 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp94 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        tmp84 = tl.load(out_ptr0 + (r0_3 + 256*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp85 = tmp84 - tmp78
        tmp86 = 256.0
        tmp87 = tmp79 / tmp86
        tmp88 = 1e-05
        tmp89 = tmp87 + tmp88
        tmp90 = libdevice.rsqrt(tmp89)
        tmp91 = tmp85 * tmp90
        tmp93 = tmp91 * tmp92
        tmp95 = tmp93 + tmp94
        tl.store(out_ptr4 + (r0_3 + 256*x0), tmp95, r0_mask & xmask)
        tl.store(out_ptr5 + (r0_3 + 256*x0), tmp85, r0_mask & xmask)
    tmp105 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp112 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp96 = 256.0
    tmp97 = tmp79 / tmp96
    tmp98 = 1e-05
    tmp99 = tmp97 + tmp98
    tmp100 = libdevice.rsqrt(tmp99)
    tmp101 = 1.003921568627451
    tmp102 = tmp97 * tmp101
    tmp103 = 0.1
    tmp104 = tmp102 * tmp103
    tmp106 = 0.9
    tmp107 = tmp105 * tmp106
    tmp108 = tmp104 + tmp107
    tmp109 = 1.0
    tmp110 = tmp108 / tmp109
    tmp111 = tmp78 * tmp103
    tmp113 = tmp112 * tmp106
    tmp114 = tmp111 + tmp113
    tmp115 = tmp114 / tmp109
    tl.store(out_ptr6 + (x0), tmp100, xmask)
    tl.store(out_ptr8 + (x0), tmp110, xmask)
    tl.store(out_ptr10 + (x0), tmp115, xmask)




# kernel path: /tmp/torchinductor_sahanp/cu/ccuhnictmqy3ktnfumcqxezuddmsjxz72i37k2rldkwfrveqdyav.py
# Topologically Sorted Source Nodes: [x_4, x_5], Original ATen: [aten.convolution, aten.mish]
# Source node to ATen node mapping:
#   x_4 => convolution_1
#   x_5 => exp_1, gt_1, log1p_1, mul_8, tanh_1, where_1
# Graph fragment:
#   %convolution_1 : [num_users=5] = call_function[target=torch.ops.aten.convolution.default](args = (%add_3, %primals_8, %primals_9, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%convolution_1,), kwargs = {})
#   %log1p_1 : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_1,), kwargs = {})
#   %gt_1 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_1, 20), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %convolution_1, %log1p_1), kwargs = {})
#   %tanh_1 : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%where_1,), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_1, %tanh_1), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_mish_2(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 256
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 20.0
    tmp4 = tmp2 > tmp3
    tmp5 = tl_math.exp(tmp2)
    tmp6 = libdevice.log1p(tmp5)
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp8 = libdevice.tanh(tmp7)
    tmp9 = tmp2 * tmp8
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp9, None)




# kernel path: /tmp/torchinductor_sahanp/6l/c6lqye7prvzzvctzqxb3svj2nzd4rloettmgzc2c6ijewqlobnni.py
# Topologically Sorted Source Nodes: [x_8], Original ATen: [aten.mish]
# Source node to ATen node mapping:
#   x_8 => exp_2, gt_2, log1p_2, mul_9, tanh_2, where_2
# Graph fragment:
#   %exp_2 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%addmm,), kwargs = {})
#   %log1p_2 : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_2,), kwargs = {})
#   %gt_2 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%addmm, 20), kwargs = {})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %addmm, %log1p_2), kwargs = {})
#   %tanh_2 : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%where_2,), kwargs = {})
#   %mul_9 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%addmm, %tanh_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_mish_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 20.0
    tmp2 = tmp0 > tmp1
    tmp3 = tl_math.exp(tmp0)
    tmp4 = libdevice.log1p(tmp3)
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp6 = libdevice.tanh(tmp5)
    tmp7 = tmp0 * tmp6
    tl.store(out_ptr0 + (x0), tmp7, xmask)




# kernel path: /tmp/torchinductor_sahanp/o6/co6v7pqtvslnkf74wtozvifuqgfn5yg5ogxds2g3hy5gf4a6brcd.py
# Topologically Sorted Source Nodes: [mean], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   mean => mean_2
# Graph fragment:
#   %mean_2 : [num_users=2] = call_function[target=torch.ops.aten.mean.default](args = (%primals_14,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_mean_4(in_out_ptr0, in_ptr0, xnumel, r0_numel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), None)
    tmp1 = tl.broadcast_to(tmp0, [R0_BLOCK])
    tmp3 = triton_helpers.promote_to_tensor(tl.sum(tmp1, 0))
    tmp4 = 256.0
    tmp5 = tmp3 / tmp4
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([1], 0, tl.int32)), tmp5, None)




# kernel path: /tmp/torchinductor_sahanp/gm/cgmx7erm2orqstupqnre6kcn6zm6tcdqo7oelquiabbsqaxvlqjl.py
# Topologically Sorted Source Nodes: [mean_1], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   mean_1 => mean_3
# Graph fragment:
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%primals_15,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_mean_5(in_ptr0, out_ptr0, xnumel, r0_numel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), None)
    tmp1 = tl.broadcast_to(tmp0, [R0_BLOCK])
    tmp3 = triton_helpers.promote_to_tensor(tl.sum(tmp1, 0))
    tl.store(out_ptr0 + (tl.full([1], 0, tl.int32)), tmp3, None)




# kernel path: /tmp/torchinductor_sahanp/cq/ccqshdvfnlp2mbzpl6aowbcu37j667enyphnfz2qs4fmw2lieruz.py
# Topologically Sorted Source Nodes: [mul, mean_1, x_10], Original ATen: [aten.mul, aten.mean, aten.add]
# Source node to ATen node mapping:
#   mean_1 => mean_3
#   mul => mul_10
#   x_10 => add_4
# Graph fragment:
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%addmm_1, %mean_2), kwargs = {})
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%primals_15,), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, %mean_3), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_mean_mul_6(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_ptr2 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp3 = tmp0 * tmp2
    tmp6 = 256.0
    tmp7 = tmp5 / tmp6
    tmp8 = tmp3 + tmp7
    tl.store(out_ptr0 + (x0), tmp8, xmask)







def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15 = args
    args.clear()
    assert_size_stride(primals_1, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (16, ), (1, ))
    assert_size_stride(primals_3, (1, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_4, (16, ), (1, ))
    assert_size_stride(primals_5, (16, ), (1, ))
    assert_size_stride(primals_6, (16, ), (1, ))
    assert_size_stride(primals_7, (16, ), (1, ))
    assert_size_stride(primals_8, (32, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_9, (32, ), (1, ))
    assert_size_stride(primals_10, (128, 8192), (8192, 1))
    assert_size_stride(primals_11, (128, ), (1, ))
    assert_size_stride(primals_12, (10, 128), (128, 1))
    assert_size_stride(primals_13, (10, ), (1, ))
    assert_size_stride(primals_14, (16, 16), (16, 1))
    assert_size_stride(primals_15, (16, 16), (16, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (1, 16, 64, 64), (65536, 4096, 64, 1))
        buf1 = buf0; del buf0  # reuse
        buf2 = empty_strided_cuda((1, 16, 64, 64), (65536, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.convolution, aten.mish]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_mish_0[grid(65536)](buf1, primals_2, buf2, 65536, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_2
        buf3 = empty_strided_cuda((1, 16, 16, 16), (4096, 256, 16, 1), torch.float32)
        buf4 = empty_strided_cuda((1, 16, 16, 16), (4096, 256, 16, 1), torch.int8)
        buf9 = empty_strided_cuda((1, 16, 16, 16), (4096, 256, 16, 1), torch.float32)
        buf20 = empty_strided_cuda((1, 16, 16, 16), (4096, 256, 16, 1), torch.float32)
        buf8 = empty_strided_cuda((1, 16, 1, 1), (16, 1, 16, 16), torch.float32)
        # Topologically Sorted Source Nodes: [x_2, x_3], Original ATen: [aten.adaptive_max_pool2d, aten._native_batch_norm_legit_functional, aten.mean, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_adaptive_max_pool2d_mean_native_batch_norm_backward_1[grid(16)](buf2, primals_6, primals_7, primals_5, primals_4, buf3, buf4, buf9, buf20, buf8, primals_5, primals_4, 16, 256, XBLOCK=2, R0_BLOCK=256, num_warps=4, num_stages=1)
        del buf3
        del primals_4
        del primals_5
        del primals_7
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, primals_8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (1, 32, 16, 16), (8192, 256, 16, 1))
        buf11 = buf10; del buf10  # reuse
        buf12 = empty_strided_cuda((1, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_4, x_5], Original ATen: [aten.convolution, aten.mish]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_mish_2[grid(8192)](buf11, primals_9, buf12, 8192, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_9
        buf13 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, reinterpret_tensor(buf12, (1, 8192), (0, 1), 0), reinterpret_tensor(primals_10, (8192, 128), (1, 8192), 0), alpha=1, beta=1, out=buf13)
        del primals_11
        buf14 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten.mish]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mish_3[grid(128)](buf13, buf14, 128, XBLOCK=128, num_warps=4, num_stages=1)
        buf15 = empty_strided_cuda((1, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_13, buf14, reinterpret_tensor(primals_12, (128, 10), (1, 128), 0), alpha=1, beta=1, out=buf15)
        del primals_13
        buf16 = empty_strided_cuda((), (), torch.float32)
        buf17 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [mean], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_4[grid(1)](buf17, primals_14, 1, 256, num_warps=2, num_stages=1)
        del primals_14
        buf18 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [mean_1], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_5[grid(1)](primals_15, buf18, 1, 256, num_warps=2, num_stages=1)
        del primals_15
        buf19 = empty_strided_cuda((1, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul, mean_1, x_10], Original ATen: [aten.mul, aten.mean, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mean_mul_6[grid(10)](buf15, buf17, buf18, buf19, 10, XBLOCK=16, num_warps=1, num_stages=1)
        del buf18
    return (buf19, primals_1, primals_3, primals_6, primals_8, buf1, buf2, buf4, reinterpret_tensor(buf8, (16, ), (1, ), 0), buf9, buf11, reinterpret_tensor(buf12, (1, 8192), (8192, 1), 0), buf13, buf14, buf15, buf17, primals_12, primals_10, buf20, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((1, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((32, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((10, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
