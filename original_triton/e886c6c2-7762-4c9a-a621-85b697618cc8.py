# AOT ID: ['6_forward']
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


# kernel path: /tmp/torchinductor_sahanp/co/ccotdhzmo3y3tllrth4aircmeia2xeadc4ruije6rrzx2vdyl3r4.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   x_2 => avg_pool2d
# Graph fragment:
#   %avg_pool2d : [num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%unsqueeze, [1, 3], [1, 2]), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_avg_pool2d_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 58953
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 19651)
    x1 = xindex // 19651
    tmp0 = (-1) + (x0 // 578)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + (((x0 // 17) % 34))
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = (-1) + (((2*x0) % 34))
    tmp9 = tmp8 >= tmp1
    tmp10 = tmp8 < tmp3
    tmp11 = tmp2 & tmp4
    tmp12 = tmp11 & tmp6
    tmp13 = tmp12 & tmp7
    tmp14 = tmp13 & tmp9
    tmp15 = tmp14 & tmp10
    tmp16 = tl.load(in_ptr0 + ((-1057) + 32*(((x0 // 17) % 34)) + 1024*(x0 // 578) + 32768*x1 + (((2*x0) % 34))), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 * tmp16
    tmp18 = (-1) + ((1 + 2*x0) // 1156)
    tmp19 = tmp18 >= tmp1
    tmp20 = tmp18 < tmp3
    tmp21 = (-1) + ((((1 + 2*x0) // 34) % 34))
    tmp22 = tmp21 >= tmp1
    tmp23 = tmp21 < tmp3
    tmp24 = (-1) + (((1 + 2*x0) % 34))
    tmp25 = tmp24 >= tmp1
    tmp26 = tmp24 < tmp3
    tmp27 = tmp19 & tmp20
    tmp28 = tmp27 & tmp22
    tmp29 = tmp28 & tmp23
    tmp30 = tmp29 & tmp25
    tmp31 = tmp30 & tmp26
    tmp32 = tl.load(in_ptr0 + ((-1057) + 32*((((1 + 2*x0) // 34) % 34)) + 1024*((1 + 2*x0) // 1156) + 32768*x1 + (((1 + 2*x0) % 34))), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp32 * tmp32
    tmp34 = tmp33 + tmp17
    tmp35 = (-1) + ((1 + x0) // 578)
    tmp36 = tmp35 >= tmp1
    tmp37 = tmp35 < tmp3
    tmp38 = (-1) + ((((1 + x0) // 17) % 34))
    tmp39 = tmp38 >= tmp1
    tmp40 = tmp38 < tmp3
    tmp41 = (-1) + (((2 + 2*x0) % 34))
    tmp42 = tmp41 >= tmp1
    tmp43 = tmp41 < tmp3
    tmp44 = tmp36 & tmp37
    tmp45 = tmp44 & tmp39
    tmp46 = tmp45 & tmp40
    tmp47 = tmp46 & tmp42
    tmp48 = tmp47 & tmp43
    tmp49 = tl.load(in_ptr0 + ((-1057) + 32*((((1 + x0) // 17) % 34)) + 1024*((1 + x0) // 578) + 32768*x1 + (((2 + 2*x0) % 34))), tmp48 & xmask, eviction_policy='evict_last', other=0.0)
    tmp50 = tmp49 * tmp49
    tmp51 = tmp50 + tmp34
    tmp52 = 0.3333333333333333
    tmp53 = tmp51 * tmp52
    tl.store(out_ptr0 + (x0 + 19680*x1), tmp53, xmask)




# kernel path: /tmp/torchinductor_sahanp/rf/crfqkoxrdcjzbq365daw3swph3cmrdhfxoiqxnj6cs4dmuojcfbm.py
# Topologically Sorted Source Nodes: [x_2, x_3], Original ATen: [aten.sign, aten.abs, aten.relu, aten.mul, aten.pow, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   x_2 => abs_1, mul, mul_1, pow_2, relu, sign
#   x_3 => var_mean
# Graph fragment:
#   %sign : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%squeeze,), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%squeeze,), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%abs_1,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign, %relu), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, 3), kwargs = {})
#   %pow_2 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_1, 0.5), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%pow_2, [0, 2]), kwargs = {correction: 0, keepdim: True})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_abs_mul_pow_relu_sign_1(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 9
    r0_numel = 6551
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 3)
    x1 = xindex // 3
    tmp28_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp28_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp28_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2 + 6551*x0
        tmp1 = tl.full([1, 1], 19651, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (r0_2 + 6551*x0 + 19680*x1), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full([1, 1], 0, tl.int32)
        tmp5 = tmp4 < tmp3
        tmp6 = tmp5.to(tl.int8)
        tmp7 = tmp3 < tmp4
        tmp8 = tmp7.to(tl.int8)
        tmp9 = tmp6 - tmp8
        tmp10 = tmp9.to(tmp3.dtype)
        tmp11 = tl_math.abs(tmp3)
        tmp12 = triton_helpers.maximum(tmp4, tmp11)
        tmp13 = tmp10 * tmp12
        tmp14 = 3.0
        tmp15 = tmp13 * tmp14
        tmp16 = libdevice.sqrt(tmp15)
        tmp17 = tl.full(tmp16.shape, 0, tmp16.dtype)
        tmp18 = tl.where(tmp2, tmp16, tmp17)
        tmp19 = 0.0
        tmp20 = tl.full(tmp19.shape, 0, tmp19.dtype)
        tmp21 = tl.where(tmp2, tmp19, tmp20)
        tmp22 = 1.0
        tmp23 = tl.full(tmp22.shape, 0, tmp22.dtype)
        tmp24 = tl.where(tmp2, tmp22, tmp23)
        tmp25 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
        tmp26 = tl.broadcast_to(tmp21, [XBLOCK, R0_BLOCK])
        tmp27 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
        tmp28_mean_next, tmp28_m2_next, tmp28_weight_next = triton_helpers.welford_combine(
            tmp28_mean, tmp28_m2, tmp28_weight,
            tmp25, tmp26, tmp27
        )
        tmp28_mean = tl.where(r0_mask & xmask, tmp28_mean_next, tmp28_mean)
        tmp28_m2 = tl.where(r0_mask & xmask, tmp28_m2_next, tmp28_m2)
        tmp28_weight = tl.where(r0_mask & xmask, tmp28_weight_next, tmp28_weight)
    tmp31, tmp32, tmp33 = triton_helpers.welford(tmp28_mean, tmp28_m2, tmp28_weight, 1)
    tmp28 = tmp31[:, None]
    tmp29 = tmp32[:, None]
    tmp30 = tmp33[:, None]
    tl.store(out_ptr0 + (x3), tmp28, xmask)
    tl.store(out_ptr1 + (x3), tmp29, xmask)
    tl.store(out_ptr2 + (x3), tmp30, xmask)




# kernel path: /tmp/torchinductor_sahanp/7g/c7gvqpssnsvbxvq7m2by6df45cyzk2rfsapnabfofsthietsrvrh.py
# Topologically Sorted Source Nodes: [x_2, x_3], Original ATen: [aten.sign, aten.abs, aten.relu, aten.mul, aten.pow, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   x_2 => abs_1, mul, mul_1, pow_2, relu, sign
#   x_3 => add_1, add_2, add_3, mul_3, mul_4, mul_5, mul_6, mul_7, rsqrt, var_mean
# Graph fragment:
#   %sign : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%squeeze,), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%squeeze,), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%abs_1,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign, %relu), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, 3), kwargs = {})
#   %pow_2 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_1, 0.5), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%pow_2, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_1, 0.1), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_3, 0.9), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %mul_4), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_3, 1.0000508905852417), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, 0.1), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_4, 0.9), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %mul_7), kwargs = {})
#   %copy__1 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_3, %add_2), kwargs = {})
#   %copy__2 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_4, %add_3), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_abs_mul_pow_relu_sign_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, out_ptr4, out_ptr6, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 3
    r0_numel = 3
    R0_BLOCK: tl.constexpr = 4
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 3*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 3*x0), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r0_1 + 3*x0), r0_mask & xmask, other=0.0)
    tmp25 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp3, 0)
    tmp8 = tl.where(r0_mask & xmask, tmp4, 0)
    tmp9 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 19651.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = 1.0000508905852417
    tmp22 = tmp17 * tmp21
    tmp23 = 0.1
    tmp24 = tmp22 * tmp23
    tmp26 = 0.9
    tmp27 = tmp25 * tmp26
    tmp28 = tmp24 + tmp27
    tmp29 = tmp13 * tmp23
    tmp31 = tmp30 * tmp26
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp28, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)




# kernel path: /tmp/torchinductor_sahanp/7b/c7bskkwuenoizntquir7yoifoftlmd222s7ezgypu3aejmh4lysy.py
# Topologically Sorted Source Nodes: [x_2, x_3, x_4, x_5, x_6, x_7], Original ATen: [aten.sign, aten.abs, aten.relu, aten.mul, aten.pow, aten._native_batch_norm_legit_functional, aten.mish, aten.softplus, aten.bernoulli, aten._to_copy, aten.add]
# Source node to ATen node mapping:
#   x_2 => abs_1, mul, mul_1, pow_2, relu, sign
#   x_3 => add_4, mul_2, mul_8, sub
#   x_4 => exp, gt, log1p, mul_9, tanh, where
#   x_5 => div, exp_1, gt_1, log1p_1, mul_10, where_1
#   x_6 => add_5, add_6, add_7, convert_element_type, inductor_lookup_seed_default, inductor_random_default_1, lt, mul_11, mul_12, mul_13
#   x_7 => pow_3
# Graph fragment:
#   %sign : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%squeeze,), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%squeeze,), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%abs_1,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign, %relu), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, 3), kwargs = {})
#   %pow_2 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_1, 0.5), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_2, %getitem_1), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %unsqueeze_1), kwargs = {})
#   %add_4 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_2), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%add_4,), kwargs = {})
#   %log1p : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp,), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_4, 20), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %add_4, %log1p), kwargs = {})
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%where,), kwargs = {})
#   %mul_9 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_4, %tanh), kwargs = {})
#   %mul_10 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, 1.0), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_10,), kwargs = {})
#   %log1p_1 : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_1,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%log1p_1, 1.0), kwargs = {})
#   %gt_1 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%mul_10, 20.0), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %mul_9, %div), kwargs = {})
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 3, 19651], %inductor_lookup_seed_default, rand), kwargs = {})
#   %lt : [num_users=2] = call_function[target=torch.ops.aten.lt.Scalar](args = (%inductor_random_default_1, 0.5), kwargs = {})
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt, torch.float32), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%convert_element_type, -1), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%add_5, 1.558387861036063), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_11, 0.7791939305180315), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%convert_element_type, 0.8864048946659319), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_1, %mul_12), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_13, %add_6), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_7, 2.0), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional__to_copy_abs_add_bernoulli_mish_mul_pow_relu_sign_softplus_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 58953
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 19651)
    x1 = xindex // 19651
    tmp5 = tl.load(in_ptr1 + (x0 + 19680*x1), xmask)
    tmp19 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x2
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.5
    tmp4 = tmp2 < tmp3
    tmp6 = tl.full([1], 0, tl.int32)
    tmp7 = tmp6 < tmp5
    tmp8 = tmp7.to(tl.int8)
    tmp9 = tmp5 < tmp6
    tmp10 = tmp9.to(tl.int8)
    tmp11 = tmp8 - tmp10
    tmp12 = tmp11.to(tmp5.dtype)
    tmp13 = tl_math.abs(tmp5)
    tmp14 = triton_helpers.maximum(tmp6, tmp13)
    tmp15 = tmp12 * tmp14
    tmp16 = 3.0
    tmp17 = tmp15 * tmp16
    tmp18 = libdevice.sqrt(tmp17)
    tmp20 = tmp18 - tmp19
    tmp22 = tmp20 * tmp21
    tmp24 = tmp22 * tmp23
    tmp26 = tmp24 + tmp25
    tmp27 = 20.0
    tmp28 = tmp26 > tmp27
    tmp29 = tl_math.exp(tmp26)
    tmp30 = libdevice.log1p(tmp29)
    tmp31 = tl.where(tmp28, tmp26, tmp30)
    tmp32 = libdevice.tanh(tmp31)
    tmp33 = tmp26 * tmp32
    tmp34 = 1.0
    tmp35 = tmp33 * tmp34
    tmp36 = tmp35 > tmp27
    tmp37 = tl_math.exp(tmp35)
    tmp38 = libdevice.log1p(tmp37)
    tmp39 = tmp38 * tmp34
    tmp40 = tl.where(tmp36, tmp33, tmp39)
    tmp41 = tmp4.to(tl.float32)
    tmp42 = 0.8864048946659319
    tmp43 = tmp41 * tmp42
    tmp44 = tmp40 * tmp43
    tmp45 = -1.0
    tmp46 = tmp41 + tmp45
    tmp47 = 1.558387861036063
    tmp48 = tmp46 * tmp47
    tmp49 = 0.7791939305180315
    tmp50 = tmp48 + tmp49
    tmp51 = tmp44 + tmp50
    tmp52 = tmp51 * tmp51
    tl.store(out_ptr1 + (x0 + 19712*x1), tmp4, xmask)
    tl.store(in_out_ptr0 + (x0 + 19680*x1), tmp52, xmask)




# kernel path: /tmp/torchinductor_sahanp/7u/c7umivkpxlxc6m7kigxbkpodbyk6466d2s62snh6eqrhfzlp6cjg.py
# Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   x_7 => avg_pool2d_1
# Graph fragment:
#   %avg_pool2d_1 : [num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%unsqueeze_3, [1, 3], [1, 2]), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_avg_pool2d_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 29475
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 9825)
    x1 = xindex // 9825
    tmp0 = tl.load(in_ptr0 + (2*x0 + 19680*x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 19680*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 2*x0 + 19680*x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp5 = 0.3333333333333333
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr0 + (x0 + 9856*x1), tmp6, xmask)




# kernel path: /tmp/torchinductor_sahanp/tc/ctcnory5si27ixbgnvgm5yhkfp6ihzosnmct3v6z4m4ll2wfbz25.py
# Topologically Sorted Source Nodes: [x_7, x_8], Original ATen: [aten.sign, aten.abs, aten.relu, aten.mul, aten.pow, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   x_7 => abs_2, mul_14, mul_15, pow_4, relu_1, sign_1
#   x_8 => var_mean_1
# Graph fragment:
#   %sign_1 : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%squeeze_4,), kwargs = {})
#   %abs_2 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%squeeze_4,), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%abs_2,), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign_1, %relu_1), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_14, 3), kwargs = {})
#   %pow_4 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_15, 0.5), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%pow_4, [0, 2]), kwargs = {correction: 0, keepdim: True})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_abs_mul_pow_relu_sign_5(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 6
    r0_numel = 4913
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 2)
    x1 = xindex // 2
    tmp28_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp28_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp28_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2 + 4913*x0
        tmp1 = tl.full([1, 1], 9825, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (r0_2 + 4913*x0 + 9856*x1), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full([1, 1], 0, tl.int32)
        tmp5 = tmp4 < tmp3
        tmp6 = tmp5.to(tl.int8)
        tmp7 = tmp3 < tmp4
        tmp8 = tmp7.to(tl.int8)
        tmp9 = tmp6 - tmp8
        tmp10 = tmp9.to(tmp3.dtype)
        tmp11 = tl_math.abs(tmp3)
        tmp12 = triton_helpers.maximum(tmp4, tmp11)
        tmp13 = tmp10 * tmp12
        tmp14 = 3.0
        tmp15 = tmp13 * tmp14
        tmp16 = libdevice.sqrt(tmp15)
        tmp17 = tl.full(tmp16.shape, 0, tmp16.dtype)
        tmp18 = tl.where(tmp2, tmp16, tmp17)
        tmp19 = 0.0
        tmp20 = tl.full(tmp19.shape, 0, tmp19.dtype)
        tmp21 = tl.where(tmp2, tmp19, tmp20)
        tmp22 = 1.0
        tmp23 = tl.full(tmp22.shape, 0, tmp22.dtype)
        tmp24 = tl.where(tmp2, tmp22, tmp23)
        tmp25 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
        tmp26 = tl.broadcast_to(tmp21, [XBLOCK, R0_BLOCK])
        tmp27 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
        tmp28_mean_next, tmp28_m2_next, tmp28_weight_next = triton_helpers.welford_combine(
            tmp28_mean, tmp28_m2, tmp28_weight,
            tmp25, tmp26, tmp27
        )
        tmp28_mean = tl.where(r0_mask & xmask, tmp28_mean_next, tmp28_mean)
        tmp28_m2 = tl.where(r0_mask & xmask, tmp28_m2_next, tmp28_m2)
        tmp28_weight = tl.where(r0_mask & xmask, tmp28_weight_next, tmp28_weight)
    tmp31, tmp32, tmp33 = triton_helpers.welford(tmp28_mean, tmp28_m2, tmp28_weight, 1)
    tmp28 = tmp31[:, None]
    tmp29 = tmp32[:, None]
    tmp30 = tmp33[:, None]
    tl.store(out_ptr0 + (x3), tmp28, xmask)
    tl.store(out_ptr1 + (x3), tmp29, xmask)
    tl.store(out_ptr2 + (x3), tmp30, xmask)




# kernel path: /tmp/torchinductor_sahanp/ga/cga6gds434cvp655b3urb5kx6jiealllsnhqszq2riromj4s2orm.py
# Topologically Sorted Source Nodes: [x_7, x_8], Original ATen: [aten.sign, aten.abs, aten.relu, aten.mul, aten.pow, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   x_7 => abs_2, mul_14, mul_15, pow_4, relu_1, sign_1
#   x_8 => add_10, add_11, add_9, mul_17, mul_18, mul_19, mul_20, mul_21, rsqrt_1, var_mean_1
# Graph fragment:
#   %sign_1 : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%squeeze_4,), kwargs = {})
#   %abs_2 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%squeeze_4,), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%abs_2,), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign_1, %relu_1), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_14, 3), kwargs = {})
#   %pow_4 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_15, 0.5), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%pow_4, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_5, 0.1), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_8, 0.9), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %mul_18), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_7, 1.0001017915309447), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, 0.1), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_9, 0.9), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, %mul_21), kwargs = {})
#   %copy__4 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_8, %add_10), kwargs = {})
#   %copy__5 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_9, %add_11), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_abs_mul_pow_relu_sign_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, out_ptr4, out_ptr6, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 3
    r0_numel = 2
    R0_BLOCK: tl.constexpr = 2
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 2*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 2*x0), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r0_1 + 2*x0), xmask, other=0.0)
    tmp25 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 9825.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = 1.0001017915309447
    tmp22 = tmp17 * tmp21
    tmp23 = 0.1
    tmp24 = tmp22 * tmp23
    tmp26 = 0.9
    tmp27 = tmp25 * tmp26
    tmp28 = tmp24 + tmp27
    tmp29 = tmp13 * tmp23
    tmp31 = tmp30 * tmp26
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp28, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)




# kernel path: /tmp/torchinductor_sahanp/r7/cr7txaycwvcsrtjyjqmeixuzahtyckpam7pnw4zhsprbqfke4fah.py
# Topologically Sorted Source Nodes: [x_7, x_8, x_9, x_10, x_11], Original ATen: [aten.sign, aten.abs, aten.relu, aten.mul, aten.pow, aten._native_batch_norm_legit_functional, aten.mish, aten.softplus, aten.bernoulli, aten._to_copy, aten.add]
# Source node to ATen node mapping:
#   x_10 => div_1, exp_3, gt_3, log1p_3, mul_24, where_3
#   x_11 => add_13, add_14, add_15, convert_element_type_1, inductor_lookup_seed_default_1, inductor_random_default, lt_1, mul_25, mul_26, mul_27
#   x_7 => abs_2, mul_14, mul_15, pow_4, relu_1, sign_1
#   x_8 => add_12, mul_16, mul_22, sub_1
#   x_9 => exp_2, gt_2, log1p_2, mul_23, tanh_1, where_2
# Graph fragment:
#   %sign_1 : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%squeeze_4,), kwargs = {})
#   %abs_2 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%squeeze_4,), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%abs_2,), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign_1, %relu_1), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_14, 3), kwargs = {})
#   %pow_4 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_15, 0.5), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_4, %getitem_3), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %unsqueeze_4), kwargs = {})
#   %add_12 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_22, %unsqueeze_5), kwargs = {})
#   %exp_2 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%add_12,), kwargs = {})
#   %log1p_2 : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_2,), kwargs = {})
#   %gt_2 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_12, 20), kwargs = {})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %add_12, %log1p_2), kwargs = {})
#   %tanh_1 : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%where_2,), kwargs = {})
#   %mul_23 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_12, %tanh_1), kwargs = {})
#   %mul_24 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_23, 1.0), kwargs = {})
#   %exp_3 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_24,), kwargs = {})
#   %log1p_3 : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_3,), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%log1p_3, 1.0), kwargs = {})
#   %gt_3 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%mul_24, 20.0), kwargs = {})
#   %where_3 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_3, %mul_23, %div_1), kwargs = {})
#   %inductor_lookup_seed_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 1), kwargs = {})
#   %inductor_random_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 3, 9825], %inductor_lookup_seed_default_1, rand), kwargs = {})
#   %lt_1 : [num_users=2] = call_function[target=torch.ops.aten.lt.Scalar](args = (%inductor_random_default, 0.5), kwargs = {})
#   %convert_element_type_1 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt_1, torch.float32), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%convert_element_type_1, -1), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%add_13, 1.558387861036063), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_25, 0.7791939305180315), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%convert_element_type_1, 0.8864048946659319), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_3, %mul_26), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_27, %add_14), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional__to_copy_abs_add_bernoulli_mish_mul_pow_relu_sign_softplus_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr3, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 29475
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 9825)
    x1 = xindex // 9825
    tmp5 = tl.load(in_ptr1 + (x0 + 9856*x1), xmask)
    tmp19 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x2
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.5
    tmp4 = tmp2 < tmp3
    tmp6 = tl.full([1], 0, tl.int32)
    tmp7 = tmp6 < tmp5
    tmp8 = tmp7.to(tl.int8)
    tmp9 = tmp5 < tmp6
    tmp10 = tmp9.to(tl.int8)
    tmp11 = tmp8 - tmp10
    tmp12 = tmp11.to(tmp5.dtype)
    tmp13 = tl_math.abs(tmp5)
    tmp14 = triton_helpers.maximum(tmp6, tmp13)
    tmp15 = tmp12 * tmp14
    tmp16 = 3.0
    tmp17 = tmp15 * tmp16
    tmp18 = libdevice.sqrt(tmp17)
    tmp20 = tmp18 - tmp19
    tmp22 = tmp20 * tmp21
    tmp24 = tmp22 * tmp23
    tmp26 = tmp24 + tmp25
    tmp27 = 20.0
    tmp28 = tmp26 > tmp27
    tmp29 = tl_math.exp(tmp26)
    tmp30 = libdevice.log1p(tmp29)
    tmp31 = tl.where(tmp28, tmp26, tmp30)
    tmp32 = libdevice.tanh(tmp31)
    tmp33 = tmp26 * tmp32
    tmp34 = 1.0
    tmp35 = tmp33 * tmp34
    tmp36 = tmp35 > tmp27
    tmp37 = tl_math.exp(tmp35)
    tmp38 = libdevice.log1p(tmp37)
    tmp39 = tmp38 * tmp34
    tmp40 = tl.where(tmp36, tmp33, tmp39)
    tmp41 = tmp4.to(tl.float32)
    tmp42 = 0.8864048946659319
    tmp43 = tmp41 * tmp42
    tmp44 = tmp40 * tmp43
    tmp45 = -1.0
    tmp46 = tmp41 + tmp45
    tmp47 = 1.558387861036063
    tmp48 = tmp46 * tmp47
    tmp49 = 0.7791939305180315
    tmp50 = tmp48 + tmp49
    tmp51 = tmp44 + tmp50
    tl.store(out_ptr1 + (x0 + 9856*x1), tmp4, xmask)
    tl.store(out_ptr3 + (x2), tmp51, xmask)




# kernel path: /tmp/torchinductor_sahanp/nv/cnv2ohoxxdrgme43ebpt5tc6rxsmrxpyitu4cbwak37hcgvypve6.py
# Topologically Sorted Source Nodes: [add_], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   add_ => add
# Graph fragment:
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_2, 1), kwargs = {})
#   %copy_ : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_2, %add), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_8(in_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp1 + tmp2
    tl.store(out_ptr1 + (tl.full([XBLOCK], 0, tl.int32)), tmp3, None)







def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11 = args
    args.clear()
    assert_size_stride(primals_1, (1, 3, 32, 32, 32), (98304, 32768, 1024, 32, 1))
    assert_size_stride(primals_2, (), ())
    assert_size_stride(primals_3, (3, ), (1, ))
    assert_size_stride(primals_4, (3, ), (1, ))
    assert_size_stride(primals_5, (3, ), (1, ))
    assert_size_stride(primals_6, (3, ), (1, ))
    assert_size_stride(primals_7, (), ())
    assert_size_stride(primals_8, (3, ), (1, ))
    assert_size_stride(primals_9, (3, ), (1, ))
    assert_size_stride(primals_10, (3, ), (1, ))
    assert_size_stride(primals_11, (3, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 3, 1, 19651), (59040, 19680, 19680, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_0[grid(58953)](primals_1, buf0, 58953, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_1
        buf1 = empty_strided_cuda((1, 3, 1, 3), (9, 3, 9, 1), torch.float32)
        buf2 = empty_strided_cuda((1, 3, 1, 3), (9, 3, 9, 1), torch.float32)
        buf3 = empty_strided_cuda((1, 3, 1, 3), (9, 3, 9, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2, x_3], Original ATen: [aten.sign, aten.abs, aten.relu, aten.mul, aten.pow, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_abs_mul_pow_relu_sign_1[grid(9)](buf0, buf1, buf2, buf3, 9, 6551, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        buf4 = empty_strided_cuda((1, 3, 1), (3, 1, 1), torch.float32)
        buf7 = empty_strided_cuda((1, 3, 1), (3, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2, x_3], Original ATen: [aten.sign, aten.abs, aten.relu, aten.mul, aten.pow, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_abs_mul_pow_relu_sign_2[grid(3)](buf1, buf2, buf3, primals_4, primals_3, buf4, buf7, primals_4, primals_3, 3, 3, XBLOCK=1, num_warps=2, num_stages=1)
        del buf1
        del buf2
        del buf3
        del primals_3
        del primals_4
        buf9 = empty_strided_cuda((2, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [2], out=buf9)
        buf11 = empty_strided_cuda((1, 3, 19651), (59136, 19712, 1), torch.bool)
        buf8 = empty_strided_cuda((1, 3, 19651), (59040, 19680, 1), torch.float32)
        buf12 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [x_2, x_3, x_4, x_5, x_6, x_7], Original ATen: [aten.sign, aten.abs, aten.relu, aten.mul, aten.pow, aten._native_batch_norm_legit_functional, aten.mish, aten.softplus, aten.bernoulli, aten._to_copy, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional__to_copy_abs_add_bernoulli_mish_mul_pow_relu_sign_softplus_3[grid(58953)](buf12, buf9, buf0, buf4, buf7, primals_5, primals_6, buf11, 0, 58953, XBLOCK=256, num_warps=4, num_stages=1)
        buf13 = empty_strided_cuda((1, 3, 1, 9825), (29568, 9856, 9856, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_4[grid(29475)](buf12, buf13, 29475, XBLOCK=128, num_warps=4, num_stages=1)
        buf14 = empty_strided_cuda((1, 3, 1, 2), (6, 2, 6, 1), torch.float32)
        buf15 = empty_strided_cuda((1, 3, 1, 2), (6, 2, 6, 1), torch.float32)
        buf16 = empty_strided_cuda((1, 3, 1, 2), (6, 2, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_7, x_8], Original ATen: [aten.sign, aten.abs, aten.relu, aten.mul, aten.pow, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_abs_mul_pow_relu_sign_5[grid(6)](buf13, buf14, buf15, buf16, 6, 4913, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        buf17 = empty_strided_cuda((1, 3, 1), (3, 1, 1), torch.float32)
        buf20 = empty_strided_cuda((1, 3, 1), (3, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_7, x_8], Original ATen: [aten.sign, aten.abs, aten.relu, aten.mul, aten.pow, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_abs_mul_pow_relu_sign_6[grid(3)](buf14, buf15, buf16, primals_9, primals_8, buf17, buf20, primals_9, primals_8, 3, 2, XBLOCK=1, num_warps=2, num_stages=1)
        del buf14
        del buf15
        del buf16
        del primals_8
        del primals_9
        buf23 = empty_strided_cuda((1, 3, 9825), (29568, 9856, 1), torch.bool)
        buf40 = empty_strided_cuda((1, 3, 9825), (29475, 9825, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_7, x_8, x_9, x_10, x_11], Original ATen: [aten.sign, aten.abs, aten.relu, aten.mul, aten.pow, aten._native_batch_norm_legit_functional, aten.mish, aten.softplus, aten.bernoulli, aten._to_copy, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional__to_copy_abs_add_bernoulli_mish_mul_pow_relu_sign_softplus_7[grid(29475)](buf9, buf13, buf17, buf20, primals_10, primals_11, buf23, buf40, 1, 29475, XBLOCK=128, num_warps=4, num_stages=1)
        del buf9
        # Topologically Sorted Source Nodes: [add_], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_8[grid(1)](primals_2, primals_2, 1, XBLOCK=1, num_warps=1, num_stages=1)
        del primals_2
        # Topologically Sorted Source Nodes: [add__1], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_8[grid(1)](primals_7, primals_7, 1, XBLOCK=1, num_warps=1, num_stages=1)
        del primals_7
    return (reinterpret_tensor(buf40, (1, 29475), (29475, 1), 0), primals_5, primals_6, primals_10, primals_11, buf0, buf4, buf7, buf11, reinterpret_tensor(buf12, (1, 3, 1, 19651), (0, 19680, 0, 1), 0), buf13, buf17, buf20, buf23, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 3, 32, 32, 32), (98304, 32768, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_3 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_8 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
