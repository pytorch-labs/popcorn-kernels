# AOT ID: ['78_inference']
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


# kernel path: /tmp/torchinductor_sahanp/i6/ci6qxuot2tikbneue3aimoh3ptl6y6ai4fd4jkoiuivr3nsfcvmd.py
# Topologically Sorted Source Nodes: [x, x_4], Original ATen: [aten.bernoulli, aten._to_copy, aten.mul]
# Source node to ATen node mapping:
#   x => inductor_lookup_seed_default, inductor_random_default_1
#   x_4 => convert_element_type_1, inductor_lookup_seed_default_1, inductor_random_default, lt_7, mul_94, mul_97
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, %arg0_1, %arg1_1, %arg2_1], %inductor_lookup_seed_default, rand), kwargs = {})
#   %inductor_lookup_seed_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 1), kwargs = {})
#   %inductor_random_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, %sym_size_int_3], %inductor_lookup_seed_default_1, rand), kwargs = {})
#   %lt_7 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%inductor_random_default, 0.5), kwargs = {})
#   %convert_element_type_1 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt_7, torch.float32), kwargs = {})
#   %mul_94 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%convert_element_type_1, 0.8864048946659319), kwargs = {})
#   %mul_97 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, %mul_94), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_bernoulli_mul_0(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, load_seed_offset, load_seed_offset1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp5 = tl.load(in_ptr1 + (x0), xmask)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = tl.load(in_ptr0 + load_seed_offset1)
    tmp4 = tl.rand(tmp3, (tmp1).to(tl.uint32))
    tmp6 = 0.5
    tmp7 = tmp2 < tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = 0.8864048946659319
    tmp10 = tmp8 * tmp9
    tmp11 = tmp5 * tmp10
    tmp12 = -1.0
    tmp13 = tmp8 + tmp12
    tmp14 = 1.558387861036063
    tmp15 = tmp13 * tmp14
    tmp16 = 0.7791939305180315
    tmp17 = tmp15 + tmp16
    tmp18 = tmp11 + tmp17
    tmp19 = 20.0
    tmp20 = tmp18 > tmp19
    tmp21 = tl_math.exp(tmp18)
    tmp22 = libdevice.log1p(tmp21)
    tmp23 = tl.where(tmp20, tmp18, tmp22)
    tmp24 = libdevice.tanh(tmp23)
    tmp25 = tmp18 * tmp24
    tmp26 = 1.0
    tmp27 = tmp25 * tmp26
    tmp28 = tmp27 > tmp19
    tmp29 = tl_math.exp(tmp27)
    tmp30 = libdevice.log1p(tmp29)
    tmp31 = tmp30 * tmp26
    tmp32 = tl.where(tmp28, tmp25, tmp31)
    tmp33 = tmp4 < tmp6
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp34 * tmp9
    tmp36 = tmp32 * tmp35
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tl.store(in_out_ptr0 + (x0), tmp36, xmask)




# kernel path: /tmp/torchinductor_sahanp/vx/cvx44atqmnkodvpp63vle3joythytg7we6otmn3wfgznypbdw3ps.py
# Topologically Sorted Source Nodes: [x_4, x_5, x_6, loss], Original ATen: [aten.bernoulli, aten._to_copy, aten.add, aten.mul, aten.mish, aten.softplus, aten.gather]
# Source node to ATen node mapping:
#   loss => gather
#   x_4 => add_75, add_84, add_97, convert_element_type_1, lt_7, mul_85
#   x_5 => exp_2, gt_5, log1p_2, mul_106, tanh_1, where_2
#   x_6 => div_1, exp_3, gt_6, log1p_3, mul_109, where_3
# Graph fragment:
#   %lt_7 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%inductor_random_default, 0.5), kwargs = {})
#   %convert_element_type_1 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt_7, torch.float32), kwargs = {})
#   %add_75 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%convert_element_type_1, -1), kwargs = {})
#   %mul_85 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%add_75, 1.558387861036063), kwargs = {})
#   %add_84 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_85, 0.7791939305180315), kwargs = {})
#   %add_97 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_97, %add_84), kwargs = {})
#   %gt_5 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_97, 20), kwargs = {})
#   %exp_2 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%add_97,), kwargs = {})
#   %log1p_2 : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_2,), kwargs = {})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_5, %add_97, %log1p_2), kwargs = {})
#   %tanh_1 : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%where_2,), kwargs = {})
#   %mul_106 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_97, %tanh_1), kwargs = {})
#   %mul_109 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_106, 1.0), kwargs = {})
#   %gt_6 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%mul_109, 20.0), kwargs = {})
#   %exp_3 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_109,), kwargs = {})
#   %log1p_3 : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_3,), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%log1p_3, 1.0), kwargs = {})
#   %where_3 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_6, %mul_106, %div_1), kwargs = {})
#   %gather : [num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%where_3, 1, %unsqueeze), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_add_bernoulli_gather_mish_mul_softplus_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, load_seed_offset, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = ks1*ks2*ks3
    tmp4 = triton_helpers.randint64(tmp0, (tmp1).to(tl.uint32), tmp2, tmp3)
    tmp5 = tmp4 + tmp3
    tmp6 = tmp4 < 0
    tmp7 = tl.where(tmp6, tmp5, tmp4)
    tl.device_assert((0 <= tmp7) & (tmp7 < ks1*ks2*ks3), "index out of bounds: 0 <= tmp7 < ks1*ks2*ks3")
    tmp9 = tl.load(in_ptr1 + (tmp7), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (tmp7), None, eviction_policy='evict_last')
    tmp11 = 0.5
    tmp12 = tmp10 < tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp14 = -1.0
    tmp15 = tmp13 + tmp14
    tmp16 = 1.558387861036063
    tmp17 = tmp15 * tmp16
    tmp18 = 0.7791939305180315
    tmp19 = tmp17 + tmp18
    tmp20 = tmp9 + tmp19
    tmp21 = 20.0
    tmp22 = tmp20 > tmp21
    tmp23 = tl_math.exp(tmp20)
    tmp24 = libdevice.log1p(tmp23)
    tmp25 = tl.where(tmp22, tmp20, tmp24)
    tmp26 = libdevice.tanh(tmp25)
    tmp27 = tmp20 * tmp26
    tmp28 = 1.0
    tmp29 = tmp27 * tmp28
    tmp30 = tmp29 > tmp21
    tmp31 = tl_math.exp(tmp29)
    tmp32 = libdevice.log1p(tmp31)
    tmp33 = tmp32 * tmp28
    tmp34 = tl.where(tmp30, tmp27, tmp33)
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp34, None)




# kernel path: /tmp/torchinductor_sahanp/2k/c2k2gj6ym724gbfcvkbtvguikmpfyfqgz6tubm652vctqsm56huz.py
# Topologically Sorted Source Nodes: [x_4, x_5, x_6, loss], Original ATen: [aten.bernoulli, aten._to_copy, aten.add, aten.mul, aten.mish, aten.softplus, aten.rsub, aten.clamp_min]
# Source node to ATen node mapping:
#   loss => add_104, clamp_min, sub_65
#   x_4 => add_75, add_84, add_97, convert_element_type_1, lt_7, mul_85
#   x_5 => exp_2, gt_5, log1p_2, mul_106, tanh_1, where_2
#   x_6 => div_1, exp_3, gt_6, log1p_3, mul_109, where_3
# Graph fragment:
#   %lt_7 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%inductor_random_default, 0.5), kwargs = {})
#   %convert_element_type_1 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt_7, torch.float32), kwargs = {})
#   %add_75 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%convert_element_type_1, -1), kwargs = {})
#   %mul_85 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%add_75, 1.558387861036063), kwargs = {})
#   %add_84 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_85, 0.7791939305180315), kwargs = {})
#   %add_97 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_97, %add_84), kwargs = {})
#   %gt_5 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_97, 20), kwargs = {})
#   %exp_2 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%add_97,), kwargs = {})
#   %log1p_2 : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_2,), kwargs = {})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_5, %add_97, %log1p_2), kwargs = {})
#   %tanh_1 : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%where_2,), kwargs = {})
#   %mul_106 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_97, %tanh_1), kwargs = {})
#   %mul_109 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_106, 1.0), kwargs = {})
#   %gt_6 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%mul_109, 20.0), kwargs = {})
#   %exp_3 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_109,), kwargs = {})
#   %log1p_3 : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_3,), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%log1p_3, 1.0), kwargs = {})
#   %where_3 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_6, %mul_106, %div_1), kwargs = {})
#   %sub_65 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %gather), kwargs = {})
#   %add_104 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub_65, %where_3), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_104, 0), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_add_bernoulli_clamp_min_mish_mul_rsub_softplus_2(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp4 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp5 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 1.0
    tmp3 = tmp2 - tmp1
    tmp6 = 0.5
    tmp7 = tmp5 < tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = -1.0
    tmp10 = tmp8 + tmp9
    tmp11 = 1.558387861036063
    tmp12 = tmp10 * tmp11
    tmp13 = 0.7791939305180315
    tmp14 = tmp12 + tmp13
    tmp15 = tmp4 + tmp14
    tmp16 = 20.0
    tmp17 = tmp15 > tmp16
    tmp18 = tl_math.exp(tmp15)
    tmp19 = libdevice.log1p(tmp18)
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tmp21 = libdevice.tanh(tmp20)
    tmp22 = tmp15 * tmp21
    tmp23 = tmp22 * tmp2
    tmp24 = tmp23 > tmp16
    tmp25 = tl_math.exp(tmp23)
    tmp26 = libdevice.log1p(tmp25)
    tmp27 = tmp26 * tmp2
    tmp28 = tl.where(tmp24, tmp22, tmp27)
    tmp29 = tmp3 + tmp28
    tmp30 = 0.0
    tmp31 = triton_helpers.maximum(tmp29, tmp30)
    tl.store(in_out_ptr0 + (x0), tmp31, xmask)




# kernel path: /tmp/torchinductor_sahanp/zs/czs34kaqe2yysaia7n4o3lxpitbk2me4q4sborymk6airefbytuy.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten.arange, aten.ne, aten.scalar_tensor, aten.where, aten.mean]
# Source node to ATen node mapping:
#   loss => full_default, iota, mean, ne_5, where_4
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (%mul_113,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %ne_5 : [num_users=1] = call_function[target=torch.ops.aten.ne.Tensor](args = (%iota, %unsqueeze), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_4 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_5, %clamp_min, %full_default), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%where_4,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_arange_mean_ne_scalar_tensor_where_3(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, load_seed_offset, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp16 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = r0_1 + x0*((1 + ks0*ks1*ks2) // 2)
        tmp1 = ks0*ks1*ks2
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + load_seed_offset)
        tmp4 = tl.full([1, 1], 0, tl.int32)
        tmp5 = tl.full([1, 1], 0, tl.int64)
        tmp6 = tl.broadcast_to(ks0*ks1*ks2, [XBLOCK, R0_BLOCK])
        tmp7 = triton_helpers.randint64(tmp3, (tmp4).to(tl.uint32), tmp5, tmp6)
        tmp8 = r0_1 + x0*((1 + ks0*ks1*ks2) // 2)
        tmp9 = tmp8 != tmp7
        tmp10 = tl.load(in_ptr1 + (r0_1 + x0*((1 + ks0*ks1*ks2) // 2)), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = 0.0
        tmp12 = tl.where(tmp9, tmp10, tmp11)
        tmp13 = tl.full(tmp12.shape, 0, tmp12.dtype)
        tmp14 = tl.where(tmp2, tmp12, tmp13)
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(r0_mask & xmask, tmp17, _tmp16)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp16, xmask)




# kernel path: /tmp/torchinductor_sahanp/tj/ctjvvcp73txnfhx5wgyt5qjl7cdx5krdbngohygiwdnzzvkzssf3.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten.arange, aten.ne, aten.scalar_tensor, aten.where, aten.mean]
# Source node to ATen node mapping:
#   loss => full_default, iota, mean, ne_5, where_4
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (%mul_113,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %ne_5 : [num_users=1] = call_function[target=torch.ops.aten.ne.Tensor](args = (%iota, %unsqueeze), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_4 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_5, %clamp_min, %full_default), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%where_4,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_arange_mean_ne_scalar_tensor_where_4(in_out_ptr0, in_ptr0, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp4 = ks0*ks1*ks2
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
        buf0 = empty_strided_cuda((3, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [3], out=buf0)
        buf1 = empty_strided_cuda((1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1), torch.float32)
        buf2 = empty_strided_cuda((1, s0*s1*s2), (s0*s1*s2, 1), torch.float32)
        buf3 = reinterpret_tensor(buf1, (1, s0*s1*s2), (s0*s1*s2, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [x, x_4], Original ATen: [aten.bernoulli, aten._to_copy, aten.mul]
        triton_poi_fused__to_copy_bernoulli_mul_0_xnumel = s0*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_bernoulli_mul_0[grid(triton_poi_fused__to_copy_bernoulli_mul_0_xnumel)](buf3, buf0, arg3_1, buf2, 0, 1, 12288, XBLOCK=128, num_warps=4, num_stages=1)
        del arg3_1
        buf4 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_4, x_5, x_6, loss], Original ATen: [aten.bernoulli, aten._to_copy, aten.add, aten.mul, aten.mish, aten.softplus, aten.gather]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_bernoulli_gather_mish_mul_softplus_1[grid(1)](buf0, buf3, buf2, buf4, 2, 3, 64, 64, 1, XBLOCK=1, num_warps=1, num_stages=1)
        buf5 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [x_4, x_5, x_6, loss], Original ATen: [aten.bernoulli, aten._to_copy, aten.add, aten.mul, aten.mish, aten.softplus, aten.rsub, aten.clamp_min]
        triton_poi_fused__to_copy_add_bernoulli_clamp_min_mish_mul_rsub_softplus_2_xnumel = s0*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_bernoulli_clamp_min_mish_mul_rsub_softplus_2[grid(triton_poi_fused__to_copy_add_bernoulli_clamp_min_mish_mul_rsub_softplus_2_xnumel)](buf5, buf4, buf2, 12288, XBLOCK=256, num_warps=4, num_stages=1)
        del buf2
        buf6 = empty_strided_cuda((2, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [loss], Original ATen: [aten.arange, aten.ne, aten.scalar_tensor, aten.where, aten.mean]
        triton_red_fused_arange_mean_ne_scalar_tensor_where_3_r0_numel = (1 + s0*s1*s2) // 2
        stream0 = get_raw_stream(0)
        triton_red_fused_arange_mean_ne_scalar_tensor_where_3[grid(2)](buf0, buf5, buf6, 3, 64, 64, 2, 2, 6144, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del buf0
        del buf5
        buf7 = reinterpret_tensor(buf4, (), (), 0); del buf4  # reuse
        buf8 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [loss], Original ATen: [aten.arange, aten.ne, aten.scalar_tensor, aten.where, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_arange_mean_ne_scalar_tensor_where_4[grid(1)](buf8, buf6, 3, 64, 64, 1, 2, XBLOCK=1, num_warps=2, num_stages=1)
        del buf6
    return (buf8, )


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
