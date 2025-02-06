# AOT ID: ['64_inference']
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


# kernel path: /tmp/torchinductor_sahanp/5r/c5rbjah6y5ob2jexek2eua5vfetiyuaxw6zwnumtvbmm7cku2ceb.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.bernoulli]
# Source node to ATen node mapping:
#   x_1 => inductor_lookup_seed_default, inductor_random_default
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, %arg0_1, 1, 1, 1], %inductor_lookup_seed_default, rand), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_bernoulli_0(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (x0), tmp2, xmask)




# kernel path: /tmp/torchinductor_sahanp/bv/cbvqhfg7tv6th525d2praa3aq7k5wi3eoq243xwkw2dxbpw5h3uv.py
# Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.constant_pad_nd, aten.bernoulli, aten._to_copy, aten.mul, aten.add]
# Source node to ATen node mapping:
#   x => constant_pad_nd
#   x_1 => add_20, add_41, add_72, convert_element_type, lt_2, mul_18, mul_27, mul_30
# Graph fragment:
#   %constant_pad_nd : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%arg4_1, [1, 1, 1, 1, 1, 1], 0.0), kwargs = {})
#   %lt_2 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%inductor_random_default, 0.5), kwargs = {})
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt_2, torch.float32), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%convert_element_type, 0.8864048946659319), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%constant_pad_nd, %mul_27), kwargs = {})
#   %add_20 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%convert_element_type, -1), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%add_20, 1.558387861036063), kwargs = {})
#   %add_41 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_18, 0.7791939305180315), kwargs = {})
#   %add_72 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_30, %add_41), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_add_bernoulli_constant_pad_nd_mul_1(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, ks6, ks7, ks8, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x6 = ((xindex // ks0) % ks1)
    x1 = ((xindex // ks3) % ks4)
    x0 = (xindex % ks3)
    x2 = ((xindex // ks7) % ks1)
    x3 = xindex // ks8
    x10 = xindex
    tmp19 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp0 = (-1) + x6
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = ks2
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = ks5
    tmp8 = tmp5 < tmp7
    tmp9 = (-1) + x0
    tmp10 = tmp9 >= tmp1
    tmp11 = ks6
    tmp12 = tmp9 < tmp11
    tmp13 = tmp2 & tmp4
    tmp14 = tmp13 & tmp6
    tmp15 = tmp14 & tmp8
    tmp16 = tmp15 & tmp10
    tmp17 = tmp16 & tmp12
    tmp18 = tl.load(in_ptr0 + ((-1) + x0 + ((-1)*ks6) + ks6*x1 + ((-1)*ks5*ks6) + ks5*ks6*x2 + ks2*ks5*ks6*x3), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = 0.5
    tmp21 = tmp19 < tmp20
    tmp22 = tmp21.to(tl.float32)
    tmp23 = 0.8864048946659319
    tmp24 = tmp22 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = -1.0
    tmp27 = tmp22 + tmp26
    tmp28 = 1.558387861036063
    tmp29 = tmp27 * tmp28
    tmp30 = 0.7791939305180315
    tmp31 = tmp29 + tmp30
    tmp32 = tmp25 + tmp31
    tl.store(out_ptr0 + (x10), tmp32, xmask)




# kernel path: /tmp/torchinductor_sahanp/qk/cqkqz2ja5zhd5amvtfkh7ywjg2mfunwrlcqutw74z7nktweua4fp.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   x_2 => pow_1, sum_1
# Graph fragment:
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%view_1, 2), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1], True), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_linalg_vector_norm_2(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 15
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = r0_1 + x0*((14 + 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 15)
        tmp1 = 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (r0_1 + x0*((14 + 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 15)), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tmp3 * tmp3
        tmp5 = tl.full(tmp4.shape, 0, tmp4.dtype)
        tmp6 = tl.where(tmp2, tmp4, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)




# kernel path: /tmp/torchinductor_sahanp/37/c37n3lc5o4dyh5v2jhzic4mu3nkn2g24xh43d7kfesdb4dnzixz3.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   x_2 => pow_1, sum_1
# Graph fragment:
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%view_1, 2), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1], True), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_linalg_vector_norm_3(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 15
    R0_BLOCK: tl.constexpr = 16
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
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp4, None)




# kernel path: /tmp/torchinductor_sahanp/pg/cpgykpoghmd74ofw7rownkhapqcrbczx5ktbg3yym5dhoqb7skgp.py
# Topologically Sorted Source Nodes: [x2, x_2], Original ATen: [aten.ones_like, aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   x2 => full
#   x_2 => pow_3, sum_2
# Graph fragment:
#   %full : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([1, %sym_size_int_5], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%full, 2), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_3, [1], True), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_linalg_vector_norm_ones_like_4(out_ptr0, ks0, ks1, ks2, ks3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 15
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = r0_1 + x0*((14 + 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 15)
        tmp1 = 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3
        tmp2 = tmp0 < tmp1
        tmp3 = 1.0
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp7, xmask)




# kernel path: /tmp/torchinductor_sahanp/oc/cocacpuaaclv6ho2krqupmtm7xmmtepokrmdesxomtdygjnv5sbj.py
# Topologically Sorted Source Nodes: [x_2, x2], Original ATen: [aten.linalg_vector_norm, aten.clamp_min, aten.div, aten.ones_like, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   x2 => full
#   x_2 => clamp_min, clamp_min_1, div, div_1, mul_56, pow_2, pow_4, sum_3
# Graph fragment:
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_2, 1e-08), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_1, %clamp_min), kwargs = {})
#   %full : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([1, %sym_size_int_5], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_2, 0.5), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_4, 1e-08), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%full, %clamp_min_1), kwargs = {})
#   %mul_56 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %div), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_56, [1]), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_clamp_min_div_linalg_vector_norm_mul_ones_like_sum_5(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, ks1, ks2, ks3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 15
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp18 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = r0_1 + x0*((14 + 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 15)
        tmp1 = 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (r0_1 + x0*((14 + 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 15)), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (tl.full([XBLOCK, R0_BLOCK], 0, tl.int32)), tmp2, eviction_policy='evict_last', other=0.0)
        tmp5 = libdevice.sqrt(tmp4)
        tmp6 = 1e-08
        tmp7 = triton_helpers.maximum(tmp5, tmp6)
        tmp8 = tmp3 / tmp7
        tmp9 = tl.load(in_ptr2 + (tl.full([XBLOCK, R0_BLOCK], 0, tl.int32)), tmp2, eviction_policy='evict_last', other=0.0)
        tmp10 = libdevice.sqrt(tmp9)
        tmp11 = triton_helpers.maximum(tmp10, tmp6)
        tmp12 = 1.0
        tmp13 = tmp12 / tmp11
        tmp14 = tmp8 * tmp13
        tmp15 = tl.full(tmp14.shape, 0, tmp14.dtype)
        tmp16 = tl.where(tmp2, tmp14, tmp15)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(r0_mask & xmask, tmp19, _tmp18)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp18, xmask)




# kernel path: /tmp/torchinductor_sahanp/25/c25tzsiehar4thzfg7mgrylvki5c6qe37w677sjkfgvi6n46fagf.py
# Topologically Sorted Source Nodes: [loss, x_2, x2], Original ATen: [aten.nll_loss_forward, aten.linalg_vector_norm, aten.clamp_min, aten.div, aten.ones_like, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   loss => convert_element_type_2, div_2, full_default_1, full_default_2, full_default_3, neg, sum_5, sum_6, where_1
#   x2 => full
#   x_2 => clamp_min, clamp_min_1, div, div_1, mul_56, pow_2, pow_4, sum_3
# Graph fragment:
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_2, 1e-08), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_1, %clamp_min), kwargs = {})
#   %full : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([1, %sym_size_int_5], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_2, 0.5), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_4, 1e-08), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%full, %clamp_min_1), kwargs = {})
#   %mul_56 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %div), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_56, [1]), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%full_default_1, %neg, %full_default_2), kwargs = {})
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_1,), kwargs = {})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%full_default_3,), kwargs = {})
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_5, torch.float32), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_6, %convert_element_type_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_clamp_min_div_linalg_vector_norm_mul_nll_loss_forward_ones_like_sum_6(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 15
    R0_BLOCK: tl.constexpr = 16
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
    tmp5 = tmp4 - tmp4
    tmp6 = tl_math.exp(tmp5)
    tmp7 = tl_math.log(tmp6)
    tmp8 = tmp5 - tmp7
    tmp9 = -tmp8
    tmp10 = tl.full([1, 1], True, tl.int1)
    tmp11 = 0.0
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tmp13 = 1.0
    tmp14 = tmp12 / tmp13
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp14, None)







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
        buf0 = empty_strided_cuda((1, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf0)
        buf1 = empty_strided_cuda((1, s0, 1, 1, 1), (s0, 1, s0, s0, s0), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.bernoulli]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bernoulli_0[grid(s0)](buf0, buf1, 0, 3, XBLOCK=4, num_warps=1, num_stages=1)
        del buf0
        ps0 = 4 + 2*s2 + 2*s3 + s2*s3
        ps1 = 2 + s1
        ps2 = 2 + s3
        ps3 = 2 + s2
        ps4 = 4 + 2*s2 + 2*s3 + s2*s3
        ps5 = 8 + 4*s1 + 4*s2 + 4*s3 + 2*s1*s2 + 2*s1*s3 + 2*s2*s3 + s1*s2*s3
        buf2 = empty_strided_cuda((1, s0, 2 + s1, 2 + s2, 2 + s3), (8*s0 + 4*s0*s1 + 4*s0*s2 + 4*s0*s3 + 2*s0*s1*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + s0*s1*s2*s3, 8 + 4*s1 + 4*s2 + 4*s3 + 2*s1*s2 + 2*s1*s3 + 2*s2*s3 + s1*s2*s3, 4 + 2*s2 + 2*s3 + s2*s3, 2 + s3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.constant_pad_nd, aten.bernoulli, aten._to_copy, aten.mul, aten.add]
        triton_poi_fused__to_copy_add_bernoulli_constant_pad_nd_mul_1_xnumel = 8*s0 + 4*s0*s1 + 4*s0*s2 + 4*s0*s3 + 2*s0*s1*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + s0*s1*s2*s3
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_bernoulli_constant_pad_nd_mul_1[grid(triton_poi_fused__to_copy_add_bernoulli_constant_pad_nd_mul_1_xnumel)](arg4_1, buf1, buf2, 1156, 34, 32, 34, 34, 32, 32, 1156, 39304, 117912, XBLOCK=512, num_warps=8, num_stages=1)
        del arg4_1
        del buf1
        buf3 = empty_strided_cuda((1, 1, 15), (15, 15, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_2_r0_numel = (14 + 8*s0 + 4*s0*s1 + 4*s0*s2 + 4*s0*s3 + 2*s0*s1*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + s0*s1*s2*s3) // 15
        stream0 = get_raw_stream(0)
        triton_red_fused_linalg_vector_norm_2[grid(15)](buf2, buf3, 3, 32, 32, 32, 15, 7861, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        buf4 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.linalg_vector_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_linalg_vector_norm_3[grid(1)](buf3, buf4, 1, 15, XBLOCK=1, num_warps=2, num_stages=1)
        buf5 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [x2, x_2], Original ATen: [aten.ones_like, aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_ones_like_4_r0_numel = (14 + 8*s0 + 4*s0*s1 + 4*s0*s2 + 4*s0*s3 + 2*s0*s1*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + s0*s1*s2*s3) // 15
        stream0 = get_raw_stream(0)
        triton_red_fused_linalg_vector_norm_ones_like_4[grid(15)](buf5, 3, 32, 32, 32, 15, 7861, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        buf6 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x2, x_2], Original ATen: [aten.ones_like, aten.linalg_vector_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_linalg_vector_norm_3[grid(1)](buf5, buf6, 1, 15, XBLOCK=1, num_warps=2, num_stages=1)
        buf7 = reinterpret_tensor(buf5, (1, 15), (15, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [x_2, x2], Original ATen: [aten.linalg_vector_norm, aten.clamp_min, aten.div, aten.ones_like, aten.mul, aten.sum]
        triton_red_fused_clamp_min_div_linalg_vector_norm_mul_ones_like_sum_5_r0_numel = (14 + 8*s0 + 4*s0*s1 + 4*s0*s2 + 4*s0*s3 + 2*s0*s1*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + s0*s1*s2*s3) // 15
        stream0 = get_raw_stream(0)
        triton_red_fused_clamp_min_div_linalg_vector_norm_mul_ones_like_sum_5[grid(15)](buf2, buf4, buf6, buf7, 3, 32, 32, 32, 15, 7861, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del buf2
        del buf4
        buf8 = reinterpret_tensor(buf6, (1, ), (1, ), 0); del buf6  # reuse
        buf9 = reinterpret_tensor(buf8, (), (), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [loss, x_2, x2], Original ATen: [aten.nll_loss_forward, aten.linalg_vector_norm, aten.clamp_min, aten.div, aten.ones_like, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_clamp_min_div_linalg_vector_norm_mul_nll_loss_forward_ones_like_sum_6[grid(1)](buf9, buf7, 1, 15, XBLOCK=1, num_warps=2, num_stages=1)
        del buf7
    return (buf9, )


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
