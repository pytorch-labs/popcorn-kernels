# AOT ID: ['55_inference']
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


# kernel path: /tmp/torchinductor_sahanp/n7/cn7qso3zkomgmxdmtreqk72wxyjpu4ykeukhc2qdthfbsnddif47.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.max_unpool3d]
# Source node to ATen node mapping:
#   x => full
# Graph fragment:
#   %full : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, %arg0_1, %sub_9, %sub_11, %sub_13], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_unpool3d_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/rf/crf2hanawqe2bryu24phriacft65ivjbh4diq7w74qyqhizcu7ik.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.max_unpool3d]
# Source node to ATen node mapping:
#   x => index_put
# Graph fragment:
#   %index_put : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%view_2, [%view_1], %view_3), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_unpool3d_1(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp8 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = 8*(ks0 // 2)*(ks1 // 2)*(ks2 // 2)*(triton_helpers.div_floor_integer(x0,  (ks0 // 2)*(ks1 // 2)*(ks2 // 2)))
    tmp2 = tmp0 + tmp1
    tmp3 = 8*ks3*(ks0 // 2)*(ks1 // 2)*(ks2 // 2)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp2 < 0
    tmp6 = tl.where(tmp5, tmp4, tmp2)
    tl.device_assert(((0 <= tmp6) & (tmp6 < 8*ks3*(ks0 // 2)*(ks1 // 2)*(ks2 // 2))) | ~(xmask), "index out of bounds: 0 <= tmp6 < 8*ks3*(ks0 // 2)*(ks1 // 2)*(ks2 // 2)")
    tl.store(out_ptr0 + (tl.broadcast_to((tmp6 % (8*ks3*(ks0 // 2)*(ks1 // 2)*(ks2 // 2))), [XBLOCK])), tmp8, xmask)




# kernel path: /tmp/torchinductor_sahanp/gz/cgzxqukvuzltjb2hv3qgkjrqzhhhv3iuamqkshsexzki6xhogz3o.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.bernoulli]
# Source node to ATen node mapping:
#   x_1 => inductor_lookup_seed_default, inductor_random_default_1
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, %arg0_1, 1, 1, 1], %inductor_lookup_seed_default, rand), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_bernoulli_2(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (x0), tmp2, xmask)




# kernel path: /tmp/torchinductor_sahanp/hv/chv7dgs35veae5v4pcxpzhoql5zyzsf3o7om5iwecgfs5qqy462t.py
# Topologically Sorted Source Nodes: [random_tensor, loss], Original ATen: [aten.randn_like, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   loss => mul_75, mul_78, mul_81, sum_1, sum_2, sum_3
#   random_tensor => inductor_lookup_seed_default_1, inductor_random_default
# Graph fragment:
#   %inductor_lookup_seed_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 1), kwargs = {})
#   %inductor_random_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, %arg0_1, %sub_9, %sub_11, %sym_size_int_4], %inductor_lookup_seed_default_1, randn), kwargs = {})
#   %mul_75 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_7, %view_6), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_75, [1]), kwargs = {})
#   %mul_78 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_7, %view_7), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_78, [1]), kwargs = {})
#   %mul_81 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_6, %view_6), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_81, [1]), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_mul_randn_like_sum_3(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, load_seed_offset, ks1, ks2, ks3, ks4, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp20 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp24 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp28 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp3 = tl.load(in_ptr1 + (2*(ks4 // 2)*((((2*(ks4 // 2)*(((r0_1 // (2*(ks4 // 2))) % (2*(ks3 // 2)))) + ((r0_1 % (2*(ks4 // 2))))) // (2*(ks4 // 2))) % (2*(ks3 // 2)))) + 4*(ks3 // 2)*(ks4 // 2)*((((2*(ks4 // 2)*(((r0_1 // (2*(ks4 // 2))) % (2*(ks3 // 2)))) + 4*(ks3 // 2)*(ks4 // 2)*((((r0_1 + 4*ks1*x0*(ks2 // 2)*(ks3 // 2)*(ks4 // 2)) // (4*(ks3 // 2)*(ks4 // 2))) % (2*(ks2 // 2)))) + ((r0_1 % (2*(ks4 // 2))))) // (4*(ks3 // 2)*(ks4 // 2))) % (2*(ks2 // 2)))) + 8*(ks2 // 2)*(ks3 // 2)*(ks4 // 2)*((((2*(ks4 // 2)*(((r0_1 // (2*(ks4 // 2))) % (2*(ks3 // 2)))) + 4*(ks3 // 2)*(ks4 // 2)*((((r0_1 + 4*ks1*x0*(ks2 // 2)*(ks3 // 2)*(ks4 // 2)) // (4*(ks3 // 2)*(ks4 // 2))) % (2*(ks2 // 2)))) + 8*(ks2 // 2)*(ks3 // 2)*(ks4 // 2)*((((r0_1 + 4*ks1*x0*(ks2 // 2)*(ks3 // 2)*(ks4 // 2)) // (8*(ks2 // 2)*(ks3 // 2)*(ks4 // 2))) % ks1)) + ((r0_1 % (2*(ks4 // 2))))) // (8*(ks2 // 2)*(ks3 // 2)*(ks4 // 2))) % ks1)) + ((((r0_1 % (2*(ks4 // 2)))) % (2*(ks4 // 2))))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + ((((r0_1 + 4*ks1*x0*(ks2 // 2)*(ks3 // 2)*(ks4 // 2)) // (8*(ks2 // 2)*(ks3 // 2)*(ks4 // 2))) % ks1)), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = tl.load(in_ptr0 + load_seed_offset)
        tmp1 = r0_1 + 4*ks1*x0*(ks2 // 2)*(ks3 // 2)*(ks4 // 2)
        tmp2 = tl.randn(tmp0, (tmp1).to(tl.uint32))
        tmp5 = 0.5
        tmp6 = tmp4 < tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = 0.8864048946659319
        tmp9 = tmp7 * tmp8
        tmp10 = tmp3 * tmp9
        tmp11 = -1.0
        tmp12 = tmp7 + tmp11
        tmp13 = 1.558387861036063
        tmp14 = tmp12 * tmp13
        tmp15 = 0.7791939305180315
        tmp16 = tmp14 + tmp15
        tmp17 = tmp10 + tmp16
        tmp18 = tmp17 * tmp2
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(r0_mask & xmask, tmp21, _tmp20)
        tmp22 = tmp17 * tmp17
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, R0_BLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(r0_mask & xmask, tmp25, _tmp24)
        tmp26 = tmp2 * tmp2
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, R0_BLOCK])
        tmp29 = _tmp28 + tmp27
        _tmp28 = tl.where(r0_mask & xmask, tmp29, _tmp28)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tmp28 = tl.sum(_tmp28, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp20, xmask)
    tl.store(out_ptr2 + (x0), tmp24, xmask)
    tl.store(out_ptr3 + (x0), tmp28, xmask)




# kernel path: /tmp/torchinductor_sahanp/5e/c5ehi25goaoyb4xnasydpnjluu3dcvwujmxfsrtlwibsvgifnsug.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten.eq, aten.fill, aten.mul, aten.sum, aten.add, aten.sqrt, aten.div, aten.sub, aten.zeros_like, aten.where, aten.clamp_min, aten.mean]
# Source node to ATen node mapping:
#   loss => add_106, add_109, add_110, clamp_min, div, full_default, full_default_1, full_default_2, full_default_3, mean, mul_75, mul_78, mul_81, mul_84, sqrt, sub_52, sub_53, sum_1, sum_2, sum_3, where, where_1
# Graph fragment:
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_75 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_7, %view_6), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_75, [1]), kwargs = {})
#   %mul_78 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_7, %view_7), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_78, [1]), kwargs = {})
#   %add_106 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%sum_2, 9.999999960041972e-13), kwargs = {})
#   %mul_81 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_6, %view_6), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_81, [1]), kwargs = {})
#   %add_109 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%sum_3, 9.999999960041972e-13), kwargs = {})
#   %mul_84 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_106, %add_109), kwargs = {})
#   %sqrt : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%mul_84,), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_1, %sqrt), kwargs = {})
#   %sub_52 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default_1, %div), kwargs = {})
#   %full_default : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([1], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%full_default_2, %sub_52, %full_default), kwargs = {})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], False), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sub_53 : [num_users=1] = call_function[target=torch.ops.aten.sub.Scalar](args = (%div, 0.0), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_53, 0), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%full_default_3, %clamp_min, %full_default), kwargs = {})
#   %add_110 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where, %where_1), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%add_110,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_clamp_min_div_eq_fill_mean_mul_sqrt_sub_sum_where_zeros_like_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp4 = tl.load(in_ptr1 + (r0_0), None)
    tmp8 = tl.load(in_ptr2 + (r0_0), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None]
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.sum(tmp5, 1)[:, None]
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp11 = tl.sum(tmp9, 1)[:, None]
    tmp12 = 9.999999960041972e-13
    tmp13 = tmp7 + tmp12
    tmp14 = tmp11 + tmp12
    tmp15 = tmp13 * tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tmp3 / tmp16
    tmp18 = 1.0
    tmp19 = tmp18 - tmp17
    tmp20 = tl.full([1, 1], True, tl.int1)
    tmp21 = 0.0
    tmp22 = tl.where(tmp20, tmp19, tmp21)
    tmp23 = tmp17 - tmp21
    tmp24 = triton_helpers.maximum(tmp23, tmp21)
    tmp25 = tl.full([1, 1], False, tl.int1)
    tmp26 = tl.where(tmp25, tmp24, tmp21)
    tmp27 = tmp22 + tmp26
    tmp28 = tmp27 / tmp18
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp28, None)







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
        # Topologically Sorted Source Nodes: [max_pool3d], Original ATen: [aten.max_pool3d_with_indices]
        buf0 = torch.ops.aten.max_pool3d_with_indices.default(arg4_1, [2, 2, 2], [2, 2, 2])
        del arg4_1
        buf1 = buf0[0]
        buf2 = buf0[1]
        del buf0
        buf3 = empty_strided_cuda((1, s0, 2*(s1 // 2), 2*(s2 // 2), 2*(s3 // 2)), (8*s0*(s1 // 2)*(s2 // 2)*(s3 // 2), 8*(s1 // 2)*(s2 // 2)*(s3 // 2), 4*(s2 // 2)*(s3 // 2), 2*(s3 // 2), 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.max_unpool3d]
        triton_poi_fused_max_unpool3d_0_xnumel = 8*s0*(s1 // 2)*(s2 // 2)*(s3 // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_unpool3d_0[grid(triton_poi_fused_max_unpool3d_0_xnumel)](buf3, 12288, XBLOCK=256, num_warps=4, num_stages=1)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.max_unpool3d]
        triton_poi_fused_max_unpool3d_1_xnumel = s0*(s1 // 2)*(s2 // 2)*(s3 // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_unpool3d_1[grid(triton_poi_fused_max_unpool3d_1_xnumel)](buf2, buf1, buf3, 16, 16, 16, 3, 1536, XBLOCK=128, num_warps=4, num_stages=1)
        del buf1
        del buf2
        buf5 = empty_strided_cuda((2, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [2], out=buf5)
        buf6 = empty_strided_cuda((1, s0, 1, 1, 1), (s0, 1, s0, s0, s0), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.bernoulli]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bernoulli_2[grid(s0)](buf5, buf6, 0, 3, XBLOCK=4, num_warps=1, num_stages=1)
        buf8 = empty_strided_cuda((1, 2), (2, 1), torch.float32)
        buf10 = empty_strided_cuda((1, 2), (2, 1), torch.float32)
        buf12 = empty_strided_cuda((1, 2), (2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [random_tensor, loss], Original ATen: [aten.randn_like, aten.mul, aten.sum]
        triton_red_fused_mul_randn_like_sum_3_r0_numel = 4*s0*(s1 // 2)*(s2 // 2)*(s3 // 2)
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_randn_like_sum_3[grid(2)](buf5, buf3, buf6, buf8, buf10, buf12, 1, 3, 16, 16, 16, 2, 6144, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del buf3
        del buf5
        del buf6
        buf9 = empty_strided_cuda((1, ), (1, ), torch.float32)
        buf14 = reinterpret_tensor(buf9, (), (), 0); del buf9  # reuse
        # Topologically Sorted Source Nodes: [loss], Original ATen: [aten.eq, aten.fill, aten.mul, aten.sum, aten.add, aten.sqrt, aten.div, aten.sub, aten.zeros_like, aten.where, aten.clamp_min, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clamp_min_div_eq_fill_mean_mul_sqrt_sub_sum_where_zeros_like_4[grid(1)](buf14, buf8, buf10, buf12, 1, 2, XBLOCK=1, num_warps=2, num_stages=1)
        del buf10
        del buf12
        del buf8
    return (buf14, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 3
    arg1_1 = 16
    arg2_1 = 16
    arg3_1 = 16
    arg4_1 = rand_strided((1, 3, 16, 16, 16), (12288, 4096, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
