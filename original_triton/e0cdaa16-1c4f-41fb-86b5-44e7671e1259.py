# AOT ID: ['109_inference']
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


# kernel path: /tmp/torchinductor_sahanp/6p/c6pgzr5l3gjbb5tveg7va6ujrmy7z5tumtn7g62hhge3jokk6m44.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   x_2 => pow_1, sum_1
# Graph fragment:
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%slice_2, 2), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1], True), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_linalg_vector_norm_0(in_ptr0, out_ptr0, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    _tmp11 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (ks2*(((-1) + ks1) * (((-1) + ks1) <= (((0) * ((0) >= ((-2) + x1)) + ((-2) + x1) * (((-2) + x1) > (0))))) + (((0) * ((0) >= ((-2) + x1)) + ((-2) + x1) * (((-2) + x1) > (0)))) * ((((0) * ((0) >= ((-2) + x1)) + ((-2) + x1) * (((-2) + x1) > (0)))) < ((-1) + ks1))) + ks1*ks2*r0_2 + (((-1) + ks2) * (((-1) + ks2) <= (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0))))) + (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) * ((((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) < ((-1) + ks2)))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.5
        tmp2 = tmp0 * tmp1
        tmp3 = 0.7071067811865476
        tmp4 = tmp0 * tmp3
        tmp5 = libdevice.erf(tmp4)
        tmp6 = 1.0
        tmp7 = tmp5 + tmp6
        tmp8 = tmp2 * tmp7
        tmp9 = tmp8 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(r0_mask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)




# kernel path: /tmp/torchinductor_sahanp/eb/cebil6skfedxyup4jqg6n5y2wfbue4n656ykvmedjnusgdeid7s7.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   x_2 => pow_3, sum_2
# Graph fragment:
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%slice_6, 2), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_3, [1], True), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_linalg_vector_norm_1(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    _tmp11 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (ks3*(((-1) + ks2) * (((-1) + ks2) <= (((0) * ((0) >= ((-2) + x1)) + ((-2) + x1) * (((-2) + x1) > (0))))) + (((0) * ((0) >= ((-2) + x1)) + ((-2) + x1) * (((-2) + x1) > (0)))) * ((((0) * ((0) >= ((-2) + x1)) + ((-2) + x1) * (((-2) + x1) > (0)))) < ((-1) + ks2))) + ks2*ks3*r0_2 + ks2*ks3*(ks1 // 2) + (((-1) + ks3) * (((-1) + ks3) <= (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0))))) + (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) * ((((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) < ((-1) + ks3)))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.5
        tmp2 = tmp0 * tmp1
        tmp3 = 0.7071067811865476
        tmp4 = tmp0 * tmp3
        tmp5 = libdevice.erf(tmp4)
        tmp6 = 1.0
        tmp7 = tmp5 + tmp6
        tmp8 = tmp2 * tmp7
        tmp9 = tmp8 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(r0_mask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)




# kernel path: /tmp/torchinductor_sahanp/4x/c4xvhodajp7rvvnsl4tmlpjlwdimppxuu6rqr5on4m7vyvu4p53b.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.linalg_vector_norm, aten.clamp_min, aten.div, aten.mul]
# Source node to ATen node mapping:
#   x_2 => clamp_min_2, clamp_min_3, div, div_1, mul_128, pow_2, pow_4
# Graph fragment:
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_2, 1e-08), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%slice_2, %clamp_min_2), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_2, 0.5), kwargs = {})
#   %clamp_min_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_4, 1e-08), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%slice_6, %clamp_min_3), kwargs = {})
#   %mul_128 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %div), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clamp_min_div_linalg_vector_norm_mul_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks1)
    x2 = xindex // ks2
    x3 = (xindex % ks2)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (ks4*(((-1) + ks3) * (((-1) + ks3) <= (((0) * ((0) >= ((-2) + x1)) + ((-2) + x1) * (((-2) + x1) > (0))))) + (((0) * ((0) >= ((-2) + x1)) + ((-2) + x1) * (((-2) + x1) > (0)))) * ((((0) * ((0) >= ((-2) + x1)) + ((-2) + x1) * (((-2) + x1) > (0)))) < ((-1) + ks3))) + ks3*ks4*x2 + (((-1) + ks4) * (((-1) + ks4) <= (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0))))) + (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) * ((((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) < ((-1) + ks4)))), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (ks4*(((-1) + ks3) * (((-1) + ks3) <= (((0) * ((0) >= ((-2) + x1)) + ((-2) + x1) * (((-2) + x1) > (0))))) + (((0) * ((0) >= ((-2) + x1)) + ((-2) + x1) * (((-2) + x1) > (0)))) * ((((0) * ((0) >= ((-2) + x1)) + ((-2) + x1) * (((-2) + x1) > (0)))) < ((-1) + ks3))) + ks3*ks4*x2 + ks3*ks4*(ks5 // 2) + (((-1) + ks4) * (((-1) + ks4) <= (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0))))) + (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) * ((((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) < ((-1) + ks4)))), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = libdevice.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tmp10 = libdevice.sqrt(tmp9)
    tmp11 = 1e-08
    tmp12 = triton_helpers.maximum(tmp10, tmp11)
    tmp13 = tmp8 / tmp12
    tmp15 = tmp14 * tmp1
    tmp16 = tmp14 * tmp3
    tmp17 = libdevice.erf(tmp16)
    tmp18 = tmp17 + tmp6
    tmp19 = tmp15 * tmp18
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = triton_helpers.maximum(tmp21, tmp11)
    tmp23 = tmp19 / tmp22
    tmp24 = tmp13 * tmp23
    tl.store(out_ptr0 + (x4), tmp24, xmask)




# kernel path: /tmp/torchinductor_sahanp/t5/ct5cjooa5jfrodxukwwuisq6tqbtpscezsm6ihjds6pgszvbybe3.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.sum]
# Source node to ATen node mapping:
#   x_2 => sum_3
# Graph fragment:
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_128, [1]), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_sum_3(in_ptr0, out_ptr0, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 16*r0_1 + 4*ks0*r0_1 + 4*ks1*r0_1 + ks0*ks1*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)




# kernel path: /tmp/torchinductor_sahanp/6b/c6b5q6oa3xwmkvwxoigpqjspp4gwt3i2o5o3rvxop44tzekp72jy.py
# Topologically Sorted Source Nodes: [loss, randint, target], Original ATen: [aten.exp, aten.randint, aten._to_copy, aten.mul, aten.sub, aten.mean]
# Source node to ATen node mapping:
#   loss => exp, mean, mul_143, sub_68
#   randint => inductor_lookup_seed_default, inductor_randint_default
#   target => convert_element_type
# Graph fragment:
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%view,), kwargs = {})
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_randint_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_randint.default](args = (0, 10, [1, %mul_138], %inductor_lookup_seed_default), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%inductor_randint_default, torch.float32), kwargs = {})
#   %mul_143 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %view), kwargs = {})
#   %sub_68 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%exp, %mul_143), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sub_68,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__to_copy_exp_mean_mul_randint_sub_4(in_out_ptr0, in_ptr0, in_ptr1, load_seed_offset, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp11 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl_math.exp(tmp0)
        tmp2 = tl.load(in_ptr1 + load_seed_offset)
        tmp3 = r0_0
        tmp4 = tl.full([1, 1], 0, tl.int64)
        tmp5 = tl.full([1, 1], 10, tl.int64)
        tmp6 = triton_helpers.randint64(tmp2, (tmp3).to(tl.uint32), tmp4, tmp5)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp7 * tmp0
        tmp9 = tmp1 - tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(r0_mask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tmp13 = 16 + 4*ks1 + 4*ks2 + ks1*ks2
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp11 / tmp14
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp15, None)







def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        ps0 = 4 + s2
        buf0 = empty_strided_cuda((1, 1, 4 + s1, 4 + s2), (16 + 4*s1 + 4*s2 + s1*s2, 16 + 4*s1 + 4*s2 + s1*s2, 4 + s2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_0_xnumel = 16 + 4*s1 + 4*s2 + s1*s2
        triton_red_fused_linalg_vector_norm_0_r0_numel = s0 // 2
        stream0 = get_raw_stream(0)
        triton_red_fused_linalg_vector_norm_0[grid(triton_red_fused_linalg_vector_norm_0_xnumel)](arg3_1, buf0, 36, 32, 32, 1296, 2, XBLOCK=64, R0_BLOCK=2, num_warps=8, num_stages=1)
        buf1 = empty_strided_cuda((1, 1, 4 + s1, 4 + s2), (16 + 4*s1 + 4*s2 + s1*s2, 16 + 4*s1 + 4*s2 + s1*s2, 4 + s2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_1_xnumel = 16 + 4*s1 + 4*s2 + s1*s2
        triton_red_fused_linalg_vector_norm_1_r0_numel = s0 + ((-1)*(s0 // 2))
        stream0 = get_raw_stream(0)
        triton_red_fused_linalg_vector_norm_1[grid(triton_red_fused_linalg_vector_norm_1_xnumel)](arg3_1, buf1, 36, 4, 32, 32, 1296, 2, XBLOCK=64, R0_BLOCK=2, num_warps=2, num_stages=1)
        ps1 = 4 + s1
        ps2 = 16 + 4*s1 + 4*s2 + s1*s2
        buf2 = empty_strided_cuda((1, s0 // 2, 4 + s1, 4 + s2), (16*(s0 // 2) + 4*s1*(s0 // 2) + 4*s2*(s0 // 2) + s1*s2*(s0 // 2), 16 + 4*s1 + 4*s2 + s1*s2, 4 + s2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.linalg_vector_norm, aten.clamp_min, aten.div, aten.mul]
        triton_poi_fused_clamp_min_div_linalg_vector_norm_mul_2_xnumel = 16*(s0 // 2) + 4*s1*(s0 // 2) + 4*s2*(s0 // 2) + s1*s2*(s0 // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_clamp_min_div_linalg_vector_norm_mul_2[grid(triton_poi_fused_clamp_min_div_linalg_vector_norm_mul_2_xnumel)](arg3_1, buf0, buf1, buf2, 36, 36, 1296, 32, 32, 4, 2592, XBLOCK=128, num_warps=4, num_stages=1)
        del arg3_1
        del buf0
        buf3 = reinterpret_tensor(buf1, (1, 4 + s1, 4 + s2), (16 + 4*s1 + 4*s2 + s1*s2, 4 + s2, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.sum]
        triton_red_fused_sum_3_xnumel = 16 + 4*s1 + 4*s2 + s1*s2
        triton_red_fused_sum_3_r0_numel = s0 // 2
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_3[grid(triton_red_fused_sum_3_xnumel)](buf2, buf3, 32, 32, 1296, 2, XBLOCK=256, R0_BLOCK=2, num_warps=4, num_stages=1)
        del buf2
        buf4 = empty_strided_cuda((1, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf4)
        buf5 = empty_strided_cuda((), (), torch.float32)
        buf6 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [loss, randint, target], Original ATen: [aten.exp, aten.randint, aten._to_copy, aten.mul, aten.sub, aten.mean]
        triton_red_fused__to_copy_exp_mean_mul_randint_sub_4_r0_numel = 16 + 4*s1 + 4*s2 + s1*s2
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_exp_mean_mul_randint_sub_4[grid(1)](buf6, buf3, buf4, 0, 32, 32, 1, 1296, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del buf3
        del buf4
    return (buf6, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 4
    arg1_1 = 32
    arg2_1 = 32
    arg3_1 = rand_strided((1, 4, 32, 32), (4096, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
