# AOT ID: ['38_inference']
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


# kernel path: /tmp/torchinductor_sahanp/jx/cjxf4slrqcg7dl57756dngqttdafkczmz4ho734xuh4xmanff5pp.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.native_dropout]
# Source node to ATen node mapping:
#   x_2 => inductor_lookup_seed_default_1, inductor_random_default_1
# Graph fragment:
#   %inductor_lookup_seed_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 1), kwargs = {})
#   %inductor_random_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, %arg0_1, %mul_11, %mul_12], %inductor_lookup_seed_default_1, rand), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_dropout_0(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (x0), tmp2, xmask)




# kernel path: /tmp/torchinductor_sahanp/hh/chhautw3h7hcoqqi2vdgnlzhm2akcm5ckpjxnpkv6aqtyaj26mti.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.native_dropout]
# Source node to ATen node mapping:
#   x => inductor_lookup_seed_default, inductor_random_default_2
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default_2 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, %arg0_1, %arg1_1, %arg2_1], %inductor_lookup_seed_default, rand), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_dropout_1(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (x0), tmp2, xmask)




# kernel path: /tmp/torchinductor_sahanp/3d/c3dlht5wzrnsrm763m7sfz3usvntta4x5dlntkq2s44ey7zsrz5o.py
# Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.native_dropout, aten._unsafe_index]
# Source node to ATen node mapping:
#   x => gt_3, mul_2, mul_3
#   x_1 => _unsafe_index
# Graph fragment:
#   %gt_3 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%inductor_random_default_2, 0.5), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gt_3, %arg3_1), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, 2.0), kwargs = {})
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%mul_3, [None, None, %unsqueeze, %convert_element_type_3]), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__unsafe_index_native_dropout_2(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // ks1) % ks2)
    x0 = (xindex % ks1)
    x2 = xindex // ks4
    x4 = xindex
    tmp0 = tl.full([1], 2.0, tl.float64)
    tmp1 = ks0
    tmp2 = tmp1.to(tl.float64)
    tmp3 = tmp0 * tmp2
    tmp4 = tmp2 / tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = x1
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp7 * tmp5
    tmp9 = tmp8.to(tl.int64)
    tmp10 = tmp9 + tmp1
    tmp11 = tmp9 < 0
    tmp12 = tl.where(tmp11, tmp10, tmp9)
    tmp13 = ks3
    tmp14 = tmp13.to(tl.float64)
    tmp15 = tmp0 * tmp14
    tmp16 = tmp14 / tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = x0
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp19 * tmp17
    tmp21 = tmp20.to(tl.int64)
    tmp22 = tmp21 + tmp13
    tmp23 = tmp21 < 0
    tmp24 = tl.where(tmp23, tmp22, tmp21)
    tmp25 = tl.load(in_ptr0 + (tmp24 + ks3*tmp12 + ks0*ks3*x2), xmask, eviction_policy='evict_last')
    tmp26 = 0.5
    tmp27 = tmp25 > tmp26
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tl.load(in_ptr1 + (tmp24 + ks3*tmp12 + ks0*ks3*x2), xmask, eviction_policy='evict_last')
    tmp30 = tmp28 * tmp29
    tmp31 = 2.0
    tmp32 = tmp30 * tmp31
    tl.store(out_ptr0 + (x4), tmp32, xmask)




# kernel path: /tmp/torchinductor_sahanp/yu/cyusykrrm3cqg7lil5lhaz242ds6evp4gevyfrib2rstk2pculo3.py
# Topologically Sorted Source Nodes: [x_2, x_3], Original ATen: [aten.native_dropout, aten._unsafe_index]
# Source node to ATen node mapping:
#   x_2 => gt_9, mul_39, mul_40
#   x_3 => _unsafe_index_1
# Graph fragment:
#   %gt_9 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%inductor_random_default_1, 0.5), kwargs = {})
#   %mul_39 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gt_9, %_unsafe_index), kwargs = {})
#   %mul_40 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_39, 2.0), kwargs = {})
#   %_unsafe_index_1 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%mul_40, [None, None, %unsqueeze_1, %convert_element_type_7]), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__unsafe_index_native_dropout_3(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, ks6, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // ks1) % ks2)
    x0 = (xindex % ks1)
    x2 = xindex // ks6
    x3 = xindex
    tmp0 = 2.0
    tmp1 = ks0
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 * tmp2
    tmp4 = tmp3.to(tl.float64)
    tmp5 = tl.full([1], 2.0, tl.float64)
    tmp6 = tmp5 * tmp4
    tmp7 = tmp4 / tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = x1
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp10 * tmp8
    tmp12 = tmp11.to(tl.int64)
    tmp13 = ks3
    tmp14 = tmp12 + tmp13
    tmp15 = tmp12 < 0
    tmp16 = tl.where(tmp15, tmp14, tmp12)
    tmp17 = ks4
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp0 * tmp18
    tmp20 = tmp19.to(tl.float64)
    tmp21 = tmp5 * tmp20
    tmp22 = tmp20 / tmp21
    tmp23 = tmp22.to(tl.float32)
    tmp24 = x0
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp25 * tmp23
    tmp27 = tmp26.to(tl.int64)
    tmp28 = ks5
    tmp29 = tmp27 + tmp28
    tmp30 = tmp27 < 0
    tmp31 = tl.where(tmp30, tmp29, tmp27)
    tmp32 = tl.load(in_ptr0 + (tmp31 + 2*ks4*tmp16 + 4*ks0*ks4*x2), xmask, eviction_policy='evict_last')
    tmp33 = 0.5
    tmp34 = tmp32 > tmp33
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tl.load(in_ptr1 + (tmp31 + 2*ks4*tmp16 + 4*ks0*ks4*x2), xmask, eviction_policy='evict_last')
    tmp37 = tmp35 * tmp36
    tmp38 = tmp37 * tmp0
    tl.store(out_ptr0 + (x3), tmp38, xmask)




# kernel path: /tmp/torchinductor_sahanp/gc/cgchlc4exrlsdnaop3i46td6dtloadbpgde6wgv23dde6tm4kaoi.py
# Topologically Sorted Source Nodes: [x_4], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   x_4 => amax, exp, sub_44, sum_1
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%_unsafe_index_1, [1], True), kwargs = {})
#   %sub_44 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_1, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_44,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__softmax_4(in_ptr0, out_ptr0, out_ptr1, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    R0_BLOCK: tl.constexpr = 128
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
    tmp0 = tl.load(in_ptr0 + (x0 + 16*ks0*ks1*r0_1), r0_mask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tmp5 = tmp0 - tmp4
    tmp6 = tl_math.exp(tmp5)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(r0_mask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr1 + (x0), tmp10, xmask)




# kernel path: /tmp/torchinductor_sahanp/yu/cyu7emigirhgz3tdqklsokopr6cstevk42htbph67x7fnsenhilm.py
# Topologically Sorted Source Nodes: [x_4], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   x_4 => div, exp, sub_44
# Graph fragment:
#   %sub_44 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_1, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_44,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__softmax_5(in_out_ptr0, in_ptr0, in_ptr1, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % ks0)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp3 = tl_math.exp(tmp2)
    tmp5 = tmp3 / tmp4
    tl.store(in_out_ptr0 + (x2), tmp5, xmask)




# kernel path: /tmp/torchinductor_sahanp/yq/cyqzxhlgntvtxboorwqr6l4op4abvneq4dlqpk7waonmca2553mw.py
# Topologically Sorted Source Nodes: [target], Original ATen: [aten.randn_like]
# Source node to ATen node mapping:
#   target => inductor_lookup_seed_default_2, inductor_random_default
# Graph fragment:
#   %inductor_lookup_seed_default_2 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 2), kwargs = {})
#   %inductor_random_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, %arg0_1, %mul_48, %mul_49], %inductor_lookup_seed_default_2, randn), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_randn_like_6(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.randn(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (x0), tmp2, xmask)




# kernel path: /tmp/torchinductor_sahanp/7e/c7el6fhzjcq5626mfbdlpr4usnc4uavyuuhowgfdzyvhytacn5ae.py
# Topologically Sorted Source Nodes: [var], Original ATen: [aten.ones_like]
# Source node to ATen node mapping:
#   var => full_default_6
# Graph fragment:
#   %full_default_6 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, %arg0_1, %mul_48, %mul_49], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_ones_like_7(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 1.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)







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
        buf1 = empty_strided_cuda((1, s0, 2*s1, 2*s2), (4*s0*s1*s2, 4*s1*s2, 2*s2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.native_dropout]
        triton_poi_fused_native_dropout_0_xnumel = 4*s0*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_dropout_0[grid(triton_poi_fused_native_dropout_0_xnumel)](buf0, buf1, 1, 12288, XBLOCK=256, num_warps=4, num_stages=1)
        buf2 = empty_strided_cuda((1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.native_dropout]
        triton_poi_fused_native_dropout_1_xnumel = s0*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_dropout_1[grid(triton_poi_fused_native_dropout_1_xnumel)](buf0, buf2, 0, 3072, XBLOCK=256, num_warps=4, num_stages=1)
        ps0 = 2*s2
        ps1 = 2*s1
        ps2 = 4*s1*s2
        buf3 = empty_strided_cuda((1, s0, 2*s1, 2*s2), (4*s0*s1*s2, 4*s1*s2, 2*s2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.native_dropout, aten._unsafe_index]
        triton_poi_fused__unsafe_index_native_dropout_2_xnumel = 4*s0*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_native_dropout_2[grid(triton_poi_fused__unsafe_index_native_dropout_2_xnumel)](buf2, arg3_1, buf3, 32, 64, 64, 32, 4096, 12288, XBLOCK=256, num_warps=4, num_stages=1)
        del arg3_1
        del buf2
        ps3 = 4*s2
        ps4 = 4*s1
        ps5 = 16*s1*s2
        buf4 = empty_strided_cuda((1, s0, 4*s1, 4*s2), (16*s0*s1*s2, 16*s1*s2, 4*s2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2, x_3], Original ATen: [aten.native_dropout, aten._unsafe_index]
        triton_poi_fused__unsafe_index_native_dropout_3_xnumel = 16*s0*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_native_dropout_3[grid(triton_poi_fused__unsafe_index_native_dropout_3_xnumel)](buf1, buf3, buf4, 32, 128, 128, 64, 32, 64, 16384, 49152, XBLOCK=512, num_warps=4, num_stages=1)
        del buf1
        del buf3
        buf5 = empty_strided_cuda((1, 1, 4*s1, 4*s2), (16*s1*s2, 16*s1*s2, 4*s2, 1), torch.float32)
        buf6 = empty_strided_cuda((1, 1, 4*s1, 4*s2), (16*s1*s2, 16*s1*s2, 4*s2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4_xnumel = 16*s1*s2
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_4[grid(triton_per_fused__softmax_4_xnumel)](buf4, buf5, buf6, 32, 32, 16384, 3, XBLOCK=8, num_warps=2, num_stages=1)
        buf7 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_5_xnumel = 16*s0*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_5[grid(triton_poi_fused__softmax_5_xnumel)](buf7, buf5, buf6, 16384, 49152, XBLOCK=512, num_warps=4, num_stages=1)
        del buf5
        del buf6
        buf8 = empty_strided_cuda((1, s0, 4*s1, 4*s2), (16*s0*s1*s2, 16*s1*s2, 4*s2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [target], Original ATen: [aten.randn_like]
        triton_poi_fused_randn_like_6_xnumel = 16*s0*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused_randn_like_6[grid(triton_poi_fused_randn_like_6_xnumel)](buf0, buf8, 2, 49152, XBLOCK=256, num_warps=4, num_stages=1)
        del buf0
        buf9 = empty_strided_cuda((1, s0, 4*s1, 4*s2), (16*s0*s1*s2, 16*s1*s2, 4*s2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [var], Original ATen: [aten.ones_like]
        triton_poi_fused_ones_like_7_xnumel = 16*s0*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused_ones_like_7[grid(triton_poi_fused_ones_like_7_xnumel)](buf9, 49152, XBLOCK=256, num_warps=4, num_stages=1)
    return (buf7, buf8, buf9, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 3
    arg1_1 = 32
    arg2_1 = 32
    arg3_1 = rand_strided((1, 3, 32, 32), (3072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
