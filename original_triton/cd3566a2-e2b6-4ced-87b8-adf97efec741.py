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


# kernel path: /tmp/torchinductor_sahanp/pk/cpkepzw2a4cm3kopsfuuky43s2zuej7aah2danyc23jkvla73i46.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.copy]
# Source node to ATen node mapping:
#   x_2 => copy
# Graph fragment:
#   %copy : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_1, %slice_2), kwargs = {})
#   %slice_scatter_default : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%empty, %copy, 1, 2, %sub_6), kwargs = {})
#   %slice_scatter_default_1 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default, %slice_7, 1, 0, 2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_copy_0(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0 + 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3
    tmp4 = tl.full([1], 2, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tl.broadcast_to(2 + 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3, [XBLOCK])
    tmp7 = tmp3 < tmp6
    tmp8 = tmp5 & tmp7
    tmp9 = tmp8 & tmp2
    tmp10 = (-1) + (((((-2) + x0 + 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // (4 + 2*ks2 + 2*ks3 + ks2*ks3)) % (2 + ks1)))
    tmp11 = tl.full([1], 0, tl.int64)
    tmp12 = tmp10 >= tmp11
    tmp13 = tl.broadcast_to(ks1, [XBLOCK])
    tmp14 = tmp10 < tmp13
    tmp15 = (-1) + (((((-2) + x0 + 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // (2 + ks3)) % (2 + ks2)))
    tmp16 = tmp15 >= tmp11
    tmp17 = tl.broadcast_to(ks2, [XBLOCK])
    tmp18 = tmp15 < tmp17
    tmp19 = (-1) + ((((-2) + x0 + 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) % (2 + ks3)))
    tmp20 = tmp19 >= tmp11
    tmp21 = tl.broadcast_to(ks3, [XBLOCK])
    tmp22 = tmp19 < tmp21
    tmp23 = tmp12 & tmp14
    tmp24 = tmp23 & tmp16
    tmp25 = tmp24 & tmp18
    tmp26 = tmp25 & tmp20
    tmp27 = tmp26 & tmp22
    tmp28 = tmp27 & tmp9
    tmp29 = tl.load(in_ptr0 + ((-1) + ((-1)*ks3) + ks3*(((((-2) + x0 + 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // (2 + ks3)) % (2 + ks2))) + ((-1)*ks2*ks3) + ks2*ks3*(((((-2) + x0 + 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // (4 + 2*ks2 + 2*ks3 + ks2*ks3)) % (2 + ks1))) + ks1*ks2*ks3*(((((-2) + x0 + 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // (8 + 4*ks1 + 4*ks2 + 4*ks3 + 2*ks1*ks2 + 2*ks1*ks3 + 2*ks2*ks3 + ks1*ks2*ks3)) % ks0)) + ((((-2) + x0 + 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) % (2 + ks3)))), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.full(tmp29.shape, 0.0, tmp29.dtype)
    tmp31 = tl.where(tmp9, tmp29, tmp30)
    tmp32 = float("nan")
    tmp33 = tl.where(tmp8, tmp31, tmp32)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp2, tmp33, tmp34)
    tmp36 = tmp0 >= tmp1
    tmp37 = 2 + 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3
    tmp38 = tmp0 < tmp37
    tmp39 = tmp36 & tmp38
    tmp40 = (-1) + (((((-2) + x0) // (4 + 2*ks2 + 2*ks3 + ks2*ks3)) % (2 + ks1)))
    tmp41 = tl.full([1], 0, tl.int64)
    tmp42 = tmp40 >= tmp41
    tmp43 = tl.broadcast_to(ks1, [XBLOCK])
    tmp44 = tmp40 < tmp43
    tmp45 = (-1) + (((((-2) + x0) // (2 + ks3)) % (2 + ks2)))
    tmp46 = tmp45 >= tmp41
    tmp47 = tl.broadcast_to(ks2, [XBLOCK])
    tmp48 = tmp45 < tmp47
    tmp49 = (-1) + ((((-2) + x0) % (2 + ks3)))
    tmp50 = tmp49 >= tmp41
    tmp51 = tl.broadcast_to(ks3, [XBLOCK])
    tmp52 = tmp49 < tmp51
    tmp53 = tmp42 & tmp44
    tmp54 = tmp53 & tmp46
    tmp55 = tmp54 & tmp48
    tmp56 = tmp55 & tmp50
    tmp57 = tmp56 & tmp52
    tmp58 = tmp57 & tmp39
    tmp59 = tl.load(in_ptr0 + ((-1) + ((-1)*ks3) + ks3*(((((-2) + x0) // (2 + ks3)) % (2 + ks2))) + ((-1)*ks2*ks3) + ks2*ks3*(((((-2) + x0) // (4 + 2*ks2 + 2*ks3 + ks2*ks3)) % (2 + ks1))) + ks1*ks2*ks3*(((((-2) + x0) // (8 + 4*ks1 + 4*ks2 + 4*ks3 + 2*ks1*ks2 + 2*ks1*ks3 + 2*ks2*ks3 + ks1*ks2*ks3)) % ks0)) + ((((-2) + x0) % (2 + ks3)))), tmp58 & xmask, eviction_policy='evict_last', other=0.0)
    tmp60 = tl.full(tmp59.shape, 0.0, tmp59.dtype)
    tmp61 = tl.where(tmp39, tmp59, tmp60)
    tmp62 = float("nan")
    tmp63 = tl.where(tmp39, tmp61, tmp62)
    tmp64 = tl.where(tmp2, tmp35, tmp63)
    tl.store(out_ptr0 + (x0), tmp64, xmask)




# kernel path: /tmp/torchinductor_sahanp/ie/ciei5azclx2c6znwm2g5xxaj4eechkkkp7a3gwzrmvjrok7noaaq.py
# Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.bernoulli]
# Source node to ATen node mapping:
#   x_4 => inductor_lookup_seed_default, inductor_random_default
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 1, 1, 1], %inductor_lookup_seed_default, rand), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_bernoulli_1(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp2, None)




# kernel path: /tmp/torchinductor_sahanp/u7/cu7iss2boyaan24wda7gzivla7h35soe6jyiqyu74jopydvugxxn.py
# Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.bernoulli, aten._to_copy, aten.div, aten.mul]
# Source node to ATen node mapping:
#   x_4 => convert_element_type, div, lt, mul_39
# Graph fragment:
#   %lt : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%inductor_random_default, 0.5), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt, torch.float32), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%convert_element_type, 0.5), kwargs = {})
#   %mul_39 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_2, %div), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_bernoulli_div_mul_2(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp4 = tl.load(in_ptr0 + (x0), xmask)
    tmp6 = tl.load(in_ptr1 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp0 = x0
    tmp1 = 2 + 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (x0 + ((-8)*ks0) + ((-4)*ks0*ks1) + ((-4)*ks0*ks2) + ((-4)*ks0*ks3) + ((-2)*ks0*ks1*ks2) + ((-2)*ks0*ks1*ks3) + ((-2)*ks0*ks2*ks3) + ((-1)*ks0*ks1*ks2*ks3)), tmp2 & xmask, other=0.0)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp8 = 0.5
    tmp9 = tmp7 < tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = 2.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp5 * tmp12
    tl.store(out_ptr0 + (x0), tmp13, xmask)




# kernel path: /tmp/torchinductor_sahanp/7q/c7qojqsmkkjkidmkddu2zxy5kvodt3tjgnl2udvrcsvzkbzfmki4.py
# Topologically Sorted Source Nodes: [x_9, x_8], Original ATen: [aten.log_sigmoid_forward, aten.replication_pad3d]
# Source node to ATen node mapping:
#   x_8 => _unsafe_index, _unsafe_index_1, _unsafe_index_2
#   x_9 => abs_1, exp, full_default, log1p, minimum, neg, sub_30
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%view_4, [None, %clamp_max, None, None]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index, [None, None, %clamp_max_1, None]), kwargs = {})
#   %_unsafe_index_2 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_1, [None, None, None, %clamp_max_2]), kwargs = {})
#   %minimum : [num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%full_default, %_unsafe_index_2), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%_unsafe_index_2,), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%abs_1,), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %log1p : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp,), kwargs = {})
#   %sub_30 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%minimum, %log1p), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_log_sigmoid_forward_replication_pad3d_3(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*((((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) * ((((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) <= (1 + 4*ks1 + 2*ks1*ks2 + 2*ks1*ks3 + 2*ks1*ks4 + ks1*ks2*ks3 + ks1*ks2*ks4 + ks1*ks3*ks4 + ((ks1*ks2*ks3*ks4) // 2))) + (1 + 4*ks1 + 2*ks1*ks2 + 2*ks1*ks3 + 2*ks1*ks4 + ks1*ks2*ks3 + ks1*ks2*ks4 + ks1*ks3*ks4 + ((ks1*ks2*ks3*ks4) // 2)) * ((1 + 4*ks1 + 2*ks1*ks2 + 2*ks1*ks3 + 2*ks1*ks4 + ks1*ks2*ks3 + ks1*ks2*ks4 + ks1*ks3*ks4 + ((ks1*ks2*ks3*ks4) // 2)) < (((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0))))))), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*((((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) * ((((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) <= (1 + 4*ks1 + 2*ks1*ks2 + 2*ks1*ks3 + 2*ks1*ks4 + ks1*ks2*ks3 + ks1*ks2*ks4 + ks1*ks3*ks4 + ((ks1*ks2*ks3*ks4) // 2))) + (1 + 4*ks1 + 2*ks1*ks2 + 2*ks1*ks3 + 2*ks1*ks4 + ks1*ks2*ks3 + ks1*ks2*ks4 + ks1*ks3*ks4 + ((ks1*ks2*ks3*ks4) // 2)) * ((1 + 4*ks1 + 2*ks1*ks2 + 2*ks1*ks3 + 2*ks1*ks4 + ks1*ks2*ks3 + ks1*ks2*ks4 + ks1*ks3*ks4 + ((ks1*ks2*ks3*ks4) // 2)) < (((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0))))))), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp3 = 0.0
    tmp4 = triton_helpers.minimum(tmp3, tmp2)
    tmp5 = tl_math.abs(tmp2)
    tmp6 = -tmp5
    tmp7 = tl_math.exp(tmp6)
    tmp8 = libdevice.log1p(tmp7)
    tmp9 = tmp4 - tmp8
    tl.store(out_ptr0 + (x2), tmp9, xmask)







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
        buf1 = empty_strided_cuda((1, 4 + 8*s0 + 4*s0*s1 + 4*s0*s2 + 4*s0*s3 + 2*s0*s1*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + s0*s1*s2*s3), (4 + 8*s0 + 4*s0*s1 + 4*s0*s2 + 4*s0*s3 + 2*s0*s1*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + s0*s1*s2*s3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.copy]
        triton_poi_fused_copy_0_xnumel = 4 + 8*s0 + 4*s0*s1 + 4*s0*s2 + 4*s0*s3 + 2*s0*s1*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + s0*s1*s2*s3
        stream0 = get_raw_stream(0)
        triton_poi_fused_copy_0[grid(triton_poi_fused_copy_0_xnumel)](arg4_1, buf1, 3, 32, 32, 32, 117916, XBLOCK=512, num_warps=8, num_stages=1)
        del arg4_1
        buf2 = empty_strided_cuda((1, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf2)
        buf3 = empty_strided_cuda((1, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.bernoulli]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bernoulli_1[grid(1)](buf2, buf3, 0, 1, XBLOCK=1, num_warps=1, num_stages=1)
        del buf2
        buf4 = empty_strided_cuda((1, 1, 1, 4 + 8*s0 + 4*s0*s1 + 4*s0*s2 + 4*s0*s3 + 2*s0*s1*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + s0*s1*s2*s3), (4 + 8*s0 + 4*s0*s1 + 4*s0*s2 + 4*s0*s3 + 2*s0*s1*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + s0*s1*s2*s3, 4 + 8*s0 + 4*s0*s1 + 4*s0*s2 + 4*s0*s3 + 2*s0*s1*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + s0*s1*s2*s3, 4 + 8*s0 + 4*s0*s1 + 4*s0*s2 + 4*s0*s3 + 2*s0*s1*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + s0*s1*s2*s3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.bernoulli, aten._to_copy, aten.div, aten.mul]
        triton_poi_fused__to_copy_bernoulli_div_mul_2_xnumel = 4 + 8*s0 + 4*s0*s1 + 4*s0*s2 + 4*s0*s3 + 2*s0*s1*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + s0*s1*s2*s3
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_bernoulli_div_mul_2[grid(triton_poi_fused__to_copy_bernoulli_div_mul_2_xnumel)](buf1, buf3, buf4, 3, 32, 32, 32, 117916, XBLOCK=512, num_warps=8, num_stages=1)
        del buf1
        del buf3
        ps0 = 4 + 4*s0 + 2*s0*s1 + 2*s0*s2 + 2*s0*s3 + s0*s1*s2 + s0*s1*s3 + s0*s2*s3 + ((s0*s1*s2*s3) // 2)
        buf5 = empty_strided_cuda((1, 3, 3, 4 + 4*s0 + 2*s0*s1 + 2*s0*s2 + 2*s0*s3 + s0*s1*s2 + s0*s1*s3 + s0*s2*s3 + ((s0*s1*s2*s3) // 2)), (36 + 9*((s0*s1*s2*s3) // 2) + 36*s0 + 18*s0*s1 + 18*s0*s2 + 18*s0*s3 + 9*s0*s1*s2 + 9*s0*s1*s3 + 9*s0*s2*s3, 12 + 3*((s0*s1*s2*s3) // 2) + 12*s0 + 6*s0*s1 + 6*s0*s2 + 6*s0*s3 + 3*s0*s1*s2 + 3*s0*s1*s3 + 3*s0*s2*s3, 4 + 4*s0 + 2*s0*s1 + 2*s0*s2 + 2*s0*s3 + s0*s1*s2 + s0*s1*s3 + s0*s2*s3 + ((s0*s1*s2*s3) // 2), 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_9, x_8], Original ATen: [aten.log_sigmoid_forward, aten.replication_pad3d]
        triton_poi_fused_log_sigmoid_forward_replication_pad3d_3_xnumel = 36 + 9*((s0*s1*s2*s3) // 2) + 36*s0 + 18*s0*s1 + 18*s0*s2 + 18*s0*s3 + 9*s0*s1*s2 + 9*s0*s1*s3 + 9*s0*s2*s3
        stream0 = get_raw_stream(0)
        triton_poi_fused_log_sigmoid_forward_replication_pad3d_3[grid(triton_poi_fused_log_sigmoid_forward_replication_pad3d_3_xnumel)](buf4, buf5, 58960, 3, 32, 32, 32, 530640, XBLOCK=512, num_warps=8, num_stages=1)
        del buf4
    return (buf5, )


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
