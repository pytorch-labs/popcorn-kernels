# AOT ID: ['16_inference']
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


# kernel path: /tmp/torchinductor_sahanp/nk/cnkxlv5no5ipydivnifk4qy6iielzt2z3jb7xwvowybcfuuli6tu.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.bernoulli]
# Source node to ATen node mapping:
#   x_3 => inductor_lookup_seed_default, inductor_random_default
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 12, 1], %inductor_lookup_seed_default, rand), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_bernoulli_0(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (x0), tmp2, xmask)




# kernel path: /tmp/torchinductor_sahanp/hm/chmewq3hzoe4352idmtchw7yd3m573wlwewjo7x7nj4c7viqbnck.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.bernoulli, aten._to_copy, aten.div, aten.mul]
# Source node to ATen node mapping:
#   x_3 => convert_element_type, div, lt, mul_1
# Graph fragment:
#   %lt : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%inductor_random_default, 0.5), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt, torch.float32), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%convert_element_type, 0.5), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_2, %div), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_bernoulli_div_mul_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 12)
    x1 = xindex // 12
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*((x1 % 32)) + 64*(((x0 // 2) % 2)) + 128*(x1 // 32) + 4096*(x0 // 4) + ((x0 % 2))), None)
    tmp8 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp1 = 20.0
    tmp2 = tmp0 > tmp1
    tmp3 = tl_math.exp(tmp0)
    tmp4 = libdevice.log1p(tmp3)
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp6 = libdevice.tanh(tmp5)
    tmp7 = tmp0 * tmp6
    tmp9 = 0.5
    tmp10 = tmp8 < tmp9
    tmp11 = tmp10.to(tl.float32)
    tmp12 = 2.0
    tmp13 = tmp11 * tmp12
    tmp14 = tmp7 * tmp13
    tl.store(out_ptr0 + (x2), tmp14, None)




# kernel path: /tmp/torchinductor_sahanp/tz/ctzhwi4nmjdafechiimrnranfity4gm7hn6ou7h2s7qkpbqwg2bm.py
# Topologically Sorted Source Nodes: [x_5, x_6, x_7, x_8], Original ATen: [aten._adaptive_avg_pool2d, aten._to_copy, aten.arange, aten.mul, aten.clamp, aten._unsafe_index, aten.sub, aten.add, aten.celu, aten.pow]
# Source node to ATen node mapping:
#   x_5 => _adaptive_avg_pool2d
#   x_6 => _unsafe_index, _unsafe_index_1, _unsafe_index_2, _unsafe_index_3, add_2, add_3, add_4, clamp_max_2, clamp_max_3, clamp_min_1, clamp_min_2, clamp_min_3, convert_element_type_2, convert_element_type_3, convert_element_type_4, iota_1, mul_3, mul_4, mul_5, mul_6, sub, sub_1, sub_2, sub_3, sub_4
#   x_7 => expm1, gt_1, where_1
#   x_8 => pow_1
# Graph fragment:
#   %_adaptive_avg_pool2d : [num_users=4] = call_function[target=torch.ops.aten._adaptive_avg_pool2d.default](args = (%view_3, [16, 16]), kwargs = {})
#   %convert_element_type_2 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_4, torch.int64), kwargs = {})
#   %iota_1 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (32,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_3 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_1, torch.float32), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_3, 0.4838709677419355), kwargs = {})
#   %clamp_min_1 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_3, 0.0), kwargs = {})
#   %convert_element_type_4 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_min_1, torch.int64), kwargs = {})
#   %_unsafe_index_3 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_adaptive_avg_pool2d, [None, None, %clamp_max, %clamp_max_1]), kwargs = {})
#   %_unsafe_index_2 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_adaptive_avg_pool2d, [None, None, %clamp_max, %convert_element_type_4]), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_3, %_unsafe_index_2), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_1, %convert_element_type_4), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub, 0.0), kwargs = {})
#   %clamp_max_2 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 1.0), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %clamp_max_2), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_2, %mul_5), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_adaptive_avg_pool2d, [None, None, %convert_element_type_2, %clamp_max_1]), kwargs = {})
#   %_unsafe_index : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_adaptive_avg_pool2d, [None, None, %convert_element_type_2, %convert_element_type_4]), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_1, %_unsafe_index), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %clamp_max_2), kwargs = {})
#   %add_2 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index, %mul_4), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %add_2), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_4, %convert_element_type_2), kwargs = {})
#   %clamp_min_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_3, 0.0), kwargs = {})
#   %clamp_max_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_3, 1.0), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %clamp_max_3), kwargs = {})
#   %add_4 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %mul_6), kwargs = {})
#   %gt_1 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_4, 0), kwargs = {})
#   %expm1 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%add_4,), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %add_4, %expm1), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%where_1, 2.0), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__adaptive_avg_pool2d__to_copy__unsafe_index_add_arange_celu_clamp_mul_pow_sub_2(in_out_ptr1, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 32) % 32)
    x0 = (xindex % 32)
    x2 = xindex // 1024
    x4 = xindex
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.4838709677419355
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full([1], 15, tl.int64)
    tmp10 = triton_helpers.minimum(tmp8, tmp9)
    tmp11 = x0
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp12 * tmp2
    tmp14 = triton_helpers.maximum(tmp13, tmp4)
    tmp15 = tmp14.to(tl.int32)
    tmp16 = tmp15 + tmp7
    tmp17 = triton_helpers.minimum(tmp16, tmp9)
    tmp18 = tl.load(in_ptr0 + (x2 + 24*tmp17 + 768*tmp10), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (12 + x2 + 24*tmp17 + 768*tmp10), None, eviction_policy='evict_last')
    tmp20 = tmp19 + tmp18
    tmp21 = tl.load(in_ptr0 + (384 + x2 + 24*tmp17 + 768*tmp10), None, eviction_policy='evict_last')
    tmp22 = tmp21 + tmp20
    tmp23 = tl.load(in_ptr0 + (396 + x2 + 24*tmp17 + 768*tmp10), None, eviction_policy='evict_last')
    tmp24 = tmp23 + tmp22
    tmp25 = 0.25
    tmp26 = tmp24 * tmp25
    tmp27 = tl.load(in_ptr0 + (x2 + 24*tmp15 + 768*tmp10), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr0 + (12 + x2 + 24*tmp15 + 768*tmp10), None, eviction_policy='evict_last')
    tmp29 = tmp28 + tmp27
    tmp30 = tl.load(in_ptr0 + (384 + x2 + 24*tmp15 + 768*tmp10), None, eviction_policy='evict_last')
    tmp31 = tmp30 + tmp29
    tmp32 = tl.load(in_ptr0 + (396 + x2 + 24*tmp15 + 768*tmp10), None, eviction_policy='evict_last')
    tmp33 = tmp32 + tmp31
    tmp34 = tmp33 * tmp25
    tmp35 = tmp26 - tmp34
    tmp36 = tl.load(in_ptr0 + (x2 + 24*tmp17 + 768*tmp6), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr0 + (12 + x2 + 24*tmp17 + 768*tmp6), None, eviction_policy='evict_last')
    tmp38 = tmp37 + tmp36
    tmp39 = tl.load(in_ptr0 + (384 + x2 + 24*tmp17 + 768*tmp6), None, eviction_policy='evict_last')
    tmp40 = tmp39 + tmp38
    tmp41 = tl.load(in_ptr0 + (396 + x2 + 24*tmp17 + 768*tmp6), None, eviction_policy='evict_last')
    tmp42 = tmp41 + tmp40
    tmp43 = tmp42 * tmp25
    tmp44 = tl.load(in_ptr0 + (x2 + 24*tmp15 + 768*tmp6), None, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr0 + (12 + x2 + 24*tmp15 + 768*tmp6), None, eviction_policy='evict_last')
    tmp46 = tmp45 + tmp44
    tmp47 = tl.load(in_ptr0 + (384 + x2 + 24*tmp15 + 768*tmp6), None, eviction_policy='evict_last')
    tmp48 = tmp47 + tmp46
    tmp49 = tl.load(in_ptr0 + (396 + x2 + 24*tmp15 + 768*tmp6), None, eviction_policy='evict_last')
    tmp50 = tmp49 + tmp48
    tmp51 = tmp50 * tmp25
    tmp52 = tmp43 - tmp51
    tmp53 = tmp15.to(tl.float32)
    tmp54 = tmp14 - tmp53
    tmp55 = triton_helpers.maximum(tmp54, tmp4)
    tmp56 = 1.0
    tmp57 = triton_helpers.minimum(tmp55, tmp56)
    tmp58 = tmp35 * tmp57
    tmp59 = tmp34 + tmp58
    tmp60 = tmp52 * tmp57
    tmp61 = tmp51 + tmp60
    tmp62 = tmp59 - tmp61
    tmp63 = tmp6.to(tl.float32)
    tmp64 = tmp5 - tmp63
    tmp65 = triton_helpers.maximum(tmp64, tmp4)
    tmp66 = triton_helpers.minimum(tmp65, tmp56)
    tmp67 = tmp62 * tmp66
    tmp68 = tmp61 + tmp67
    tmp69 = tmp68 > tmp4
    tmp70 = libdevice.expm1(tmp68)
    tmp71 = tl.where(tmp69, tmp68, tmp70)
    tmp72 = tmp71 * tmp71
    tl.store(in_out_ptr1 + (x4), tmp72, None)




# kernel path: /tmp/torchinductor_sahanp/2m/c2mwj2yo5f6bh4apjjagvjapnz4ayon7qesqdnkp4rcvlw3orq4s.py
# Topologically Sorted Source Nodes: [x_6, x_7, x_8, x_9, x_10], Original ATen: [aten._to_copy, aten.sub, aten.clamp, aten.mul, aten.add, aten.celu, aten.pow, aten.avg_pool2d, aten.sign, aten.abs, aten.relu]
# Source node to ATen node mapping:
#   x_10 => pow_3
#   x_6 => add_4, clamp_max_3, clamp_min_3, convert_element_type_2, mul_6, sub_3, sub_4
#   x_7 => expm1, gt_1, where_1
#   x_8 => abs_1, avg_pool2d, mul_7, mul_8, pow_1, pow_2, relu, sign
#   x_9 => expm1_1, gt_2, where_2
# Graph fragment:
#   %convert_element_type_2 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_4, torch.int64), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %add_2), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_4, %convert_element_type_2), kwargs = {})
#   %clamp_min_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_3, 0.0), kwargs = {})
#   %clamp_max_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_3, 1.0), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %clamp_max_3), kwargs = {})
#   %add_4 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %mul_6), kwargs = {})
#   %gt_1 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_4, 0), kwargs = {})
#   %expm1 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%add_4,), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %add_4, %expm1), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%where_1, 2.0), kwargs = {})
#   %avg_pool2d : [num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%pow_1, [2, 2], [2, 2]), kwargs = {})
#   %sign : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%avg_pool2d,), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%avg_pool2d,), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%abs_1,), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign, %relu), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, 4), kwargs = {})
#   %pow_2 : [num_users=3] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_8, 0.5), kwargs = {})
#   %gt_2 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%pow_2, 0), kwargs = {})
#   %expm1_1 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%pow_2,), kwargs = {})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %pow_2, %expm1_1), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%where_2, 2.0), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_abs_add_avg_pool2d_celu_clamp_mul_pow_relu_sign_sub_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 64*x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 64*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (32 + 2*x0 + 64*x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (33 + 2*x0 + 64*x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tmp9 = tl.full([1], 0, tl.int32)
    tmp10 = tmp9 < tmp8
    tmp11 = tmp10.to(tl.int8)
    tmp12 = tmp8 < tmp9
    tmp13 = tmp12.to(tl.int8)
    tmp14 = tmp11 - tmp13
    tmp15 = tmp14.to(tmp8.dtype)
    tmp16 = tl_math.abs(tmp8)
    tmp17 = triton_helpers.maximum(tmp9, tmp16)
    tmp18 = tmp15 * tmp17
    tmp19 = 4.0
    tmp20 = tmp18 * tmp19
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = 0.0
    tmp23 = tmp21 > tmp22
    tmp24 = libdevice.expm1(tmp21)
    tmp25 = tl.where(tmp23, tmp21, tmp24)
    tmp26 = tmp25 * tmp25
    tl.store(out_ptr0 + (x2), tmp26, xmask)




# kernel path: /tmp/torchinductor_sahanp/zs/czsidqby6il4cwwsmm6e33ej5sjcenh3yjm6q762tdn2qnznpnmx.py
# Topologically Sorted Source Nodes: [x_6, x_7, x_8, x_9, x_10, x_11, x_12], Original ATen: [aten._to_copy, aten.sub, aten.clamp, aten.mul, aten.add, aten.celu, aten.pow, aten.avg_pool2d, aten.sign, aten.abs, aten.relu]
# Source node to ATen node mapping:
#   x_10 => abs_2, avg_pool2d_1, mul_10, mul_9, pow_3, pow_4, relu_1, sign_1
#   x_11 => expm1_2, gt_3, where_3
#   x_12 => pow_5
#   x_6 => add_4, clamp_max_3, clamp_min_3, convert_element_type_2, mul_6, sub_3, sub_4
#   x_7 => expm1, gt_1, where_1
#   x_8 => abs_1, avg_pool2d, mul_7, mul_8, pow_1, pow_2, relu, sign
#   x_9 => expm1_1, gt_2, where_2
# Graph fragment:
#   %convert_element_type_2 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_4, torch.int64), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %add_2), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_4, %convert_element_type_2), kwargs = {})
#   %clamp_min_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_3, 0.0), kwargs = {})
#   %clamp_max_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_3, 1.0), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %clamp_max_3), kwargs = {})
#   %add_4 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %mul_6), kwargs = {})
#   %gt_1 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_4, 0), kwargs = {})
#   %expm1 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%add_4,), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %add_4, %expm1), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%where_1, 2.0), kwargs = {})
#   %avg_pool2d : [num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%pow_1, [2, 2], [2, 2]), kwargs = {})
#   %sign : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%avg_pool2d,), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%avg_pool2d,), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%abs_1,), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign, %relu), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, 4), kwargs = {})
#   %pow_2 : [num_users=3] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_8, 0.5), kwargs = {})
#   %gt_2 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%pow_2, 0), kwargs = {})
#   %expm1_1 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%pow_2,), kwargs = {})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %pow_2, %expm1_1), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%where_2, 2.0), kwargs = {})
#   %avg_pool2d_1 : [num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%pow_3, [2, 2], [2, 2]), kwargs = {})
#   %sign_1 : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%avg_pool2d_1,), kwargs = {})
#   %abs_2 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%avg_pool2d_1,), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%abs_2,), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign_1, %relu_1), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, 4), kwargs = {})
#   %pow_4 : [num_users=3] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_10, 0.5), kwargs = {})
#   %gt_3 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%pow_4, 0), kwargs = {})
#   %expm1_2 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%pow_4,), kwargs = {})
#   %where_3 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_3, %pow_4, %expm1_2), kwargs = {})
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%where_3, 2.0), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_abs_add_avg_pool2d_celu_clamp_mul_pow_relu_sign_sub_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 8)
    x1 = xindex // 8
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 32*x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 32*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (16 + 2*x0 + 32*x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (17 + 2*x0 + 32*x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tmp9 = tl.full([1], 0, tl.int32)
    tmp10 = tmp9 < tmp8
    tmp11 = tmp10.to(tl.int8)
    tmp12 = tmp8 < tmp9
    tmp13 = tmp12.to(tl.int8)
    tmp14 = tmp11 - tmp13
    tmp15 = tmp14.to(tmp8.dtype)
    tmp16 = tl_math.abs(tmp8)
    tmp17 = triton_helpers.maximum(tmp9, tmp16)
    tmp18 = tmp15 * tmp17
    tmp19 = 4.0
    tmp20 = tmp18 * tmp19
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = 0.0
    tmp23 = tmp21 > tmp22
    tmp24 = libdevice.expm1(tmp21)
    tmp25 = tl.where(tmp23, tmp21, tmp24)
    tmp26 = tmp25 * tmp25
    tl.store(out_ptr0 + (x2), tmp26, xmask)




# kernel path: /tmp/torchinductor_sahanp/7e/c7ebobhq6tb2j3uvomse6odq24oeaq2mqmd3wkcjpvhqeovujads.py
# Topologically Sorted Source Nodes: [x_6, x_7, x_8, x_9, x_10, x_11, x_12], Original ATen: [aten._to_copy, aten.sub, aten.clamp, aten.mul, aten.add, aten.celu, aten.pow, aten.avg_pool2d, aten.sign, aten.abs, aten.relu]
# Source node to ATen node mapping:
#   x_10 => abs_2, avg_pool2d_1, mul_10, mul_9, pow_3, pow_4, relu_1, sign_1
#   x_11 => expm1_2, gt_3, where_3
#   x_12 => abs_3, avg_pool2d_2, mul_11, mul_12, pow_5, pow_6, relu_2, sign_2
#   x_6 => add_4, clamp_max_3, clamp_min_3, convert_element_type_2, mul_6, sub_3, sub_4
#   x_7 => expm1, gt_1, where_1
#   x_8 => abs_1, avg_pool2d, mul_7, mul_8, pow_1, pow_2, relu, sign
#   x_9 => expm1_1, gt_2, where_2
# Graph fragment:
#   %convert_element_type_2 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_4, torch.int64), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %add_2), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_4, %convert_element_type_2), kwargs = {})
#   %clamp_min_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_3, 0.0), kwargs = {})
#   %clamp_max_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_3, 1.0), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %clamp_max_3), kwargs = {})
#   %add_4 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %mul_6), kwargs = {})
#   %gt_1 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_4, 0), kwargs = {})
#   %expm1 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%add_4,), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %add_4, %expm1), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%where_1, 2.0), kwargs = {})
#   %avg_pool2d : [num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%pow_1, [2, 2], [2, 2]), kwargs = {})
#   %sign : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%avg_pool2d,), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%avg_pool2d,), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%abs_1,), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign, %relu), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, 4), kwargs = {})
#   %pow_2 : [num_users=3] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_8, 0.5), kwargs = {})
#   %gt_2 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%pow_2, 0), kwargs = {})
#   %expm1_1 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%pow_2,), kwargs = {})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %pow_2, %expm1_1), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%where_2, 2.0), kwargs = {})
#   %avg_pool2d_1 : [num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%pow_3, [2, 2], [2, 2]), kwargs = {})
#   %sign_1 : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%avg_pool2d_1,), kwargs = {})
#   %abs_2 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%avg_pool2d_1,), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%abs_2,), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign_1, %relu_1), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, 4), kwargs = {})
#   %pow_4 : [num_users=3] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_10, 0.5), kwargs = {})
#   %gt_3 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%pow_4, 0), kwargs = {})
#   %expm1_2 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%pow_4,), kwargs = {})
#   %where_3 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_3, %pow_4, %expm1_2), kwargs = {})
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%where_3, 2.0), kwargs = {})
#   %avg_pool2d_2 : [num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%pow_5, [2, 2], [2, 2]), kwargs = {})
#   %sign_2 : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%avg_pool2d_2,), kwargs = {})
#   %abs_3 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%avg_pool2d_2,), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%abs_3,), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign_2, %relu_2), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_11, 4), kwargs = {})
#   %pow_6 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_12, 0.5), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_abs_add_avg_pool2d_celu_clamp_mul_pow_relu_sign_sub_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = xindex // 4
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 16*x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 16*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (8 + 2*x0 + 16*x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (9 + 2*x0 + 16*x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tmp9 = tl.full([1], 0, tl.int32)
    tmp10 = tmp9 < tmp8
    tmp11 = tmp10.to(tl.int8)
    tmp12 = tmp8 < tmp9
    tmp13 = tmp12.to(tl.int8)
    tmp14 = tmp11 - tmp13
    tmp15 = tmp14.to(tmp8.dtype)
    tmp16 = tl_math.abs(tmp8)
    tmp17 = triton_helpers.maximum(tmp9, tmp16)
    tmp18 = tmp15 * tmp17
    tmp19 = 4.0
    tmp20 = tmp18 * tmp19
    tmp21 = libdevice.sqrt(tmp20)
    tl.store(out_ptr0 + (x2), tmp21, xmask)







def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (1, 3, 64, 64), (12288, 4096, 64, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf0)
        buf1 = empty_strided_cuda((1, 12, 1), (12, 1, 12), torch.float32)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.bernoulli]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bernoulli_0[grid(12)](buf0, buf1, 0, 12, XBLOCK=16, num_warps=1, num_stages=1)
        del buf0
        buf2 = empty_strided_cuda((1, 12, 1024), (12288, 1, 12), torch.float32)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.bernoulli, aten._to_copy, aten.div, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_bernoulli_div_mul_1[grid(12288)](arg0_1, buf1, buf2, 12288, XBLOCK=128, num_warps=4, num_stages=1)
        del arg0_1
        del buf1
        buf5 = empty_strided_cuda((1, 12, 32, 32), (12288, 1024, 32, 1), torch.float32)
        buf6 = buf5; del buf5  # reuse
        buf7 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [x_5, x_6, x_7, x_8], Original ATen: [aten._adaptive_avg_pool2d, aten._to_copy, aten.arange, aten.mul, aten.clamp, aten._unsafe_index, aten.sub, aten.add, aten.celu, aten.pow]
        stream0 = get_raw_stream(0)
        triton_poi_fused__adaptive_avg_pool2d__to_copy__unsafe_index_add_arange_celu_clamp_mul_pow_sub_2[grid(12288)](buf7, buf2, 12288, XBLOCK=128, num_warps=4, num_stages=1)
        del buf2
        buf8 = empty_strided_cuda((1, 12, 16, 16), (3072, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_6, x_7, x_8, x_9, x_10], Original ATen: [aten._to_copy, aten.sub, aten.clamp, aten.mul, aten.add, aten.celu, aten.pow, aten.avg_pool2d, aten.sign, aten.abs, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_abs_add_avg_pool2d_celu_clamp_mul_pow_relu_sign_sub_3[grid(3072)](buf7, buf8, 3072, XBLOCK=128, num_warps=4, num_stages=1)
        del buf7
        buf9 = empty_strided_cuda((1, 12, 8, 8), (768, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_6, x_7, x_8, x_9, x_10, x_11, x_12], Original ATen: [aten._to_copy, aten.sub, aten.clamp, aten.mul, aten.add, aten.celu, aten.pow, aten.avg_pool2d, aten.sign, aten.abs, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_abs_add_avg_pool2d_celu_clamp_mul_pow_relu_sign_sub_4[grid(768)](buf8, buf9, 768, XBLOCK=128, num_warps=4, num_stages=1)
        del buf8
        buf10 = empty_strided_cuda((1, 12, 4, 4), (192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_6, x_7, x_8, x_9, x_10, x_11, x_12], Original ATen: [aten._to_copy, aten.sub, aten.clamp, aten.mul, aten.add, aten.celu, aten.pow, aten.avg_pool2d, aten.sign, aten.abs, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_abs_add_avg_pool2d_celu_clamp_mul_pow_relu_sign_sub_5[grid(192)](buf9, buf10, 192, XBLOCK=256, num_warps=4, num_stages=1)
        del buf9
    return (buf10, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
