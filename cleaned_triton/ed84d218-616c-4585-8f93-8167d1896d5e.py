# AOT ID: ['15_forward']
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


# kernel path: /tmp/torchinductor_sahanp/67/c67wi27xkqox4owrykfyewsxpxusy63isiqsextwit56u4r2czit.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x => constant_pad_nd
# Graph fragment:
#   %constant_pad_nd : [num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%primals_1, [2, 2, 2, 2], 3.0), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_constant_pad_nd_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 13872
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 68) % 68)
    x0 = (xindex % 68)
    x2 = xindex // 4624
    x4 = xindex
    tmp0 = (-2) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + x0
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-130) + x0 + 64*x1 + 4096*x2), tmp10 & xmask, other=3.0)
    tl.store(out_ptr0 + (x4), tmp11, xmask)




# kernel path: /tmp/torchinductor_sahanp/nz/cnz2wkh54rbl3ijvzgx5iollngpdoq4oa5a3uwiclq2sdnqulxsk.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_1 => convolution
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd, %primals_2, %primals_3, [1, 1], [0, 0], [1, 1], True, [0, 0], 1), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 156800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4900
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)




# kernel path: /tmp/torchinductor_sahanp/ui/cuicn45ups3n34ovzrgm5obqvfcbka4y7oxttt5aj47gbulh7af6.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_2 => getitem, getitem_1
# Graph fragment:
#   %getitem : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 0), kwargs = {})
#   %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 1), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_2(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 39200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 35)
    x3 = xindex // 35
    x2 = xindex // 1225
    x4 = (xindex % 1225)
    tmp0 = tl.load(in_ptr0 + (2*x0 + 140*x3), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 140*x3), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (70 + 2*x0 + 140*x3), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (71 + 2*x0 + 140*x3), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = tl.full([1], 1, tl.int8)
    tmp9 = tl.full([1], 0, tl.int8)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = tl.full([1], 2, tl.int8)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = tl.full([1], 3, tl.int8)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + (x4 + 1248*x2), tmp6, xmask)
    tl.store(out_ptr1 + (x4 + 1280*x2), tmp16, xmask)




# kernel path: /tmp/torchinductor_sahanp/as/casvtwlenv6emtzgbnylebbgitklbiiya7wbowpqyzp2j6jkrcrl.py
# Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.bernoulli]
# Source node to ATen node mapping:
#   x_4 => inductor_lookup_seed_default, inductor_random_default, lt
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 32, 1, 1], %inductor_lookup_seed_default, rand), kwargs = {})
#   %lt : [num_users=2] = call_function[target=torch.ops.aten.lt.Scalar](args = (%inductor_random_default, 0.5), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_bernoulli_3(in_ptr0, out_ptr1, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.5
    tmp4 = tmp2 < tmp3
    tl.store(out_ptr1 + (x0), tmp4, xmask)




# kernel path: /tmp/torchinductor_sahanp/r2/cr2uy74zaxc2wyrdqh5jdmnzyk6o76ssnm7wsma7nokgw4ql7dck.py
# Topologically Sorted Source Nodes: [x_4], Original ATen: [aten._to_copy, aten.add, aten.mul]
# Source node to ATen node mapping:
#   x_4 => add, add_1, add_2, convert_element_type, mul, mul_1, mul_2
# Graph fragment:
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt, torch.float32), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%convert_element_type, -1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%add, 1.558387861036063), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul, 0.7791939305180315), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%convert_element_type, 0.8864048946659319), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_2, %mul_1), kwargs = {})
#   %add_2 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %add_1), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_add_mul_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 25
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 0.8864048946659319
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp6 = -1.0
    tmp7 = tmp2 + tmp6
    tmp8 = 1.558387861036063
    tmp9 = tmp7 * tmp8
    tmp10 = 0.7791939305180315
    tmp11 = tmp9 + tmp10
    tmp12 = tmp5 + tmp11
    tl.store(in_out_ptr0 + (x2), tmp12, xmask)




# kernel path: /tmp/torchinductor_sahanp/cm/ccmgpxfqheeffargwcpal2owoluh3pukjwm6y3u5337wucffd5dz.py
# Topologically Sorted Source Nodes: [x_5, x_6, x_7], Original ATen: [aten.convolution, aten.abs, aten.add, aten.div, aten.log_sigmoid_forward]
# Source node to ATen node mapping:
#   x_5 => convolution_1
#   x_6 => abs_1, add_3, div
#   x_7 => abs_2, exp, full_default, log1p, minimum, neg, sub
# Graph fragment:
#   %convolution_1 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%add_2, %primals_4, %primals_5, [1, 1], [0, 0], [1, 1], True, [0, 0], 1), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%convolution_1,), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%abs_1, 1), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%convolution_1, %add_3), kwargs = {})
#   %full_default : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %minimum : [num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%full_default, %div), kwargs = {})
#   %abs_2 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%div,), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%abs_2,), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %log1p : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp,), kwargs = {})
#   %sub : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%minimum, %log1p), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_abs_add_convolution_div_log_sigmoid_forward_5(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 49
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl_math.abs(tmp2)
    tmp4 = 1.0
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 / tmp5
    tmp7 = 0.0
    tmp8 = triton_helpers.minimum(tmp7, tmp6)
    tmp9 = tl_math.abs(tmp6)
    tmp10 = -tmp9
    tmp11 = tl_math.exp(tmp10)
    tmp12 = libdevice.log1p(tmp11)
    tmp13 = tmp8 - tmp12
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp13, xmask)




# kernel path: /tmp/torchinductor_sahanp/7a/c7ayxndk5mp7wearflfqgsku67cejkrlgxcy7jc4wpmy23icwke7.py
# Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.convolution, aten.abs, aten.add, aten.div]
# Source node to ATen node mapping:
#   input_1 => convolution_2
#   input_2 => abs_3, add_4, div_1
# Graph fragment:
#   %convolution_2 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%sub, %primals_6, %primals_7, [1, 1], [0, 0], [1, 1], True, [0, 0], 1), kwargs = {})
#   %abs_3 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%convolution_2,), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%abs_3, 1), kwargs = {})
#   %div_1 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%convolution_2, %add_4), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_abs_add_convolution_div_6(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 81
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl_math.abs(tmp2)
    tmp4 = 1.0
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 / tmp5
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp6, xmask)




# kernel path: /tmp/torchinductor_sahanp/xd/cxd5owdh4gl6bxmm7cacxmkm5q3yf72e7ucz6wls3xktbvypgv6v.py
# Topologically Sorted Source Nodes: [x_7, input_3, input_4, loss], Original ATen: [aten.log_sigmoid_forward, aten.convolution, aten.sub, aten.add, aten.norm, aten.scalar_tensor, aten.div, aten.eq, aten.masked_fill]
# Source node to ATen node mapping:
#   input_3 => convolution_3
#   input_4 => abs_4, exp_1, log1p_1, minimum_1, neg_1, sub_1
#   loss => add_5, pow_1, sub_2, sum_1
#   x_7 => full_default
# Graph fragment:
#   %full_default : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %convolution_3 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %primals_8, %primals_9, [1, 1], [0, 0], [1, 1], True, [0, 0], 1), kwargs = {})
#   %minimum_1 : [num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%full_default, %convolution_3), kwargs = {})
#   %abs_4 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%convolution_3,), kwargs = {})
#   %neg_1 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%abs_4,), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_1,), kwargs = {})
#   %log1p_1 : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_1,), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%minimum_1, %log1p_1), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_1, %sub_1), kwargs = {})
#   %add_5 : [num_users=2] = call_function[target=torch.ops.aten.add.Scalar](args = (%sub_2, 1e-06), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_5, 2.0), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [3]), kwargs = {})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_5, %unsqueeze_1), kwargs = {})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%unsqueeze_1, 0), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default_2, %div_3), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_convolution_div_eq_log_sigmoid_forward_masked_fill_norm_scalar_tensor_sub_7(in_out_ptr0, in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 2816
    r0_numel = 11
    R0_BLOCK: tl.constexpr = 16
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
    r0_2 = r0_index
    x3 = xindex
    x1 = xindex // 11
    tmp0 = tl.load(in_out_ptr0 + (r0_2 + 11*x3), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.minimum(tmp3, tmp2)
    tmp5 = tl_math.abs(tmp2)
    tmp6 = -tmp5
    tmp7 = tl_math.exp(tmp6)
    tmp8 = libdevice.log1p(tmp7)
    tmp9 = tmp4 - tmp8
    tmp10 = tmp9 - tmp9
    tmp11 = 1e-06
    tmp12 = tmp10 + tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.where(r0_mask & xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tmp18 = libdevice.sqrt(tmp17)
    tmp19 = tmp18 == tmp3
    tmp20 = tmp12 / tmp18
    tmp21 = tl.where(tmp19, tmp3, tmp20)
    tl.store(in_out_ptr0 + (r0_2 + 11*x3), tmp2, r0_mask & xmask)
    tl.store(out_ptr1 + (r0_2 + 11*x3), tmp21, r0_mask & xmask)
    tl.store(out_ptr0 + (x3), tmp17, xmask)




# kernel path: /tmp/torchinductor_sahanp/2l/c2l5zzwofdcyyrrx47v67mgz456icfpykjpco3c46k3sgug734gb.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten.norm, aten.add, aten.sub, aten.clamp_min, aten.mean, aten.ge]
# Source node to ATen node mapping:
#   loss => add_7, clamp_min, mean, pow_2, sub_4
# Graph fragment:
#   %pow_2 : [num_users=3] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%pow_2, 1.0), kwargs = {})
#   %sub_4 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_7, %pow_2), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_4, 0), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%clamp_min,), kwargs = {})
#   %ge : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_4, 0), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_clamp_min_ge_mean_norm_sub_8(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 2816
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = libdevice.sqrt(tmp0)
        tmp2 = 1.0
        tmp3 = tmp1 + tmp2
        tmp4 = tmp3 - tmp1
        tmp5 = 0.0
        tmp6 = triton_helpers.maximum(tmp4, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask, tmp9, _tmp8)
        tmp10 = tmp4 >= tmp5
        tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp10, r0_mask)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tmp11 = 2816.0
    tmp12 = tmp8 / tmp11
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp12, None)







def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9 = args
    args.clear()
    assert_size_stride(primals_1, (1, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_2, (3, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_3, (32, ), (1, ))
    assert_size_stride(primals_4, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_7, (128, ), (1, ))
    assert_size_stride(primals_8, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_9, (256, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 3, 68, 68), (13872, 4624, 68, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_0[grid(13872)](primals_1, buf0, 13872, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_1
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(buf0, primals_2, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (1, 32, 70, 70), (156800, 4900, 70, 1))
        buf2 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_1[grid(156800)](buf2, primals_3, 156800, XBLOCK=512, num_warps=8, num_stages=1)
        del primals_3
        buf3 = empty_strided_cuda((1, 32, 35, 35), (39936, 1248, 35, 1), torch.float32)
        buf4 = empty_strided_cuda((1, 32, 35, 35), (40960, 1280, 35, 1), torch.int8)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_2[grid(39200)](buf2, buf3, buf4, 39200, XBLOCK=256, num_warps=4, num_stages=1)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.adaptive_max_pool2d]
        buf5 = torch.ops.aten.max_pool2d_with_indices.default(buf3, [7, 7])
        buf6 = buf5[0]
        buf7 = buf5[1]
        del buf5
        buf8 = empty_strided_cuda((1, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf8)
        buf10 = empty_strided_cuda((1, 32, 1, 1), (32, 1, 1, 1), torch.bool)
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.bernoulli]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bernoulli_3[grid(32)](buf8, buf10, 0, 32, XBLOCK=32, num_warps=1, num_stages=1)
        del buf8
        buf11 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten._to_copy, aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_mul_4[grid(800)](buf11, buf10, 800, XBLOCK=128, num_warps=4, num_stages=1)
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, primals_4, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (1, 64, 7, 7), (3136, 49, 7, 1))
        buf13 = buf12; del buf12  # reuse
        buf14 = empty_strided_cuda((1, 64, 7, 7), (3136, 49, 7, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_5, x_6, x_7], Original ATen: [aten.convolution, aten.abs, aten.add, aten.div, aten.log_sigmoid_forward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_abs_add_convolution_div_log_sigmoid_forward_5[grid(3136)](buf13, primals_5, buf14, 3136, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_5
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf14, primals_6, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (1, 128, 9, 9), (10368, 81, 9, 1))
        buf16 = buf15; del buf15  # reuse
        buf17 = empty_strided_cuda((1, 128, 9, 9), (10368, 81, 9, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.convolution, aten.abs, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_abs_add_convolution_div_6[grid(10368)](buf16, primals_7, buf17, 10368, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_7
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, primals_8, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (1, 256, 11, 11), (30976, 121, 11, 1))
        buf19 = buf18; del buf18  # reuse
        buf20 = empty_strided_cuda((1, 256, 11), (2816, 11, 1), torch.float32)
        buf23 = empty_strided_cuda((1, 256, 11, 11), (30976, 121, 11, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_7, input_3, input_4, loss], Original ATen: [aten.log_sigmoid_forward, aten.convolution, aten.sub, aten.add, aten.norm, aten.scalar_tensor, aten.div, aten.eq, aten.masked_fill]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_convolution_div_eq_log_sigmoid_forward_masked_fill_norm_scalar_tensor_sub_7[grid(2816)](buf19, primals_9, buf20, buf23, 2816, 11, XBLOCK=8, num_warps=2, num_stages=1)
        del primals_9
        buf21 = empty_strided_cuda((), (), torch.float32)
        buf22 = empty_strided_cuda((1, 256, 11), (2816, 11, 1), torch.bool)
        buf24 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [loss], Original ATen: [aten.norm, aten.add, aten.sub, aten.clamp_min, aten.mean, aten.ge]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_clamp_min_ge_mean_norm_sub_8[grid(1)](buf24, buf20, buf22, 1, 2816, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del buf20
    return (buf24, primals_2, primals_4, primals_6, primals_8, buf0, buf2, buf3, buf4, buf7, buf10, buf11, buf13, buf14, buf16, buf17, buf19, buf22, buf23, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((3, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
