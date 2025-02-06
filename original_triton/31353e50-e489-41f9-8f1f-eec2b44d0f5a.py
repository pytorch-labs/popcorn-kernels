# AOT ID: ['19_forward']
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


# kernel path: /tmp/torchinductor_sahanp/3w/c3wfs7jf6xzzqhmkelvuypohiimcs2zpp2uz2ekqzpclcyv7cpai.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_1 => _low_memory_max_pool2d_with_offsets
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%unsqueeze, [1, 2], [1, 2], [0, 0], [1, 1], False), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 64)
    x1 = xindex // 64
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x1), xmask)
    tmp1 = tl.load(in_ptr0 + (64 + x0 + 128*x1), xmask)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(out_ptr0 + (x2), tmp2, xmask)




# kernel path: /tmp/torchinductor_sahanp/qc/cqcdq6txda5gt4riwqfvju6hdzhbhp2mv5k46kytqcpxl3ryrao5.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   multi_head_attention_forward => clone
# Graph fragment:
#   %clone : [num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze_2,), kwargs = {memory_format: torch.contiguous_format})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 16)
    x2 = xindex // 1024
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x2 + 192*x1), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, xmask)




# kernel path: /tmp/torchinductor_sahanp/m4/cm4zvjvizwvuasjsxu7dk6mmuvw7lml2w6irv5tl4n4enqvoxh4b.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.view]
# Source node to ATen node mapping:
#   multi_head_attention_forward => view_6
# Graph fragment:
#   %view_6 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_4, [16, 8, 1, 8]), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_view_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/pe/cpefphwt263uwilpbtknqitx2ztsj76hnaj466px22r6xcvcewn7.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.view]
# Source node to ATen node mapping:
#   multi_head_attention_forward => view_7
# Graph fragment:
#   %view_7 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_5, [16, 8, 1, 8]), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_view_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (1024 + x2), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/kg/ckgdrbe5q75dpmr33fhj2xbbg473ggn2oxom62aaxcdt64otwvxb.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.view]
# Source node to ATen node mapping:
#   multi_head_attention_forward => view_8
# Graph fragment:
#   %view_8 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_6, [16, 8, 1, 8]), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_view_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2048 + x2), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/ry/cryc62y7s52qvzqkb544n356haecpc6xbupcu5zesfjr2iwpuo3x.py
# Topologically Sorted Source Nodes: [dropout, add, x_3], Original ATen: [aten.native_dropout, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   add => add
#   dropout => gt, inductor_lookup_seed_default, inductor_random_default_5, mul, mul_1
#   x_3 => add_1, add_2, mul_2, mul_3, rsqrt, sub, var_mean
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default_5 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 16, 64], %inductor_lookup_seed_default, rand), kwargs = {})
#   %gt : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%inductor_random_default_5, 0.1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gt, %view_10), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, 1.1111111111111112), kwargs = {})
#   %add : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_1, %mul_1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %getitem_7), kwargs = {})
#   %mul_2 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %primals_6), kwargs = {})
#   %add_2 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %primals_7), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt, 64), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr4, out_ptr5, load_seed_offset, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
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
    tmp5 = tl.load(in_ptr1 + (r0_1 + 64*x0), xmask, other=0.0)
    tmp7 = tl.load(in_out_ptr0 + (r0_1 + 64*x0), xmask, other=0.0)
    tmp8 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr4 + (r0_1), None, eviction_policy='evict_last')
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = r0_1 + 64*x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp6 = tmp4.to(tl.float32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp5 + tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.where(xmask, tmp14, 0)
    tmp17 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp20 / tmp22
    tmp24 = tmp14 - tmp23
    tmp25 = tmp24 * tmp24
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
    tmp28 = tl.where(xmask, tmp26, 0)
    tmp29 = tl.sum(tmp28, 1)[:, None]
    tmp30 = tmp13 - tmp23
    tmp31 = 64.0
    tmp32 = tmp29 / tmp31
    tmp33 = 1e-05
    tmp34 = tmp32 + tmp33
    tmp35 = libdevice.rsqrt(tmp34)
    tmp36 = tmp30 * tmp35
    tmp38 = tmp36 * tmp37
    tmp40 = tmp38 + tmp39
    tmp41 = 0.015625
    tmp42 = tmp35 * tmp41
    tl.store(out_ptr1 + (r0_1 + 64*x0), tmp4, xmask)
    tl.store(in_out_ptr0 + (r0_1 + 64*x0), tmp36, xmask)
    tl.store(out_ptr4 + (r0_1 + 64*x0), tmp40, xmask)
    tl.store(out_ptr5 + (x0), tmp42, xmask)




# kernel path: /tmp/torchinductor_sahanp/gf/cgfhahh5yyupkd2pm2atpn54whsjbhwmwmmnj6irnbnxjcdfjee6.py
# Topologically Sorted Source Nodes: [relu, dropout_1], Original ATen: [aten.relu, aten.native_dropout, aten.threshold_backward]
# Source node to ATen node mapping:
#   dropout_1 => gt_1, inductor_lookup_seed_default_1, inductor_random_default_4, mul_4, mul_5
#   relu => relu
# Graph fragment:
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%view_12,), kwargs = {})
#   %inductor_lookup_seed_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 1), kwargs = {})
#   %inductor_random_default_4 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 16, 2048], %inductor_lookup_seed_default_1, rand), kwargs = {})
#   %gt_1 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%inductor_random_default_4, 0.1), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gt_1, %relu), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, 1.1111111111111112), kwargs = {})
#   %le_1 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu, 0), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_dropout_relu_threshold_backward_6(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    x1 = (xindex % 2048)
    tmp6 = tl.load(in_ptr1 + (x0), None)
    tmp7 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full([1], 0, tl.int32)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp11 = tmp5 * tmp10
    tmp12 = 1.1111111111111112
    tmp13 = tmp11 * tmp12
    tmp14 = 0.0
    tmp15 = tmp10 <= tmp14
    tl.store(out_ptr1 + (x0), tmp4, None)
    tl.store(out_ptr2 + (x0), tmp13, None)
    tl.store(out_ptr3 + (x0), tmp15, None)




# kernel path: /tmp/torchinductor_sahanp/v3/cv3rxuwon7fpw7luoln37gcww63g5diuliadmry2espaf3pyngib.py
# Topologically Sorted Source Nodes: [dropout_2, add_1, x_5, x_6], Original ATen: [aten.native_dropout, aten.add, aten.native_layer_norm, aten.gelu, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   add_1 => add_3
#   dropout_2 => gt_2, inductor_lookup_seed_default_2, inductor_random_default_3, mul_6, mul_7
#   x_5 => add_4, add_5, mul_8, mul_9, rsqrt_1, sub_1, var_mean_1
#   x_6 => add_6, erf, mul_10, mul_11, mul_12
# Graph fragment:
#   %inductor_lookup_seed_default_2 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 2), kwargs = {})
#   %inductor_random_default_3 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 16, 64], %inductor_lookup_seed_default_2, rand), kwargs = {})
#   %gt_2 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%inductor_random_default_3, 0.1), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gt_2, %view_14), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, 1.1111111111111112), kwargs = {})
#   %add_3 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %mul_7), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_3, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_8, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %getitem_9), kwargs = {})
#   %mul_8 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, %primals_12), kwargs = {})
#   %add_5 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %primals_13), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, 0.5), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_11,), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_12 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %add_6), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_1, 64), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_gelu_native_dropout_native_layer_norm_native_layer_norm_backward_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr4, out_ptr5, load_seed_offset, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
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
    tmp5 = tl.load(in_ptr1 + (r0_1 + 64*x0), xmask, other=0.0)
    tmp7 = tl.load(in_out_ptr0 + (r0_1 + 64*x0), xmask, other=0.0)
    tmp8 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr4 + (r0_1), None, eviction_policy='evict_last')
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = r0_1 + 64*x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp6 = tmp4.to(tl.float32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp5 + tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.where(xmask, tmp14, 0)
    tmp17 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp20 / tmp22
    tmp24 = tmp14 - tmp23
    tmp25 = tmp24 * tmp24
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
    tmp28 = tl.where(xmask, tmp26, 0)
    tmp29 = tl.sum(tmp28, 1)[:, None]
    tmp30 = tmp13 - tmp23
    tmp31 = 64.0
    tmp32 = tmp29 / tmp31
    tmp33 = 1e-05
    tmp34 = tmp32 + tmp33
    tmp35 = libdevice.rsqrt(tmp34)
    tmp36 = tmp30 * tmp35
    tmp38 = tmp36 * tmp37
    tmp40 = tmp38 + tmp39
    tmp41 = 0.5
    tmp42 = tmp40 * tmp41
    tmp43 = 0.7071067811865476
    tmp44 = tmp40 * tmp43
    tmp45 = libdevice.erf(tmp44)
    tmp46 = 1.0
    tmp47 = tmp45 + tmp46
    tmp48 = tmp42 * tmp47
    tmp49 = 0.015625
    tmp50 = tmp35 * tmp49
    tl.store(out_ptr1 + (r0_1 + 64*x0), tmp4, xmask)
    tl.store(in_out_ptr0 + (r0_1 + 64*x0), tmp36, xmask)
    tl.store(out_ptr4 + (r0_1 + 64*x0), tmp48, xmask)
    tl.store(out_ptr5 + (x0), tmp50, xmask)




# kernel path: /tmp/torchinductor_sahanp/sp/cspmt2hbkewdj4h6mgvxy7w3adejhs3muibd4eqpikdpj5y5dknx.py
# Topologically Sorted Source Nodes: [relu_1, dropout_4], Original ATen: [aten.relu, aten.native_dropout, aten.threshold_backward]
# Source node to ATen node mapping:
#   dropout_4 => gt_4, inductor_lookup_seed_default_4, inductor_random_default_1, mul_17, mul_18
#   relu_1 => relu_1
# Graph fragment:
#   %relu_1 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%view_27,), kwargs = {})
#   %inductor_lookup_seed_default_4 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 4), kwargs = {})
#   %inductor_random_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 16, 2048], %inductor_lookup_seed_default_4, rand), kwargs = {})
#   %gt_4 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%inductor_random_default_1, 0.1), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gt_4, %relu_1), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_17, 1.1111111111111112), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_1, 0), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_dropout_relu_threshold_backward_8(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    x1 = (xindex % 2048)
    tmp6 = tl.load(in_ptr1 + (x0), None)
    tmp7 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full([1], 0, tl.int32)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp11 = tmp5 * tmp10
    tmp12 = 1.1111111111111112
    tmp13 = tmp11 * tmp12
    tmp14 = 0.0
    tmp15 = tmp10 <= tmp14
    tl.store(out_ptr1 + (x0), tmp4, None)
    tl.store(out_ptr2 + (x0), tmp13, None)
    tl.store(out_ptr3 + (x0), tmp15, None)




# kernel path: /tmp/torchinductor_sahanp/7m/c7mddhuir53aleadfxravo2lgiwt67tohlc7fzs6ykxeudih73nq.py
# Topologically Sorted Source Nodes: [x_11, x_13], Original ATen: [aten.max_pool2d_with_indices, aten.gelu, aten.gelu_backward]
# Source node to ATen node mapping:
#   x_11 => getitem_19
#   x_13 => add_13, erf_1, mul_23, mul_24, mul_25
# Graph fragment:
#   %getitem_19 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_1, 1), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_21, 0.5), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_21, 0.7071067811865476), kwargs = {})
#   %erf_1 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_24,), kwargs = {})
#   %add_13 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_1, 1), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_23, %add_13), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_13, 0.5), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_21, %permute_21), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, -0.5), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_29,), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp, 0.3989422804014327), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_21, %mul_30), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_27, %mul_31), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_gelu_gelu_backward_max_pool2d_with_indices_9(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 64)
    x1 = xindex // 64
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x1), xmask)
    tmp1 = tl.load(in_ptr0 + (64 + x0 + 128*x1), xmask)
    tmp2 = tmp1 > tmp0
    tmp3 = tl.full([1], 1, tl.int8)
    tmp4 = tl.full([1], 0, tl.int8)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = triton_helpers.maximum(tmp1, tmp0)
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = 0.7071067811865476
    tmp10 = tmp6 * tmp9
    tmp11 = libdevice.erf(tmp10)
    tmp12 = 1.0
    tmp13 = tmp11 + tmp12
    tmp14 = tmp8 * tmp13
    tmp15 = tmp13 * tmp7
    tmp16 = tmp6 * tmp6
    tmp17 = -0.5
    tmp18 = tmp16 * tmp17
    tmp19 = tl_math.exp(tmp18)
    tmp20 = 0.3989422804014327
    tmp21 = tmp19 * tmp20
    tmp22 = tmp6 * tmp21
    tmp23 = tmp15 + tmp22
    tl.store(out_ptr0 + (x2), tmp5, xmask)
    tl.store(out_ptr1 + (x2), tmp14, xmask)
    tl.store(out_ptr2 + (x2), tmp23, xmask)







def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25 = args
    args.clear()
    assert_size_stride(primals_1, (1, 32, 64), (2048, 64, 1))
    assert_size_stride(primals_2, (192, ), (1, ))
    assert_size_stride(primals_3, (192, 64), (64, 1))
    assert_size_stride(primals_4, (64, 64), (64, 1))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_7, (64, ), (1, ))
    assert_size_stride(primals_8, (2048, 64), (64, 1))
    assert_size_stride(primals_9, (2048, ), (1, ))
    assert_size_stride(primals_10, (64, 2048), (2048, 1))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_12, (64, ), (1, ))
    assert_size_stride(primals_13, (64, ), (1, ))
    assert_size_stride(primals_14, (192, ), (1, ))
    assert_size_stride(primals_15, (192, 64), (64, 1))
    assert_size_stride(primals_16, (64, 64), (64, 1))
    assert_size_stride(primals_17, (64, ), (1, ))
    assert_size_stride(primals_18, (64, ), (1, ))
    assert_size_stride(primals_19, (64, ), (1, ))
    assert_size_stride(primals_20, (2048, 64), (64, 1))
    assert_size_stride(primals_21, (2048, ), (1, ))
    assert_size_stride(primals_22, (64, 2048), (2048, 1))
    assert_size_stride(primals_23, (64, ), (1, ))
    assert_size_stride(primals_24, (64, ), (1, ))
    assert_size_stride(primals_25, (64, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 64, 1, 16), (1024, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_0[grid(1024)](primals_1, buf0, 1024, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_1
        buf1 = empty_strided_cuda((16, 192), (192, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf0, (16, 64), (64, 1), 0), reinterpret_tensor(primals_3, (64, 192), (1, 64), 0), out=buf1)
        del primals_3
        buf2 = empty_strided_cuda((3, 1, 16, 64), (1024, 1, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1[grid(3072)](buf1, primals_2, buf2, 3072, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_2
        buf3 = empty_strided_cuda((16, 8, 1, 8), (64, 8, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_2[grid(1024)](buf2, buf3, 1024, XBLOCK=256, num_warps=4, num_stages=1)
        buf4 = empty_strided_cuda((16, 8, 1, 8), (64, 8, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3[grid(1024)](buf2, buf4, 1024, XBLOCK=256, num_warps=4, num_stages=1)
        buf5 = empty_strided_cuda((16, 8, 1, 8), (64, 8, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_4[grid(1024)](buf2, buf5, 1024, XBLOCK=256, num_warps=4, num_stages=1)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf3, buf4, buf5, None, True, 0.1)
        buf7 = buf6[0]
        buf8 = buf6[1]
        buf9 = buf6[2]
        buf10 = buf6[3]
        del buf6
        buf11 = empty_strided_cuda((16, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf7, (16, 64), (64, 1), 0), reinterpret_tensor(primals_4, (64, 64), (1, 64), 0), out=buf11)
        buf12 = empty_strided_cuda((6, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [6], out=buf12)
        buf14 = empty_strided_cuda((1, 16, 64), (1024, 64, 1), torch.bool)
        buf18 = reinterpret_tensor(buf11, (1, 16, 64), (1024, 64, 1), 0); del buf11  # reuse
        buf19 = empty_strided_cuda((1, 16, 64), (1024, 64, 1), torch.float32)
        buf70 = empty_strided_cuda((1, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [dropout, add, x_3], Original ATen: [aten.native_dropout, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_5[grid(16)](buf18, buf12, buf0, primals_5, primals_6, primals_7, buf14, buf19, buf70, 5, 16, 64, XBLOCK=1, num_warps=2, num_stages=1)
        del primals_5
        del primals_7
        buf20 = empty_strided_cuda((16, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf19, (16, 64), (64, 1), 0), reinterpret_tensor(primals_8, (64, 2048), (1, 64), 0), out=buf20)
        buf22 = empty_strided_cuda((1, 16, 2048), (32768, 2048, 1), torch.bool)
        buf23 = empty_strided_cuda((1, 16, 2048), (32768, 2048, 1), torch.float32)
        buf69 = empty_strided_cuda((1, 16, 2048), (32768, 2048, 1), torch.bool)
        # Topologically Sorted Source Nodes: [relu, dropout_1], Original ATen: [aten.relu, aten.native_dropout, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_dropout_relu_threshold_backward_6[grid(32768)](buf12, buf20, primals_9, buf22, buf23, buf69, 1, 32768, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_9
        buf24 = empty_strided_cuda((16, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf23, (16, 2048), (2048, 1), 0), reinterpret_tensor(primals_10, (2048, 64), (1, 2048), 0), out=buf24)
        buf26 = empty_strided_cuda((1, 16, 64), (1024, 64, 1), torch.bool)
        buf30 = reinterpret_tensor(buf24, (1, 16, 64), (1024, 64, 1), 0); del buf24  # reuse
        buf31 = empty_strided_cuda((1, 16, 64), (1024, 64, 1), torch.float32)
        buf68 = empty_strided_cuda((1, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [dropout_2, add_1, x_5, x_6], Original ATen: [aten.native_dropout, aten.add, aten.native_layer_norm, aten.gelu, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_gelu_native_dropout_native_layer_norm_native_layer_norm_backward_7[grid(16)](buf30, buf12, buf19, primals_11, primals_12, primals_13, buf26, buf31, buf68, 2, 16, 64, XBLOCK=1, num_warps=2, num_stages=1)
        del primals_11
        buf32 = reinterpret_tensor(buf2, (16, 192), (192, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf31, (16, 64), (64, 1), 0), reinterpret_tensor(primals_15, (64, 192), (1, 64), 0), out=buf32)
        buf33 = reinterpret_tensor(buf1, (3, 1, 16, 64), (1024, 1, 64, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1[grid(3072)](buf32, primals_14, buf33, 3072, XBLOCK=256, num_warps=4, num_stages=1)
        del buf32
        del primals_14
        buf34 = empty_strided_cuda((16, 8, 1, 8), (64, 8, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_2[grid(1024)](buf33, buf34, 1024, XBLOCK=256, num_warps=4, num_stages=1)
        buf35 = empty_strided_cuda((16, 8, 1, 8), (64, 8, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3[grid(1024)](buf33, buf35, 1024, XBLOCK=256, num_warps=4, num_stages=1)
        buf36 = empty_strided_cuda((16, 8, 1, 8), (64, 8, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_4[grid(1024)](buf33, buf36, 1024, XBLOCK=256, num_warps=4, num_stages=1)
        del buf33
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf37 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf34, buf35, buf36, None, True, 0.1)
        buf38 = buf37[0]
        buf39 = buf37[1]
        buf40 = buf37[2]
        buf41 = buf37[3]
        del buf37
        buf42 = empty_strided_cuda((16, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf38, (16, 64), (64, 1), 0), reinterpret_tensor(primals_16, (64, 64), (1, 64), 0), out=buf42)
        buf44 = empty_strided_cuda((1, 16, 64), (1024, 64, 1), torch.bool)
        buf48 = reinterpret_tensor(buf42, (1, 16, 64), (1024, 64, 1), 0); del buf42  # reuse
        buf49 = empty_strided_cuda((1, 16, 64), (1024, 64, 1), torch.float32)
        buf67 = empty_strided_cuda((1, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [dropout_3, add_2, x_7], Original ATen: [aten.native_dropout, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_5[grid(16)](buf48, buf12, buf31, primals_17, primals_18, primals_19, buf44, buf49, buf67, 5, 16, 64, XBLOCK=1, num_warps=2, num_stages=1)
        del primals_17
        del primals_19
        buf50 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf49, (16, 64), (64, 1), 0), reinterpret_tensor(primals_20, (64, 2048), (1, 64), 0), out=buf50)
        buf52 = empty_strided_cuda((1, 16, 2048), (32768, 2048, 1), torch.bool)
        buf53 = empty_strided_cuda((1, 16, 2048), (32768, 2048, 1), torch.float32)
        buf66 = empty_strided_cuda((1, 16, 2048), (32768, 2048, 1), torch.bool)
        # Topologically Sorted Source Nodes: [relu_1, dropout_4], Original ATen: [aten.relu, aten.native_dropout, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_dropout_relu_threshold_backward_8[grid(32768)](buf12, buf50, primals_21, buf52, buf53, buf66, 4, 32768, XBLOCK=256, num_warps=4, num_stages=1)
        del buf50
        del primals_21
        buf54 = empty_strided_cuda((16, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf53, (16, 2048), (2048, 1), 0), reinterpret_tensor(primals_22, (2048, 64), (1, 2048), 0), out=buf54)
        buf56 = empty_strided_cuda((1, 16, 64), (1024, 64, 1), torch.bool)
        buf60 = reinterpret_tensor(buf54, (1, 16, 64), (1024, 64, 1), 0); del buf54  # reuse
        buf61 = empty_strided_cuda((1, 16, 64), (1024, 64, 1), torch.float32)
        buf65 = empty_strided_cuda((1, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [dropout_5, add_3, x_9], Original ATen: [aten.native_dropout, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_5[grid(16)](buf60, buf12, buf49, primals_23, primals_24, primals_25, buf56, buf61, buf65, 5, 16, 64, XBLOCK=1, num_warps=2, num_stages=1)
        del buf12
        del primals_23
        del primals_25
        buf62 = empty_strided_cuda((1, 64, 1, 8), (512, 1, 512, 64), torch.int8)
        buf63 = empty_strided_cuda((1, 8, 64), (512, 64, 1), torch.float32)
        buf64 = empty_strided_cuda((1, 8, 64), (512, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_11, x_13], Original ATen: [aten.max_pool2d_with_indices, aten.gelu, aten.gelu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_gelu_backward_max_pool2d_with_indices_9[grid(512)](buf61, buf62, buf63, buf64, 512, XBLOCK=256, num_warps=4, num_stages=1)
    return (buf63, primals_6, primals_12, primals_13, primals_18, primals_24, reinterpret_tensor(buf0, (16, 64), (64, 1), 0), buf3, buf4, buf5, buf7, buf8, buf9, buf10, buf14, buf18, reinterpret_tensor(buf19, (16, 64), (64, 1), 0), buf22, reinterpret_tensor(buf23, (16, 2048), (2048, 1), 0), buf26, buf30, reinterpret_tensor(buf31, (16, 64), (64, 1), 0), buf34, buf35, buf36, buf38, buf39, buf40, buf41, buf44, buf48, reinterpret_tensor(buf49, (16, 64), (64, 1), 0), buf52, reinterpret_tensor(buf53, (16, 2048), (2048, 1), 0), buf56, buf60, reinterpret_tensor(buf61, (1, 64, 1, 16), (1024, 1, 1024, 64), 0), buf62, buf64, buf65, primals_22, buf66, primals_20, buf67, primals_16, primals_15, buf68, primals_10, buf69, primals_8, buf70, primals_4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 32, 64), (2048, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((192, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((2048, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((192, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((2048, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((64, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
