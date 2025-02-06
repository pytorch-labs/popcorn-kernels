# AOT ID: ['72_forward']
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


# kernel path: /tmp/torchinductor_sahanp/sk/csk3hu5dl75drbx4zr2kr6pixnzg64mqjlxzmlga4ukyjns3wmhn.py
# Topologically Sorted Source Nodes: [dropout, add, x_4], Original ATen: [aten.native_dropout, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   add => add
#   dropout => gt, inductor_lookup_seed_default, inductor_random_default_3, mul, mul_1
#   x_4 => add_1, add_2, mul_2, mul_3, rsqrt, sub, var_mean
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default_3 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 1, 128], %inductor_lookup_seed_default, rand), kwargs = {})
#   %gt : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%inductor_random_default_3, 0.1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gt, %view_11), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, 1.1111111111111112), kwargs = {})
#   %add : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze, %mul_1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %getitem_7), kwargs = {})
#   %mul_2 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %primals_11), kwargs = {})
#   %add_2 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %primals_12), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt, 128), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr4, out_ptr5, load_seed_offset, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
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
    tmp5 = tl.load(in_ptr1 + (r0_0), None)
    tmp7 = tl.load(in_out_ptr0 + (r0_0), None)
    tmp8 = tl.load(in_ptr2 + (r0_0), None)
    tmp34 = tl.load(in_ptr3 + (r0_0), None)
    tmp36 = tl.load(in_ptr4 + (r0_0), None)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = r0_0
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
    tmp16 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp18 = tl.sum(tmp16, 1)[:, None]
    tmp19 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 / tmp20
    tmp22 = tmp14 - tmp21
    tmp23 = tmp22 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK, R0_BLOCK])
    tmp26 = tl.sum(tmp24, 1)[:, None]
    tmp27 = tmp13 - tmp21
    tmp28 = 128.0
    tmp29 = tmp26 / tmp28
    tmp30 = 1e-05
    tmp31 = tmp29 + tmp30
    tmp32 = libdevice.rsqrt(tmp31)
    tmp33 = tmp27 * tmp32
    tmp35 = tmp33 * tmp34
    tmp37 = tmp35 + tmp36
    tmp38 = 0.0078125
    tmp39 = tmp32 * tmp38
    tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp4, None)
    tl.store(in_out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp33, None)
    tl.store(out_ptr4 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp37, None)
    tl.store(out_ptr5 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp39, None)




# kernel path: /tmp/torchinductor_sahanp/5a/c5annfjrwtoed2wwezx2gx2ndcukmyctccrz5bmddf65l25ztyy7.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.view]
# Source node to ATen node mapping:
#   multi_head_attention_forward_1 => full_default
# Graph fragment:
#   %full_default : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([1, 128], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_view_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/t6/ct6je3spiw4jkwesduhxnlcgzlkykaxmktb7mbrcdf6p65756335.py
# Topologically Sorted Source Nodes: [dropout_1, add_1, x_5], Original ATen: [aten.native_dropout, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   add_1 => add_3
#   dropout_1 => gt_1, inductor_lookup_seed_default_1, inductor_random_default_2, mul_4, mul_5
#   x_5 => add_4, add_5, mul_6, mul_7, rsqrt_1, sub_1, var_mean_1
# Graph fragment:
#   %inductor_lookup_seed_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 1), kwargs = {})
#   %inductor_random_default_2 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 1, 128], %inductor_lookup_seed_default_1, rand), kwargs = {})
#   %gt_1 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%inductor_random_default_2, 0.1), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gt_1, %view_24), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, 1.1111111111111112), kwargs = {})
#   %add_3 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %mul_5), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_3, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_16, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %getitem_17), kwargs = {})
#   %mul_6 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %primals_17), kwargs = {})
#   %add_5 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %primals_18), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_1, 128), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr4, out_ptr5, load_seed_offset, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
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
    tmp5 = tl.load(in_ptr1 + (r0_0), None)
    tmp7 = tl.load(in_out_ptr0 + (r0_0), None)
    tmp8 = tl.load(in_ptr2 + (r0_0), None)
    tmp34 = tl.load(in_ptr3 + (r0_0), None)
    tmp36 = tl.load(in_ptr4 + (r0_0), None)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = r0_0
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
    tmp16 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp18 = tl.sum(tmp16, 1)[:, None]
    tmp19 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 / tmp20
    tmp22 = tmp14 - tmp21
    tmp23 = tmp22 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK, R0_BLOCK])
    tmp26 = tl.sum(tmp24, 1)[:, None]
    tmp27 = tmp13 - tmp21
    tmp28 = 128.0
    tmp29 = tmp26 / tmp28
    tmp30 = 1e-05
    tmp31 = tmp29 + tmp30
    tmp32 = libdevice.rsqrt(tmp31)
    tmp33 = tmp27 * tmp32
    tmp35 = tmp33 * tmp34
    tmp37 = tmp35 + tmp36
    tmp38 = 0.0078125
    tmp39 = tmp32 * tmp38
    tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp4, None)
    tl.store(in_out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp33, None)
    tl.store(out_ptr4 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp37, None)
    tl.store(out_ptr5 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp39, None)




# kernel path: /tmp/torchinductor_sahanp/jj/cjjabtageqlna7pma55ulz2ztthlvh6le5df624bfkjzhwnc2drv.py
# Topologically Sorted Source Nodes: [relu, dropout_2], Original ATen: [aten.relu, aten.native_dropout, aten.threshold_backward]
# Source node to ATen node mapping:
#   dropout_2 => gt_2, inductor_lookup_seed_default_2, inductor_random_default_1, mul_8, mul_9
#   relu => relu
# Graph fragment:
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%view_26,), kwargs = {})
#   %inductor_lookup_seed_default_2 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 2), kwargs = {})
#   %inductor_random_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 1, 2048], %inductor_lookup_seed_default_2, rand), kwargs = {})
#   %gt_2 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%inductor_random_default_1, 0.1), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gt_2, %relu), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, 1.1111111111111112), kwargs = {})
#   %le_2 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu, 0), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_dropout_relu_threshold_backward_3(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp7 = tl.load(in_ptr2 + (x0), xmask)
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
    tl.store(out_ptr1 + (x0), tmp4, xmask)
    tl.store(out_ptr2 + (x0), tmp13, xmask)
    tl.store(out_ptr3 + (x0), tmp15, xmask)




# kernel path: /tmp/torchinductor_sahanp/t3/ct3rvyleayubxysmg6wk2rh5mw6klic3jgkcgoznj5mzdyzwrnba.py
# Topologically Sorted Source Nodes: [x_9, x_10], Original ATen: [aten.rrelu_with_noise_functional, aten.abs, aten.le, aten.scalar_tensor, aten.where]
# Source node to ATen node mapping:
#   x_10 => abs_1, full_default_2, le_1, where_2
#   x_9 => le, mul_14, where
# Graph fragment:
#   %le : [num_users=2] = call_function[target=torch.ops.aten.le.Scalar](args = (%squeeze_2, 0), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_2, %uniform), kwargs = {})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le, %mul_14, %squeeze_2), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%where,), kwargs = {})
#   %le_1 : [num_users=2] = call_function[target=torch.ops.aten.le.Scalar](args = (%abs_1, 0.5), kwargs = {})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_2 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_1, %full_default_2, %where), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_abs_le_rrelu_with_noise_functional_scalar_tensor_where_4(in_out_ptr0, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp0 * tmp3
    tmp5 = tl.where(tmp2, tmp4, tmp0)
    tmp6 = tl_math.abs(tmp5)
    tmp7 = 0.5
    tmp8 = tmp6 <= tmp7
    tmp9 = tl.where(tmp8, tmp1, tmp5)
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tl.store(out_ptr1 + (x0), tmp8, xmask)
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)







def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26 = args
    args.clear()
    s0 = primals_1
    s1 = primals_2
    s2 = primals_3
    assert_size_stride(primals_4, (1, 1, s0, s1, s2), (s0*s1*s2, s0*s1*s2, s1*s2, s2, 1))
    assert_size_stride(primals_5, (128, 125), (125, 1))
    assert_size_stride(primals_6, (128, ), (1, ))
    assert_size_stride(primals_7, (384, ), (1, ))
    assert_size_stride(primals_8, (384, 128), (128, 1))
    assert_size_stride(primals_9, (128, 128), (128, 1))
    assert_size_stride(primals_10, (128, ), (1, ))
    assert_size_stride(primals_11, (128, ), (1, ))
    assert_size_stride(primals_12, (128, ), (1, ))
    assert_size_stride(primals_13, (384, 128), (128, 1))
    assert_size_stride(primals_14, (384, ), (1, ))
    assert_size_stride(primals_15, (128, 128), (128, 1))
    assert_size_stride(primals_16, (128, ), (1, ))
    assert_size_stride(primals_17, (128, ), (1, ))
    assert_size_stride(primals_18, (128, ), (1, ))
    assert_size_stride(primals_19, (2048, 128), (128, 1))
    assert_size_stride(primals_20, (2048, ), (1, ))
    assert_size_stride(primals_21, (128, 2048), (2048, 1))
    assert_size_stride(primals_22, (128, ), (1, ))
    assert_size_stride(primals_23, (128, ), (1, ))
    assert_size_stride(primals_24, (128, ), (1, ))
    assert_size_stride(primals_25, (10, 128), (128, 1))
    assert_size_stride(primals_26, (10, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.adaptive_max_pool3d]
        buf0 = torch.ops.aten.adaptive_max_pool3d.default(primals_4, [5, 5, 5])
        del primals_4
        buf1 = buf0[0]
        del buf0
        buf3 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_6, reinterpret_tensor(buf1, (1, 125), (125, 1), 0), reinterpret_tensor(primals_5, (125, 128), (1, 125), 0), alpha=1, beta=1, out=buf3)
        del primals_5
        del primals_6
        buf4 = empty_strided_cuda((1, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_7, buf3, reinterpret_tensor(primals_8, (128, 384), (1, 128), 0), alpha=1, beta=1, out=buf4)
        del primals_7
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf4, (1, 8, 1, 16), (128, 16, 384, 1), 0), reinterpret_tensor(buf4, (1, 8, 1, 16), (128, 16, 384, 1), 128), reinterpret_tensor(buf4, (1, 8, 1, 16), (128, 16, 384, 1), 256), None, True, 0.1)
        buf6 = buf5[0]
        buf7 = buf5[1]
        buf8 = buf5[2]
        buf9 = buf5[3]
        del buf5
        buf10 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf6, (1, 128), (128, 1), 0), reinterpret_tensor(primals_9, (128, 128), (1, 128), 0), out=buf10)
        buf11 = empty_strided_cuda((4, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [4], out=buf11)
        buf13 = empty_strided_cuda((1, 1, 128), (128, 128, 1), torch.bool)
        buf17 = reinterpret_tensor(buf10, (1, 1, 128), (128, 128, 1), 0); del buf10  # reuse
        buf18 = empty_strided_cuda((1, 1, 128), (128, 128, 1), torch.float32)
        buf56 = empty_strided_cuda((1, 1, 1), (1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [dropout, add, x_4], Original ATen: [aten.native_dropout, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_0[grid(1)](buf17, buf11, buf3, primals_10, primals_11, primals_12, buf13, buf18, buf56, 3, 1, 128, XBLOCK=1, num_warps=2, num_stages=1)
        del primals_10
        del primals_12
        buf19 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(reinterpret_tensor(primals_14, (128, ), (1, ), 0), reinterpret_tensor(buf18, (1, 128), (0, 1), 0), reinterpret_tensor(primals_13, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf19)
        buf20 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_1[grid(128)](buf20, 128, XBLOCK=128, num_warps=4, num_stages=1)
        buf21 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(reinterpret_tensor(primals_14, (256, ), (1, ), 128), buf20, reinterpret_tensor(primals_13, (128, 256), (1, 128), 16384), alpha=1, beta=1, out=buf21)
        del primals_14
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf22 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf19, (1, 8, 1, 16), (128, 16, 128, 1), 0), reinterpret_tensor(buf21, (1, 8, 1, 16), (128, 16, 256, 1), 0), reinterpret_tensor(buf21, (1, 8, 1, 16), (128, 16, 256, 1), 128), None, True, 0.1)
        buf23 = buf22[0]
        buf24 = buf22[1]
        buf25 = buf22[2]
        buf26 = buf22[3]
        del buf22
        buf27 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf23, (1, 128), (128, 1), 0), reinterpret_tensor(primals_15, (128, 128), (1, 128), 0), out=buf27)
        buf29 = empty_strided_cuda((1, 1, 128), (128, 128, 1), torch.bool)
        buf33 = reinterpret_tensor(buf27, (1, 1, 128), (128, 128, 1), 0); del buf27  # reuse
        buf34 = empty_strided_cuda((1, 1, 128), (128, 128, 1), torch.float32)
        buf55 = empty_strided_cuda((1, 1, 1), (1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [dropout_1, add_1, x_5], Original ATen: [aten.native_dropout, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_2[grid(1)](buf33, buf11, buf18, primals_16, primals_17, primals_18, buf29, buf34, buf55, 1, 1, 128, XBLOCK=1, num_warps=2, num_stages=1)
        del primals_16
        del primals_18
        buf35 = empty_strided_cuda((1, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf34, (1, 128), (0, 1), 0), reinterpret_tensor(primals_19, (128, 2048), (1, 128), 0), out=buf35)
        buf37 = empty_strided_cuda((1, 1, 2048), (2048, 2048, 1), torch.bool)
        buf38 = empty_strided_cuda((1, 1, 2048), (2048, 2048, 1), torch.float32)
        buf53 = empty_strided_cuda((1, 1, 2048), (2048, 2048, 1), torch.bool)
        # Topologically Sorted Source Nodes: [relu, dropout_2], Original ATen: [aten.relu, aten.native_dropout, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_dropout_relu_threshold_backward_3[grid(2048)](buf11, buf35, primals_20, buf37, buf38, buf53, 2, 2048, XBLOCK=256, num_warps=4, num_stages=1)
        del buf35
        del primals_20
        buf39 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf38, (1, 2048), (0, 1), 0), reinterpret_tensor(primals_21, (2048, 128), (1, 2048), 0), out=buf39)
        buf41 = empty_strided_cuda((1, 1, 128), (128, 128, 1), torch.bool)
        buf45 = reinterpret_tensor(buf39, (1, 1, 128), (128, 128, 1), 0); del buf39  # reuse
        buf46 = empty_strided_cuda((1, 1, 128), (128, 128, 1), torch.float32)
        buf54 = empty_strided_cuda((1, 1, 1), (1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [dropout_3, add_2, x_7], Original ATen: [aten.native_dropout, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_0[grid(1)](buf45, buf11, buf34, primals_22, primals_23, primals_24, buf41, buf46, buf54, 3, 1, 128, XBLOCK=1, num_warps=2, num_stages=1)
        del buf11
        del primals_22
        del primals_24
        # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.rrelu_with_noise_functional]
        buf48 = torch.ops.aten.uniform.default(reinterpret_tensor(buf46, (1, 128), (0, 1), 0), 0.125, 0.3333333333333333)
        buf49 = buf48
        del buf48
        buf47 = empty_strided_cuda((1, 128), (128, 1), torch.bool)
        buf50 = empty_strided_cuda((1, 128), (128, 1), torch.bool)
        buf51 = reinterpret_tensor(buf46, (1, 128), (128, 1), 0); del buf46  # reuse
        # Topologically Sorted Source Nodes: [x_9, x_10], Original ATen: [aten.rrelu_with_noise_functional, aten.abs, aten.le, aten.scalar_tensor, aten.where]
        stream0 = get_raw_stream(0)
        triton_poi_fused_abs_le_rrelu_with_noise_functional_scalar_tensor_where_4[grid(128)](buf51, buf49, buf47, buf50, 128, XBLOCK=128, num_warps=4, num_stages=1)
        buf52 = empty_strided_cuda((1, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_11], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_26, buf51, reinterpret_tensor(primals_25, (128, 10), (1, 128), 0), alpha=1, beta=1, out=buf52)
        del primals_26
    return (buf52, primals_11, primals_17, primals_23, reinterpret_tensor(buf1, (1, 125), (125, 1), 0), buf3, reinterpret_tensor(buf4, (1, 8, 1, 16), (128, 16, 384, 1), 0), reinterpret_tensor(buf4, (1, 8, 1, 16), (128, 16, 384, 1), 128), reinterpret_tensor(buf4, (1, 8, 1, 16), (128, 16, 384, 1), 256), buf6, buf7, buf8, buf9, buf13, buf17, reinterpret_tensor(buf18, (1, 128), (128, 1), 0), buf20, reinterpret_tensor(buf19, (1, 8, 1, 16), (128, 16, 128, 1), 0), reinterpret_tensor(buf21, (1, 8, 1, 16), (128, 16, 256, 1), 0), reinterpret_tensor(buf21, (1, 8, 1, 16), (128, 16, 256, 1), 128), buf23, buf24, buf25, buf26, buf29, buf33, reinterpret_tensor(buf34, (1, 128), (128, 1), 0), buf37, reinterpret_tensor(buf38, (1, 2048), (2048, 1), 0), buf41, buf45, buf47, buf49, buf50, buf51, primals_25, buf54, primals_21, buf53, primals_19, buf55, primals_15, reinterpret_tensor(primals_13, (128, 128), (128, 1), 0), buf56, primals_9, primals_8, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = 10
    primals_2 = 10
    primals_3 = 10
    primals_4 = rand_strided((1, 1, 10, 10, 10), (1000, 1000, 100, 10, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((128, 125), (125, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((2048, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((128, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((10, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
