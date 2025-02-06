# AOT ID: ['128_forward']
import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import (
    grid,
)
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


# kernel path: /tmp/torchinductor_sahanp/ep/cepibrsx4dyamqayjdypbbzvc35o2367e3zfpfxa5dfif25bxtgo.py
# Topologically Sorted Source Nodes: [relu, x_1], Original ATen: [aten.relu, aten.native_layer_norm]
# Source node to ATen node mapping:
#   relu => relu
#   x_1 => add, add_1, mul, mul_1, rsqrt, sub, var_mean
# Graph fragment:
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%addmm,), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%relu, [1]), kwargs = {correction: 0, keepdim: True})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%relu, %getitem_1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %primals_4), kwargs = {})
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %primals_5), kwargs = {})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_layer_norm_relu_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel):
    XBLOCK: tl.constexpr = 1
    R0_BLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    tl.full([1], xoffset, tl.int32)
    tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    tl.full([R0_BLOCK], True, tl.int1)
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), None)
    tmp23 = tl.load(in_ptr1 + (r0_0), None)
    tmp25 = tl.load(in_ptr2 + (r0_0), None)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp3 = tl.broadcast_to(tmp2, [R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp3, [R0_BLOCK])
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tl.full([1], 256, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp3 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [R0_BLOCK])
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = 256.0
    tmp17 = tmp15 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = tmp2 - tmp10
    tmp22 = tmp21 * tmp20
    tmp24 = tmp22 * tmp23
    tmp26 = tmp24 + tmp25
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([1], 0, tl.int32)), tmp20, None)
    tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [R0_BLOCK])), tmp26, None)
    tl.store(out_ptr0 + (tl.full([1], 0, tl.int32)), tmp10, None)


# kernel path: /tmp/torchinductor_sahanp/pw/cpwl2vzahuxkfg6ppm5thq6lv7eqd5ejqq5yknhil67xzm2wl7rv.py
# Topologically Sorted Source Nodes: [relu_1, x_2], Original ATen: [aten.relu, aten.native_layer_norm]
# Source node to ATen node mapping:
#   relu_1 => relu_1
#   x_2 => add_2, add_3, mul_2, mul_3, rsqrt_1, sub_1, var_mean_1
# Graph fragment:
#   %relu_1 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%addmm_1,), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%relu_1, [1]), kwargs = {correction: 0, keepdim: True})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%relu_1, %getitem_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %primals_8), kwargs = {})
#   %add_3 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %primals_9), kwargs = {})
import triton

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_layer_norm_relu_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel):
    XBLOCK: tl.constexpr = 1
    R0_BLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    tl.full([1], xoffset, tl.int32)
    tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    tl.full([R0_BLOCK], True, tl.int1)
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), None)
    tmp23 = tl.load(in_ptr1 + (r0_0), None)
    tmp25 = tl.load(in_ptr2 + (r0_0), None)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp3 = tl.broadcast_to(tmp2, [R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp3, [R0_BLOCK])
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tl.full([1], 512, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp3 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [R0_BLOCK])
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = 512.0
    tmp17 = tmp15 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = tmp2 - tmp10
    tmp22 = tmp21 * tmp20
    tmp24 = tmp22 * tmp23
    tmp26 = tmp24 + tmp25
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([1], 0, tl.int32)), tmp20, None)
    tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [R0_BLOCK])), tmp26, None)
    tl.store(out_ptr0 + (tl.full([1], 0, tl.int32)), tmp10, None)


# kernel path: /tmp/torchinductor_sahanp/pq/cpqyjdk6bs3yedbogjktnmkpwy7ksg7efpdwxjo6i2xoxmz5cuiq.py
# Topologically Sorted Source Nodes: [relu_3, x_4], Original ATen: [aten.relu, aten.native_layer_norm]
# Source node to ATen node mapping:
#   relu_3 => relu_3
#   x_4 => add_6, add_7, mul_6, mul_7, rsqrt_3, sub_3, var_mean_3
# Graph fragment:
#   %relu_3 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%addmm_3,), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%relu_3, [1]), kwargs = {correction: 0, keepdim: True})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt_3 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_6,), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%relu_3, %getitem_7), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_3), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %primals_16), kwargs = {})
#   %add_7 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %primals_17), kwargs = {})
import triton

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_layer_norm_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    R0_BLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), None)
    tmp23 = tl.load(in_ptr1 + (r0_0), None)
    tmp25 = tl.load(in_ptr2 + (r0_0), None)
    tmp1 = tl.full([1, 1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp7 = tl.sum(tmp5, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp3 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.sum(tmp13, 1)[:, None]
    tmp16 = 128.0
    tmp17 = tmp15 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = tmp2 - tmp10
    tmp22 = tmp21 * tmp20
    tmp24 = tmp22 * tmp23
    tmp26 = tmp24 + tmp25
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp20, None)
    tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp26, None)
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp10, None)


# kernel path: /tmp/torchinductor_sahanp/45/c45kwkgkymcs3lkef5qwtktqjfwaavv7lynkrbu3dtduoq5jfyci.py
# Topologically Sorted Source Nodes: [relu_4, x_5, x_6], Original ATen: [aten.relu, aten.native_layer_norm, aten.abs, aten.le]
# Source node to ATen node mapping:
#   relu_4 => relu_4
#   x_5 => add_8, add_9, mul_8, mul_9, rsqrt_4, sub_4, var_mean_4
#   x_6 => abs_1, le
# Graph fragment:
#   %relu_4 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%addmm_4,), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%relu_4, [1]), kwargs = {correction: 0, keepdim: True})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_8, 1e-05), kwargs = {})
#   %rsqrt_4 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_8,), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%relu_4, %getitem_9), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_4), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, %primals_20), kwargs = {})
#   %add_9 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %primals_21), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%add_9,), kwargs = {})
#   %le : [num_users=2] = call_function[target=torch.ops.aten.le.Scalar](args = (%abs_1, 0.5), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_abs_le_native_layer_norm_relu_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    R0_BLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), None)
    tmp23 = tl.load(in_ptr1 + (r0_0), None)
    tmp25 = tl.load(in_ptr2 + (r0_0), None)
    tmp1 = tl.full([1, 1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp7 = tl.sum(tmp5, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp3 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.sum(tmp13, 1)[:, None]
    tmp16 = 64.0
    tmp17 = tmp15 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = tmp2 - tmp10
    tmp22 = tmp21 * tmp20
    tmp24 = tmp22 * tmp23
    tmp26 = tmp24 + tmp25
    tmp27 = tl_math.abs(tmp26)
    tmp28 = 0.5
    tmp29 = tmp27 <= tmp28
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp20, None)
    tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp26, None)
    tl.store(out_ptr2 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp29, None)
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp10, None)


# kernel path: /tmp/torchinductor_sahanp/qm/cqmmg5rmvgylqoyxh7okljxt2svcybnzy7avovyuu5mj6umw6owv.py
# Topologically Sorted Source Nodes: [x_8], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   x_8 => add_10, add_11, convert_element_type, convert_element_type_1, iota, mul_10, mul_11
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, 1), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, 0), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_10, torch.float32), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, 0.0), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_11, 0.5), kwargs = {})
#   %convert_element_type_1 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_11, torch.int64), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_4(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)


# kernel path: /tmp/torchinductor_sahanp/xl/cxlo5dojl4ts3mgvomoipfzcc72yjhvtnmv65w2sgspxpb4tijsd.py
# Topologically Sorted Source Nodes: [x_8, loss], Original ATen: [aten._unsafe_index, aten.mul, aten.sum, aten.add, aten.sqrt, aten.div, aten.zeros_like, aten.fill, aten.sub, aten.clamp_min, aten.eq, aten.where, aten.mean]
# Source node to ATen node mapping:
#   loss => add_14, add_16, clamp_min, div, full_default_1, full_default_2, full_default_3, full_default_4, mean, mul_14, mul_17, sqrt, sub_5, sub_6, sum_1, where_1, where_2
#   x_8 => _unsafe_index
# Graph fragment:
#   %_unsafe_index : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%view_1, [None, None, %unsqueeze, %convert_element_type_1]), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_2, %view_2), kwargs = {})
#   %sum_1 : [num_users=3] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_14, [1]), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%sum_1, 9.999999960041972e-13), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_14, %add_14), kwargs = {})
#   %sqrt : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%mul_17,), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_1, %sqrt), kwargs = {})
#   %full_default_1 : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([1], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default_2, %div), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Scalar](args = (%div, 0.0), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_6, 0), kwargs = {})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%full_default_3, %sub_5, %full_default_1), kwargs = {})
#   %full_default_4 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], False), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%full_default_4, %clamp_min, %full_default_1), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_1, %where_2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%add_16,), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__unsafe_index_add_clamp_min_div_eq_fill_mean_mul_sqrt_sub_sum_where_zeros_like_5(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel):
    XBLOCK: tl.constexpr = 1
    R0_BLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    tl.full([1], xoffset, tl.int32)
    tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    tl.full([R0_BLOCK], True, tl.int1)
    r0_1 = r0_index // 16
    r0_0 = (r0_index % 16)
    r0_2 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (r0_0), None, eviction_policy='evict_last')
    tmp1 = tl.full([R0_BLOCK], 8, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr1 + (tmp8 + 8*tmp4), None, eviction_policy='evict_last').to(tl.int1)
    tmp10 = tl.load(in_ptr2 + (tmp8 + 8*tmp4), None, eviction_policy='evict_last')
    tmp11 = 0.0
    tmp12 = tl.where(tmp9, tmp11, tmp10)
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [R0_BLOCK])
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp17 = 9.999999960041972e-13
    tmp18 = tmp16 + tmp17
    tmp19 = tmp18 * tmp18
    tmp20 = libdevice.sqrt(tmp19)
    tmp21 = tmp16 / tmp20
    tmp22 = 1.0
    tmp23 = tmp22 - tmp21
    tmp24 = tl.full([1], True, tl.int1)
    tmp25 = tl.where(tmp24, tmp23, tmp11)
    tmp26 = tmp21 - tmp11
    tmp27 = triton_helpers.maximum(tmp26, tmp11)
    tmp28 = tl.full([1], False, tl.int1)
    tmp29 = tl.where(tmp28, tmp27, tmp11)
    tmp30 = tmp25 + tmp29
    tmp31 = tmp30 / tmp22
    tl.store(out_ptr0 + (tl.broadcast_to(r0_2, [R0_BLOCK])), tmp12, None)
    tl.store(out_ptr2 + (tl.full([1], 0, tl.int32)), tmp31, None)
    tl.store(out_ptr1 + (tl.full([1], 0, tl.int32)), tmp16, None)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21 = args
    args.clear()
    assert_size_stride(primals_1, (1, 128), (128, 1))
    assert_size_stride(primals_2, (256, 128), (128, 1))
    assert_size_stride(primals_3, (256, ), (1, ))
    assert_size_stride(primals_4, (256, ), (1, ))
    assert_size_stride(primals_5, (256, ), (1, ))
    assert_size_stride(primals_6, (512, 256), (256, 1))
    assert_size_stride(primals_7, (512, ), (1, ))
    assert_size_stride(primals_8, (512, ), (1, ))
    assert_size_stride(primals_9, (512, ), (1, ))
    assert_size_stride(primals_10, (256, 512), (512, 1))
    assert_size_stride(primals_11, (256, ), (1, ))
    assert_size_stride(primals_12, (256, ), (1, ))
    assert_size_stride(primals_13, (256, ), (1, ))
    assert_size_stride(primals_14, (128, 256), (256, 1))
    assert_size_stride(primals_15, (128, ), (1, ))
    assert_size_stride(primals_16, (128, ), (1, ))
    assert_size_stride(primals_17, (128, ), (1, ))
    assert_size_stride(primals_18, (64, 128), (128, 1))
    assert_size_stride(primals_19, (64, ), (1, ))
    assert_size_stride(primals_20, (64, ), (1, ))
    assert_size_stride(primals_21, (64, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_3, primals_1, reinterpret_tensor(primals_2, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf0)
        del primals_2
        del primals_3
        buf1 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        buf2 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        buf4 = buf2; del buf2  # reuse
        buf5 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [relu, x_1], Original ATen: [aten.relu, aten.native_layer_norm]
        get_raw_stream(0)
        triton_per_fused_native_layer_norm_relu_0[grid(1)](buf4, buf0, primals_4, primals_5, buf1, buf5, 1, 256, num_warps=2, num_stages=1)
        del primals_5
        buf6 = empty_strided_cuda((1, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_7, buf5, reinterpret_tensor(primals_6, (256, 512), (1, 256), 0), alpha=1, beta=1, out=buf6)
        del primals_7
        buf7 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        buf8 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        buf10 = buf8; del buf8  # reuse
        buf11 = empty_strided_cuda((1, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [relu_1, x_2], Original ATen: [aten.relu, aten.native_layer_norm]
        get_raw_stream(0)
        triton_per_fused_native_layer_norm_relu_1[grid(1)](buf10, buf6, primals_8, primals_9, buf7, buf11, 1, 512, num_warps=4, num_stages=1)
        del primals_9
        buf12 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, buf11, reinterpret_tensor(primals_10, (512, 256), (1, 512), 0), alpha=1, beta=1, out=buf12)
        del primals_11
        buf13 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        buf14 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        buf16 = buf14; del buf14  # reuse
        buf17 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [relu_2, x_3], Original ATen: [aten.relu, aten.native_layer_norm]
        get_raw_stream(0)
        triton_per_fused_native_layer_norm_relu_0[grid(1)](buf16, buf12, primals_12, primals_13, buf13, buf17, 1, 256, num_warps=2, num_stages=1)
        del primals_13
        buf18 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_15, buf17, reinterpret_tensor(primals_14, (256, 128), (1, 256), 0), alpha=1, beta=1, out=buf18)
        del primals_15
        buf19 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        buf20 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        buf22 = buf20; del buf20  # reuse
        buf23 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [relu_3, x_4], Original ATen: [aten.relu, aten.native_layer_norm]
        get_raw_stream(0)
        triton_per_fused_native_layer_norm_relu_2[grid(1)](buf22, buf18, primals_16, primals_17, buf19, buf23, 1, 128, XBLOCK=1, num_warps=2, num_stages=1)
        del primals_17
        buf24 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_19, buf23, reinterpret_tensor(primals_18, (128, 64), (1, 128), 0), alpha=1, beta=1, out=buf24)
        del primals_19
        buf25 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        buf26 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        buf28 = buf26; del buf26  # reuse
        buf29 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
        buf30 = empty_strided_cuda((1, 64), (64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [relu_4, x_5, x_6], Original ATen: [aten.relu, aten.native_layer_norm, aten.abs, aten.le]
        get_raw_stream(0)
        triton_per_fused_abs_le_native_layer_norm_relu_3[grid(1)](buf28, buf24, primals_20, primals_21, buf25, buf29, buf30, 1, 64, XBLOCK=1, num_warps=2, num_stages=1)
        del primals_21
        buf31 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_4[grid(16)](buf31, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf32 = empty_strided_cuda((1, 1, 16, 16), (256, 256, 16, 1), torch.float32)
        buf33 = empty_strided_cuda((1, ), (1, ), torch.float32)
        buf34 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [x_8, loss], Original ATen: [aten._unsafe_index, aten.mul, aten.sum, aten.add, aten.sqrt, aten.div, aten.zeros_like, aten.fill, aten.sub, aten.clamp_min, aten.eq, aten.where, aten.mean]
        get_raw_stream(0)
        triton_per_fused__unsafe_index_add_clamp_min_div_eq_fill_mean_mul_sqrt_sub_sum_where_zeros_like_5[grid(1)](buf31, buf30, buf29, buf32, buf33, buf34, 1, 256, num_warps=2, num_stages=1)
        del buf29
    return (buf34, primals_4, primals_8, primals_12, primals_16, primals_20, primals_1, buf0, buf1, buf4, buf5, buf6, buf7, buf10, buf11, buf12, buf13, buf16, buf17, buf18, buf19, buf22, buf23, buf24, buf25, buf28, buf30, buf31, buf32, buf33, primals_18, primals_14, primals_10, primals_6, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((64, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
