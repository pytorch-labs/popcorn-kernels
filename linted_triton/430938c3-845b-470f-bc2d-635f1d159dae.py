# AOT ID: ['62_forward']
import torch
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


# kernel path: /tmp/torchinductor_sahanp/gc/cgcyz3pmdmqwd4jgtrx2jrlp5ubg3iaqiphthy2sk37madrjyvj4.py
# Topologically Sorted Source Nodes: [x, input_1], Original ATen: [aten.glu, aten._native_batch_norm_legit_functional, aten.mean]
# Source node to ATen node mapping:
#   input_1 => add, add_3, mean, mean_1, mul, mul_6, rsqrt, sub, var_mean
#   x => glu
# Graph fragment:
#   %glu : [num_users=3] = call_function[target=torch.ops.aten.glu.default](args = (%primals_1, 1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%glu, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%glu, %getitem_1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %unsqueeze_1), kwargs = {})
#   %add_3 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %unsqueeze_3), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_2, [0]), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_4, [0]), kwargs = {})
#   %copy_ : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_2, %mean), kwargs = {})
#   %copy__1 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_3, %mean_1), kwargs = {})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_glu_mean_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr3, out_ptr5, out_ptr7, out_ptr8, xnumel, r0_numel):
    XBLOCK: tl.constexpr = 1
    R0_BLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    tl.full([R0_BLOCK], True, tl.int1)
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 1024*x0), None)
    tmp1 = tl.load(in_ptr0 + (32768 + r0_1 + 1024*x0), None)
    tmp26 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [R0_BLOCK])
    tmp6 = tl.broadcast_to(tmp4, [R0_BLOCK])
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp9 = tl.full([1], 1024, tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 / tmp10
    tmp12 = tmp4 - tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [R0_BLOCK])
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp17 = 1024.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tmp22 = 1.0009775171065494
    tmp23 = tmp18 * tmp22
    tmp24 = 0.1
    tmp25 = tmp23 * tmp24
    tmp27 = 0.9
    tmp28 = tmp26 * tmp27
    tmp29 = tmp25 + tmp28
    tmp30 = 1.0
    tmp31 = tmp29 / tmp30
    tmp32 = tmp11 * tmp24
    tmp34 = tmp33 * tmp27
    tmp35 = tmp32 + tmp34
    tmp36 = tmp35 / tmp30
    tmp37 = tmp3 - tmp11
    tmp38 = tmp37 * tmp21
    tmp40 = tmp38 * tmp39
    tmp42 = tmp40 + tmp41
    tl.store(out_ptr0 + (r0_1 + 1024*x0), tmp3, None)
    tl.store(out_ptr3 + (x0), tmp21, None)
    tl.store(out_ptr5 + (x0), tmp31, None)
    tl.store(out_ptr7 + (x0), tmp36, None)
    tl.store(out_ptr8 + (r0_1 + 1024*x0), tmp42, None)
    tl.store(out_ptr1 + (x0), tmp11, None)


# kernel path: /tmp/torchinductor_sahanp/wd/cwdeb2ln4fieh6nukxservvivfcmwb5pznoxmzpacwsrnmhdnti6.py
# Topologically Sorted Source Nodes: [input_2, input_3, input_4, input_5], Original ATen: [aten.abs, aten.le, aten.scalar_tensor, aten.where, aten.glu, aten._native_batch_norm_legit_functional, aten.mean]
# Source node to ATen node mapping:
#   input_2 => abs_1, full_default, le, where
#   input_3 => glu_1
#   input_4 => add_4, add_7, mean_2, mean_3, mul_13, mul_7, rsqrt_1, sub_1, var_mean_1
#   input_5 => abs_2, le_1, where_1
# Graph fragment:
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%add_3,), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%abs_1, 0.5), kwargs = {})
#   %full_default : [num_users=5] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le, %full_default, %add_3), kwargs = {})
#   %glu_1 : [num_users=3] = call_function[target=torch.ops.aten.glu.default](args = (%where, 1), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%glu_1, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%glu_1, %getitem_3), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_5), kwargs = {})
#   %add_7 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_13, %unsqueeze_7), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_8, [0]), kwargs = {})
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_10, [0]), kwargs = {})
#   %abs_2 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%add_7,), kwargs = {})
#   %le_1 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%abs_2, 0.5), kwargs = {})
#   %where_1 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_1, %full_default, %add_7), kwargs = {})
#   %copy__2 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_6, %mean_2), kwargs = {})
#   %copy__3 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_7, %mean_3), kwargs = {})
import triton

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_abs_glu_le_mean_scalar_tensor_where_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr3, out_ptr5, out_ptr7, xnumel, r0_numel):
    XBLOCK: tl.constexpr = 1
    R0_BLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    tl.full([R0_BLOCK], True, tl.int1)
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 1024*x0), None)
    tmp6 = tl.load(in_ptr0 + (16384 + r0_1 + 1024*x0), None)
    tmp34 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl_math.abs(tmp0)
    tmp2 = 0.5
    tmp3 = tmp1 <= tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp3, tmp4, tmp0)
    tmp7 = tl_math.abs(tmp6)
    tmp8 = tmp7 <= tmp2
    tmp9 = tl.where(tmp8, tmp4, tmp6)
    tmp10 = tl.sigmoid(tmp9)
    tmp11 = tmp5 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [R0_BLOCK])
    tmp14 = tl.broadcast_to(tmp12, [R0_BLOCK])
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp17 = tl.full([1], 1024, tl.int32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 / tmp18
    tmp20 = tmp12 - tmp19
    tmp21 = tmp20 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [R0_BLOCK])
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp25 = 1024.0
    tmp26 = tmp24 / tmp25
    tmp27 = 1e-05
    tmp28 = tmp26 + tmp27
    tmp29 = libdevice.rsqrt(tmp28)
    tmp30 = 1.0009775171065494
    tmp31 = tmp26 * tmp30
    tmp32 = 0.1
    tmp33 = tmp31 * tmp32
    tmp35 = 0.9
    tmp36 = tmp34 * tmp35
    tmp37 = tmp33 + tmp36
    tmp38 = 1.0
    tmp39 = tmp37 / tmp38
    tmp40 = tmp19 * tmp32
    tmp42 = tmp41 * tmp35
    tmp43 = tmp40 + tmp42
    tmp44 = tmp43 / tmp38
    tmp45 = tmp11 - tmp19
    tmp46 = tmp45 * tmp29
    tmp48 = tmp46 * tmp47
    tmp50 = tmp48 + tmp49
    tmp51 = tl_math.abs(tmp50)
    tmp52 = tmp51 <= tmp2
    tmp53 = tl.where(tmp52, tmp4, tmp50)
    tl.store(out_ptr0 + (r0_1 + 1024*x0), tmp11, None)
    tl.store(out_ptr3 + (x0), tmp29, None)
    tl.store(out_ptr5 + (x0), tmp39, None)
    tl.store(out_ptr7 + (x0), tmp44, None)
    tl.store(in_out_ptr0 + (r0_1 + 1024*x0), tmp53, None)
    tl.store(out_ptr1 + (x0), tmp19, None)


# kernel path: /tmp/torchinductor_sahanp/x3/cx3taxtbofyoleelqnljtf6otkk3mlfvxyss5w5ptop2g54zrxhg.py
# Topologically Sorted Source Nodes: [input_2, input_6, input_7, input_8], Original ATen: [aten.scalar_tensor, aten.glu, aten._native_batch_norm_legit_functional, aten.mean, aten.abs, aten.le, aten.where]
# Source node to ATen node mapping:
#   input_2 => full_default
#   input_6 => glu_2
#   input_7 => add_11, add_8, mean_4, mean_5, mul_14, mul_20, rsqrt_2, sub_2, var_mean_2
#   input_8 => abs_3, le_2, where_2
# Graph fragment:
#   %full_default : [num_users=5] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %glu_2 : [num_users=3] = call_function[target=torch.ops.aten.glu.default](args = (%where_1, 1), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%glu_2, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-05), kwargs = {})
#   %rsqrt_2 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_8,), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%glu_2, %getitem_5), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_2), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_14, %unsqueeze_9), kwargs = {})
#   %add_11 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, %unsqueeze_11), kwargs = {})
#   %mean_4 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_14, [0]), kwargs = {})
#   %mean_5 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_16, [0]), kwargs = {})
#   %abs_3 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%add_11,), kwargs = {})
#   %le_2 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%abs_3, 0.5), kwargs = {})
#   %where_2 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_2, %full_default, %add_11), kwargs = {})
#   %copy__4 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_10, %mean_4), kwargs = {})
#   %copy__5 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_11, %mean_5), kwargs = {})
import triton

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_abs_glu_le_mean_scalar_tensor_where_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr3, out_ptr5, out_ptr7, xnumel, r0_numel):
    XBLOCK: tl.constexpr = 1
    R0_BLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    tl.full([R0_BLOCK], True, tl.int1)
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 1024*x0), None)
    tmp1 = tl.load(in_ptr0 + (8192 + r0_1 + 1024*x0), None)
    tmp26 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [R0_BLOCK])
    tmp6 = tl.broadcast_to(tmp4, [R0_BLOCK])
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp9 = tl.full([1], 1024, tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 / tmp10
    tmp12 = tmp4 - tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [R0_BLOCK])
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp17 = 1024.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tmp22 = 1.0009775171065494
    tmp23 = tmp18 * tmp22
    tmp24 = 0.1
    tmp25 = tmp23 * tmp24
    tmp27 = 0.9
    tmp28 = tmp26 * tmp27
    tmp29 = tmp25 + tmp28
    tmp30 = 1.0
    tmp31 = tmp29 / tmp30
    tmp32 = tmp11 * tmp24
    tmp34 = tmp33 * tmp27
    tmp35 = tmp32 + tmp34
    tmp36 = tmp35 / tmp30
    tmp37 = tmp3 - tmp11
    tmp38 = tmp37 * tmp21
    tmp40 = tmp38 * tmp39
    tmp42 = tmp40 + tmp41
    tmp43 = tl_math.abs(tmp42)
    tmp44 = 0.5
    tmp45 = tmp43 <= tmp44
    tmp46 = 0.0
    tmp47 = tl.where(tmp45, tmp46, tmp42)
    tl.store(out_ptr0 + (r0_1 + 1024*x0), tmp3, None)
    tl.store(out_ptr3 + (x0), tmp21, None)
    tl.store(out_ptr5 + (x0), tmp31, None)
    tl.store(out_ptr7 + (x0), tmp36, None)
    tl.store(in_out_ptr0 + (r0_1 + 1024*x0), tmp47, None)
    tl.store(out_ptr1 + (x0), tmp11, None)


# kernel path: /tmp/torchinductor_sahanp/sh/cshvmnwhg3dv43rmg3xohrpfr5i2bur24yk3wu2pazht45y67djl.py
# Topologically Sorted Source Nodes: [input_2, input_9, input_10, input_11], Original ATen: [aten.scalar_tensor, aten.glu, aten._native_batch_norm_legit_functional, aten.mean, aten.abs, aten.le, aten.where]
# Source node to ATen node mapping:
#   input_10 => add_12, add_15, mean_6, mean_7, mul_21, mul_27, rsqrt_3, sub_3, var_mean_3
#   input_11 => abs_4, le_3, where_3
#   input_2 => full_default
#   input_9 => glu_3
# Graph fragment:
#   %full_default : [num_users=5] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %glu_3 : [num_users=3] = call_function[target=torch.ops.aten.glu.default](args = (%where_2, 1), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%glu_3, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt_3 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_12,), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%glu_3, %getitem_7), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_3), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_21, %unsqueeze_13), kwargs = {})
#   %add_15 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_27, %unsqueeze_15), kwargs = {})
#   %mean_6 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_20, [0]), kwargs = {})
#   %mean_7 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_22, [0]), kwargs = {})
#   %abs_4 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%add_15,), kwargs = {})
#   %le_3 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%abs_4, 0.5), kwargs = {})
#   %where_3 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_3, %full_default, %add_15), kwargs = {})
#   %copy__6 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_14, %mean_6), kwargs = {})
#   %copy__7 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_15, %mean_7), kwargs = {})
import triton

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_abs_glu_le_mean_scalar_tensor_where_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr3, out_ptr5, out_ptr7, xnumel, r0_numel):
    XBLOCK: tl.constexpr = 1
    R0_BLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    tl.full([R0_BLOCK], True, tl.int1)
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 1024*x0), None)
    tmp1 = tl.load(in_ptr0 + (4096 + r0_1 + 1024*x0), None)
    tmp26 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [R0_BLOCK])
    tmp6 = tl.broadcast_to(tmp4, [R0_BLOCK])
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp9 = tl.full([1], 1024, tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 / tmp10
    tmp12 = tmp4 - tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [R0_BLOCK])
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp17 = 1024.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tmp22 = 1.0009775171065494
    tmp23 = tmp18 * tmp22
    tmp24 = 0.1
    tmp25 = tmp23 * tmp24
    tmp27 = 0.9
    tmp28 = tmp26 * tmp27
    tmp29 = tmp25 + tmp28
    tmp30 = 1.0
    tmp31 = tmp29 / tmp30
    tmp32 = tmp11 * tmp24
    tmp34 = tmp33 * tmp27
    tmp35 = tmp32 + tmp34
    tmp36 = tmp35 / tmp30
    tmp37 = tmp3 - tmp11
    tmp38 = tmp37 * tmp21
    tmp40 = tmp38 * tmp39
    tmp42 = tmp40 + tmp41
    tmp43 = tl_math.abs(tmp42)
    tmp44 = 0.5
    tmp45 = tmp43 <= tmp44
    tmp46 = 0.0
    tmp47 = tl.where(tmp45, tmp46, tmp42)
    tl.store(out_ptr0 + (r0_1 + 1024*x0), tmp3, None)
    tl.store(out_ptr3 + (x0), tmp21, None)
    tl.store(out_ptr5 + (x0), tmp31, None)
    tl.store(out_ptr7 + (x0), tmp36, None)
    tl.store(in_out_ptr0 + (r0_1 + 1024*x0), tmp47, None)
    tl.store(out_ptr1 + (x0), tmp11, None)


# kernel path: /tmp/torchinductor_sahanp/zi/czi7wsig7et5yehbuawakkei6jbqvekaggqadq3gbka2v5iuqprw.py
# Topologically Sorted Source Nodes: [input_2, input_12, input_13, input_14], Original ATen: [aten.scalar_tensor, aten.glu, aten._native_batch_norm_legit_functional, aten.mean, aten.abs, aten.le, aten.where]
# Source node to ATen node mapping:
#   input_12 => glu_4
#   input_13 => add_16, add_19, mean_8, mean_9, mul_28, mul_34, rsqrt_4, sub_4, var_mean_4
#   input_14 => abs_5, le_4, where_4
#   input_2 => full_default
# Graph fragment:
#   %full_default : [num_users=5] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %glu_4 : [num_users=3] = call_function[target=torch.ops.aten.glu.default](args = (%where_3, 1), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%glu_4, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_8, 1e-05), kwargs = {})
#   %rsqrt_4 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_16,), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%glu_4, %getitem_9), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_4), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, %unsqueeze_17), kwargs = {})
#   %add_19 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_34, %unsqueeze_19), kwargs = {})
#   %mean_8 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_26, [0]), kwargs = {})
#   %mean_9 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_28, [0]), kwargs = {})
#   %abs_5 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%add_19,), kwargs = {})
#   %le_4 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%abs_5, 0.5), kwargs = {})
#   %where_4 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_4, %full_default, %add_19), kwargs = {})
#   %copy__8 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_18, %mean_8), kwargs = {})
#   %copy__9 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_19, %mean_9), kwargs = {})
import triton

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_abs_glu_le_mean_scalar_tensor_where_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr3, out_ptr5, out_ptr7, xnumel, r0_numel):
    XBLOCK: tl.constexpr = 1
    R0_BLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    tl.full([R0_BLOCK], True, tl.int1)
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 1024*x0), None)
    tmp1 = tl.load(in_ptr0 + (2048 + r0_1 + 1024*x0), None)
    tmp24 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [R0_BLOCK])
    tmp6 = tl.broadcast_to(tmp4, [R0_BLOCK])
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp9 = tl.full([1], 1024, tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 / tmp10
    tmp12 = tmp4 - tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [R0_BLOCK])
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp17 = 1024.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tmp22 = 0.1
    tmp23 = tmp11 * tmp22
    tmp25 = 0.9
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 + tmp26
    tmp28 = 1.0
    tmp29 = tmp27 / tmp28
    tmp30 = 1.0009775171065494
    tmp31 = tmp18 * tmp30
    tmp32 = tmp31 * tmp22
    tmp34 = tmp33 * tmp25
    tmp35 = tmp32 + tmp34
    tmp36 = tmp35 / tmp28
    tmp37 = tmp3 - tmp11
    tmp38 = tmp37 * tmp21
    tmp40 = tmp38 * tmp39
    tmp42 = tmp40 + tmp41
    tmp43 = tl_math.abs(tmp42)
    tmp44 = 0.5
    tmp45 = tmp43 <= tmp44
    tmp46 = 0.0
    tmp47 = tl.where(tmp45, tmp46, tmp42)
    tl.store(out_ptr0 + (r0_1 + 1024*x0), tmp3, None)
    tl.store(out_ptr3 + (x0), tmp21, None)
    tl.store(out_ptr5 + (x0), tmp29, None)
    tl.store(out_ptr7 + (x0), tmp36, None)
    tl.store(in_out_ptr0 + (r0_1 + 1024*x0), tmp47, None)
    tl.store(out_ptr1 + (x0), tmp11, None)


# kernel path: /tmp/torchinductor_sahanp/tm/ctmyscpnat73fbhit3ygq3pv3llnj6p4wbgc7a4ef2qwt3zhiqrn.py
# Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.glu]
# Source node to ATen node mapping:
#   input_15 => glu_5
# Graph fragment:
#   %glu_5 : [num_users=1] = call_function[target=torch.ops.aten.glu.default](args = (%where_4, 1), kwargs = {})
import triton

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_glu_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (1024 + x0), xmask)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x0), tmp3, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21 = args
    args.clear()
    assert_size_stride(primals_1, (1, 64, 32, 32), (65536, 1024, 32, 1))
    assert_size_stride(primals_2, (32, ), (1, ))
    assert_size_stride(primals_3, (32, ), (1, ))
    assert_size_stride(primals_4, (32, ), (1, ))
    assert_size_stride(primals_5, (32, ), (1, ))
    assert_size_stride(primals_6, (16, ), (1, ))
    assert_size_stride(primals_7, (16, ), (1, ))
    assert_size_stride(primals_8, (16, ), (1, ))
    assert_size_stride(primals_9, (16, ), (1, ))
    assert_size_stride(primals_10, (8, ), (1, ))
    assert_size_stride(primals_11, (8, ), (1, ))
    assert_size_stride(primals_12, (8, ), (1, ))
    assert_size_stride(primals_13, (8, ), (1, ))
    assert_size_stride(primals_14, (4, ), (1, ))
    assert_size_stride(primals_15, (4, ), (1, ))
    assert_size_stride(primals_16, (4, ), (1, ))
    assert_size_stride(primals_17, (4, ), (1, ))
    assert_size_stride(primals_18, (2, ), (1, ))
    assert_size_stride(primals_19, (2, ), (1, ))
    assert_size_stride(primals_20, (2, ), (1, ))
    assert_size_stride(primals_21, (2, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        buf1 = empty_strided_cuda((1, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf4 = empty_strided_cuda((1, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf5 = empty_strided_cuda((1, 32, 32, 32), (32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, input_1], Original ATen: [aten.glu, aten._native_batch_norm_legit_functional, aten.mean]
        get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_glu_mean_0[grid(32)](primals_1, primals_3, primals_2, primals_4, primals_5, buf0, buf1, buf4, primals_3, primals_2, buf5, 32, 1024, num_warps=8, num_stages=1)
        del primals_1
        del primals_2
        del primals_3
        buf6 = empty_strided_cuda((1, 16, 32, 32), (16384, 1024, 32, 1), torch.float32)
        buf7 = empty_strided_cuda((1, 16, 1, 1), (16, 1, 1, 1), torch.float32)
        buf10 = empty_strided_cuda((1, 16, 1, 1), (16, 1, 1, 1), torch.float32)
        buf11 = empty_strided_cuda((1, 16, 32, 32), (16384, 1024, 32, 1), torch.float32)
        buf12 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [input_2, input_3, input_4, input_5], Original ATen: [aten.abs, aten.le, aten.scalar_tensor, aten.where, aten.glu, aten._native_batch_norm_legit_functional, aten.mean]
        get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_abs_glu_le_mean_scalar_tensor_where_1[grid(16)](buf12, buf5, primals_7, primals_6, primals_8, primals_9, buf6, buf7, buf10, primals_7, primals_6, 16, 1024, num_warps=8, num_stages=1)
        del buf5
        del primals_6
        del primals_7
        buf13 = empty_strided_cuda((1, 8, 32, 32), (8192, 1024, 32, 1), torch.float32)
        buf14 = empty_strided_cuda((1, 8, 1, 1), (8, 1, 1, 1), torch.float32)
        buf17 = empty_strided_cuda((1, 8, 1, 1), (8, 1, 1, 1), torch.float32)
        buf18 = empty_strided_cuda((1, 8, 32, 32), (8192, 1024, 32, 1), torch.float32)
        buf19 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [input_2, input_6, input_7, input_8], Original ATen: [aten.scalar_tensor, aten.glu, aten._native_batch_norm_legit_functional, aten.mean, aten.abs, aten.le, aten.where]
        get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_abs_glu_le_mean_scalar_tensor_where_2[grid(8)](buf19, buf12, primals_11, primals_10, primals_12, primals_13, buf13, buf14, buf17, primals_11, primals_10, 8, 1024, num_warps=8, num_stages=1)
        del primals_10
        del primals_11
        buf20 = empty_strided_cuda((1, 4, 32, 32), (4096, 1024, 32, 1), torch.float32)
        buf21 = empty_strided_cuda((1, 4, 1, 1), (4, 1, 1, 1), torch.float32)
        buf24 = empty_strided_cuda((1, 4, 1, 1), (4, 1, 1, 1), torch.float32)
        buf25 = empty_strided_cuda((1, 4, 32, 32), (4096, 1024, 32, 1), torch.float32)
        buf26 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [input_2, input_9, input_10, input_11], Original ATen: [aten.scalar_tensor, aten.glu, aten._native_batch_norm_legit_functional, aten.mean, aten.abs, aten.le, aten.where]
        get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_abs_glu_le_mean_scalar_tensor_where_3[grid(4)](buf26, buf19, primals_15, primals_14, primals_16, primals_17, buf20, buf21, buf24, primals_15, primals_14, 4, 1024, num_warps=8, num_stages=1)
        del primals_14
        del primals_15
        buf27 = empty_strided_cuda((1, 2, 32, 32), (2048, 1024, 32, 1), torch.float32)
        buf28 = empty_strided_cuda((1, 2, 1, 1), (2, 1, 1, 1), torch.float32)
        buf31 = empty_strided_cuda((1, 2, 1, 1), (2, 1, 1, 1), torch.float32)
        buf32 = empty_strided_cuda((1, 2, 32, 32), (2048, 1024, 32, 1), torch.float32)
        buf33 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [input_2, input_12, input_13, input_14], Original ATen: [aten.scalar_tensor, aten.glu, aten._native_batch_norm_legit_functional, aten.mean, aten.abs, aten.le, aten.where]
        get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_abs_glu_le_mean_scalar_tensor_where_4[grid(2)](buf33, buf26, primals_18, primals_19, primals_20, primals_21, buf27, buf28, buf31, primals_18, primals_19, 2, 1024, num_warps=8, num_stages=1)
        del primals_18
        del primals_19
        buf34 = empty_strided_cuda((1, 1, 32, 32), (1024, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.glu]
        get_raw_stream(0)
        triton_poi_fused_glu_5[grid(1024)](buf33, buf34, 1024, XBLOCK=256, num_warps=4, num_stages=1)
    return (buf34, primals_4, primals_5, primals_8, primals_9, primals_12, primals_13, primals_16, primals_17, primals_20, primals_21, buf0, buf1, buf4, buf6, buf7, buf10, buf12, buf13, buf14, buf17, buf19, buf20, buf21, buf24, buf26, buf27, buf28, buf31, buf33, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 64, 32, 32), (65536, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
