# AOT ID: ['177_forward']
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


# kernel path: /tmp/torchinductor_sahanp/w7/cw765y2x2trktb35sct72ouxgstilg32vgamulfxifgzejvfjrlp.py
# Topologically Sorted Source Nodes: [x, x_1, x_3, x_5, x_6, x_7, x_9], Original ATen: [aten._native_batch_norm_legit, aten._native_batch_norm_legit_functional, aten.abs, aten.gt, aten.sign, aten.mul, aten.sub, aten.where]
# Source node to ATen node mapping:
#   x => add, mul, rsqrt, sub, var_mean
#   x_1 => add_2, add_3, add_4, add_5, mul_1, mul_2, mul_3, mul_4, mul_5, mul_6, mul_7, rsqrt_1, sub_1, var_mean_1
#   x_3 => add_10, add_7, add_8, add_9, mul_10, mul_11, mul_12, mul_13, mul_14, mul_8, mul_9, rsqrt_2, sub_2, var_mean_2
#   x_5 => abs_1, gt, mul_15, mul_16, sign, sub_3, where
#   x_6 => add_11, mul_17, rsqrt_3, sub_4, var_mean_3
#   x_7 => add_13, add_14, add_15, add_16, mul_18, mul_19, mul_20, mul_21, mul_22, mul_23, mul_24, rsqrt_4, sub_5, var_mean_4
#   x_9 => add_18, add_19, add_20, mul_26, mul_27, mul_28, mul_29, mul_30, rsqrt_5, var_mean_5
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%primals_1, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_1, %getitem_1), kwargs = {})
#   %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%mul, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul, %getitem_3), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_2, 0.1), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_3, 0.9), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %mul_3), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_4, 1.0009775171065494), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, 0.1), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_4, 0.9), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %mul_6), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_1), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %unsqueeze_3), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%unsqueeze_4, [0, 2, 3, 4]), kwargs = {correction: 0, keepdim: True})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-05), kwargs = {})
#   %rsqrt_2 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_7,), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze_4, %getitem_5), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_2), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_5, 0.1), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_8, 0.9), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %mul_10), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_7, 1.0009775171065494), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_11, 0.1), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_9, 0.9), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_12, %mul_13), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, %unsqueeze_7), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_10), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%squeeze_8,), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%abs_1, 0.5), kwargs = {})
#   %sign : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%squeeze_8,), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign, 0.5), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%squeeze_8, %mul_15), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_8, 0), kwargs = {})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt, %sub_3, %mul_16), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%where, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt_3 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_11,), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where, %getitem_7), kwargs = {})
#   %mul_17 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_3), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%mul_17, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_8, 1e-05), kwargs = {})
#   %rsqrt_4 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_13,), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_17, %getitem_9), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %rsqrt_4), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_11, 0.1), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_13, 0.9), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_19, %mul_20), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_13, 1.0009775171065494), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_21, 0.1), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_14, 0.9), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_22, %mul_23), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_18, %unsqueeze_12), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_24, %unsqueeze_14), kwargs = {})
#   %var_mean_5 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%unsqueeze_15, [0, 2, 3, 4]), kwargs = {correction: 0, keepdim: True})
#   %add_18 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_10, 1e-05), kwargs = {})
#   %rsqrt_5 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_18,), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_14, 0.1), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_18, 0.9), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %mul_27), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_16, 1.0009775171065494), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, 0.1), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_19, 0.9), kwargs = {})
#   %add_20 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_29, %mul_30), kwargs = {})
#   %copy__1 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_3, %add_3), kwargs = {})
#   %copy__2 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_4, %add_4), kwargs = {})
#   %copy__4 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_8, %add_8), kwargs = {})
#   %copy__5 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_9, %add_9), kwargs = {})
#   %copy__7 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_13, %add_14), kwargs = {})
#   %copy__8 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_14, %add_15), kwargs = {})
#   %copy__10 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_18, %add_19), kwargs = {})
#   %copy__11 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_19, %add_20), kwargs = {})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit__native_batch_norm_legit_functional_abs_gt_mul_sign_sub_where_0(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, out_ptr0, out_ptr1, out_ptr3, out_ptr5, out_ptr7, out_ptr8, out_ptr10, out_ptr12, out_ptr14, out_ptr15, out_ptr16, out_ptr18, out_ptr20, out_ptr22, out_ptr23, out_ptr24, out_ptr25, out_ptr27, out_ptr29, xnumel, r0_numel):
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
    tmp39 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp69 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp73 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp78 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp80 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp129 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp133 = tl.load(in_ptr10 + (x0), None, eviction_policy='evict_last')
    tmp138 = tl.load(in_ptr11 + (x0), None, eviction_policy='evict_last')
    tmp140 = tl.load(in_ptr12 + (x0), None, eviction_policy='evict_last')
    tmp158 = tl.load(in_ptr13 + (x0), None, eviction_policy='evict_last')
    tmp162 = tl.load(in_ptr14 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [R0_BLOCK])
    tmp3 = tl.broadcast_to(tmp1, [R0_BLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tl.full([1], 1024, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 / tmp7
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [R0_BLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 1024.0
    tmp15 = tmp13 / tmp14
    tmp16 = 1e-05
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tmp19 = tmp0 - tmp8
    tmp20 = tmp19 * tmp18
    tmp21 = tl.broadcast_to(tmp20, [R0_BLOCK])
    tmp23 = tl.broadcast_to(tmp21, [R0_BLOCK])
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp26 = tmp25 / tmp7
    tmp27 = tmp21 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [R0_BLOCK])
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp29, 0))
    tmp32 = tmp31 / tmp14
    tmp33 = tmp32 + tmp16
    tmp34 = libdevice.rsqrt(tmp33)
    tmp35 = 1.0009775171065494
    tmp36 = tmp32 * tmp35
    tmp37 = 0.1
    tmp38 = tmp36 * tmp37
    tmp40 = 0.9
    tmp41 = tmp39 * tmp40
    tmp42 = tmp38 + tmp41
    tmp43 = tmp26 * tmp37
    tmp45 = tmp44 * tmp40
    tmp46 = tmp43 + tmp45
    tmp47 = tmp20 - tmp26
    tmp48 = tmp47 * tmp34
    tmp50 = tmp48 * tmp49
    tmp52 = tmp50 + tmp51
    tmp53 = tl.broadcast_to(tmp52, [R0_BLOCK])
    tmp55 = tl.broadcast_to(tmp53, [R0_BLOCK])
    tmp57 = triton_helpers.promote_to_tensor(tl.sum(tmp55, 0))
    tmp58 = tmp57 / tmp7
    tmp59 = tmp53 - tmp58
    tmp60 = tmp59 * tmp59
    tmp61 = tl.broadcast_to(tmp60, [R0_BLOCK])
    tmp63 = triton_helpers.promote_to_tensor(tl.sum(tmp61, 0))
    tmp64 = tmp63 / tmp14
    tmp65 = tmp64 + tmp16
    tmp66 = libdevice.rsqrt(tmp65)
    tmp67 = tmp64 * tmp35
    tmp68 = tmp67 * tmp37
    tmp70 = tmp69 * tmp40
    tmp71 = tmp68 + tmp70
    tmp72 = tmp58 * tmp37
    tmp74 = tmp73 * tmp40
    tmp75 = tmp72 + tmp74
    tmp76 = tmp52 - tmp58
    tmp77 = tmp76 * tmp66
    tmp79 = tmp77 * tmp78
    tmp81 = tmp79 + tmp80
    tmp82 = tl_math.abs(tmp81)
    tmp83 = 0.5
    tmp84 = tmp82 > tmp83
    tmp85 = tl.full([1], 0, tl.int32)
    tmp86 = tmp85 < tmp81
    tmp87 = tmp86.to(tl.int8)
    tmp88 = tmp81 < tmp85
    tmp89 = tmp88.to(tl.int8)
    tmp90 = tmp87 - tmp89
    tmp91 = tmp90.to(tmp81.dtype)
    tmp92 = tmp91 * tmp83
    tmp93 = tmp81 - tmp92
    tmp94 = 0.0
    tmp95 = tmp81 * tmp94
    tmp96 = tl.where(tmp84, tmp93, tmp95)
    tmp97 = tl.broadcast_to(tmp96, [R0_BLOCK])
    tmp99 = tl.broadcast_to(tmp97, [R0_BLOCK])
    tmp101 = triton_helpers.promote_to_tensor(tl.sum(tmp99, 0))
    tmp102 = tmp101 / tmp7
    tmp103 = tmp97 - tmp102
    tmp104 = tmp103 * tmp103
    tmp105 = tl.broadcast_to(tmp104, [R0_BLOCK])
    tmp107 = triton_helpers.promote_to_tensor(tl.sum(tmp105, 0))
    tmp108 = tmp107 / tmp14
    tmp109 = tmp108 + tmp16
    tmp110 = libdevice.rsqrt(tmp109)
    tmp111 = tmp96 - tmp102
    tmp112 = tmp111 * tmp110
    tmp113 = tl.broadcast_to(tmp112, [R0_BLOCK])
    tmp115 = tl.broadcast_to(tmp113, [R0_BLOCK])
    tmp117 = triton_helpers.promote_to_tensor(tl.sum(tmp115, 0))
    tmp118 = tmp117 / tmp7
    tmp119 = tmp113 - tmp118
    tmp120 = tmp119 * tmp119
    tmp121 = tl.broadcast_to(tmp120, [R0_BLOCK])
    tmp123 = triton_helpers.promote_to_tensor(tl.sum(tmp121, 0))
    tmp124 = tmp123 / tmp14
    tmp125 = tmp124 + tmp16
    tmp126 = libdevice.rsqrt(tmp125)
    tmp127 = tmp124 * tmp35
    tmp128 = tmp127 * tmp37
    tmp130 = tmp129 * tmp40
    tmp131 = tmp128 + tmp130
    tmp132 = tmp118 * tmp37
    tmp134 = tmp133 * tmp40
    tmp135 = tmp132 + tmp134
    tmp136 = tmp112 - tmp118
    tmp137 = tmp136 * tmp126
    tmp139 = tmp137 * tmp138
    tmp141 = tmp139 + tmp140
    tmp142 = tl.broadcast_to(tmp141, [R0_BLOCK])
    tmp144 = tl.broadcast_to(tmp142, [R0_BLOCK])
    tmp146 = triton_helpers.promote_to_tensor(tl.sum(tmp144, 0))
    tmp147 = tmp146 / tmp7
    tmp148 = tmp142 - tmp147
    tmp149 = tmp148 * tmp148
    tmp150 = tl.broadcast_to(tmp149, [R0_BLOCK])
    tmp152 = triton_helpers.promote_to_tensor(tl.sum(tmp150, 0))
    tmp153 = tmp152 / tmp14
    tmp154 = tmp153 + tmp16
    tmp155 = libdevice.rsqrt(tmp154)
    tmp156 = tmp153 * tmp35
    tmp157 = tmp156 * tmp37
    tmp159 = tmp158 * tmp40
    tmp160 = tmp157 + tmp159
    tmp161 = tmp147 * tmp37
    tmp163 = tmp162 * tmp40
    tmp164 = tmp161 + tmp163
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp18, None)
    tl.store(out_ptr3 + (x0), tmp34, None)
    tl.store(out_ptr5 + (x0), tmp42, None)
    tl.store(out_ptr7 + (x0), tmp46, None)
    tl.store(out_ptr10 + (x0), tmp66, None)
    tl.store(out_ptr12 + (x0), tmp71, None)
    tl.store(out_ptr14 + (x0), tmp75, None)
    tl.debug_barrier()
    tl.store(in_out_ptr2 + (x0), tmp110, None)
    tl.store(out_ptr18 + (x0), tmp126, None)
    tl.store(out_ptr20 + (x0), tmp131, None)
    tl.store(out_ptr22 + (x0), tmp135, None)
    tl.store(in_out_ptr1 + (r0_1 + 1024*x0), tmp141, None)
    tl.store(out_ptr25 + (x0), tmp155, None)
    tl.store(out_ptr27 + (x0), tmp160, None)
    tl.store(out_ptr29 + (x0), tmp164, None)
    tl.store(out_ptr0 + (x0), tmp8, None)
    tl.store(out_ptr1 + (x0), tmp26, None)
    tl.store(out_ptr8 + (x0), tmp58, None)
    tl.store(out_ptr15 + (x0), tmp102, None)
    tl.store(out_ptr16 + (x0), tmp118, None)
    tl.store(out_ptr23 + (x0), tmp147, None)
    tl.store(out_ptr24 + (x0), tmp152, None)


# kernel path: /tmp/torchinductor_sahanp/bw/cbwvvzao6qvlu7nlv7uyctlxbm2g4q5twxyxdpc532r6fuundita.py
# Topologically Sorted Source Nodes: [x_9, x_11, randn_like, mul, positive, randn_like_1, mul_1, negative, loss], Original ATen: [aten._native_batch_norm_legit_functional, aten.abs, aten.gt, aten.sign, aten.mul, aten.sub, aten.where, aten.randn_like, aten.add, aten.norm, aten.scalar_tensor, aten.div, aten.eq, aten.masked_fill]
# Source node to ATen node mapping:
#   loss => add_24, add_25, pow_1, pow_3, sub_8, sub_9, sum_1, sum_2
#   mul => mul_34
#   mul_1 => mul_35
#   negative => add_23
#   positive => add_22
#   randn_like => inductor_lookup_seed_default, inductor_random_default_1
#   randn_like_1 => inductor_lookup_seed_default_1, inductor_random_default
#   x_11 => abs_2, gt_1, mul_32, mul_33, sign_1, sub_7, where_1
#   x_9 => add_18, add_21, mul_25, mul_31, rsqrt_5, sub_6, var_mean_5
# Graph fragment:
#   %var_mean_5 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%unsqueeze_15, [0, 2, 3, 4]), kwargs = {correction: 0, keepdim: True})
#   %add_18 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_10, 1e-05), kwargs = {})
#   %rsqrt_5 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_18,), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze_15, %getitem_11), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %rsqrt_5), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %unsqueeze_18), kwargs = {})
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_31, %unsqueeze_21), kwargs = {})
#   %abs_2 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%squeeze_17,), kwargs = {})
#   %gt_1 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%abs_2, 0.5), kwargs = {})
#   %sign_1 : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%squeeze_17,), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign_1, 0.5), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%squeeze_17, %mul_32), kwargs = {})
#   %mul_33 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_17, 0), kwargs = {})
#   %where_1 : [num_users=4] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %sub_7, %mul_33), kwargs = {})
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 10, 32, 32], %inductor_lookup_seed_default, randn), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%inductor_random_default_1, 0.1), kwargs = {})
#   %add_22 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_1, %mul_34), kwargs = {})
#   %inductor_lookup_seed_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 1), kwargs = {})
#   %inductor_random_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 10, 32, 32], %inductor_lookup_seed_default_1, randn), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%inductor_random_default, 0.2), kwargs = {})
#   %add_23 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_1, %mul_35), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_1, %add_22), kwargs = {})
#   %add_24 : [num_users=2] = call_function[target=torch.ops.aten.add.Scalar](args = (%sub_8, 1e-06), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_24, 2.0), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [3]), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_1, %add_23), kwargs = {})
#   %add_25 : [num_users=2] = call_function[target=torch.ops.aten.add.Scalar](args = (%sub_9, 1e-06), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_25, 2.0), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_3, [3]), kwargs = {})
#   %full_default : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_25, %unsqueeze_23), kwargs = {})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%unsqueeze_23, 0), kwargs = {})
#   %where_3 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default, %div_1), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_24, %unsqueeze_25), kwargs = {})
#   %eq_1 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%unsqueeze_25, 0), kwargs = {})
#   %where_4 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_1, %full_default, %div_2), kwargs = {})
import triton

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_abs_add_div_eq_gt_masked_fill_mul_norm_randn_like_scalar_tensor_sign_sub_where_1(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, load_seed_offset, load_seed_offset1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 320
    R0_BLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_1 = r0_index
    x0 = xindex
    x3 = xindex // 32
    tmp5 = tl.load(in_out_ptr0 + (r0_1 + 32*x0), xmask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = r0_1 + 32*x0
    tmp2 = tl.randn(tmp0, (tmp1).to(tl.uint32))
    tmp3 = tl.load(in_ptr0 + load_seed_offset1)
    tmp4 = tl.randn(tmp3, (tmp1).to(tl.uint32))
    tmp7 = tmp5 - tmp6
    tmp9 = 1024.0
    tmp10 = tmp8 / tmp9
    tmp11 = 1e-05
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.rsqrt(tmp12)
    tmp14 = tmp7 * tmp13
    tmp16 = tmp14 * tmp15
    tmp18 = tmp16 + tmp17
    tmp19 = tl_math.abs(tmp18)
    tmp20 = 0.5
    tmp21 = tmp19 > tmp20
    tmp22 = tl.full([1, 1], 0, tl.int32)
    tmp23 = tmp22 < tmp18
    tmp24 = tmp23.to(tl.int8)
    tmp25 = tmp18 < tmp22
    tmp26 = tmp25.to(tl.int8)
    tmp27 = tmp24 - tmp26
    tmp28 = tmp27.to(tmp18.dtype)
    tmp29 = tmp28 * tmp20
    tmp30 = tmp18 - tmp29
    tmp31 = 0.0
    tmp32 = tmp18 * tmp31
    tmp33 = tl.where(tmp21, tmp30, tmp32)
    tmp34 = 0.1
    tmp35 = tmp4 * tmp34
    tmp36 = tmp33 + tmp35
    tmp37 = tmp33 - tmp36
    tmp38 = 1e-06
    tmp39 = tmp37 + tmp38
    tmp40 = tmp39 * tmp39
    tmp41 = tl.broadcast_to(tmp40, [XBLOCK, R0_BLOCK])
    tmp43 = tl.where(xmask, tmp41, 0)
    tmp44 = tl.sum(tmp43, 1)[:, None]
    tmp45 = 0.2
    tmp46 = tmp2 * tmp45
    tmp47 = tmp33 + tmp46
    tmp48 = tmp33 - tmp47
    tmp49 = tmp48 + tmp38
    tmp50 = tmp49 * tmp49
    tmp51 = tl.broadcast_to(tmp50, [XBLOCK, R0_BLOCK])
    tmp53 = tl.where(xmask, tmp51, 0)
    tmp54 = tl.sum(tmp53, 1)[:, None]
    tmp55 = libdevice.sqrt(tmp54)
    tmp56 = tmp55 == tmp31
    tmp57 = tmp49 / tmp55
    tmp58 = tl.where(tmp56, tmp31, tmp57)
    tmp59 = libdevice.sqrt(tmp44)
    tmp60 = tmp59 == tmp31
    tmp61 = tmp39 / tmp59
    tmp62 = tl.where(tmp60, tmp31, tmp61)
    tl.store(out_ptr0 + (r0_1 + 32*x0), tmp21, xmask)
    tl.store(in_out_ptr1 + (r0_1 + 32*x0), tmp58, xmask)
    tl.store(in_out_ptr2 + (r0_1 + 32*x0), tmp62, xmask)
    tl.store(out_ptr1 + (x0), tmp44, xmask)
    tl.store(out_ptr2 + (x0), tmp54, xmask)


# kernel path: /tmp/torchinductor_sahanp/5g/c5gsyvyy5h7wqlr4siqhc4hvyix22cobx2qqdt5srvchdt343q5u.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten.norm, aten.add, aten.sub, aten.clamp_min, aten.mean, aten.ge]
# Source node to ATen node mapping:
#   loss => add_26, clamp_min, mean, pow_2, pow_4, sub_10
# Graph fragment:
#   %pow_2 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %pow_4 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_2, 0.5), kwargs = {})
#   %add_26 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%pow_2, 1.0), kwargs = {})
#   %sub_10 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_26, %pow_4), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_10, 0), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%clamp_min,), kwargs = {})
#   %ge : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_10, 0), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_clamp_min_ge_mean_norm_sub_2(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel):
    XBLOCK: tl.constexpr = 1
    r0_numel = 320
    R0_BLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    tl.full([1], xoffset, tl.int32)
    tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_mask = r0_index < r0_numel
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, other=0.0)
    tmp4 = tl.load(in_ptr1 + (r0_0), r0_mask, other=0.0)
    tmp1 = libdevice.sqrt(tmp0)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp5 = libdevice.sqrt(tmp4)
    tmp6 = tmp3 - tmp5
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tl.broadcast_to(tmp8, [R0_BLOCK])
    tmp11 = tl.where(r0_mask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = tmp6 >= tmp7
    tmp14 = 320.0
    tmp15 = tmp12 / tmp14
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [R0_BLOCK])), tmp13, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([1], 0, tl.int32)), tmp15, None)


# kernel path: /tmp/torchinductor_sahanp/36/c36fsokdqemetuxowygjraqhmkvfqomdt7tvtednpyk77ioklyqe.py
# Topologically Sorted Source Nodes: [add_], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   add_ => add_1
# Graph fragment:
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_2, 1), kwargs = {})
#   %copy_ : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_2, %add_1), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_3(in_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp1 + tmp2
    tl.store(out_ptr1 + (tl.full([XBLOCK], 0, tl.int32)), tmp3, None)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21 = args
    args.clear()
    assert_size_stride(primals_1, (1, 10, 32, 32), (10240, 1024, 32, 1))
    assert_size_stride(primals_2, (), ())
    assert_size_stride(primals_3, (10, ), (1, ))
    assert_size_stride(primals_4, (10, ), (1, ))
    assert_size_stride(primals_5, (10, ), (1, ))
    assert_size_stride(primals_6, (10, ), (1, ))
    assert_size_stride(primals_7, (), ())
    assert_size_stride(primals_8, (10, ), (1, ))
    assert_size_stride(primals_9, (10, ), (1, ))
    assert_size_stride(primals_10, (10, ), (1, ))
    assert_size_stride(primals_11, (10, ), (1, ))
    assert_size_stride(primals_12, (), ())
    assert_size_stride(primals_13, (10, ), (1, ))
    assert_size_stride(primals_14, (10, ), (1, ))
    assert_size_stride(primals_15, (10, ), (1, ))
    assert_size_stride(primals_16, (10, ), (1, ))
    assert_size_stride(primals_17, (), ())
    assert_size_stride(primals_18, (10, ), (1, ))
    assert_size_stride(primals_19, (10, ), (1, ))
    assert_size_stride(primals_20, (10, ), (1, ))
    assert_size_stride(primals_21, (10, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((1, 10, 1, 1), (10, 1, 10, 10), torch.float32)
        buf3 = reinterpret_tensor(buf1, (1, 10, 1, 1), (10, 1, 1, 1), 0); del buf1  # reuse
        buf0 = empty_strided_cuda((1, 10, 1, 1), (10, 1, 1, 1), torch.float32)
        buf4 = empty_strided_cuda((1, 10, 1, 1), (10, 1, 1, 1), torch.float32)
        buf7 = empty_strided_cuda((1, 10, 1, 1), (10, 1, 1, 1), torch.float32)
        buf8 = empty_strided_cuda((1, 10, 32, 32), (10240, 1024, 32, 1), torch.float32)
        buf9 = empty_strided_cuda((1, 10, 1, 1, 1), (10, 1, 1, 1, 1), torch.float32)
        buf12 = empty_strided_cuda((1, 10, 1, 1, 1), (10, 1, 1, 1, 1), torch.float32)
        buf13 = reinterpret_tensor(buf8, (1, 10, 1, 32, 32), (10240, 1024, 10240, 32, 1), 0); del buf8  # reuse
        buf15 = empty_strided_cuda((1, 10, 1, 1), (10, 1, 10, 10), torch.float32)
        buf17 = reinterpret_tensor(buf15, (1, 10, 1, 1), (10, 1, 1, 1), 0); del buf15  # reuse
        buf14 = empty_strided_cuda((1, 10, 1, 1), (10, 1, 1, 1), torch.float32)
        buf18 = empty_strided_cuda((1, 10, 1, 1), (10, 1, 1, 1), torch.float32)
        buf21 = empty_strided_cuda((1, 10, 1, 1), (10, 1, 1, 1), torch.float32)
        buf22 = reinterpret_tensor(buf13, (1, 10, 32, 32), (10240, 1024, 32, 1), 0); del buf13  # reuse
        buf23 = empty_strided_cuda((1, 10, 1, 1, 1), (10, 1, 10, 10, 10), torch.float32)
        buf24 = empty_strided_cuda((1, 10, 1, 1, 1), (10, 1, 10, 10, 10), torch.float32)
        buf26 = empty_strided_cuda((1, 10, 1, 1, 1), (10, 1, 10, 10, 10), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_1, x_3, x_5, x_6, x_7, x_9], Original ATen: [aten._native_batch_norm_legit, aten._native_batch_norm_legit_functional, aten.abs, aten.gt, aten.sign, aten.mul, aten.sub, aten.where]
        get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit__native_batch_norm_legit_functional_abs_gt_mul_sign_sub_where_0[grid(10)](buf3, buf22, buf17, primals_1, primals_4, primals_3, primals_5, primals_6, primals_9, primals_8, primals_10, primals_11, primals_14, primals_13, primals_15, primals_16, primals_19, primals_18, buf0, buf4, buf7, primals_4, primals_3, buf9, buf12, primals_9, primals_8, buf14, buf18, buf21, primals_14, primals_13, buf23, buf24, buf26, primals_19, primals_18, 10, 1024, num_warps=8, num_stages=1)
        del primals_13
        del primals_14
        del primals_18
        del primals_19
        del primals_3
        del primals_4
        del primals_8
        del primals_9
        buf29 = empty_strided_cuda((2, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [2], out=buf29)
        buf31 = empty_strided_cuda((1, 10, 32, 32), (10240, 1024, 32, 1), torch.float32)
        buf30 = empty_strided_cuda((1, 10, 32, 32), (10240, 1024, 32, 1), torch.float32)
        buf27 = reinterpret_tensor(buf22, (1, 10, 1, 32, 32), (10240, 1024, 10240, 32, 1), 0); del buf22  # reuse
        buf28 = empty_strided_cuda((1, 10, 32, 32), (10240, 1024, 32, 1), torch.bool)
        buf32 = empty_strided_cuda((1, 10, 32), (320, 32, 1), torch.float32)
        buf33 = empty_strided_cuda((1, 10, 32), (320, 32, 1), torch.float32)
        buf36 = buf31; del buf31  # reuse
        buf37 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [x_9, x_11, randn_like, mul, positive, randn_like_1, mul_1, negative, loss], Original ATen: [aten._native_batch_norm_legit_functional, aten.abs, aten.gt, aten.sign, aten.mul, aten.sub, aten.where, aten.randn_like, aten.add, aten.norm, aten.scalar_tensor, aten.div, aten.eq, aten.masked_fill]
        get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_abs_add_div_eq_gt_masked_fill_mul_norm_randn_like_scalar_tensor_sign_sub_where_1[grid(320)](buf27, buf36, buf37, buf29, buf23, buf24, primals_20, primals_21, buf28, buf32, buf33, 1, 0, 320, 32, XBLOCK=1, num_warps=2, num_stages=1)
        del buf24
        del buf27
        del buf29
        del primals_21
        buf34 = empty_strided_cuda((), (), torch.float32)
        buf35 = empty_strided_cuda((1, 10, 32), (320, 32, 1), torch.bool)
        buf70 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [loss], Original ATen: [aten.norm, aten.add, aten.sub, aten.clamp_min, aten.mean, aten.ge]
        get_raw_stream(0)
        triton_per_fused_add_clamp_min_ge_mean_norm_sub_2[grid(1)](buf70, buf32, buf33, buf35, 1, 320, num_warps=4, num_stages=1)
        del buf32
        del buf33
        # Topologically Sorted Source Nodes: [add_], Original ATen: [aten.add]
        get_raw_stream(0)
        triton_poi_fused_add_3[grid(1)](primals_2, primals_2, 1, XBLOCK=1, num_warps=1, num_stages=1)
        del primals_2
        # Topologically Sorted Source Nodes: [add__1], Original ATen: [aten.add]
        get_raw_stream(0)
        triton_poi_fused_add_3[grid(1)](primals_7, primals_7, 1, XBLOCK=1, num_warps=1, num_stages=1)
        del primals_7
        # Topologically Sorted Source Nodes: [add__2], Original ATen: [aten.add]
        get_raw_stream(0)
        triton_poi_fused_add_3[grid(1)](primals_12, primals_12, 1, XBLOCK=1, num_warps=1, num_stages=1)
        del primals_12
        # Topologically Sorted Source Nodes: [add__3], Original ATen: [aten.add]
        get_raw_stream(0)
        triton_poi_fused_add_3[grid(1)](primals_17, primals_17, 1, XBLOCK=1, num_warps=1, num_stages=1)
        del primals_17
    return (buf70, primals_1, primals_5, primals_6, primals_10, primals_11, primals_15, primals_16, primals_20, buf0, buf3, buf4, buf7, buf9, buf12, buf14, buf17, buf18, buf21, reinterpret_tensor(buf26, (10, ), (1, ), 0), buf28, buf35, buf36, buf37, reinterpret_tensor(buf23, (1, 10, 1, 1, 1), (10, 1, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 10, 32, 32), (10240, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_3 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_8 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_13 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_18 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
