# AOT ID: ['2_forward']
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


# kernel path: /tmp/torchinductor_sahanp/iu/ciutaees7d6sbpkbylpjv6eelpqdockibjbguhbg2n4ehhxzwmaj.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x => constant_pad_nd
# Graph fragment:
#   %constant_pad_nd : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%primals_1, [1, 1, 1, 1], 0.5), kwargs = {})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_constant_pad_nd_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 13068
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 66) % 66)
    x0 = (xindex % 66)
    x2 = xindex // 4356
    x4 = xindex
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x0
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-65) + x0 + 64*x1 + 4096*x2), tmp10 & xmask, other=0.5)
    tl.store(out_ptr0 + (x4), tmp11, xmask)


# kernel path: /tmp/torchinductor_sahanp/sg/csgj7vt7twguaj2vcf4ehrz4ndlnrub775f7mao3qienyebrket3.py
# Topologically Sorted Source Nodes: [x_2, x_3, x_4], Original ATen: [aten.convolution, aten._native_batch_norm_legit_functional, aten.mish]
# Source node to ATen node mapping:
#   x_2 => convolution
#   x_3 => add_1, add_2, add_3, add_4, mul, mul_1, mul_2, mul_3, mul_4, mul_5, mul_6, rsqrt, sub, var_mean
#   x_4 => exp, gt, log1p, mul_7, tanh, where
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%view, %primals_2, %primals_3, [1], [0], [1], False, [0], 1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convolution, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %getitem_1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze, 0.1), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_5, 0.9), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %mul_2), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_2, 1.000229726625316), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, 0.1), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_6, 0.9), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %mul_5), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %unsqueeze), kwargs = {})
#   %add_4 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %unsqueeze_1), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%add_4,), kwargs = {})
#   %log1p : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp,), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_4, 20), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %add_4, %log1p), kwargs = {})
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%where,), kwargs = {})
#   %mul_7 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_4, %tanh), kwargs = {})
#   %copy__1 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_5, %add_2), kwargs = {})
#   %copy__2 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_6, %add_3), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_convolution_mish_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, out_ptr4, out_ptr6, out_ptr8, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32
    r0_numel = 4354
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp4_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        r0_1 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 4354*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight, roffset == 0
        )
        tmp4_mean = tl.where(r0_mask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(r0_mask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(r0_mask & xmask, tmp4_weight_next, tmp4_weight)
        tl.store(in_out_ptr0 + (r0_1 + 4354*x0), tmp2, r0_mask & xmask)
    tmp7, tmp8, tmp9 = triton_helpers.welford(tmp4_mean, tmp4_m2, tmp4_weight, 1)
    tmp4 = tmp7[:, None]
    tmp5 = tmp8[:, None]
    tmp9[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tmp19 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = 4354.0
    tmp11 = tmp5 / tmp10
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = 1.000229726625316
    tmp16 = tmp11 * tmp15
    tmp17 = 0.1
    tmp18 = tmp16 * tmp17
    tmp20 = 0.9
    tmp21 = tmp19 * tmp20
    tmp22 = tmp18 + tmp21
    tmp23 = tmp4 * tmp17
    tmp25 = tmp24 * tmp20
    tmp26 = tmp23 + tmp25
    tl.store(out_ptr2 + (x0), tmp14, xmask)
    tl.store(out_ptr4 + (x0), tmp22, xmask)
    tl.store(out_ptr6 + (x0), tmp26, xmask)
    tmp30 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        r0_1 = r0_index
        tmp27 = tl.load(in_out_ptr0 + (r0_1 + 4354*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp28 = tmp27 - tmp4
        tmp29 = tmp28 * tmp14
        tmp31 = tmp29 * tmp30
        tmp33 = tmp31 + tmp32
        tmp34 = 20.0
        tmp35 = tmp33 > tmp34
        tmp36 = tl_math.exp(tmp33)
        tmp37 = libdevice.log1p(tmp36)
        tmp38 = tl.where(tmp35, tmp33, tmp37)
        tmp39 = libdevice.tanh(tmp38)
        tmp40 = tmp33 * tmp39
        tl.store(out_ptr8 + (r0_1 + 4354*x0), tmp40, r0_mask & xmask)


# kernel path: /tmp/torchinductor_sahanp/bj/cbjbhigp6okasn4tgfvhsra63slgubuz4wmyy3kvh6nwexhiwb6n.py
# Topologically Sorted Source Nodes: [x_5, x_6, x_7], Original ATen: [aten.convolution, aten._native_batch_norm_legit_functional, aten.hardswish]
# Source node to ATen node mapping:
#   x_5 => convolution_1
#   x_6 => add_6, add_7, add_8, add_9, mul_10, mul_11, mul_12, mul_13, mul_14, mul_8, mul_9, rsqrt_1, sub_1, var_mean_1
#   x_7 => add_10, clamp_max, clamp_min, div, mul_15
# Graph fragment:
#   %convolution_1 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_7, %primals_9, %primals_10, [1], [0], [1], False, [0], 1), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convolution_1, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_6,), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %getitem_3), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_3, 0.1), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_12, 0.9), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %mul_10), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_5, 1.0002298322224776), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_11, 0.1), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_13, 0.9), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_12, %mul_13), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, %unsqueeze_2), kwargs = {})
#   %add_9 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_3), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_9, 3), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_10, 0), kwargs = {})
#   %clamp_max : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 6), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_9, %clamp_max), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_15, 6), kwargs = {})
#   %copy__4 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_12, %add_7), kwargs = {})
#   %copy__5 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_13, %add_8), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_convolution_hardswish_2(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, out_ptr4, out_ptr6, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 64
    r0_numel = 4352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp4_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        r0_1 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 4352*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight, roffset == 0
        )
        tmp4_mean = tl.where(r0_mask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(r0_mask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(r0_mask & xmask, tmp4_weight_next, tmp4_weight)
        tl.store(in_out_ptr0 + (r0_1 + 4352*x0), tmp2, r0_mask & xmask)
    tmp7, tmp8, tmp9 = triton_helpers.welford(tmp4_mean, tmp4_m2, tmp4_weight, 1)
    tmp4 = tmp7[:, None]
    tmp5 = tmp8[:, None]
    tmp9[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tmp19 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = 4352.0
    tmp11 = tmp5 / tmp10
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = 1.0002298322224776
    tmp16 = tmp11 * tmp15
    tmp17 = 0.1
    tmp18 = tmp16 * tmp17
    tmp20 = 0.9
    tmp21 = tmp19 * tmp20
    tmp22 = tmp18 + tmp21
    tmp23 = tmp4 * tmp17
    tmp25 = tmp24 * tmp20
    tmp26 = tmp23 + tmp25
    tl.store(out_ptr2 + (x0), tmp14, xmask)
    tl.store(out_ptr4 + (x0), tmp22, xmask)
    tl.store(out_ptr6 + (x0), tmp26, xmask)
    tmp30 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        r0_1 = r0_index
        tmp27 = tl.load(in_out_ptr0 + (r0_1 + 4352*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp28 = tmp27 - tmp4
        tmp29 = tmp28 * tmp14
        tmp31 = tmp29 * tmp30
        tmp33 = tmp31 + tmp32
        tmp34 = 3.0
        tmp35 = tmp33 + tmp34
        tmp36 = 0.0
        tmp37 = triton_helpers.maximum(tmp35, tmp36)
        tmp38 = 6.0
        tmp39 = triton_helpers.minimum(tmp37, tmp38)
        tmp40 = tmp33 * tmp39
        tmp41 = 0.16666666666666666
        tmp42 = tmp40 * tmp41
        tl.store(in_out_ptr1 + (r0_1 + 4352*x0), tmp42, r0_mask & xmask)


# kernel path: /tmp/torchinductor_sahanp/vw/cvwgbo32qyoki772v7lrrncux7pmml6aj2bwzst55vbogti65oqq.py
# Topologically Sorted Source Nodes: [x_8, x_9, x_10], Original ATen: [aten.convolution, aten._native_batch_norm_legit_functional, aten.mish]
# Source node to ATen node mapping:
#   x_10 => exp_1, gt_1, log1p_1, mul_23, tanh_1, where_1
#   x_8 => convolution_2
#   x_9 => add_12, add_13, add_14, add_15, mul_16, mul_17, mul_18, mul_19, mul_20, mul_21, mul_22, rsqrt_2, sub_2, var_mean_2
# Graph fragment:
#   %convolution_2 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %primals_16, %primals_17, [1], [0], [1], False, [0], 1), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convolution_2, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-05), kwargs = {})
#   %rsqrt_2 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_12,), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %getitem_5), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_2), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_6, 0.1), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_19, 0.9), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %mul_18), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_8, 1.0002299379167625), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, 0.1), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_20, 0.9), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, %mul_21), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %unsqueeze_4), kwargs = {})
#   %add_15 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_22, %unsqueeze_5), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%add_15,), kwargs = {})
#   %log1p_1 : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_1,), kwargs = {})
#   %gt_1 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_15, 20), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %add_15, %log1p_1), kwargs = {})
#   %tanh_1 : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%where_1,), kwargs = {})
#   %mul_23 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_15, %tanh_1), kwargs = {})
#   %copy__7 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_19, %add_13), kwargs = {})
#   %copy__8 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_20, %add_14), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_convolution_mish_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, out_ptr4, out_ptr6, out_ptr8, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 4350
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp4_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        r0_1 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 4350*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight, roffset == 0
        )
        tmp4_mean = tl.where(r0_mask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(r0_mask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(r0_mask & xmask, tmp4_weight_next, tmp4_weight)
        tl.store(in_out_ptr0 + (r0_1 + 4350*x0), tmp2, r0_mask & xmask)
    tmp7, tmp8, tmp9 = triton_helpers.welford(tmp4_mean, tmp4_m2, tmp4_weight, 1)
    tmp4 = tmp7[:, None]
    tmp5 = tmp8[:, None]
    tmp9[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tmp19 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = 4350.0
    tmp11 = tmp5 / tmp10
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = 1.0002299379167625
    tmp16 = tmp11 * tmp15
    tmp17 = 0.1
    tmp18 = tmp16 * tmp17
    tmp20 = 0.9
    tmp21 = tmp19 * tmp20
    tmp22 = tmp18 + tmp21
    tmp23 = tmp4 * tmp17
    tmp25 = tmp24 * tmp20
    tmp26 = tmp23 + tmp25
    tl.store(out_ptr2 + (x0), tmp14, xmask)
    tl.store(out_ptr4 + (x0), tmp22, xmask)
    tl.store(out_ptr6 + (x0), tmp26, xmask)
    tmp30 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        r0_1 = r0_index
        tmp27 = tl.load(in_out_ptr0 + (r0_1 + 4350*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp28 = tmp27 - tmp4
        tmp29 = tmp28 * tmp14
        tmp31 = tmp29 * tmp30
        tmp33 = tmp31 + tmp32
        tmp34 = 20.0
        tmp35 = tmp33 > tmp34
        tmp36 = tl_math.exp(tmp33)
        tmp37 = libdevice.log1p(tmp36)
        tmp38 = tl.where(tmp35, tmp33, tmp37)
        tmp39 = libdevice.tanh(tmp38)
        tmp40 = tmp33 * tmp39
        tl.store(out_ptr8 + (r0_1 + 4350*x0), tmp40, r0_mask & xmask)


# kernel path: /tmp/torchinductor_sahanp/mu/cmulhwlgzri3keivdmoaj6acmhxsbid2t6cxck2podlfgbe253mm.py
# Topologically Sorted Source Nodes: [x_11, x_12, x_13], Original ATen: [aten.convolution, aten._native_batch_norm_legit_functional, aten.hardswish]
# Source node to ATen node mapping:
#   x_11 => convolution_3
#   x_12 => add_17, add_18, add_19, add_20, mul_24, mul_25, mul_26, mul_27, mul_28, mul_29, mul_30, rsqrt_3, sub_3, var_mean_3
#   x_13 => add_21, clamp_max_1, clamp_min_1, div_1, mul_31
# Graph fragment:
#   %convolution_3 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_23, %primals_23, %primals_24, [1], [0], [1], False, [0], 1), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convolution_3, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt_3 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_17,), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %getitem_7), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_3), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_9, 0.1), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_26, 0.9), kwargs = {})
#   %add_18 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_25, %mul_26), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_11, 1.0002300437083045), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_27, 0.1), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_27, 0.9), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_28, %mul_29), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_24, %unsqueeze_6), kwargs = {})
#   %add_20 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_30, %unsqueeze_7), kwargs = {})
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_20, 3), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_21, 0), kwargs = {})
#   %clamp_max_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_1, 6), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_20, %clamp_max_1), kwargs = {})
#   %div_1 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_31, 6), kwargs = {})
#   %copy__10 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_26, %add_18), kwargs = {})
#   %copy__11 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_27, %add_19), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_convolution_hardswish_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, out_ptr4, out_ptr6, out_ptr8, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 4348
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp4_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        r0_1 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 4348*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight, roffset == 0
        )
        tmp4_mean = tl.where(r0_mask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(r0_mask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(r0_mask & xmask, tmp4_weight_next, tmp4_weight)
        tl.store(in_out_ptr0 + (r0_1 + 4348*x0), tmp2, r0_mask & xmask)
    tmp7, tmp8, tmp9 = triton_helpers.welford(tmp4_mean, tmp4_m2, tmp4_weight, 1)
    tmp4 = tmp7[:, None]
    tmp5 = tmp8[:, None]
    tmp9[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tmp19 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = 4348.0
    tmp11 = tmp5 / tmp10
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = 1.0002300437083045
    tmp16 = tmp11 * tmp15
    tmp17 = 0.1
    tmp18 = tmp16 * tmp17
    tmp20 = 0.9
    tmp21 = tmp19 * tmp20
    tmp22 = tmp18 + tmp21
    tmp23 = tmp4 * tmp17
    tmp25 = tmp24 * tmp20
    tmp26 = tmp23 + tmp25
    tl.store(out_ptr2 + (x0), tmp14, xmask)
    tl.store(out_ptr4 + (x0), tmp22, xmask)
    tl.store(out_ptr6 + (x0), tmp26, xmask)
    tmp30 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        r0_1 = r0_index
        tmp27 = tl.load(in_out_ptr0 + (r0_1 + 4348*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp28 = tmp27 - tmp4
        tmp29 = tmp28 * tmp14
        tmp31 = tmp29 * tmp30
        tmp33 = tmp31 + tmp32
        tmp34 = 3.0
        tmp35 = tmp33 + tmp34
        tmp36 = 0.0
        tmp37 = triton_helpers.maximum(tmp35, tmp36)
        tmp38 = 6.0
        tmp39 = triton_helpers.minimum(tmp37, tmp38)
        tmp40 = tmp33 * tmp39
        tmp41 = 0.16666666666666666
        tmp42 = tmp40 * tmp41
        tl.store(out_ptr8 + (r0_1 + 4348*x0), tmp42, r0_mask & xmask)


# kernel path: /tmp/torchinductor_sahanp/di/cdixoaj5awirdqu3rm7l6lu22c4xl2krpmmc4rtk7fa4slkrc43b.py
# Topologically Sorted Source Nodes: [x_14, x_15, x_16], Original ATen: [aten.convolution, aten._native_batch_norm_legit_functional, aten.log_sigmoid_forward]
# Source node to ATen node mapping:
#   x_14 => convolution_4
#   x_15 => add_23, add_24, add_25, add_26, mul_32, mul_33, mul_34, mul_35, mul_36, mul_37, mul_38, rsqrt_4, sub_4, var_mean_4
#   x_16 => abs_1, exp_2, full_default, log1p_2, minimum, neg, sub_5
# Graph fragment:
#   %convolution_4 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%div_1, %primals_30, %primals_31, [1], [0], [1], False, [0], 1), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convolution_4, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %add_23 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_8, 1e-05), kwargs = {})
#   %rsqrt_4 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_23,), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %getitem_9), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_4), kwargs = {})
#   %mul_33 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_12, 0.1), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_33, 0.9), kwargs = {})
#   %add_24 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_33, %mul_34), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_14, 1.000230149597238), kwargs = {})
#   %mul_36 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_35, 0.1), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_34, 0.9), kwargs = {})
#   %add_25 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_36, %mul_37), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_32, %unsqueeze_8), kwargs = {})
#   %add_26 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_38, %unsqueeze_9), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %minimum : [num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%full_default, %add_26), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%add_26,), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%abs_1,), kwargs = {})
#   %exp_2 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %log1p_2 : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_2,), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%minimum, %log1p_2), kwargs = {})
#   %copy__13 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_33, %add_24), kwargs = {})
#   %copy__14 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_34, %add_25), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_convolution_log_sigmoid_forward_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, out_ptr4, out_ptr6, out_ptr8, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 4346
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp4_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        r0_1 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 4346*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight, roffset == 0
        )
        tmp4_mean = tl.where(r0_mask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(r0_mask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(r0_mask & xmask, tmp4_weight_next, tmp4_weight)
        tl.store(in_out_ptr0 + (r0_1 + 4346*x0), tmp2, r0_mask & xmask)
    tmp7, tmp8, tmp9 = triton_helpers.welford(tmp4_mean, tmp4_m2, tmp4_weight, 1)
    tmp4 = tmp7[:, None]
    tmp5 = tmp8[:, None]
    tmp9[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tmp19 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = 4346.0
    tmp11 = tmp5 / tmp10
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = 1.000230149597238
    tmp16 = tmp11 * tmp15
    tmp17 = 0.1
    tmp18 = tmp16 * tmp17
    tmp20 = 0.9
    tmp21 = tmp19 * tmp20
    tmp22 = tmp18 + tmp21
    tmp23 = tmp4 * tmp17
    tmp25 = tmp24 * tmp20
    tmp26 = tmp23 + tmp25
    tl.store(out_ptr2 + (x0), tmp14, xmask)
    tl.store(out_ptr4 + (x0), tmp22, xmask)
    tl.store(out_ptr6 + (x0), tmp26, xmask)
    tmp30 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        r0_1 = r0_index
        tmp27 = tl.load(in_out_ptr0 + (r0_1 + 4346*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp28 = tmp27 - tmp4
        tmp29 = tmp28 * tmp14
        tmp31 = tmp29 * tmp30
        tmp33 = tmp31 + tmp32
        tmp34 = 0.0
        tmp35 = triton_helpers.minimum(tmp34, tmp33)
        tmp36 = tl_math.abs(tmp33)
        tmp37 = -tmp36
        tmp38 = tl_math.exp(tmp37)
        tmp39 = libdevice.log1p(tmp38)
        tmp40 = tmp35 - tmp39
        tl.store(out_ptr8 + (r0_1 + 4346*x0), tmp40, r0_mask & xmask)


# kernel path: /tmp/torchinductor_sahanp/ux/cuxzbc4jtpnw2mt2tk5ctjfyjtpflzbowjitxsuq7siiwgss3pyq.py
# Topologically Sorted Source Nodes: [add_], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   add_ => add
# Graph fragment:
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_4, 1), kwargs = {})
#   %copy_ : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_4, %add), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_6(in_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp1 + tmp2
    tl.store(out_ptr1 + (tl.full([XBLOCK], 0, tl.int32)), tmp3, None)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36 = args
    args.clear()
    assert_size_stride(primals_1, (1, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_2, (32, 3, 3), (9, 3, 1))
    assert_size_stride(primals_3, (32, ), (1, ))
    assert_size_stride(primals_4, (), ())
    assert_size_stride(primals_5, (32, ), (1, ))
    assert_size_stride(primals_6, (32, ), (1, ))
    assert_size_stride(primals_7, (32, ), (1, ))
    assert_size_stride(primals_8, (32, ), (1, ))
    assert_size_stride(primals_9, (64, 32, 3), (96, 3, 1))
    assert_size_stride(primals_10, (64, ), (1, ))
    assert_size_stride(primals_11, (), ())
    assert_size_stride(primals_12, (64, ), (1, ))
    assert_size_stride(primals_13, (64, ), (1, ))
    assert_size_stride(primals_14, (64, ), (1, ))
    assert_size_stride(primals_15, (64, ), (1, ))
    assert_size_stride(primals_16, (128, 64, 3), (192, 3, 1))
    assert_size_stride(primals_17, (128, ), (1, ))
    assert_size_stride(primals_18, (), ())
    assert_size_stride(primals_19, (128, ), (1, ))
    assert_size_stride(primals_20, (128, ), (1, ))
    assert_size_stride(primals_21, (128, ), (1, ))
    assert_size_stride(primals_22, (128, ), (1, ))
    assert_size_stride(primals_23, (256, 128, 3), (384, 3, 1))
    assert_size_stride(primals_24, (256, ), (1, ))
    assert_size_stride(primals_25, (), ())
    assert_size_stride(primals_26, (256, ), (1, ))
    assert_size_stride(primals_27, (256, ), (1, ))
    assert_size_stride(primals_28, (256, ), (1, ))
    assert_size_stride(primals_29, (256, ), (1, ))
    assert_size_stride(primals_30, (512, 256, 3), (768, 3, 1))
    assert_size_stride(primals_31, (512, ), (1, ))
    assert_size_stride(primals_32, (), ())
    assert_size_stride(primals_33, (512, ), (1, ))
    assert_size_stride(primals_34, (512, ), (1, ))
    assert_size_stride(primals_35, (512, ), (1, ))
    assert_size_stride(primals_36, (512, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 3, 66, 66), (13068, 4356, 66, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.constant_pad_nd]
        get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_0[grid(13068)](primals_1, buf0, 13068, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_1
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(reinterpret_tensor(buf0, (1, 3, 4356), (0, 4356, 1), 0), primals_2, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf1, (1, 32, 4354), (139328, 4354, 1))
        buf2 = buf1; del buf1  # reuse
        buf3 = empty_strided_cuda((1, 32, 1), (32, 1, 1), torch.float32)
        buf6 = empty_strided_cuda((1, 32, 1), (32, 1, 1), torch.float32)
        buf8 = empty_strided_cuda((1, 32, 4354), (139328, 4354, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2, x_3, x_4], Original ATen: [aten.convolution, aten._native_batch_norm_legit_functional, aten.mish]
        get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_convolution_mish_1[grid(32)](buf2, primals_3, primals_6, primals_5, primals_7, primals_8, buf3, buf6, primals_6, primals_5, buf8, 32, 4354, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del primals_3
        del primals_5
        del primals_6
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf8, primals_9, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf9, (1, 64, 4352), (278528, 4352, 1))
        buf10 = buf9; del buf9  # reuse
        buf11 = empty_strided_cuda((1, 64, 1), (64, 1, 1), torch.float32)
        buf14 = empty_strided_cuda((1, 64, 1), (64, 1, 1), torch.float32)
        buf15 = empty_strided_cuda((1, 64, 4352), (278528, 4352, 1), torch.float32)
        buf16 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [x_5, x_6, x_7], Original ATen: [aten.convolution, aten._native_batch_norm_legit_functional, aten.hardswish]
        get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_convolution_hardswish_2[grid(64)](buf10, buf16, primals_10, primals_13, primals_12, primals_14, primals_15, buf11, buf14, primals_13, primals_12, 64, 4352, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del primals_10
        del primals_12
        del primals_13
        # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf16, primals_16, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf17, (1, 128, 4350), (556800, 4350, 1))
        buf18 = buf17; del buf17  # reuse
        buf19 = empty_strided_cuda((1, 128, 1), (128, 1, 1), torch.float32)
        buf22 = empty_strided_cuda((1, 128, 1), (128, 1, 1), torch.float32)
        buf24 = empty_strided_cuda((1, 128, 4350), (556800, 4350, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_8, x_9, x_10], Original ATen: [aten.convolution, aten._native_batch_norm_legit_functional, aten.mish]
        get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_convolution_mish_3[grid(128)](buf18, primals_17, primals_20, primals_19, primals_21, primals_22, buf19, buf22, primals_20, primals_19, buf24, 128, 4350, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del primals_17
        del primals_19
        del primals_20
        # Topologically Sorted Source Nodes: [x_11], Original ATen: [aten.convolution]
        buf25 = extern_kernels.convolution(buf24, primals_23, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf25, (1, 256, 4348), (1113088, 4348, 1))
        buf26 = buf25; del buf25  # reuse
        buf27 = empty_strided_cuda((1, 256, 1), (256, 1, 1), torch.float32)
        buf30 = empty_strided_cuda((1, 256, 1), (256, 1, 1), torch.float32)
        buf32 = empty_strided_cuda((1, 256, 4348), (1113088, 4348, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_11, x_12, x_13], Original ATen: [aten.convolution, aten._native_batch_norm_legit_functional, aten.hardswish]
        get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_convolution_hardswish_4[grid(256)](buf26, primals_24, primals_27, primals_26, primals_28, primals_29, buf27, buf30, primals_27, primals_26, buf32, 256, 4348, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del primals_24
        del primals_26
        del primals_27
        # Topologically Sorted Source Nodes: [x_14], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf32, primals_30, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf33, (1, 512, 4346), (2225152, 4346, 1))
        buf34 = buf33; del buf33  # reuse
        buf35 = empty_strided_cuda((1, 512, 1), (512, 1, 1), torch.float32)
        buf38 = empty_strided_cuda((1, 512, 1), (512, 1, 1), torch.float32)
        buf40 = empty_strided_cuda((1, 512, 4346), (2225152, 4346, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_14, x_15, x_16], Original ATen: [aten.convolution, aten._native_batch_norm_legit_functional, aten.log_sigmoid_forward]
        get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_convolution_log_sigmoid_forward_5[grid(512)](buf34, primals_31, primals_34, primals_33, primals_35, primals_36, buf35, buf38, primals_34, primals_33, buf40, 512, 4346, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del primals_31
        del primals_33
        del primals_34
        # Topologically Sorted Source Nodes: [add_], Original ATen: [aten.add]
        get_raw_stream(0)
        triton_poi_fused_add_6[grid(1)](primals_4, primals_4, 1, XBLOCK=1, num_warps=1, num_stages=1)
        del primals_4
        # Topologically Sorted Source Nodes: [add__1], Original ATen: [aten.add]
        get_raw_stream(0)
        triton_poi_fused_add_6[grid(1)](primals_11, primals_11, 1, XBLOCK=1, num_warps=1, num_stages=1)
        del primals_11
        # Topologically Sorted Source Nodes: [add__2], Original ATen: [aten.add]
        get_raw_stream(0)
        triton_poi_fused_add_6[grid(1)](primals_18, primals_18, 1, XBLOCK=1, num_warps=1, num_stages=1)
        del primals_18
        # Topologically Sorted Source Nodes: [add__3], Original ATen: [aten.add]
        get_raw_stream(0)
        triton_poi_fused_add_6[grid(1)](primals_25, primals_25, 1, XBLOCK=1, num_warps=1, num_stages=1)
        del primals_25
        # Topologically Sorted Source Nodes: [add__4], Original ATen: [aten.add]
        get_raw_stream(0)
        triton_poi_fused_add_6[grid(1)](primals_32, primals_32, 1, XBLOCK=1, num_warps=1, num_stages=1)
        del primals_32
    return (buf40, primals_2, primals_7, primals_8, primals_9, primals_14, primals_15, primals_16, primals_21, primals_22, primals_23, primals_28, primals_29, primals_30, primals_35, primals_36, reinterpret_tensor(buf0, (1, 3, 4356), (13068, 4356, 1), 0), buf2, buf3, buf6, buf8, buf10, buf11, buf14, buf16, buf18, buf19, buf22, buf24, buf26, buf27, buf30, buf32, buf34, buf35, buf38, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((32, 3, 3), (9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_5 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, 32, 3), (96, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_12 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((128, 64, 3), (192, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_19 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((256, 128, 3), (384, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_26 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((512, 256, 3), (768, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_33 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
