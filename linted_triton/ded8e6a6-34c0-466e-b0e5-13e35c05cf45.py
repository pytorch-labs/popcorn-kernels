# AOT ID: ['178_forward']
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


# kernel path: /tmp/torchinductor_sahanp/3x/c3xzelsjfpqja3v5otcuqucoxrg5biwwesal5xuw4gvqxsnz62es.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   multi_head_attention_forward => clone
# Graph fragment:
#   %clone : [num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze,), kwargs = {memory_format: torch.contiguous_format})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 64)
    x2 = xindex // 4096
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x2 + 192*x1), None)
    tmp1 = tl.load(in_ptr1 + (x0 + 64*x2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)


# kernel path: /tmp/torchinductor_sahanp/kt/cktnp3dfzugje5lap3cskukubk2n52ka6hjnyidwohmxno5726yw.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   multi_head_attention_forward => mul
# Graph fragment:
#   %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_2, 0.3535533905932738), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_mul_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 0.3535533905932738
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, None)


# kernel path: /tmp/torchinductor_sahanp/oo/coonyeckyccaaw3t4iuiixcwdfk3wc6bmllnkhgya57tzntduoly.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   multi_head_attention_forward => amax, div, exp, sub, sum_1
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%bmm, [-1], True), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%bmm, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__softmax_2(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tmp0 - tmp0
    tmp2 = tl_math.exp(tmp1)
    tmp3 = tmp2 / tmp2
    tl.store(in_out_ptr0 + (x0), tmp3, xmask)


# kernel path: /tmp/torchinductor_sahanp/tl/ctlfbqxievuhi2pyrhgqrh5jgyxi24u6usdnarudacolve66asf5.py
# Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten.silu, aten.mish]
# Source node to ATen node mapping:
#   x_1 => mul_1, sigmoid
#   x_2 => exp_1, gt, log1p, mul_2, tanh, where
# Graph fragment:
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_8,), kwargs = {})
#   %mul_1 : [num_users=4] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_8, %sigmoid), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_1,), kwargs = {})
#   %log1p : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_1,), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%mul_1, 20), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %mul_1, %log1p), kwargs = {})
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%where,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %tanh), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_mish_silu_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp3 = 20.0
    tmp4 = tmp2 > tmp3
    tmp5 = tl_math.exp(tmp2)
    tmp6 = libdevice.log1p(tmp5)
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp8 = libdevice.tanh(tmp7)
    tmp9 = tmp2 * tmp8
    tl.store(out_ptr0 + (x0), tmp9, None)


# kernel path: /tmp/torchinductor_sahanp/rp/crpmcf4ezmap6rtymnpwtunum2qr3352nqpzc5x5xkf3wxgla3on.py
# Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   x_5 => add, add_1, convert_element_type, convert_element_type_1, iota, mul_3, mul_4
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (34,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, 1), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, 0), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add, torch.float32), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, 0.0), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, 0.5), kwargs = {})
#   %convert_element_type_1 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_4, torch.int64), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_4(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 34
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


# kernel path: /tmp/torchinductor_sahanp/th/cthmxheqmaml6minbtsenetwdwq3xxvslwbm5lenjy3na6pebc2d.py
# Topologically Sorted Source Nodes: [x_4, x_5, x_7, x_12], Original ATen: [aten.convolution, aten._unsafe_index, aten._native_batch_norm_legit, aten.neg, aten._softmax]
# Source node to ATen node mapping:
#   x_12 => amax_1, neg
#   x_4 => convolution
#   x_5 => _unsafe_index
#   x_7 => add_4, rsqrt, var_mean
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_10, %primals_6, %primals_7, [2, 2], [0, 0], [1, 1], True, [0, 0], 1), kwargs = {})
#   %_unsafe_index : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution, [None, None, %unsqueeze_1, %convert_element_type_1]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%unsqueeze_2, [0, 2, 3, 4]), kwargs = {correction: 0, keepdim: True})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
#   %neg : [num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%view_13,), kwargs = {})
#   %amax_1 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%neg, [1], True), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit__softmax__unsafe_index_convolution_neg_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32
    r0_numel = 1156
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    tmp10 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp13_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp13_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        r0_2 = r0_index // 34
        r0_1 = (r0_index % 34)
        r0_3 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.full([XBLOCK, R0_BLOCK], 17, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tmp6 = tmp5 + tmp1
        tmp7 = tmp5 < 0
        tmp8 = tl.where(tmp7, tmp6, tmp5)
        tmp9 = tl.load(in_ptr1 + (tmp8 + 17*tmp4 + 289*x0), r0_mask & xmask, eviction_policy='evict_last')
        tmp11 = tmp9 + tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
        tmp13_mean_next, tmp13_m2_next, tmp13_weight_next = triton_helpers.welford_reduce(
            tmp12, tmp13_mean, tmp13_m2, tmp13_weight, roffset == 0
        )
        tmp13_mean = tl.where(r0_mask & xmask, tmp13_mean_next, tmp13_mean)
        tmp13_m2 = tl.where(r0_mask & xmask, tmp13_m2_next, tmp13_m2)
        tmp13_weight = tl.where(r0_mask & xmask, tmp13_weight_next, tmp13_weight)
        tl.store(out_ptr0 + (r0_3 + 1184*x0), tmp11, r0_mask & xmask)
    tmp16, tmp17, tmp18 = triton_helpers.welford(tmp13_mean, tmp13_m2, tmp13_weight, 1)
    tmp16[:, None]
    tmp14 = tmp17[:, None]
    tmp18[:, None]
    tmp19 = 1156.0
    tmp20 = tmp14 / tmp19
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp23, xmask)
    tmp26_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp26_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp26_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        r0_3 = r0_index
        tmp24 = tl.load(out_ptr0 + (r0_3 + 1184*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
        tmp26_mean_next, tmp26_m2_next, tmp26_weight_next = triton_helpers.welford_reduce(
            tmp25, tmp26_mean, tmp26_m2, tmp26_weight, roffset == 0
        )
        tmp26_mean = tl.where(r0_mask & xmask, tmp26_mean_next, tmp26_mean)
        tmp26_m2 = tl.where(r0_mask & xmask, tmp26_m2_next, tmp26_m2)
        tmp26_weight = tl.where(r0_mask & xmask, tmp26_weight_next, tmp26_weight)
    tmp29, tmp30, tmp31 = triton_helpers.welford(tmp26_mean, tmp26_m2, tmp26_weight, 1)
    tmp26 = tmp29[:, None]
    tmp30[:, None]
    tmp31[:, None]
    tl.store(out_ptr1 + (x0), tmp26, xmask)
    _tmp40 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        r0_3 = r0_index
        tmp32 = tl.load(out_ptr0 + (r0_3 + 1184*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp33 = tmp32 - tmp26
        tmp34 = tmp33 * tmp23
        tmp35 = libdevice.tanh(tmp34)
        tmp36 = tl.full([1, 1], 0, tl.int32)
        tmp37 = triton_helpers.maximum(tmp36, tmp35)
        tmp38 = -tmp37
        tmp39 = tl.broadcast_to(tmp38, [XBLOCK, R0_BLOCK])
        tmp41 = triton_helpers.maximum(_tmp40, tmp39)
        _tmp40 = tl.where(r0_mask & xmask, tmp41, _tmp40)
    tmp40 = triton_helpers.max2(_tmp40, 1)[:, None]
    tl.store(out_ptr2 + (x0), tmp40, xmask)


# kernel path: /tmp/torchinductor_sahanp/4v/c4vmtxusq3knao3zhly62rtd5kex2oyffb4kjgh46htzqcj5mvu7.py
# Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.neg, aten._softmax]
# Source node to ATen node mapping:
#   x_12 => amax_1, neg
# Graph fragment:
#   %neg : [num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%view_13,), kwargs = {})
#   %amax_1 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%neg, [1], True), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__softmax_neg_6(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    R0_BLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = triton_helpers.max2(tmp1, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp3, None)


# kernel path: /tmp/torchinductor_sahanp/pw/cpw33onzb7kgnyqhgor5magu57vwjuvcxkowie7k6fejq2ejq7o2.py
# Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.neg, aten._softmax]
# Source node to ATen node mapping:
#   x_12 => exp_2, neg, sub_2, sum_2
# Graph fragment:
#   %neg : [num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%view_13,), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%neg, %amax_1), kwargs = {})
#   %exp_2 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_2,), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_2, [1], True), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__softmax_neg_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 5
    r0_numel = 7399
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    _tmp18 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp0 = r0_1 + 7399*x0
        tmp1 = tl.full([1, 1], 36992, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (1184*((((r0_1 + 7399*x0) // 1156) % 32)) + (((r0_1 + 7399*x0) % 1156))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + ((((r0_1 + 7399*x0) // 1156) % 32)), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 - tmp4
        tmp6 = tl.load(in_ptr2 + ((((r0_1 + 7399*x0) // 1156) % 32)), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tmp5 * tmp6
        tmp8 = libdevice.tanh(tmp7)
        tmp9 = tl.full([1, 1], 0, tl.int32)
        tmp10 = triton_helpers.maximum(tmp9, tmp8)
        tmp11 = -tmp10
        tmp12 = tl.load(in_ptr3 + (tl.full([XBLOCK, R0_BLOCK], 0, tl.int32)), tmp2, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp11 - tmp12
        tmp14 = tl_math.exp(tmp13)
        tmp15 = tl.full(tmp14.shape, 0, tmp14.dtype)
        tmp16 = tl.where(tmp2, tmp14, tmp15)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(r0_mask & xmask, tmp19, _tmp18)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp18, xmask)


# kernel path: /tmp/torchinductor_sahanp/7r/c7rojmgwzmgydri73qxi4zurti5r33w6eizbjf7swppx6ssfcapt.py
# Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.neg, aten._softmax]
# Source node to ATen node mapping:
#   x_12 => exp_2, neg, sub_2, sum_2
# Graph fragment:
#   %neg : [num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%view_13,), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%neg, %amax_1), kwargs = {})
#   %exp_2 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_2,), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_2, [1], True), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__softmax_neg_8(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 5
    R0_BLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_mask = r0_index < r0_numel
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp4, None)


# kernel path: /tmp/torchinductor_sahanp/mm/cmmn5x6shk7blzzn5kk75p7dlp3c3kdm2twiv4j6vv3fy7bbshvt.py
# Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.neg, aten._softmax]
# Source node to ATen node mapping:
#   x_12 => div_1, exp_2, neg, sub_2
# Graph fragment:
#   %neg : [num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%view_13,), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%neg, %amax_1), kwargs = {})
#   %exp_2 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_2,), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_2, %sum_2), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__softmax_neg_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36992
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (1184*(x0 // 1156) + ((x0 % 1156))), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 // 1156), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 // 1156), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (0))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK])
    tmp13 = tl.load(in_ptr4 + (0))
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK])
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = libdevice.tanh(tmp4)
    tmp6 = tl.full([1], 0, tl.int32)
    tmp7 = triton_helpers.maximum(tmp6, tmp5)
    tmp8 = -tmp7
    tmp11 = tmp8 - tmp10
    tmp12 = tl_math.exp(tmp11)
    tmp15 = tmp12 / tmp14
    tl.store(out_ptr0 + (x0), tmp15, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7 = args
    args.clear()
    assert_size_stride(primals_1, (1, 64, 64), (4096, 64, 1))
    assert_size_stride(primals_2, (192, ), (1, ))
    assert_size_stride(primals_3, (192, 64), (64, 1))
    assert_size_stride(primals_4, (64, 64), (64, 1))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_7, (32, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 192), (192, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (64, 64), (64, 1), 0), reinterpret_tensor(primals_3, (64, 192), (1, 64), 0), out=buf0)
        del primals_3
        buf1 = empty_strided_cuda((3, 1, 64, 64), (4096, 1, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.clone]
        get_raw_stream(0)
        triton_poi_fused_clone_0[grid(12288)](buf0, primals_2, buf1, 12288, XBLOCK=256, num_warps=4, num_stages=1)
        del buf0
        del primals_2
        buf2 = empty_strided_cuda((512, 1, 8), (8, 4096, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.mul]
        get_raw_stream(0)
        triton_poi_fused_mul_1[grid(4096)](buf1, buf2, 4096, XBLOCK=256, num_warps=4, num_stages=1)
        buf3 = empty_strided_cuda((512, 1, 1), (1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.mul, aten.bmm]
        extern_kernels.bmm(buf2, reinterpret_tensor(buf1, (512, 8, 1), (8, 1, 0), 4096), out=buf3)
        buf4 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._softmax]
        get_raw_stream(0)
        triton_poi_fused__softmax_2[grid(512)](buf4, 512, XBLOCK=128, num_warps=4, num_stages=1)
        buf5 = empty_strided_cuda((512, 1, 8), (8, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf4, reinterpret_tensor(buf1, (512, 1, 8), (8, 0, 1), 8192), out=buf5)
        buf6 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_5, reinterpret_tensor(buf5, (64, 64), (64, 1), 0), reinterpret_tensor(primals_4, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf6)
        del primals_5
        buf7 = empty_strided_cuda((1, 64, 64), (4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten.silu, aten.mish]
        get_raw_stream(0)
        triton_poi_fused_mish_silu_3[grid(4096)](buf6, buf7, 4096, XBLOCK=128, num_warps=4, num_stages=1)
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(reinterpret_tensor(buf7, (1, 64, 8, 8), (0, 64, 8, 1), 0), primals_6, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (1, 32, 17, 17), (9248, 289, 17, 1))
        buf9 = empty_strided_cuda((34, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_4[grid(34)](buf9, 34, XBLOCK=64, num_warps=1, num_stages=1)
        buf10 = empty_strided_cuda((1, 32, 34, 34), (37888, 1184, 34, 1), torch.float32)
        buf12 = empty_strided_cuda((1, 32, 1, 1, 1), (32, 1, 32, 32, 32), torch.float32)
        buf14 = reinterpret_tensor(buf12, (1, 32, 1, 1, 1), (32, 1, 1, 1, 1), 0); del buf12  # reuse
        buf11 = empty_strided_cuda((1, 32, 1, 1, 1), (32, 1, 1, 1, 1), torch.float32)
        buf15 = empty_strided_cuda((1, 1, 1, 32, 1, 1, 1), (32, 32, 32, 1, 32, 32, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_4, x_5, x_7, x_12], Original ATen: [aten.convolution, aten._unsafe_index, aten._native_batch_norm_legit, aten.neg, aten._softmax]
        get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit__softmax__unsafe_index_convolution_neg_5[grid(32)](buf14, buf9, buf8, primals_7, buf10, buf11, buf15, 32, 1156, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del buf8
        del primals_7
        buf16 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.neg, aten._softmax]
        get_raw_stream(0)
        triton_per_fused__softmax_neg_6[grid(1)](buf15, buf16, 1, 32, XBLOCK=1, num_warps=2, num_stages=1)
        del buf15
        buf17 = empty_strided_cuda((1, 1, 5), (5, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.neg, aten._softmax]
        get_raw_stream(0)
        triton_red_fused__softmax_neg_7[grid(5)](buf10, buf11, buf14, buf16, buf17, 5, 7399, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        buf18 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.neg, aten._softmax]
        get_raw_stream(0)
        triton_per_fused__softmax_neg_8[grid(1)](buf17, buf18, 1, 5, XBLOCK=1, num_warps=2, num_stages=1)
        del buf17
        buf19 = empty_strided_cuda((1, 36992), (36992, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.neg, aten._softmax]
        get_raw_stream(0)
        triton_poi_fused__softmax_neg_9[grid(36992)](buf10, buf11, buf14, buf16, buf18, buf19, 36992, XBLOCK=256, num_warps=4, num_stages=1)
        del buf16
        del buf18
    return (buf19, primals_6, reinterpret_tensor(primals_1, (64, 64), (64, 1), 0), buf4, reinterpret_tensor(buf5, (64, 64), (64, 1), 0), buf6, reinterpret_tensor(buf7, (1, 64, 8, 8), (4096, 64, 8, 1), 0), buf9, buf10, buf11, buf14, buf19, primals_4, reinterpret_tensor(buf1, (512, 8, 1), (8, 1, 4096), 8192), reinterpret_tensor(buf2, (512, 8, 1), (8, 1, 8), 0), reinterpret_tensor(buf1, (512, 1, 8), (8, 4096, 1), 4096), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 64, 64), (4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((192, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
