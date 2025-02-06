# AOT ID: ['160_inference']
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


# kernel path: /tmp/torchinductor_sahanp/5r/c5rbjah6y5ob2jexek2eua5vfetiyuaxw6zwnumtvbmm7cku2ceb.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.bernoulli]
# Source node to ATen node mapping:
#   x => inductor_lookup_seed_default, inductor_random_default
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, %arg0_1, 1, 1, 1], %inductor_lookup_seed_default, rand), kwargs = {})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_bernoulli_0(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (x0), tmp2, xmask)


# kernel path: /tmp/torchinductor_sahanp/n3/cn3faxm3wyapkw2btsrjdozuysn37eybypamtkhtmjkdu5zl2mw3.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   x_2 => amax
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view, [1], True), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__softmax_1(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, ks3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 12
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    _tmp14 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp0 = r0_1 + x0*((11 + ks0*ks1*ks2*ks3) // 12)
        tmp1 = ks0*ks1*ks2*ks3
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (((r0_1 + x0*((11 + ks0*ks1*ks2*ks3) // 12)) % (ks0*ks1*ks2*ks3))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + ((((r0_1 + x0*((11 + ks0*ks1*ks2*ks3) // 12)) // (ks1*ks2*ks3)) % ks0)), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = 0.5
        tmp6 = tmp4 < tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = 2.0
        tmp9 = tmp7 * tmp8
        tmp10 = tmp3 * tmp9
        tmp11 = tl.full(tmp10.shape, float("-inf"), tmp10.dtype)
        tmp12 = tl.where(tmp2, tmp10, tmp11)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
        tmp15 = triton_helpers.maximum(_tmp14, tmp13)
        _tmp14 = tl.where(r0_mask & xmask, tmp15, _tmp14)
    tmp14 = triton_helpers.max2(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp14, xmask)


# kernel path: /tmp/torchinductor_sahanp/z4/cz4qlbj74yi7wq4jfbxznwxcssuaywwh5lwzwxoxkw2hwb5ilxkl.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   x_2 => amax
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view, [1], True), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__softmax_2(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 12
    R0_BLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_mask = r0_index < r0_numel
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp4, None)


# kernel path: /tmp/torchinductor_sahanp/32/c32m77ksjkt4qgpldkqi5oppjrxgjrszius3bchf3s2d47roxun5.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   x_2 => exp, sub_10, sum_1
# Graph fragment:
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_10,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__softmax_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, ks1, ks2, ks3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 12
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    _tmp17 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp0 = r0_1 + x0*((11 + ks0*ks1*ks2*ks3) // 12)
        tmp1 = ks0*ks1*ks2*ks3
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (((r0_1 + x0*((11 + ks0*ks1*ks2*ks3) // 12)) % (ks0*ks1*ks2*ks3))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + ((((r0_1 + x0*((11 + ks0*ks1*ks2*ks3) // 12)) // (ks1*ks2*ks3)) % ks0)), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = 0.5
        tmp6 = tmp4 < tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = 2.0
        tmp9 = tmp7 * tmp8
        tmp10 = tmp3 * tmp9
        tmp11 = tl.load(in_ptr2 + (tl.full([XBLOCK, R0_BLOCK], 0, tl.int32)), tmp2, eviction_policy='evict_last', other=0.0)
        tmp12 = tmp10 - tmp11
        tmp13 = tl_math.exp(tmp12)
        tmp14 = tl.full(tmp13.shape, 0, tmp13.dtype)
        tmp15 = tl.where(tmp2, tmp13, tmp14)
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(r0_mask & xmask, tmp18, _tmp17)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp17, xmask)


# kernel path: /tmp/torchinductor_sahanp/q2/cq2zbthjhnjuutpdmz36gdelph6jjf2ueoi7enn7o43znmwirdly.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   x_2 => exp, sub_10, sum_1
# Graph fragment:
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_10,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__softmax_4(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 12
    R0_BLOCK: tl.constexpr = 16
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


# kernel path: /tmp/torchinductor_sahanp/v2/cv2zklwt42ruskzncuwibs5hm5jskgk7kjse523jseyz4sxp7z5h.py
# Topologically Sorted Source Nodes: [x_2, x_3, target, loss], Original ATen: [aten._softmax, aten.view, aten.zeros_like, aten.smooth_l1_loss]
# Source node to ATen node mapping:
#   loss => abs_1, div_2, lt_4, mean, mul_30, pow_1, sub_14, sub_15, where
#   target => full
#   x_2 => div_1, exp, sub_10
#   x_3 => view_1
# Graph fragment:
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_10,), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
#   %view_1 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%div_1, [1, %mul_25]), kwargs = {})
#   %full : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, %sym_size_int_4], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_1, %full), kwargs = {})
#   %abs_1 : [num_users=3] = call_function[target=torch.ops.aten.abs.default](args = (%sub_14,), kwargs = {})
#   %lt_4 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%abs_1, 1.0), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%abs_1, 2), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_1, 0.5), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_30, 1.0), kwargs = {})
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%abs_1, 0.5), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%lt_4, %div_2, %sub_15), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%where,), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__softmax_smooth_l1_loss_view_zeros_like_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ks0, ks1, ks2, ks3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 12
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    _tmp29 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp0 = r0_1 + x0*((11 + ks0*ks1*ks2*ks3) // 12)
        tmp1 = ks0*ks1*ks2*ks3
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (((r0_1 + x0*((11 + ks0*ks1*ks2*ks3) // 12)) % (ks0*ks1*ks2*ks3))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + ((((r0_1 + x0*((11 + ks0*ks1*ks2*ks3) // 12)) // (ks1*ks2*ks3)) % ks0)), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = 0.5
        tmp6 = tmp4 < tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = 2.0
        tmp9 = tmp7 * tmp8
        tmp10 = tmp3 * tmp9
        tmp11 = tl.load(in_ptr2 + (tl.full([XBLOCK, R0_BLOCK], 0, tl.int32)), tmp2, eviction_policy='evict_last', other=0.0)
        tmp12 = tmp10 - tmp11
        tmp13 = tl_math.exp(tmp12)
        tmp14 = tl.load(in_ptr3 + (tl.full([XBLOCK, R0_BLOCK], 0, tl.int32)), tmp2, eviction_policy='evict_last', other=0.0)
        tmp15 = tmp13 / tmp14
        tmp16 = 0.0
        tmp17 = tmp15 - tmp16
        tmp18 = tl_math.abs(tmp17)
        tmp19 = 1.0
        tmp20 = tmp18 < tmp19
        tmp21 = tmp18 * tmp18
        tmp22 = tmp21 * tmp5
        tmp23 = tmp22 * tmp19
        tmp24 = tmp18 - tmp5
        tmp25 = tl.where(tmp20, tmp23, tmp24)
        tmp26 = tl.full(tmp25.shape, 0, tmp25.dtype)
        tmp27 = tl.where(tmp2, tmp25, tmp26)
        tmp28 = tl.broadcast_to(tmp27, [XBLOCK, R0_BLOCK])
        tmp30 = _tmp29 + tmp28
        _tmp29 = tl.where(r0_mask & xmask, tmp30, _tmp29)
    tmp29 = tl.sum(_tmp29, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp29, xmask)


# kernel path: /tmp/torchinductor_sahanp/vr/cvrniurtwoauvpyfaciu6zdaxrid3sffqsjilmd2qftjlov5ucbh.py
# Topologically Sorted Source Nodes: [x_2, x_3, target, loss], Original ATen: [aten._softmax, aten.view, aten.zeros_like, aten.smooth_l1_loss]
# Source node to ATen node mapping:
#   loss => abs_1, div_2, lt_4, mean, mul_30, pow_1, sub_14, sub_15, where
#   target => full
#   x_2 => div_1, exp, sub_10
#   x_3 => view_1
# Graph fragment:
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_10,), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
#   %view_1 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%div_1, [1, %mul_25]), kwargs = {})
#   %full : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, %sym_size_int_4], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_1, %full), kwargs = {})
#   %abs_1 : [num_users=3] = call_function[target=torch.ops.aten.abs.default](args = (%sub_14,), kwargs = {})
#   %lt_4 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%abs_1, 1.0), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%abs_1, 2), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_1, 0.5), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_30, 1.0), kwargs = {})
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%abs_1, 0.5), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%lt_4, %div_2, %sub_15), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%where,), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__softmax_smooth_l1_loss_view_zeros_like_6(in_out_ptr0, in_ptr0, ks0, ks1, ks2, ks3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 12
    R0_BLOCK: tl.constexpr = 16
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
    tmp5 = ks0*ks1*ks2*ks3
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp4 / tmp6
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp7, None)


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
        buf0 = empty_strided_cuda((1, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf0)
        buf1 = empty_strided_cuda((1, s0, 1, 1, 1), (s0, 1, s0, s0, s0), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.bernoulli]
        get_raw_stream(0)
        triton_poi_fused_bernoulli_0[grid(s0)](buf0, buf1, 0, 3, XBLOCK=4, num_warps=1, num_stages=1)
        del buf0
        buf2 = empty_strided_cuda((1, 1, 12), (12, 12, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten._softmax]
        (11 + s0*s1*s2*s3) // 12
        get_raw_stream(0)
        triton_red_fused__softmax_1[grid(12)](arg4_1, buf1, buf2, 3, 32, 32, 32, 12, 8192, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        buf3 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten._softmax]
        get_raw_stream(0)
        triton_per_fused__softmax_2[grid(1)](buf2, buf3, 1, 12, XBLOCK=1, num_warps=2, num_stages=1)
        buf4 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten._softmax]
        (11 + s0*s1*s2*s3) // 12
        get_raw_stream(0)
        triton_red_fused__softmax_3[grid(12)](arg4_1, buf1, buf3, buf4, 3, 32, 32, 32, 12, 8192, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        buf5 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten._softmax]
        get_raw_stream(0)
        triton_per_fused__softmax_4[grid(1)](buf4, buf5, 1, 12, XBLOCK=1, num_warps=2, num_stages=1)
        buf6 = reinterpret_tensor(buf4, (12, ), (1, ), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [x_2, x_3, target, loss], Original ATen: [aten._softmax, aten.view, aten.zeros_like, aten.smooth_l1_loss]
        (11 + s0*s1*s2*s3) // 12
        get_raw_stream(0)
        triton_red_fused__softmax_smooth_l1_loss_view_zeros_like_5[grid(12)](arg4_1, buf1, buf3, buf5, buf6, 3, 32, 32, 32, 12, 8192, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del arg4_1
        del buf1
        del buf3
        buf7 = reinterpret_tensor(buf5, (), (), 0); del buf5  # reuse
        buf8 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [x_2, x_3, target, loss], Original ATen: [aten._softmax, aten.view, aten.zeros_like, aten.smooth_l1_loss]
        get_raw_stream(0)
        triton_per_fused__softmax_smooth_l1_loss_view_zeros_like_6[grid(1)](buf8, buf6, 3, 32, 32, 32, 1, 12, XBLOCK=1, num_warps=2, num_stages=1)
        del buf6
    return (buf8, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 3
    arg1_1 = 32
    arg2_1 = 32
    arg3_1 = 32
    arg4_1 = rand_strided((1, 3, 32, 32, 32), (98304, 32768, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
