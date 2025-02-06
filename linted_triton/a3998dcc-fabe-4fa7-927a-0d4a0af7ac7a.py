# AOT ID: ['27_inference']
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


# kernel path: /tmp/torchinductor_sahanp/ma/cma7euc2vn2pp2d7yll52cc2qcco6bauvyskmpovekgeitgbxq5w.py
# Topologically Sorted Source Nodes: [x, x_1, x_2, x_3], Original ATen: [aten.replication_pad1d, aten.elu, aten.mish, aten.neg, aten._softmax]
# Source node to ATen node mapping:
#   x => _unsafe_index
#   x_1 => expm1, gt, mul_3, mul_4, mul_5, where
#   x_2 => exp, gt_1, log1p, mul_9, tanh, where_1
#   x_3 => amax, exp_1, neg, sub_11, sum_1
# Graph fragment:
#   %_unsafe_index : [num_users=3] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg2_1, [None, None, %clamp_max]), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%_unsafe_index, 0), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index, 1.0507009873554805), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index, 1.0), kwargs = {})
#   %expm1 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%mul_4,), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expm1, 1.7580993408473766), kwargs = {})
#   %where : [num_users=4] = call_function[target=torch.ops.aten.where.self](args = (%gt, %mul_3, %mul_5), kwargs = {})
#   %gt_1 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%where, 20), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%where,), kwargs = {})
#   %log1p : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp,), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %where, %log1p), kwargs = {})
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%where_1,), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where, %tanh), kwargs = {})
#   %neg : [num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%mul_9,), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%neg, [1], True), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%neg, %amax), kwargs = {})
#   %exp_1 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_11,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_1, [1], True), kwargs = {})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__softmax_elu_mish_neg_replication_pad1d_0(in_ptr0, out_ptr0, out_ptr1, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    _tmp20 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (ks0*r0_1 + (((-1) + ks0) * (((-1) + ks0) <= (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0))))) + (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) * ((((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) < ((-1) + ks0)))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 > tmp1
        tmp3 = 1.0507009873554805
        tmp4 = tmp0 * tmp3
        tmp5 = 1.0
        tmp6 = tmp0 * tmp5
        tmp7 = libdevice.expm1(tmp6)
        tmp8 = 1.7580993408473766
        tmp9 = tmp7 * tmp8
        tmp10 = tl.where(tmp2, tmp4, tmp9)
        tmp11 = 20.0
        tmp12 = tmp10 > tmp11
        tmp13 = tl_math.exp(tmp10)
        tmp14 = libdevice.log1p(tmp13)
        tmp15 = tl.where(tmp12, tmp10, tmp14)
        tmp16 = libdevice.tanh(tmp15)
        tmp17 = tmp10 * tmp16
        tmp18 = -tmp17
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
        tmp21 = triton_helpers.maximum(_tmp20, tmp19)
        _tmp20 = tl.where(r0_mask & xmask, tmp21, _tmp20)
    tmp20 = triton_helpers.max2(_tmp20, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp20, xmask)
    _tmp44 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp22 = tl.load(in_ptr0 + (ks0*r0_1 + (((-1) + ks0) * (((-1) + ks0) <= (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0))))) + (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) * ((((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) < ((-1) + ks0)))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp23 = 0.0
        tmp24 = tmp22 > tmp23
        tmp25 = 1.0507009873554805
        tmp26 = tmp22 * tmp25
        tmp27 = 1.0
        tmp28 = tmp22 * tmp27
        tmp29 = libdevice.expm1(tmp28)
        tmp30 = 1.7580993408473766
        tmp31 = tmp29 * tmp30
        tmp32 = tl.where(tmp24, tmp26, tmp31)
        tmp33 = 20.0
        tmp34 = tmp32 > tmp33
        tmp35 = tl_math.exp(tmp32)
        tmp36 = libdevice.log1p(tmp35)
        tmp37 = tl.where(tmp34, tmp32, tmp36)
        tmp38 = libdevice.tanh(tmp37)
        tmp39 = tmp32 * tmp38
        tmp40 = -tmp39
        tmp41 = tmp40 - tmp20
        tmp42 = tl_math.exp(tmp41)
        tmp43 = tl.broadcast_to(tmp42, [XBLOCK, R0_BLOCK])
        tmp45 = _tmp44 + tmp43
        _tmp44 = tl.where(r0_mask & xmask, tmp45, _tmp44)
    tmp44 = tl.sum(_tmp44, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp44, xmask)


# kernel path: /tmp/torchinductor_sahanp/yd/cyd2dsw3hdbftn56wuy3qot3kxgpv2hdjxhy3ocfes3w6hyzeh3r.py
# Topologically Sorted Source Nodes: [x, x_1, x_2, x_3], Original ATen: [aten.replication_pad1d, aten.elu, aten.mish, aten.neg, aten._softmax]
# Source node to ATen node mapping:
#   x => _unsafe_index
#   x_1 => expm1, gt, mul_3, mul_4, mul_5, where
#   x_2 => exp, gt_1, log1p, mul_9, tanh, where_1
#   x_3 => div, exp_1, neg, sub_11
# Graph fragment:
#   %_unsafe_index : [num_users=3] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg2_1, [None, None, %clamp_max]), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%_unsafe_index, 0), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index, 1.0507009873554805), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index, 1.0), kwargs = {})
#   %expm1 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%mul_4,), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expm1, 1.7580993408473766), kwargs = {})
#   %where : [num_users=4] = call_function[target=torch.ops.aten.where.self](args = (%gt, %mul_3, %mul_5), kwargs = {})
#   %gt_1 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%where, 20), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%where,), kwargs = {})
#   %log1p : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp,), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %where, %log1p), kwargs = {})
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%where_1,), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where, %tanh), kwargs = {})
#   %neg : [num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%mul_9,), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%neg, %amax), kwargs = {})
#   %exp_1 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_11,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_1, %sum_1), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__softmax_elu_mish_neg_replication_pad1d_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (ks1*x1 + (((-1) + ks1) * (((-1) + ks1) <= (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0))))) + (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) * ((((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) < ((-1) + ks1)))), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 1.0507009873554805
    tmp4 = tmp0 * tmp3
    tmp5 = 1.0
    tmp6 = tmp0 * tmp5
    tmp7 = libdevice.expm1(tmp6)
    tmp8 = 1.7580993408473766
    tmp9 = tmp7 * tmp8
    tmp10 = tl.where(tmp2, tmp4, tmp9)
    tmp11 = 20.0
    tmp12 = tmp10 > tmp11
    tmp13 = tl_math.exp(tmp10)
    tmp14 = libdevice.log1p(tmp13)
    tmp15 = tl.where(tmp12, tmp10, tmp14)
    tmp16 = libdevice.tanh(tmp15)
    tmp17 = tmp10 * tmp16
    tmp18 = -tmp17
    tmp20 = tmp18 - tmp19
    tmp21 = tl_math.exp(tmp20)
    tmp23 = tmp21 / tmp22
    tl.store(out_ptr0 + (x2), tmp23, xmask)


def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    assert_size_stride(arg2_1, (1, s0, s1), (s0*s1, s1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 1, 4 + s1), (4 + s1, 4 + s1, 1), torch.float32)
        buf1 = empty_strided_cuda((1, 1, 4 + s1), (4 + s1, 4 + s1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_1, x_2, x_3], Original ATen: [aten.replication_pad1d, aten.elu, aten.mish, aten.neg, aten._softmax]
        triton_red_fused__softmax_elu_mish_neg_replication_pad1d_0_xnumel = 4 + s1
        get_raw_stream(0)
        triton_red_fused__softmax_elu_mish_neg_replication_pad1d_0[grid(triton_red_fused__softmax_elu_mish_neg_replication_pad1d_0_xnumel)](arg2_1, buf0, buf1, 32, 36, 10, XBLOCK=1, R0_BLOCK=16, num_warps=2, num_stages=1)
        4 + s1
        buf2 = empty_strided_cuda((1, s0, 4 + s1), (4*s0 + s0*s1, 4 + s1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_1, x_2, x_3], Original ATen: [aten.replication_pad1d, aten.elu, aten.mish, aten.neg, aten._softmax]
        triton_poi_fused__softmax_elu_mish_neg_replication_pad1d_1_xnumel = 4*s0 + s0*s1
        get_raw_stream(0)
        triton_poi_fused__softmax_elu_mish_neg_replication_pad1d_1[grid(triton_poi_fused__softmax_elu_mish_neg_replication_pad1d_1_xnumel)](arg2_1, buf0, buf1, buf2, 36, 32, 360, XBLOCK=128, num_warps=4, num_stages=1)
        del arg2_1
        del buf0
        del buf1
    return (buf2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 10
    arg1_1 = 32
    arg2_1 = rand_strided((1, 10, 32), (320, 32, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
