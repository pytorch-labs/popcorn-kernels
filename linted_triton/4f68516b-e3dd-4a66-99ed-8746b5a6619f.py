# AOT ID: ['94_inference']
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


# kernel path: /tmp/torchinductor_sahanp/5n/c5niek6tl6xlexxaysojtjy5sqy3wymzb2ztnmenwoy4ynwg6fn5.py
# Topologically Sorted Source Nodes: [x_2, x, x_1], Original ATen: [aten.log_sigmoid_forward, aten.sign, aten.abs, aten.relu, aten.mul, aten.pow, aten.softplus]
# Source node to ATen node mapping:
#   x => abs_1, mul_24, mul_28, pow_2, relu, sign
#   x_1 => div, exp, gt, log1p, mul_35, where
#   x_2 => abs_2, exp_1, full_default, log1p_1, minimum, neg, sub_22
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sign : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%squeeze,), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%squeeze,), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%abs_1,), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign, %relu), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_24, 3), kwargs = {})
#   %pow_2 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_28, 0.5), kwargs = {})
#   %mul_35 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_2, 1.0), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%mul_35, 20.0), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_35,), kwargs = {})
#   %log1p : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%log1p, 1.0), kwargs = {})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt, %pow_2, %div), kwargs = {})
#   %minimum : [num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%full_default, %where), kwargs = {})
#   %abs_2 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%where,), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%abs_2,), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %log1p_1 : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_1,), kwargs = {})
#   %sub_22 : [num_users=4] = call_function[target=torch.ops.aten.sub.Tensor](args = (%minimum, %log1p_1), kwargs = {})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_abs_log_sigmoid_forward_mul_pow_relu_sign_softplus_0(in_ptr0, out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    tmp0 = tl.load(in_ptr0 + (2*x0 + ks1*x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 2*x0 + ks1*x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (2 + 2*x0 + ks1*x1), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 * tmp0
    tmp3 = tmp2 * tmp2
    tmp4 = tmp3 + tmp1
    tmp6 = tmp5 * tmp5
    tmp7 = tmp6 + tmp4
    tmp8 = 0.3333333333333333
    tmp9 = tmp7 * tmp8
    tmp10 = tl.full([1], 0, tl.int32)
    tmp11 = tmp10 < tmp9
    tmp12 = tmp11.to(tl.int8)
    tmp13 = tmp9 < tmp10
    tmp14 = tmp13.to(tl.int8)
    tmp15 = tmp12 - tmp14
    tmp16 = tmp15.to(tmp9.dtype)
    tmp17 = tl_math.abs(tmp9)
    tmp18 = triton_helpers.maximum(tmp10, tmp17)
    tmp19 = tmp16 * tmp18
    tmp20 = 3.0
    tmp21 = tmp19 * tmp20
    tmp22 = libdevice.sqrt(tmp21)
    tmp23 = 1.0
    tmp24 = tmp22 * tmp23
    tmp25 = 20.0
    tmp26 = tmp24 > tmp25
    tmp27 = tl_math.exp(tmp24)
    tmp28 = libdevice.log1p(tmp27)
    tmp29 = tmp28 * tmp23
    tmp30 = tl.where(tmp26, tmp22, tmp29)
    tmp31 = 0.0
    tmp32 = triton_helpers.minimum(tmp31, tmp30)
    tmp33 = tl_math.abs(tmp30)
    tmp34 = -tmp33
    tmp35 = tl_math.exp(tmp34)
    tmp36 = libdevice.log1p(tmp35)
    tmp37 = tmp32 - tmp36
    tl.store(out_ptr0 + (x0 + x1 + x1*(triton_helpers.div_floor_integer((-3) + ks1,  2))), tmp37, xmask)


# kernel path: /tmp/torchinductor_sahanp/ip/cipmnqorskp2k3jclc6ienhbchb2dyrtkwhrcv65ldfoijs6cd2f.py
# Topologically Sorted Source Nodes: [dist_pos, negative, dist_neg], Original ATen: [aten.sub, aten.add, aten.norm, aten.roll]
# Source node to ATen node mapping:
#   dist_neg => add_55, pow_5, sub_36, sum_2
#   dist_pos => add_46, pow_3, sub_30, sum_1
#   negative => index
# Graph fragment:
#   %sub_30 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_22, %sub_22), kwargs = {})
#   %add_46 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%sub_30, 1e-06), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_46, 2.0), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_3, [2]), kwargs = {})
#   %index : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%sub_22, [None, None, %fmod]), kwargs = {})
#   %sub_36 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_22, %index), kwargs = {})
#   %add_55 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%sub_36, 1e-06), kwargs = {})
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_55, 2.0), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_5, [2]), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_norm_roll_sub_1(in_ptr0, out_ptr0, out_ptr1, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp14 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + x0 + x0*(triton_helpers.div_floor_integer((-3) + ks0,  2))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tl.device_assert((((r0_1 + ((triton_helpers.div_floor_integer((-3) + ks0,  2)) % (1 + (triton_helpers.div_floor_integer((-3) + ks0,  2))))) % (1 + (triton_helpers.div_floor_integer((-3) + ks0,  2)))) < triton_helpers.div_floor_integer((-1) + ks0,  2)) | ~(r0_mask), "index out of bounds: ((r0_1 + ((triton_helpers.div_floor_integer((-3) + ks0,  2)) % (1 + (triton_helpers.div_floor_integer((-3) + ks0,  2))))) % (1 + (triton_helpers.div_floor_integer((-3) + ks0,  2)))) < triton_helpers.div_floor_integer((-1) + ks0,  2)")
        tmp9 = tl.load(in_ptr0 + (x0 + x0*(triton_helpers.div_floor_integer((-3) + ks0,  2)) + (((r0_1 + ((triton_helpers.div_floor_integer((-3) + ks0,  2)) % (1 + (triton_helpers.div_floor_integer((-3) + ks0,  2))))) % (1 + (triton_helpers.div_floor_integer((-3) + ks0,  2)))))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0 - tmp0
        tmp2 = 1e-06
        tmp3 = tmp1 + tmp2
        tmp4 = tmp3 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask & xmask, tmp7, _tmp6)
        tmp10 = tmp0 - tmp9
        tmp11 = tmp10 + tmp2
        tmp12 = tmp11 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(r0_mask & xmask, tmp15, _tmp14)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)


# kernel path: /tmp/torchinductor_sahanp/sk/cskscswj2fyhy7fwam6xqnaq5br5ozyit7w7u2hguit5vnrqfqdd.py
# Topologically Sorted Source Nodes: [dist_pos, add, dist_neg, sub, loss, loss_1], Original ATen: [aten.norm, aten.add, aten.sub, aten.clamp_min, aten.mean]
# Source node to ATen node mapping:
#   add => add_61
#   dist_neg => pow_6
#   dist_pos => pow_4
#   loss => clamp_min
#   loss_1 => mean
#   sub => sub_43
# Graph fragment:
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %add_61 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_4, 1.0), kwargs = {})
#   %pow_6 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_2, 0.5), kwargs = {})
#   %sub_43 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_61, %pow_6), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_43, 0), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%clamp_min,), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_clamp_min_mean_norm_sub_2(in_out_ptr0, in_ptr0, in_ptr1, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = libdevice.sqrt(tmp0)
        tmp2 = 1.0
        tmp3 = tmp1 + tmp2
        tmp5 = libdevice.sqrt(tmp4)
        tmp6 = tmp3 - tmp5
        tmp7 = 0.0
        tmp8 = triton_helpers.maximum(tmp6, tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tmp12 = ks0
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp10 / tmp13
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp14, None)


def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    assert_size_stride(arg2_1, (1, s0, s1), (s0*s1, s1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        ((-1) + s1) // 2
        buf0 = empty_strided_cuda((1, s0, ((-1) + s1) // 2), (s0 + s0*(((-3) + s1) // 2), 1 + (((-3) + s1) // 2), 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2, x, x_1], Original ATen: [aten.log_sigmoid_forward, aten.sign, aten.abs, aten.relu, aten.mul, aten.pow, aten.softplus]
        triton_poi_fused_abs_log_sigmoid_forward_mul_pow_relu_sign_softplus_0_xnumel = s0*(((-1) + s1) // 2)
        get_raw_stream(0)
        triton_poi_fused_abs_log_sigmoid_forward_mul_pow_relu_sign_softplus_0[grid(triton_poi_fused_abs_log_sigmoid_forward_mul_pow_relu_sign_softplus_0_xnumel)](arg2_1, buf0, 31, 64, 310, XBLOCK=128, num_warps=4, num_stages=1)
        del arg2_1
        buf1 = empty_strided_cuda((1, s0), (s0, 1), torch.float32)
        buf2 = empty_strided_cuda((1, s0), (s0, 1), torch.float32)
        # Topologically Sorted Source Nodes: [dist_pos, negative, dist_neg], Original ATen: [aten.sub, aten.add, aten.norm, aten.roll]
        ((-1) + s1) // 2
        get_raw_stream(0)
        triton_red_fused_add_norm_roll_sub_1[grid(s0)](buf0, buf1, buf2, 64, 10, 31, XBLOCK=1, R0_BLOCK=32, num_warps=2, num_stages=1)
        buf3 = empty_strided_cuda((), (), torch.float32)
        buf4 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [dist_pos, add, dist_neg, sub, loss, loss_1], Original ATen: [aten.norm, aten.add, aten.sub, aten.clamp_min, aten.mean]
        get_raw_stream(0)
        triton_red_fused_add_clamp_min_mean_norm_sub_2[grid(1)](buf4, buf1, buf2, 10, 1, 10, XBLOCK=1, R0_BLOCK=16, num_warps=2, num_stages=1)
        del buf1
        del buf2
    return (buf4, buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 10
    arg1_1 = 64
    arg2_1 = rand_strided((1, 10, 64), (640, 64, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
