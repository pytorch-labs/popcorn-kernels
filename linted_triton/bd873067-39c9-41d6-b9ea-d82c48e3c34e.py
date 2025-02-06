# AOT ID: ['92_inference']
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


# kernel path: /tmp/torchinductor_sahanp/ic/cicj6exgtjrryp2imlatu2ymqemy2nydibcnc5ywea2hfqwerfvp.py
# Topologically Sorted Source Nodes: [x, x_1, loss], Original ATen: [aten.abs, aten.gt, aten.sign, aten.mul, aten.sub, aten.where, aten.hardtanh, aten.mse_loss]
# Source node to ATen node mapping:
#   loss => clamp_max, mean, pow_1
#   x => abs_1, gt, mul_11, mul_20, sign, sub_12, where
#   x_1 => clamp_min
# Graph fragment:
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%arg3_1,), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%abs_1, 0.5), kwargs = {})
#   %sign : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%arg3_1,), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign, 0.5), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg3_1, %mul_11), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg3_1, 0), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %sub_12, %mul_20), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%where, -1.0), kwargs = {})
#   %clamp_max : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 1.0), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%clamp_max, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_1,), kwargs = {})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_abs_gt_hardtanh_mse_loss_mul_sign_sub_where_0(in_ptr0, out_ptr0, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    _tmp27 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp0 = r0_1 + x0*((1 + ks0*ks1*ks2) // 2)
        tmp1 = ks0*ks1*ks2
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (((r0_1 + x0*((1 + ks0*ks1*ks2) // 2)) % (ks0*ks1*ks2))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl_math.abs(tmp3)
        tmp5 = 0.5
        tmp6 = tmp4 > tmp5
        tmp7 = tl.full([1, 1], 0, tl.int32)
        tmp8 = tmp7 < tmp3
        tmp9 = tmp8.to(tl.int8)
        tmp10 = tmp3 < tmp7
        tmp11 = tmp10.to(tl.int8)
        tmp12 = tmp9 - tmp11
        tmp13 = tmp12.to(tmp3.dtype)
        tmp14 = tmp13 * tmp5
        tmp15 = tmp3 - tmp14
        tmp16 = 0.0
        tmp17 = tmp3 * tmp16
        tmp18 = tl.where(tmp6, tmp15, tmp17)
        tmp19 = -1.0
        tmp20 = triton_helpers.maximum(tmp18, tmp19)
        tmp21 = 1.0
        tmp22 = triton_helpers.minimum(tmp20, tmp21)
        tmp23 = tmp22 * tmp22
        tmp24 = tl.full(tmp23.shape, 0, tmp23.dtype)
        tmp25 = tl.where(tmp2, tmp23, tmp24)
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(r0_mask & xmask, tmp28, _tmp27)
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp27, xmask)


# kernel path: /tmp/torchinductor_sahanp/gd/cgdh7s3rjmfdpddyco3f63w2xc5uyua6temewkw3g44nzceboz37.py
# Topologically Sorted Source Nodes: [x, x_1, loss], Original ATen: [aten.abs, aten.gt, aten.sign, aten.mul, aten.sub, aten.where, aten.hardtanh, aten.mse_loss]
# Source node to ATen node mapping:
#   loss => clamp_max, mean, pow_1
#   x => abs_1, gt, mul_11, mul_20, sign, sub_12, where
#   x_1 => clamp_min
# Graph fragment:
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%arg3_1,), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%abs_1, 0.5), kwargs = {})
#   %sign : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%arg3_1,), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign, 0.5), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg3_1, %mul_11), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg3_1, 0), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %sub_12, %mul_20), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%where, -1.0), kwargs = {})
#   %clamp_max : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 1.0), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%clamp_max, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_1,), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_abs_gt_hardtanh_mse_loss_mul_sign_sub_where_1(in_out_ptr0, in_ptr0, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    R0_BLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None]
    tmp4 = ks0*ks1*ks2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp6, None)


def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_1, loss], Original ATen: [aten.abs, aten.gt, aten.sign, aten.mul, aten.sub, aten.where, aten.hardtanh, aten.mse_loss]
        (1 + s0*s1*s2) // 2
        get_raw_stream(0)
        triton_red_fused_abs_gt_hardtanh_mse_loss_mul_sign_sub_where_0[grid(2)](arg3_1, buf0, 3, 64, 64, 2, 6144, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del arg3_1
        buf1 = empty_strided_cuda((), (), torch.float32)
        buf2 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [x, x_1, loss], Original ATen: [aten.abs, aten.gt, aten.sign, aten.mul, aten.sub, aten.where, aten.hardtanh, aten.mse_loss]
        get_raw_stream(0)
        triton_per_fused_abs_gt_hardtanh_mse_loss_mul_sign_sub_where_1[grid(1)](buf2, buf0, 3, 64, 64, 1, 2, XBLOCK=1, num_warps=2, num_stages=1)
        del buf0
    return (buf2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 3
    arg1_1 = 64
    arg2_1 = 64
    arg3_1 = rand_strided((1, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
