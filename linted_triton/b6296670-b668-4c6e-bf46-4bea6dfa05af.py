
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


from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_0(in_ptr0, out_ptr0, out_ptr1, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 10
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + ks0*ks1*ks2*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(r0_mask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(r0_mask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(r0_mask & xmask, tmp2_weight_next, tmp2_weight)
    tmp5, tmp6, tmp7 = triton_helpers.welford(tmp2_mean, tmp2_m2, tmp2_weight, 1)
    tmp2 = tmp5[:, None]
    tmp3 = tmp6[:, None]
    tmp7[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tl.store(out_ptr1 + (x0), tmp3, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__log_softmax_exp_mean_mul_sub_1(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_0 // (ks0*ks1*ks2)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r0_0 // (ks0*ks1*ks2)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 - tmp1
        tmp4 = ks0*ks1*ks2
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 / tmp5
        tmp7 = 1e-05
        tmp8 = tmp6 + tmp7
        tmp9 = libdevice.rsqrt(tmp8)
        tmp10 = tmp2 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = triton_helpers.maximum(_tmp12, tmp11)
        _tmp12 = tl.where(r0_mask, tmp13, _tmp12)
    tmp12 = triton_helpers.max2(_tmp12, 1)[:, None]
    _tmp28 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_0 = r0_index
        tmp14 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.load(in_ptr1 + (r0_0 // (ks0*ks1*ks2)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr2 + (r0_0 // (ks0*ks1*ks2)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp16 = tmp14 - tmp15
        tmp18 = ks0*ks1*ks2
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp17 / tmp19
        tmp21 = 1e-05
        tmp22 = tmp20 + tmp21
        tmp23 = libdevice.rsqrt(tmp22)
        tmp24 = tmp16 * tmp23
        tmp25 = tmp24 - tmp12
        tmp26 = tl_math.exp(tmp25)
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, R0_BLOCK])
        tmp29 = _tmp28 + tmp27
        _tmp28 = tl.where(r0_mask, tmp29, _tmp28)
    tmp28 = tl.sum(_tmp28, 1)[:, None]
    _tmp48 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_0 = r0_index
        tmp30 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp31 = tl.load(in_ptr1 + (r0_0 // (ks0*ks1*ks2)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp33 = tl.load(in_ptr2 + (r0_0 // (ks0*ks1*ks2)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp32 = tmp30 - tmp31
        tmp34 = ks0*ks1*ks2
        tmp35 = tmp34.to(tl.float32)
        tmp36 = tmp33 / tmp35
        tmp37 = 1e-05
        tmp38 = tmp36 + tmp37
        tmp39 = libdevice.rsqrt(tmp38)
        tmp40 = tmp32 * tmp39
        tmp41 = tmp40 - tmp12
        tmp42 = tl_math.log(tmp28)
        tmp43 = tmp41 - tmp42
        tmp44 = tl_math.exp(tmp43)
        tmp45 = tmp44 * tmp43
        tmp46 = tmp44 - tmp45
        tmp47 = tl.broadcast_to(tmp46, [XBLOCK, R0_BLOCK])
        tmp49 = _tmp48 + tmp47
        _tmp48 = tl.where(r0_mask, tmp49, _tmp48)
        tl.store(out_ptr2 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp43, r0_mask)
    tmp48 = tl.sum(_tmp48, 1)[:, None]
    tl.store(out_ptr3 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp48, None)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_clamp_min_exp_mean_mul_norm_sub_2(in_out_ptr0, in_ptr0, in_ptr1, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp14 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr0 + (r0_0 + ks0*ks1*ks2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr0 + (r0_0 + 2*ks0*ks1*ks2), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 - tmp1
        tmp3 = 1e-06
        tmp4 = tmp2 + tmp3
        tmp5 = tmp4 * tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask, tmp8, _tmp7)
        tmp10 = tmp0 - tmp9
        tmp11 = tmp10 + tmp3
        tmp12 = tmp11 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(r0_mask, tmp15, _tmp14)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tmp24 = tl.load(in_ptr1 + (0))
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, 1])
    tmp16 = libdevice.sqrt(tmp7)
    tmp17 = 1.0
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.sqrt(tmp14)
    tmp20 = tmp18 - tmp19
    tmp21 = 0.0
    tmp22 = triton_helpers.maximum(tmp20, tmp21)
    tmp23 = tmp22 / tmp17
    tmp26 = 10*ks0*ks1*ks2
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp25 / tmp27
    tmp29 = tmp23 + tmp28
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp29, None)


def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s1 = arg0_1
    s2 = arg1_1
    s3 = arg2_1
    assert_size_stride(arg3_1, (1, 10, s1, s2, s3), (10*s1*s2*s3, s1*s2*s3, s2*s3, s3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 10, 1, 1, 1), (10, 1, 10, 10, 10), torch.float32)
        buf1 = empty_strided_cuda((1, 10, 1, 1, 1), (10, 1, 10, 10, 10), torch.float32)

        s1*s2*s3
        get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_0[grid(10)](arg3_1, buf0, buf1, 5, 5, 5, 10, 125, XBLOCK=1, R0_BLOCK=128, num_warps=2, num_stages=1)
        buf5 = empty_strided_cuda((1, 10*s1*s2*s3), (10*s1*s2*s3, 1), torch.float32)
        buf8 = empty_strided_cuda((), (), torch.float32)

        10*s1*s2*s3
        get_raw_stream(0)
        triton_red_fused__log_softmax_exp_mean_mul_sub_1[grid(1)](arg3_1, buf0, buf1, buf5, buf8, 5, 5, 5, 1, 1250, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del arg3_1
        del buf0
        del buf1
        buf6 = empty_strided_cuda((1, ), (1, ), torch.float32)
        buf9 = reinterpret_tensor(buf6, (), (), 0); del buf6

        s1*s2*s3
        get_raw_stream(0)
        triton_red_fused_add_clamp_min_exp_mean_mul_norm_sub_2[grid(1)](buf9, buf5, buf8, 5, 5, 5, 1, 125, XBLOCK=1, R0_BLOCK=128, num_warps=2, num_stages=1)
        del buf5
        del buf8
    return (buf9, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 5
    arg1_1 = 5
    arg2_1 = 5
    arg3_1 = rand_strided((1, 10, 5, 5, 5), (1250, 125, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
