
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
from torch._inductor.runtime.triton_helpers import libdevice
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_0(in_ptr0, out_ptr0, out_ptr1, out_ptr2, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp13_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp13_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp13_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_2 = r0_index
        tmp0 = r0_2 + x0*((31 + ks0*ks1*ks2) // 32)
        tmp1 = ks0*ks1*ks2
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (ks0*ks1*ks2*x1 + (((r0_2 + x0*((31 + ks0*ks1*ks2) // 32)) % (ks0*ks1*ks2)))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tl.full(tmp4.shape, 0, tmp4.dtype)
        tmp6 = tl.where(tmp2, tmp4, tmp5)
        tmp7 = 1.0
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp11 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp12 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp13_mean_next, tmp13_m2_next, tmp13_weight_next = triton_helpers.welford_combine(
            tmp13_mean, tmp13_m2, tmp13_weight,
            tmp10, tmp11, tmp12
        )
        tmp13_mean = tl.where(r0_mask & xmask, tmp13_mean_next, tmp13_mean)
        tmp13_m2 = tl.where(r0_mask & xmask, tmp13_m2_next, tmp13_m2)
        tmp13_weight = tl.where(r0_mask & xmask, tmp13_weight_next, tmp13_weight)
    tmp16, tmp17, tmp18 = triton_helpers.welford(tmp13_mean, tmp13_m2, tmp13_weight, 1)
    tmp13 = tmp16[:, None]
    tmp14 = tmp17[:, None]
    tmp15 = tmp18[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
    tl.store(out_ptr2 + (x3), tmp15, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32
    R0_BLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 32*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 32*x0), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r0_1 + 32*x0), xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp12[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // ks0
    tmp0 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = ks0
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp2 * tmp9
    tl.store(out_ptr0 + (x2), tmp10, xmask)


def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s1 = arg0_1
    s2 = arg1_1
    s3 = arg2_1
    assert_size_stride(arg3_1, (1, 32, s1, s2, s3), (32*s1*s2*s3, s1*s2*s3, s2*s3, s3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 32, 1, 1, 1, 32), (1024, 32, 1024, 1024, 1024, 1), torch.float32)
        buf1 = empty_strided_cuda((1, 32, 1, 1, 1, 32), (1024, 32, 1024, 1024, 1024, 1), torch.float32)
        buf2 = empty_strided_cuda((1, 32, 1, 1, 1, 32), (1024, 32, 1024, 1024, 1024, 1), torch.float32)

        (31 + s1*s2*s3) // 32
        get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_0[grid(1024)](arg3_1, buf0, buf1, buf2, 64, 64, 64, 1024, 8192, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        buf3 = empty_strided_cuda((1, 32, 1, 1, 1), (32, 1, 32, 32, 32), torch.float32)
        buf4 = empty_strided_cuda((1, 32, 1, 1, 1), (32, 1, 32, 32, 32), torch.float32)

        get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_1[grid(32)](buf0, buf1, buf2, buf3, buf4, 32, 32, XBLOCK=1, num_warps=2, num_stages=1)
        del buf0
        del buf1
        del buf2
        s1*s2*s3
        buf6 = empty_strided_cuda((1, 32, s1, s2, s3), (32*s1*s2*s3, s1*s2*s3, s2*s3, s3, 1), torch.float32)

        triton_poi_fused__native_batch_norm_legit_2_xnumel = 32*s1*s2*s3
        get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_2[grid(triton_poi_fused__native_batch_norm_legit_2_xnumel)](arg3_1, buf3, buf4, buf6, 262144, 8388608, XBLOCK=512, num_warps=8, num_stages=1)
        del arg3_1
        del buf3
        del buf4
    return (reinterpret_tensor(buf6, (1, 32, s1*s2*s3), (32*s1*s2*s3, s1*s2*s3, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 64
    arg1_1 = 64
    arg2_1 = 64
    arg3_1 = rand_strided((1, 32, 64, 64, 64), (8388608, 262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
