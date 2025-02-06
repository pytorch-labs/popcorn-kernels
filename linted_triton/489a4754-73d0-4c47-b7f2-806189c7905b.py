
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
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_dropout_0(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (x0), tmp2, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_mean_native_dropout_1(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = (xindex % 32)
    x1 = xindex // 32
    _tmp14 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_2 = r0_index
        tmp0 = r0_2 + x0*((31 + ks0*ks1*ks2) // 32)
        tmp1 = ks0*ks1*ks2
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (ks0*ks1*ks2*x1 + (((r0_2 + x0*((31 + ks0*ks1*ks2) // 32)) % (ks0*ks1*ks2)))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.5
        tmp5 = tmp3 > tmp4
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tl.load(in_ptr1 + (ks0*ks1*ks2*x1 + (((r0_2 + x0*((31 + ks0*ks1*ks2) // 32)) % (ks0*ks1*ks2)))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 * tmp7
        tmp9 = 2.0
        tmp10 = tmp8 * tmp9
        tmp11 = tl.full(tmp10.shape, 0, tmp10.dtype)
        tmp12 = tl.where(tmp2, tmp10, tmp11)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(r0_mask & xmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_mean_native_dropout_2(in_out_ptr0, in_ptr0, in_ptr1, load_seed_offset, ks1, ks2, ks3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    R0_BLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 32*x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = tl.load(in_ptr1 + load_seed_offset)
    tmp6 = x0
    tmp7 = tl.rand(tmp5, (tmp6).to(tl.uint32))
    tmp8 = 0.5
    tmp9 = tmp7 > tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = ks1*ks2*ks3
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp4 / tmp12
    tmp14 = tmp10 * tmp13
    tmp15 = 2.0
    tmp16 = tmp14 * tmp15
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp16, xmask)


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
        buf0 = empty_strided_cuda((2, ), (1, ), torch.int64)

        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [2], out=buf0)
        buf2 = empty_strided_cuda((1, s0, s1, s2, s3), (s0*s1*s2*s3, s1*s2*s3, s2*s3, s3, 1), torch.float32)

        triton_poi_fused_native_dropout_0_xnumel = s0*s1*s2*s3
        get_raw_stream(0)
        triton_poi_fused_native_dropout_0[grid(triton_poi_fused_native_dropout_0_xnumel)](buf0, buf2, 0, 786432, XBLOCK=1024, num_warps=4, num_stages=1)
        buf3 = empty_strided_cuda((1, s0, 1, 1, 1, 32), (32*s0, 32, 32*s0, 32*s0, 32*s0, 1), torch.float32)

        triton_red_fused_mean_native_dropout_1_xnumel = 32*s0
        (31 + s1*s2*s3) // 32
        get_raw_stream(0)
        triton_red_fused_mean_native_dropout_1[grid(triton_red_fused_mean_native_dropout_1_xnumel)](buf2, arg4_1, buf3, 64, 64, 64, 96, 8192, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del arg4_1
        del buf2
        buf1 = empty_strided_cuda((1, s0), (s0, 1), torch.float32)
        buf5 = buf1; del buf1

        get_raw_stream(0)
        triton_per_fused_mean_native_dropout_2[grid(s0)](buf5, buf3, buf0, 1, 64, 64, 64, 3, 32, XBLOCK=1, num_warps=2, num_stages=1)
        del buf0
        del buf3
    return (buf5, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 3
    arg1_1 = 64
    arg2_1 = 64
    arg3_1 = 64
    arg4_1 = rand_strided((1, 3, 64, 64, 64), (786432, 262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
