
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
from torch._inductor.runtime.triton_helpers import math as tl_math
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__softmax_0(in_ptr0, out_ptr0, out_ptr1, ks0, ks1, ks2, ks3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (ks3*(tl.where((-1) + ks2 + ((-1)*tl_math.abs(1 + ((-1)*ks2) + tl_math.abs((-1) + (x0 // (2 + ks3))))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks2) + tl_math.abs((-1) + (x0 // (2 + ks3))))) + 2*ks2, (-1) + ks2 + ((-1)*tl_math.abs(1 + ((-1)*ks2) + tl_math.abs((-1) + (x0 // (2 + ks3))))))) + ks2*ks3*(tl.where((-1) + ks1 + ((-1)*tl_math.abs(1 + ((-1)*ks1) + tl_math.abs((-1) + x1))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks1) + tl_math.abs((-1) + x1))) + 2*ks1, (-1) + ks1 + ((-1)*tl_math.abs(1 + ((-1)*ks1) + tl_math.abs((-1) + x1))))) + ks1*ks2*ks3*r0_2 + (tl.where((-1) + ks3 + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs((-1) + ((x0 % (2 + ks3)))))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs((-1) + ((x0 % (2 + ks3)))))) + 2*ks3, (-1) + ks3 + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs((-1) + ((x0 % (2 + ks3))))))))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = triton_helpers.maximum(_tmp2, tmp1)
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = triton_helpers.max2(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_2 = r0_index
        tmp4 = tl.load(in_ptr0 + (ks3*(tl.where((-1) + ks2 + ((-1)*tl_math.abs(1 + ((-1)*ks2) + tl_math.abs((-1) + (x0 // (2 + ks3))))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks2) + tl_math.abs((-1) + (x0 // (2 + ks3))))) + 2*ks2, (-1) + ks2 + ((-1)*tl_math.abs(1 + ((-1)*ks2) + tl_math.abs((-1) + (x0 // (2 + ks3))))))) + ks2*ks3*(tl.where((-1) + ks1 + ((-1)*tl_math.abs(1 + ((-1)*ks1) + tl_math.abs((-1) + x1))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks1) + tl_math.abs((-1) + x1))) + 2*ks1, (-1) + ks1 + ((-1)*tl_math.abs(1 + ((-1)*ks1) + tl_math.abs((-1) + x1))))) + ks1*ks2*ks3*r0_2 + (tl.where((-1) + ks3 + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs((-1) + ((x0 % (2 + ks3)))))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs((-1) + ((x0 % (2 + ks3)))))) + 2*ks3, (-1) + ks3 + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs((-1) + ((x0 % (2 + ks3))))))))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp4 - tmp2
        tmp6 = tl_math.exp(tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp8, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__softmax_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks1)
    x2 = xindex // ks2
    x3 = (xindex % ks2)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (ks5*(tl.where((-1) + ks4 + ((-1)*tl_math.abs(1 + ((-1)*ks4) + tl_math.abs((-1) + (x0 // (2 + ks5))))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks4) + tl_math.abs((-1) + (x0 // (2 + ks5))))) + 2*ks4, (-1) + ks4 + ((-1)*tl_math.abs(1 + ((-1)*ks4) + tl_math.abs((-1) + (x0 // (2 + ks5))))))) + ks4*ks5*(tl.where((-1) + ks3 + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs((-1) + x1))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs((-1) + x1))) + 2*ks3, (-1) + ks3 + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs((-1) + x1))))) + ks3*ks4*ks5*x2 + (tl.where((-1) + ks5 + ((-1)*tl_math.abs(1 + ((-1)*ks5) + tl_math.abs((-1) + ((x0 % (2 + ks5)))))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks5) + tl_math.abs((-1) + ((x0 % (2 + ks5)))))) + 2*ks5, (-1) + ks5 + ((-1)*tl_math.abs(1 + ((-1)*ks5) + tl_math.abs((-1) + ((x0 % (2 + ks5))))))))), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp3 = tl_math.exp(tmp2)
    tmp5 = tmp3 / tmp4
    tl.store(out_ptr0 + (x4), tmp5, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_reflection_pad3d_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 12)
    x1 = ((xindex // 12) % 12)
    x2 = ((xindex // 144) % 12)
    x3 = xindex // 1728
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (511 + ((-1)*tl_math.abs((-7) + tl_math.abs((-2) + x0))) + ((-64)*tl_math.abs((-7) + tl_math.abs((-2) + x2))) + ((-8)*tl_math.abs((-7) + tl_math.abs((-2) + x1))) + 512*x3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x4), tmp0, xmask)


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
        4 + 2*s2 + 2*s3 + s2*s3
        buf0 = empty_strided_cuda((1, 1, 2 + s1, 4 + 2*s2 + 2*s3 + s2*s3), (8 + 4*s1 + 4*s2 + 4*s3 + 2*s1*s2 + 2*s1*s3 + 2*s2*s3 + s1*s2*s3, 8 + 4*s1 + 4*s2 + 4*s3 + 2*s1*s2 + 2*s1*s3 + 2*s2*s3 + s1*s2*s3, 4 + 2*s2 + 2*s3 + s2*s3, 1), torch.float32)
        buf1 = empty_strided_cuda((1, 1, 2 + s1, 4 + 2*s2 + 2*s3 + s2*s3), (8 + 4*s1 + 4*s2 + 4*s3 + 2*s1*s2 + 2*s1*s3 + 2*s2*s3 + s1*s2*s3, 8 + 4*s1 + 4*s2 + 4*s3 + 2*s1*s2 + 2*s1*s3 + 2*s2*s3 + s1*s2*s3, 4 + 2*s2 + 2*s3 + s2*s3, 1), torch.float32)

        triton_red_fused__softmax_0_xnumel = 8 + 4*s1 + 4*s2 + 4*s3 + 2*s1*s2 + 2*s1*s3 + 2*s2*s3 + s1*s2*s3
        get_raw_stream(0)
        triton_red_fused__softmax_0[grid(triton_red_fused__softmax_0_xnumel)](arg4_1, buf0, buf1, 324, 16, 16, 16, 5832, 3, XBLOCK=128, R0_BLOCK=4, num_warps=4, num_stages=1)
        2 + s1
        8 + 4*s1 + 4*s2 + 4*s3 + 2*s1*s2 + 2*s1*s3 + 2*s2*s3 + s1*s2*s3
        buf2 = empty_strided_cuda((1, s0, 2 + s1, 4 + 2*s2 + 2*s3 + s2*s3), (8*s0 + 4*s0*s1 + 4*s0*s2 + 4*s0*s3 + 2*s0*s1*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + s0*s1*s2*s3, 8 + 4*s1 + 4*s2 + 4*s3 + 2*s1*s2 + 2*s1*s3 + 2*s2*s3 + s1*s2*s3, 4 + 2*s2 + 2*s3 + s2*s3, 1), torch.float32)

        triton_poi_fused__softmax_1_xnumel = 8*s0 + 4*s0*s1 + 4*s0*s2 + 4*s0*s3 + 2*s0*s1*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + s0*s1*s2*s3
        get_raw_stream(0)
        triton_poi_fused__softmax_1[grid(triton_poi_fused__softmax_1_xnumel)](arg4_1, buf0, buf1, buf2, 324, 18, 5832, 16, 16, 16, 17496, XBLOCK=256, num_warps=4, num_stages=1)
        del arg4_1
        del buf0
        del buf1

        buf3 = torch.ops.aten.adaptive_max_pool3d.default(reinterpret_tensor(buf2, (1, s0, 2 + s1, 4 + 2*s2 + 2*s3 + s2*s3, 1), (0, 8 + 4*s1 + 4*s2 + 4*s3 + 2*s1*s2 + 2*s1*s3 + 2*s2*s3 + s1*s2*s3, 4 + 2*s2 + 2*s3 + s2*s3, 1, 0), 0), [8, 8, 8])
        del buf2
        buf4 = buf3[0]
        del buf3
        buf6 = empty_strided_cuda((1, s0, 12, 12, 12), (1728*s0, 1728, 144, 12, 1), torch.float32)

        triton_poi_fused_reflection_pad3d_2_xnumel = 1728*s0
        get_raw_stream(0)
        triton_poi_fused_reflection_pad3d_2[grid(triton_poi_fused_reflection_pad3d_2_xnumel)](buf4, buf6, 5184, XBLOCK=256, num_warps=4, num_stages=1)
        del buf4

        buf7 = torch.ops.aten.adaptive_max_pool3d.default(buf6, [4, 4, 4])
        del buf6
        buf8 = buf7[0]
        del buf7
    return (buf8, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 3
    arg1_1 = 16
    arg2_1 = 16
    arg3_1 = 16
    arg4_1 = rand_strided((1, 3, 16, 16, 16), (12288, 4096, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
