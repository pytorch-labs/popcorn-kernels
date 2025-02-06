
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
def triton_poi_fused_constant_pad_nd_mish_reflection_pad2d_tanh_0(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // ks0) % ks1)
    x0 = (xindex % ks0)
    x2 = xindex // ks4
    x4 = xindex
    tmp0 = (-2) + (tl.where(3 + ks2 + ((-1)*tl_math.abs(3 + ks2 + ((-1)*tl_math.abs((-2) + x1)))) < 0, 7 + ((-1)*tl_math.abs(3 + ks2 + ((-1)*tl_math.abs((-2) + x1)))) + 2*ks2, 3 + ks2 + ((-1)*tl_math.abs(3 + ks2 + ((-1)*tl_math.abs((-2) + x1))))))
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = ks2
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + (tl.where(3 + ks3 + ((-1)*tl_math.abs(3 + ks3 + ((-1)*tl_math.abs((-2) + x0)))) < 0, 7 + ((-1)*tl_math.abs(3 + ks3 + ((-1)*tl_math.abs((-2) + x0)))) + 2*ks3, 3 + ks3 + ((-1)*tl_math.abs(3 + ks3 + ((-1)*tl_math.abs((-2) + x0))))))
    tmp6 = tmp5 >= tmp1
    tmp7 = ks3
    tmp8 = tmp5 < tmp7
    tmp9 = tmp2 & tmp4
    tmp10 = tmp9 & tmp6
    tmp11 = tmp10 & tmp8
    tmp12 = tl.load(in_ptr0 + ((-2) + ((-2)*ks3) + ks3*(tl.where(3 + ks2 + ((-1)*tl_math.abs(3 + ks2 + ((-1)*tl_math.abs((-2) + x1)))) < 0, 7 + ((-1)*tl_math.abs(3 + ks2 + ((-1)*tl_math.abs((-2) + x1)))) + 2*ks2, 3 + ks2 + ((-1)*tl_math.abs(3 + ks2 + ((-1)*tl_math.abs((-2) + x1)))))) + ks2*ks3*x2 + (tl.where(3 + ks3 + ((-1)*tl_math.abs(3 + ks3 + ((-1)*tl_math.abs((-2) + x0)))) < 0, 7 + ((-1)*tl_math.abs(3 + ks3 + ((-1)*tl_math.abs((-2) + x0)))) + 2*ks3, 3 + ks3 + ((-1)*tl_math.abs(3 + ks3 + ((-1)*tl_math.abs((-2) + x0))))))), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = libdevice.tanh(tmp12)
    tmp14 = 20.0
    tmp15 = tmp13 > tmp14
    tmp16 = tl_math.exp(tmp13)
    tmp17 = libdevice.log1p(tmp16)
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tmp19 = libdevice.tanh(tmp18)
    tmp20 = tmp13 * tmp19
    tl.store(out_ptr0 + (x4), tmp20, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_binary_cross_entropy_with_logits_ones_like_1(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    _tmp24 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp0 = r0_1 + x0*((1 + ks0*ks1*ks2) // 2) + 32*ks0*x0 + 4*ks0*ks1*x0 + 4*ks0*ks2*x0
        tmp1 = 64*ks0 + 8*ks0*ks1 + 8*ks0*ks2 + ks0*ks1*ks2
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (8*((((r0_1 + x0*((1 + ks0*ks1*ks2) // 2) + 32*ks0*x0 + 4*ks0*ks1*x0 + 4*ks0*ks2*x0) // ks3) % ks4)) + 64*((((r0_1 + x0*((1 + ks0*ks1*ks2) // 2) + 32*ks0*x0 + 4*ks0*ks1*x0 + 4*ks0*ks2*x0) // (64 + 8*ks1 + 8*ks2 + ks1*ks2)) % ks0)) + ks2*((((r0_1 + x0*((1 + ks0*ks1*ks2) // 2) + 32*ks0*x0 + 4*ks0*ks1*x0 + 4*ks0*ks2*x0) // ks3) % ks4)) + 8*ks1*((((r0_1 + x0*((1 + ks0*ks1*ks2) // 2) + 32*ks0*x0 + 4*ks0*ks1*x0 + 4*ks0*ks2*x0) // (64 + 8*ks1 + 8*ks2 + ks1*ks2)) % ks0)) + 8*ks2*((((r0_1 + x0*((1 + ks0*ks1*ks2) // 2) + 32*ks0*x0 + 4*ks0*ks1*x0 + 4*ks0*ks2*x0) // (64 + 8*ks1 + 8*ks2 + ks1*ks2)) % ks0)) + ks1*ks2*((((r0_1 + x0*((1 + ks0*ks1*ks2) // 2) + 32*ks0*x0 + 4*ks0*ks1*x0 + 4*ks0*ks2*x0) // (64 + 8*ks1 + 8*ks2 + ks1*ks2)) % ks0)) + (((r0_1 + x0*((1 + ks0*ks1*ks2) // 2) + 32*ks0*x0 + 4*ks0*ks1*x0 + 4*ks0*ks2*x0) % ks3))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl_math.abs(tmp3)
        tmp5 = 1.0
        tmp6 = tmp4 + tmp5
        tmp7 = tmp3 / tmp6
        tmp8 = tl.sigmoid(tmp7)
        tmp9 = tmp7 * tmp8
        tmp10 = libdevice.tanh(tmp9)
        tmp11 = tmp9 - tmp10
        tmp12 = 0.0
        tmp13 = tmp12 * tmp11
        tmp14 = triton_helpers.minimum(tmp12, tmp11)
        tmp15 = tl_math.abs(tmp11)
        tmp16 = -tmp15
        tmp17 = tl_math.exp(tmp16)
        tmp18 = libdevice.log1p(tmp17)
        tmp19 = tmp14 - tmp18
        tmp20 = tmp13 - tmp19
        tmp21 = tl.full(tmp20.shape, 0, tmp20.dtype)
        tmp22 = tl.where(tmp2, tmp20, tmp21)
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, R0_BLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(r0_mask & xmask, tmp25, _tmp24)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp24, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_binary_cross_entropy_with_logits_ones_like_2(in_out_ptr0, in_ptr0, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp4 = 64*ks0 + 8*ks0*ks1 + 8*ks0*ks2 + ks0*ks1*ks2
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
        8 + s2
        8 + s1
        64 + 8*s1 + 8*s2 + s1*s2
        buf0 = empty_strided_cuda((1, s0, 8 + s1, 8 + s2), (64*s0 + 8*s0*s1 + 8*s0*s2 + s0*s1*s2, 64 + 8*s1 + 8*s2 + s1*s2, 8 + s2, 1), torch.float32)

        triton_poi_fused_constant_pad_nd_mish_reflection_pad2d_tanh_0_xnumel = 64*s0 + 8*s0*s1 + 8*s0*s2 + s0*s1*s2
        get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_mish_reflection_pad2d_tanh_0[grid(triton_poi_fused_constant_pad_nd_mish_reflection_pad2d_tanh_0_xnumel)](arg3_1, buf0, 72, 72, 64, 64, 5184, 15552, XBLOCK=128, num_warps=4, num_stages=1)
        del arg3_1
        buf1 = empty_strided_cuda((2, ), (1, ), torch.float32)

        32*s0 + 4*s0*s1 + 4*s0*s2 + ((1 + s0*s1*s2) // 2)
        get_raw_stream(0)
        triton_red_fused_binary_cross_entropy_with_logits_ones_like_1[grid(2)](buf0, buf1, 3, 64, 64, 72, 72, 2, 7776, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del buf0
        buf2 = empty_strided_cuda((), (), torch.float32)
        buf3 = buf2; del buf2

        get_raw_stream(0)
        triton_per_fused_binary_cross_entropy_with_logits_ones_like_2[grid(1)](buf3, buf1, 3, 64, 64, 1, 2, XBLOCK=1, num_warps=2, num_stages=1)
        del buf1
    return (buf3, )


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
