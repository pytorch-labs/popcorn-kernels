
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
def triton_poi_fused_abs_add_avg_pool2d_div_0(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks1)
    x2 = xindex // ks2
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 2*ks4*x1 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (1 + 2*x0 + 2*ks4*x1 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (ks4 + 2*x0 + 2*ks4*x1 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (1 + ks4 + 2*x0 + 2*ks4*x1 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl_math.abs(tmp0)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 / tmp3
    tmp6 = tl_math.abs(tmp5)
    tmp7 = tmp6 + tmp2
    tmp8 = tmp5 / tmp7
    tmp9 = tmp8 + tmp4
    tmp11 = tl_math.abs(tmp10)
    tmp12 = tmp11 + tmp2
    tmp13 = tmp10 / tmp12
    tmp14 = tmp13 + tmp9
    tmp16 = tl_math.abs(tmp15)
    tmp17 = tmp16 + tmp2
    tmp18 = tmp15 / tmp17
    tmp19 = tmp18 + tmp14
    tmp20 = 0.25
    tmp21 = tmp19 * tmp20
    tl.store(out_ptr0 + (x3), tmp21, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_constant_pad_nd_1(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex // ks0
    x0 = (xindex % ks2)
    x1 = ((xindex // ks2) % ks3)
    x4 = xindex
    tmp0 = (-2) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = ks1
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (2*x0 + ((-2)*ks4*ks5) + 2*ks4*x1 + ks4*ks5*x2), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr0 + (1 + 2*x0 + ((-2)*ks4*ks5) + 2*ks4*x1 + ks4*ks5*x2), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp7 + tmp6
    tmp9 = tl.load(in_ptr0 + (ks4 + 2*x0 + ((-2)*ks4*ks5) + 2*ks4*x1 + ks4*ks5*x2), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp9 + tmp8
    tmp11 = tl.load(in_ptr0 + (1 + ks4 + 2*x0 + ((-2)*ks4*ks5) + 2*ks4*x1 + ks4*ks5*x2), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 + tmp10
    tmp13 = 0.25
    tmp14 = tmp12 * tmp13
    tmp15 = tmp14 * tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp5, tmp15, tmp16)
    tl.store(out_ptr0 + (x4), tmp17, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_abs_add_avg_pool2d_div_mul_pow_2(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks1)
    x2 = xindex // ks2
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 2*ks3*x1 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 2*ks3*x1 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (ks3 + 2*x0 + 2*ks3*x1 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (1 + ks3 + 2*x0 + 2*ks3*x1 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr1 + (ks2 + x3), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (x3 + 2*ks0*ks1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr1 + (x3 + 3*ks0*ks1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr1 + (x3 + 4*ks0*ks1), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tmp11 = tmp10 + tmp9
    tmp13 = tmp12 + tmp11
    tmp15 = tmp14 + tmp13
    tmp17 = tmp16 + tmp15
    tmp18 = 0.2
    tmp19 = tmp17 * tmp18
    tmp20 = 0.0001
    tmp21 = tmp19 * tmp20
    tmp22 = 1.0
    tmp23 = tmp21 + tmp22
    tmp24 = 0.75
    tmp25 = libdevice.pow(tmp23, tmp24)
    tmp26 = tmp8 / tmp25
    tl.store(out_ptr0 + (x3), tmp26, xmask)


def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        s2 // 2
        s1 // 2
        (s1 // 2)*(s2 // 2)
        buf0 = empty_strided_cuda((1, s0, s1 // 2, s2 // 2), (s0*(s1 // 2)*(s2 // 2), (s1 // 2)*(s2 // 2), s2 // 2, 1), torch.float32)

        triton_poi_fused_abs_add_avg_pool2d_div_0_xnumel = s0*(s1 // 2)*(s2 // 2)
        get_raw_stream(0)
        triton_poi_fused_abs_add_avg_pool2d_div_0[grid(triton_poi_fused_abs_add_avg_pool2d_div_0_xnumel)](arg3_1, buf0, 32, 32, 1024, 64, 64, 3072, XBLOCK=256, num_warps=4, num_stages=1)
        del arg3_1
        (s1 // 4)*(s2 // 4)
        s2 // 4
        s1 // 4
        buf1 = empty_strided_cuda((1, 1, 4 + s0, s1 // 4, s2 // 4), (4*(s1 // 4)*(s2 // 4) + s0*(s1 // 4)*(s2 // 4), 4*(s1 // 4)*(s2 // 4) + s0*(s1 // 4)*(s2 // 4), (s1 // 4)*(s2 // 4), s2 // 4, 1), torch.float32)

        triton_poi_fused_constant_pad_nd_1_xnumel = 4*(s1 // 4)*(s2 // 4) + s0*(s1 // 4)*(s2 // 4)
        get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_1[grid(triton_poi_fused_constant_pad_nd_1_xnumel)](buf0, buf1, 256, 3, 16, 16, 32, 32, 1792, XBLOCK=128, num_warps=4, num_stages=1)
        buf2 = empty_strided_cuda((1, s0, s1 // 4, s2 // 4), (s0*(s1 // 4)*(s2 // 4), (s1 // 4)*(s2 // 4), s2 // 4, 1), torch.float32)

        triton_poi_fused_abs_add_avg_pool2d_div_mul_pow_2_xnumel = s0*(s1 // 4)*(s2 // 4)
        get_raw_stream(0)
        triton_poi_fused_abs_add_avg_pool2d_div_mul_pow_2[grid(triton_poi_fused_abs_add_avg_pool2d_div_mul_pow_2_xnumel)](buf0, buf1, buf2, 16, 16, 256, 32, 32, 768, XBLOCK=128, num_warps=4, num_stages=1)
        del buf0
        del buf1
    return (reinterpret_tensor(buf2, (1, s0*(s1 // 4), s2 // 4), (s0*(s1 // 4)*(s2 // 4), s2 // 4, 1), 0), )


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
