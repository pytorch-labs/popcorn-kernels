
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
def triton_poi_fused_max_unpool2d_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_max_unpool2d_1(in_ptr0, out_ptr1, ks0, ks1, ks2, ks3, ks4, ks5, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks1)
    x2 = xindex // ks2
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 2*ks4*x1 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 2*ks4*x1 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (ks4 + 2*x0 + 2*ks4*x1 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (1 + ks4 + 2*x0 + 2*ks4*x1 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr0 + (2*((x3 % ks0)) + 2*ks4*(((x3 // ks0) % ks1)) + ks3*ks4*(x3 // ks2)), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr0 + (1 + 2*((x3 % ks0)) + 2*ks4*(((x3 // ks0) % ks1)) + ks3*ks4*(x3 // ks2)), xmask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr0 + (ks4 + 2*((x3 % ks0)) + 2*ks4*(((x3 // ks0) % ks1)) + ks3*ks4*(x3 // ks2)), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr0 + (1 + ks4 + 2*((x3 % ks0)) + 2*ks4*(((x3 // ks0) % ks1)) + ks3*ks4*(x3 // ks2)), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 > tmp0
    tmp3 = tl.full([1], 1, tl.int8)
    tmp4 = tl.full([1], 0, tl.int8)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = triton_helpers.maximum(tmp1, tmp0)
    tmp8 = tmp7 > tmp6
    tmp9 = tl.full([1], 2, tl.int8)
    tmp10 = tl.where(tmp8, tmp9, tmp5)
    tmp11 = triton_helpers.maximum(tmp7, tmp6)
    tmp13 = tmp12 > tmp11
    tmp14 = tl.full([1], 3, tl.int8)
    tmp15 = tl.where(tmp13, tmp14, tmp10)
    triton_helpers.maximum(tmp12, tmp11)
    tmp17 = tl.full([1], 2, tl.int32)
    tmp18 = tl.where((tmp15 < 0) != (tmp17 < 0), tl.where(tmp15 % tmp17 != 0, tmp15 // tmp17 - 1, tmp15 // tmp17), tmp15 // tmp17)
    tmp19 = tmp18 * tmp17
    tmp20 = tmp15 - tmp19
    tmp21 = 2*x1
    tmp22 = tmp21 + tmp18
    tmp23 = 2*x0
    tmp24 = tmp23 + tmp20
    tmp25 = ks4
    tmp26 = tmp22 * tmp25
    tmp27 = tmp26 + tmp24
    tmp28 = 4*ks0*ks1*x2
    tmp29 = tmp27 + tmp28
    tmp30 = 4*ks0*ks1*ks5
    tmp31 = tmp29 + tmp30
    tmp32 = tmp29 < 0
    tmp33 = tl.where(tmp32, tmp31, tmp29)
    tl.device_assert(((0 <= tmp33) & (tmp33 < 4*ks5*(ks3 // 2)*(ks4 // 2))) | ~(xmask), "index out of bounds: 0 <= tmp33 < 4*ks5*(ks3 // 2)*(ks4 // 2)")
    tmp37 = triton_helpers.maximum(tmp36, tmp35)
    tmp39 = triton_helpers.maximum(tmp38, tmp37)
    tmp41 = triton_helpers.maximum(tmp40, tmp39)
    tl.store(out_ptr1 + (tl.broadcast_to((tmp33 % (4*ks0*ks1*ks5)), [XBLOCK])), tmp41, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_avg_pool2d_2(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks1)
    x2 = xindex // ks2
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 2*ks0*((((x0 + 2*ks0*x1) // ks0) % (2*ks1))) + 4*ks0*ks1*((((x0 + 2*ks0*x1 + 2*ks0*ks1*x2) // (2*ks0*ks1)) % ks3))), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (2*ks0*((((1 + 2*x0 + 4*ks0*x1) // (2*ks0)) % (2*ks1))) + 4*ks0*ks1*((((1 + 2*x0 + 4*ks0*x1 + 4*ks0*ks1*x2) // (4*ks0*ks1)) % ks3)) + (((1 + 2*x0) % (2*ks0)))), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2*x0 + 2*ks0*((((ks0 + x0 + 2*ks0*x1) // ks0) % (2*ks1))) + 4*ks0*ks1*((((ks0 + x0 + 2*ks0*x1 + 2*ks0*ks1*x2) // (2*ks0*ks1)) % ks3))), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (2*ks0*((((1 + 2*ks0 + 2*x0 + 4*ks0*x1) // (2*ks0)) % (2*ks1))) + 4*ks0*ks1*((((1 + 2*ks0 + 2*x0 + 4*ks0*x1 + 4*ks0*ks1*x2) // (4*ks0*ks1)) % ks3)) + (((1 + 2*x0) % (2*ks0)))), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x3), tmp8, xmask)


def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((1, s0, 2*(s1 // 2), 2*(s2 // 2)), (4*s0*(s1 // 2)*(s2 // 2), 4*(s1 // 2)*(s2 // 2), 2*(s2 // 2), 1), torch.float32)

        triton_poi_fused_max_unpool2d_0_xnumel = 4*s0*(s1 // 2)*(s2 // 2)
        get_raw_stream(0)
        triton_poi_fused_max_unpool2d_0[grid(triton_poi_fused_max_unpool2d_0_xnumel)](buf1, 12288, XBLOCK=256, num_warps=4, num_stages=1)
        s2 // 2
        s1 // 2
        (s1 // 2)*(s2 // 2)

        triton_poi_fused_max_pool2d_with_indices_max_unpool2d_1_xnumel = s0*(s1 // 2)*(s2 // 2)
        get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_max_unpool2d_1[grid(triton_poi_fused_max_pool2d_with_indices_max_unpool2d_1_xnumel)](arg3_1, buf1, 32, 32, 1024, 64, 64, 3, 3072, XBLOCK=256, num_warps=4, num_stages=1)
        del arg3_1
        buf3 = empty_strided_cuda((1, s0, s1 // 2, s2 // 2), (s0*(s1 // 2)*(s2 // 2), (s1 // 2)*(s2 // 2), s2 // 2, 1), torch.float32)

        triton_poi_fused_avg_pool2d_2_xnumel = s0*(s1 // 2)*(s2 // 2)
        get_raw_stream(0)
        triton_poi_fused_avg_pool2d_2[grid(triton_poi_fused_avg_pool2d_2_xnumel)](buf1, buf3, 32, 32, 1024, 3, 3072, XBLOCK=128, num_warps=4, num_stages=1)
        del buf1
    return (reinterpret_tensor(buf3, (1, (s1 // 2)*(s2 // 2), s0), (s0*(s1 // 2)*(s2 // 2), 1, (s1 // 2)*(s2 // 2)), 0), )


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
