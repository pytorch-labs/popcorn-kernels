
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
def triton_poi_fused_bernoulli_0(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
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
def triton_poi_fused__to_copy_bernoulli_div_mul_1(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // ks0) % ks1)
    x0 = (xindex % ks0)
    x2 = xindex // ks4
    x4 = xindex
    tmp13 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp0 = (-2) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = ks2
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + x0
    tmp6 = tmp5 >= tmp1
    tmp7 = ks3
    tmp8 = tmp5 < tmp7
    tmp9 = tmp2 & tmp4
    tmp10 = tmp9 & tmp6
    tmp11 = tmp10 & tmp8
    tmp12 = tl.load(in_ptr0 + ((-2) + x0 + ((-2)*ks3) + ks3*x1 + ks2*ks3*x2), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = 0.5
    tmp15 = tmp13 < tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = 2.0
    tmp18 = tmp16 * tmp17
    tmp19 = tmp12 * tmp18
    tl.store(out_ptr0 + (x4), tmp19, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_bernoulli_div_mul_view_2(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (4*(x0 // ks1) + 16*x1 + ks3*(x0 // ks1) + 4*ks2*x1 + 4*ks3*x1 + ks2*ks3*x1 + ((x0 % ks1))), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp0, xmask)


def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, ), (1, ), torch.int64)

        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf0)
        buf1 = empty_strided_cuda((1, s0, 1, 1, 1), (s0, 1, s0, s0, s0), torch.float32)

        get_raw_stream(0)
        triton_poi_fused_bernoulli_0[grid(s0)](buf0, buf1, 0, 3, XBLOCK=4, num_warps=1, num_stages=1)
        del buf0
        4 + s2
        4 + s1
        16 + 4*s1 + 4*s2 + s1*s2
        buf2 = empty_strided_cuda((1, s0, 1, 4 + s1, 4 + s2), (16*s0 + 4*s0*s1 + 4*s0*s2 + s0*s1*s2, 16 + 4*s1 + 4*s2 + s1*s2, 16 + 4*s1 + 4*s2 + s1*s2, 4 + s2, 1), torch.float32)

        triton_poi_fused__to_copy_bernoulli_div_mul_1_xnumel = 16*s0 + 4*s0*s1 + 4*s0*s2 + s0*s1*s2
        get_raw_stream(0)
        triton_poi_fused__to_copy_bernoulli_div_mul_1[grid(triton_poi_fused__to_copy_bernoulli_div_mul_1_xnumel)](arg3_1, buf1, buf2, 68, 68, 64, 64, 4624, 13872, XBLOCK=256, num_warps=4, num_stages=1)
        del arg3_1
        del buf1
        16 + 4*s1 + 4*s2 + s1*s2
        buf3 = empty_strided_cuda((1, s0, 16 + 4*s1 + 4*s2 + s1*s2), (16*s0 + 4*s0*s1 + 4*s0*s2 + s0*s1*s2, 16 + 4*s1 + 4*s2 + s1*s2, 1), torch.float32)

        triton_poi_fused__to_copy_bernoulli_div_mul_view_2_xnumel = 16*s0 + 4*s0*s1 + 4*s0*s2 + s0*s1*s2
        get_raw_stream(0)
        triton_poi_fused__to_copy_bernoulli_div_mul_view_2[grid(triton_poi_fused__to_copy_bernoulli_div_mul_view_2_xnumel)](buf2, buf3, 4624, 68, 64, 64, 13872, XBLOCK=256, num_warps=4, num_stages=1)
        del buf2
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
