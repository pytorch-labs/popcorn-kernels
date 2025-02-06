
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
def triton_poi_fused__to_copy_bernoulli_div_mul_1(in_ptr0, in_ptr1, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // ks0
    tmp0 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = 0.5
    tmp3 = tmp1 < tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 2.0
    tmp6 = tmp4 * tmp5
    tmp7 = tmp0 * tmp6
    tl.store(out_ptr0 + (x2), tmp7, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_bernoulli_2(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
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
def triton_poi_fused__to_copy_adaptive_max_pool2d_add_bernoulli_mul_3(in_out_ptr0, in_ptr0, in_ptr1, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (10*x0 + ks1*x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 10*x0 + ks1*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 10*x0 + ks1*x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 10*x0 + ks1*x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4 + 10*x0 + ks1*x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (5 + 10*x0 + ks1*x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (6 + 10*x0 + ks1*x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (7 + 10*x0 + ks1*x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (8 + 10*x0 + ks1*x1), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (9 + 10*x0 + ks1*x1), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tmp18 = triton_helpers.maximum(tmp17, tmp16)
    tmp20 = 0.5
    tmp21 = tmp19 < tmp20
    tmp22 = tmp21.to(tl.float32)
    tmp23 = 0.8864048946659319
    tmp24 = tmp22 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = -1.0
    tmp27 = tmp22 + tmp26
    tmp28 = 1.558387861036063
    tmp29 = tmp27 * tmp28
    tmp30 = 0.7791939305180315
    tmp31 = tmp29 + tmp30
    tmp32 = tmp25 + tmp31
    tl.store(in_out_ptr0 + (x2), tmp32, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_add_bernoulli_mul_4(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + ks1*x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + ks1*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = 0.5
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 0.8864048946659319
    tmp8 = tmp6 * tmp7
    tmp9 = tmp2 * tmp8
    tmp10 = -1.0
    tmp11 = tmp6 + tmp10
    tmp12 = 1.558387861036063
    tmp13 = tmp11 * tmp12
    tmp14 = 0.7791939305180315
    tmp15 = tmp13 + tmp14
    tmp16 = tmp9 + tmp15
    tl.store(out_ptr0 + (x2), tmp16, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_5(in_ptr0, out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + ks1*x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + ks1*x1), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(out_ptr0 + (x2), tmp2, xmask)


def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    assert_size_stride(arg2_1, (1, s0, s1), (s0*s1, s1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((3, ), (1, ), torch.int64)

        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [3], out=buf0)
        buf1 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)

        get_raw_stream(0)
        triton_poi_fused_bernoulli_0[grid(s0)](buf0, buf1, 2, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf2 = empty_strided_cuda((1, s0, s1), (s0*s1, s1, 1), torch.float32)

        triton_poi_fused__to_copy_bernoulli_div_mul_1_xnumel = s0*s1
        get_raw_stream(0)
        triton_poi_fused__to_copy_bernoulli_div_mul_1[grid(triton_poi_fused__to_copy_bernoulli_div_mul_1_xnumel)](arg2_1, buf1, buf2, 100, 3200, XBLOCK=256, num_warps=4, num_stages=1)
        del arg2_1
        buf4 = buf1; del buf1

        get_raw_stream(0)
        triton_poi_fused_bernoulli_2[grid(s0)](buf0, buf4, 1, 32, XBLOCK=32, num_warps=1, num_stages=1)
        s1 // 10
        buf3 = empty_strided_cuda((1, s0, 1, s1 // 10), (s0*(s1 // 10), s1 // 10, s1 // 10, 1), torch.float32)
        buf5 = reinterpret_tensor(buf3, (1, s0, s1 // 10), (s0*(s1 // 10), s1 // 10, 1), 0); del buf3

        triton_poi_fused__to_copy_adaptive_max_pool2d_add_bernoulli_mul_3_xnumel = s0*(s1 // 10)
        get_raw_stream(0)
        triton_poi_fused__to_copy_adaptive_max_pool2d_add_bernoulli_mul_3[grid(triton_poi_fused__to_copy_adaptive_max_pool2d_add_bernoulli_mul_3_xnumel)](buf5, buf2, buf4, 10, 100, 320, XBLOCK=128, num_warps=4, num_stages=1)
        del buf2
        buf6 = buf4; del buf4

        get_raw_stream(0)
        triton_poi_fused_bernoulli_0[grid(s0)](buf0, buf6, 2, 32, XBLOCK=32, num_warps=1, num_stages=1)
        del buf0
        s1 // 20
        buf7 = empty_strided_cuda((1, s0, s1 // 20), (s0*(s1 // 20), s1 // 20, 1), torch.float32)

        triton_poi_fused__to_copy_add_bernoulli_mul_4_xnumel = s0*(s1 // 20)
        get_raw_stream(0)
        triton_poi_fused__to_copy_add_bernoulli_mul_4[grid(triton_poi_fused__to_copy_add_bernoulli_mul_4_xnumel)](buf5, buf6, buf7, 5, 10, 160, XBLOCK=128, num_warps=4, num_stages=1)
        del buf5
        del buf6
        s1 // 40
        buf8 = empty_strided_cuda((1, s0, 1, s1 // 40), (s0*(s1 // 40), s1 // 40, s1 // 40, 1), torch.float32)

        triton_poi_fused_max_pool2d_with_indices_5_xnumel = s0*(s1 // 40)
        get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_5[grid(triton_poi_fused_max_pool2d_with_indices_5_xnumel)](buf7, buf8, 2, 5, 64, XBLOCK=64, num_warps=1, num_stages=1)
        del buf7
    return (reinterpret_tensor(buf8, (1, s0, s1 // 40), (s0*(s1 // 40), s1 // 40, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 32
    arg1_1 = 100
    arg2_1 = rand_strided((1, 32, 100), (3200, 100, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
