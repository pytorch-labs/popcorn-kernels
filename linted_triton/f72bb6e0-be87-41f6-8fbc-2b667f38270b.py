
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
def triton_poi_fused__to_copy_abs_add_avg_pool2d_bernoulli_mul_pow_relu_sign_1(in_out_ptr0, in_ptr0, in_ptr1, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    x2 = xindex
    tmp53 = tl.load(in_ptr1 + (x0 + x1 + ks1*x1 + ks2*x1 + x1*((1 + ks1*ks2) // 2)), xmask, eviction_policy='evict_last')
    tmp0 = (-1) + ((((2*x0) // (2 + ks2)) % (2 + ks1)))
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = ks1
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + (((2*x0) % (2 + ks2)))
    tmp6 = tmp5 >= tmp1
    tmp7 = ks2
    tmp8 = tmp5 < tmp7
    tmp9 = tmp2 & tmp4
    tmp10 = tmp9 & tmp6
    tmp11 = tmp10 & tmp8
    tmp12 = tl.load(in_ptr0 + ((-1) + ((-1)*ks2) + ks2*((((2*x0) // (2 + ks2)) % (2 + ks1))) + ks1*ks2*x1 + (((2*x0) % (2 + ks2)))), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp12 * tmp12
    tmp14 = (-1) + ((((1 + 2*x0) // (2 + ks2)) % (2 + ks1)))
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = (-1) + (((1 + 2*x0) % (2 + ks2)))
    tmp18 = tmp17 >= tmp1
    tmp19 = tmp17 < tmp7
    tmp20 = tmp15 & tmp16
    tmp21 = tmp20 & tmp18
    tmp22 = tmp21 & tmp19
    tmp23 = tl.load(in_ptr0 + ((-1) + ((-1)*ks2) + ks2*((((1 + 2*x0) // (2 + ks2)) % (2 + ks1))) + ks1*ks2*x1 + (((1 + 2*x0) % (2 + ks2)))), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp23 * tmp23
    tmp25 = tmp24 + tmp13
    tmp26 = (-1) + ((((2 + 2*x0) // (2 + ks2)) % (2 + ks1)))
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = (-1) + (((2 + 2*x0) % (2 + ks2)))
    tmp30 = tmp29 >= tmp1
    tmp31 = tmp29 < tmp7
    tmp32 = tmp27 & tmp28
    tmp33 = tmp32 & tmp30
    tmp34 = tmp33 & tmp31
    tmp35 = tl.load(in_ptr0 + ((-1) + ((-1)*ks2) + ks2*((((2 + 2*x0) // (2 + ks2)) % (2 + ks1))) + ks1*ks2*x1 + (((2 + 2*x0) % (2 + ks2)))), tmp34 & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tmp35 * tmp35
    tmp37 = tmp36 + tmp25
    tmp38 = 0.3333333333333333
    tmp39 = tmp37 * tmp38
    tmp40 = tl.full([1], 0, tl.int32)
    tmp41 = tmp40 < tmp39
    tmp42 = tmp41.to(tl.int8)
    tmp43 = tmp39 < tmp40
    tmp44 = tmp43.to(tl.int8)
    tmp45 = tmp42 - tmp44
    tmp46 = tmp45.to(tmp39.dtype)
    tmp47 = tl_math.abs(tmp39)
    tmp48 = triton_helpers.maximum(tmp40, tmp47)
    tmp49 = tmp46 * tmp48
    tmp50 = 3.0
    tmp51 = tmp49 * tmp50
    tmp52 = libdevice.sqrt(tmp51)
    tmp54 = 0.5
    tmp55 = tmp53 < tmp54
    tmp56 = tmp55.to(tl.float32)
    tmp57 = 0.8864048946659319
    tmp58 = tmp56 * tmp57
    tmp59 = tmp52 * tmp58
    tmp60 = -1.0
    tmp61 = tmp56 + tmp60
    tmp62 = 1.558387861036063
    tmp63 = tmp61 * tmp62
    tmp64 = 0.7791939305180315
    tmp65 = tmp63 + tmp64
    tmp66 = tmp59 + tmp65
    tl.store(in_out_ptr0 + (x2), tmp66, xmask)


def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((1, ), (1, ), torch.int64)

        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf1)
        buf2 = empty_strided_cuda((1, s0, 1 + s1 + s2 + ((1 + s1*s2) // 2)), (s0 + s0*s1 + s0*s2 + s0*((1 + s1*s2) // 2), 1 + s1 + s2 + ((1 + s1*s2) // 2), 1), torch.float32)

        triton_poi_fused_bernoulli_0_xnumel = s0 + s0*s1 + s0*s2 + s0*((1 + s1*s2) // 2)
        get_raw_stream(0)
        triton_poi_fused_bernoulli_0[grid(triton_poi_fused_bernoulli_0_xnumel)](buf1, buf2, 0, 6531, XBLOCK=128, num_warps=4, num_stages=1)
        del buf1
        s1 + s2 + ((3 + s1*s2) // 2)
        buf0 = empty_strided_cuda((1, s0, 1, s1 + s2 + ((3 + s1*s2) // 2)), (s0*s1 + s0*s2 + s0*((3 + s1*s2) // 2), s1 + s2 + ((3 + s1*s2) // 2), s1 + s2 + ((3 + s1*s2) // 2), 1), torch.float32)
        buf3 = reinterpret_tensor(buf0, (1, s0, s1 + s2 + ((3 + s1*s2) // 2)), (s0*s1 + s0*s2 + s0*((3 + s1*s2) // 2), s1 + s2 + ((3 + s1*s2) // 2), 1), 0); del buf0

        triton_poi_fused__to_copy_abs_add_avg_pool2d_bernoulli_mul_pow_relu_sign_1_xnumel = s0*s1 + s0*s2 + s0*((3 + s1*s2) // 2)
        get_raw_stream(0)
        triton_poi_fused__to_copy_abs_add_avg_pool2d_bernoulli_mul_pow_relu_sign_1[grid(triton_poi_fused__to_copy_abs_add_avg_pool2d_bernoulli_mul_pow_relu_sign_1_xnumel)](buf3, arg3_1, buf2, 2177, 64, 64, 6531, XBLOCK=128, num_warps=4, num_stages=1)
        del arg3_1
        del buf2
    return (reinterpret_tensor(buf3, (1, s1 + s2 + ((3 + s1*s2) // 2), s0), (s0 + s0*s1 + s0*s2 + s0*((1 + s1*s2) // 2), 1, s1 + s2 + ((3 + s1*s2) // 2)), 0), )


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
