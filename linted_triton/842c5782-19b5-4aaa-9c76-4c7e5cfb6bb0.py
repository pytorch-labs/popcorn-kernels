
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
def triton_poi_fused_rand_0(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
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
def triton_poi_fused_fractional_max_pool2d_1(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex // 196
    x1 = ((xindex // 14) % 14)
    x0 = (xindex % 14)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr0 + (1 + 2*x2), xmask, eviction_policy='evict_last')
    tmp1 = ((-2) + ks0) / 13
    tmp2 = tmp1.to(tl.float32)
    tmp3 = x1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp4 + tmp0
    tmp6 = tmp5 * tmp2
    tmp7 = libdevice.floor(tmp6)
    tmp8 = tmp0 * tmp2
    tmp9 = libdevice.floor(tmp8)
    tmp10 = tmp7 - tmp9
    tmp11 = tmp10.to(tl.int64)
    tmp12 = tl.full([1], 13, tl.int64)
    tmp13 = tmp4 < tmp12
    tmp14 = (-2) + ks0
    tmp15 = tl.where(tmp13, tmp11, tmp14)
    tmp16 = ks0
    tmp17 = tmp15 + tmp16
    tmp18 = tmp15 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp15)
    tl.device_assert(((0 <= tmp19) & (tmp19 < ks0)) | ~(xmask), "index out of bounds: 0 <= tmp19 < ks0")
    tmp22 = ((-2) + ks1) / 13
    tmp23 = tmp22.to(tl.float32)
    tmp24 = x0
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp25 + tmp21
    tmp27 = tmp26 * tmp23
    tmp28 = libdevice.floor(tmp27)
    tmp29 = tmp21 * tmp23
    tmp30 = libdevice.floor(tmp29)
    tmp31 = tmp28 - tmp30
    tmp32 = tmp31.to(tl.int64)
    tmp33 = tmp25 < tmp12
    tmp34 = (-2) + ks1
    tmp35 = tl.where(tmp33, tmp32, tmp34)
    tmp36 = ks1
    tmp37 = tmp35 + tmp36
    tmp38 = tmp35 < 0
    tmp39 = tl.where(tmp38, tmp37, tmp35)
    tl.device_assert(((0 <= tmp39) & (tmp39 < ks1)) | ~(xmask), "index out of bounds: 0 <= tmp39 < ks1")
    tmp41 = tl.load(in_ptr1 + (tmp39 + ks1*tmp19 + ks0*ks1*x2), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr1 + (1 + tmp39 + ks1*tmp19 + ks0*ks1*x2), xmask, eviction_policy='evict_last')
    tmp43 = triton_helpers.maximum(tmp42, tmp41)
    tmp44 = tl.load(in_ptr1 + (ks1 + tmp39 + ks1*tmp19 + ks0*ks1*x2), xmask, eviction_policy='evict_last')
    tmp45 = triton_helpers.maximum(tmp44, tmp43)
    tmp46 = tl.load(in_ptr1 + (1 + ks1 + tmp39 + ks1*tmp19 + ks0*ks1*x2), xmask, eviction_policy='evict_last')
    tmp47 = triton_helpers.maximum(tmp46, tmp45)
    tl.store(out_ptr0 + (x4), tmp47, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 28) % 28)
    x0 = (xindex % 28)
    x2 = xindex // 784
    x4 = xindex
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.48148148148148145
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full([1], 13, tl.int64)
    tmp10 = triton_helpers.minimum(tmp8, tmp9)
    tmp11 = x0
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp12 * tmp2
    tmp14 = triton_helpers.maximum(tmp13, tmp4)
    tmp15 = tmp14.to(tl.int32)
    tmp16 = tl.load(in_ptr0 + (tmp15 + 14*tmp10 + 196*x2), xmask, eviction_policy='evict_last')
    tmp17 = tmp15 + tmp7
    tmp18 = triton_helpers.minimum(tmp17, tmp9)
    tmp19 = tl.load(in_ptr0 + (tmp18 + 14*tmp10 + 196*x2), xmask, eviction_policy='evict_last')
    tmp20 = tmp19 - tmp16
    tmp21 = tmp15.to(tl.float32)
    tmp22 = tmp14 - tmp21
    tmp23 = triton_helpers.maximum(tmp22, tmp4)
    tmp24 = 1.0
    tmp25 = triton_helpers.minimum(tmp23, tmp24)
    tmp26 = tmp20 * tmp25
    tmp27 = tmp16 + tmp26
    tmp28 = tl.load(in_ptr0 + (tmp15 + 14*tmp6 + 196*x2), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr0 + (tmp18 + 14*tmp6 + 196*x2), xmask, eviction_policy='evict_last')
    tmp30 = tmp29 - tmp28
    tmp31 = tmp30 * tmp25
    tmp32 = tmp28 + tmp31
    tmp33 = tmp27 - tmp32
    tmp34 = tmp6.to(tl.float32)
    tmp35 = tmp5 - tmp34
    tmp36 = triton_helpers.maximum(tmp35, tmp4)
    tmp37 = triton_helpers.minimum(tmp36, tmp24)
    tmp38 = tmp33 * tmp37
    tmp39 = tmp32 + tmp38
    tl.store(in_out_ptr0 + (x4), tmp39, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_constant_pad_nd_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 32) % 32)
    x0 = (xindex % 32)
    x2 = xindex // 1024
    x4 = xindex
    tmp0 = (-2) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 28, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + x0
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-58) + x0 + 28*x1 + 784*x2), tmp10 & xmask, other=3.0)
    tl.store(out_ptr0 + (x4), tmp11, xmask)


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
        buf1 = empty_strided_cuda((1, s0, 2), (2*s0, 2, 1), torch.float32)

        triton_poi_fused_rand_0_xnumel = 2*s0
        get_raw_stream(0)
        triton_poi_fused_rand_0[grid(triton_poi_fused_rand_0_xnumel)](buf0, buf1, 0, 6, XBLOCK=8, num_warps=1, num_stages=1)
        del buf0
        buf2 = empty_strided_cuda((1, s0, 14, 14), (196*s0, 196, 14, 1), torch.float32)

        triton_poi_fused_fractional_max_pool2d_1_xnumel = 196*s0
        get_raw_stream(0)
        triton_poi_fused_fractional_max_pool2d_1[grid(triton_poi_fused_fractional_max_pool2d_1_xnumel)](buf1, arg3_1, buf2, 28, 28, 588, XBLOCK=128, num_warps=4, num_stages=1)
        del arg3_1
        del buf1
        buf3 = empty_strided_cuda((1, s0, 28, 28), (784*s0, 784, 28, 1), torch.float32)
        buf4 = buf3; del buf3

        triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_2_xnumel = 784*s0
        get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_2[grid(triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_2_xnumel)](buf4, buf2, 2352, XBLOCK=256, num_warps=4, num_stages=1)
        del buf2
        buf5 = empty_strided_cuda((1, s0, 32, 32), (1024*s0, 1024, 32, 1), torch.float32)

        triton_poi_fused_constant_pad_nd_3_xnumel = 1024*s0
        get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_3[grid(triton_poi_fused_constant_pad_nd_3_xnumel)](buf4, buf5, 3072, XBLOCK=256, num_warps=4, num_stages=1)
        del buf4
    return (reinterpret_tensor(buf5, (1, 1024, s0), (1024*s0, 1, 1024), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 3
    arg1_1 = 28
    arg2_1 = 28
    arg3_1 = rand_strided((1, 3, 28, 28), (2352, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
