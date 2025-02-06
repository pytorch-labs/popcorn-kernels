
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
def triton_poi_fused_abs_add_constant_pad_nd_div_le_scalar_tensor_where_0(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // ks0) % ks1)
    x0 = (xindex % ks0)
    x2 = xindex // ks4
    x4 = xindex
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
    tmp12 = tl.load(in_ptr0 + ((-2) + x0 + ((-2)*ks3) + ks3*x1 + ks2*ks3*x2), tmp11 & xmask, eviction_policy='evict_last', other=3.0)
    tmp13 = tl_math.abs(tmp12)
    tmp14 = 1.0
    tmp15 = tmp13 + tmp14
    tmp16 = tmp12 / tmp15
    tmp17 = tl_math.abs(tmp16)
    tmp18 = 0.5
    tmp19 = tmp17 <= tmp18
    tmp20 = 0.0
    tmp21 = tl.where(tmp19, tmp20, tmp16)
    tl.store(out_ptr0 + (x4), tmp21, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__adaptive_avg_pool2d_1(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
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
        4 + s2
        4 + s1
        16 + 4*s1 + 4*s2 + s1*s2
        buf0 = empty_strided_cuda((1, s0, 4 + s1, 4 + s2), (16*s0 + 4*s0*s1 + 4*s0*s2 + s0*s1*s2, 16 + 4*s1 + 4*s2 + s1*s2, 4 + s2, 1), torch.float32)

        triton_poi_fused_abs_add_constant_pad_nd_div_le_scalar_tensor_where_0_xnumel = 16*s0 + 4*s0*s1 + 4*s0*s2 + s0*s1*s2
        get_raw_stream(0)
        triton_poi_fused_abs_add_constant_pad_nd_div_le_scalar_tensor_where_0[grid(triton_poi_fused_abs_add_constant_pad_nd_div_le_scalar_tensor_where_0_xnumel)](arg3_1, buf0, 32, 32, 28, 28, 1024, 3072, XBLOCK=256, num_warps=4, num_stages=1)
        del arg3_1
        16 + 4*s1 + 4*s2 + s1*s2
        buf1 = empty_strided_cuda((1, s0, 1, 16 + 4*s1 + 4*s2 + s1*s2), (16*s0 + 4*s0*s1 + 4*s0*s2 + s0*s1*s2, 16 + 4*s1 + 4*s2 + s1*s2, 16*s0 + 4*s0*s1 + 4*s0*s2 + s0*s1*s2, 1), torch.float32)

        triton_poi_fused__adaptive_avg_pool2d_1_xnumel = 16*s0 + 4*s0*s1 + 4*s0*s2 + s0*s1*s2
        get_raw_stream(0)
        triton_poi_fused__adaptive_avg_pool2d_1[grid(triton_poi_fused__adaptive_avg_pool2d_1_xnumel)](buf0, buf1, 1024, 32, 28, 28, 3072, XBLOCK=128, num_warps=4, num_stages=1)
        del buf0

        buf2 = torch.ops.aten._adaptive_avg_pool2d.default(buf1, [1, 10])
        del buf1
        buf3 = buf2
        del buf2
    return (reinterpret_tensor(buf3, (1, s0, 10), (10*s0, 10, 1), 0), )


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
