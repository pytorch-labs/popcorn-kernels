
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
def triton_poi_fused_reflection_pad2d_0(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // ks0
    x0 = (xindex % ks0)
    x2 = xindex
    tmp0 = (-1) + (tl.where(1 + ks1 + ((-1)*tl_math.abs(1 + ks1 + ((-1)*tl_math.abs((-2) + x1)))) < 0, 3 + ((-1)*tl_math.abs(1 + ks1 + ((-1)*tl_math.abs((-2) + x1)))) + 2*ks1, 1 + ks1 + ((-1)*tl_math.abs(1 + ks1 + ((-1)*tl_math.abs((-2) + x1))))))
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = ks1
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + ((((tl.where(3 + ((-1)*tl_math.abs(3 + ((-1)*tl_math.abs((-2) + x0)) + 2*ks2 + 2*ks3 + ks2*ks3)) + 2*ks2 + 2*ks3 + ks2*ks3 < 0, 7 + ((-1)*tl_math.abs(3 + ((-1)*tl_math.abs((-2) + x0)) + 2*ks2 + 2*ks3 + ks2*ks3)) + 4*ks2 + 4*ks3 + 2*ks2*ks3, 3 + ((-1)*tl_math.abs(3 + ((-1)*tl_math.abs((-2) + x0)) + 2*ks2 + 2*ks3 + ks2*ks3)) + 2*ks2 + 2*ks3 + ks2*ks3)) // (2 + ks3)) % (2 + ks2)))
    tmp6 = tmp5 >= tmp1
    tmp7 = ks2
    tmp8 = tmp5 < tmp7
    tmp9 = (-1) + (((tl.where(3 + ((-1)*tl_math.abs(3 + ((-1)*tl_math.abs((-2) + x0)) + 2*ks2 + 2*ks3 + ks2*ks3)) + 2*ks2 + 2*ks3 + ks2*ks3 < 0, 7 + ((-1)*tl_math.abs(3 + ((-1)*tl_math.abs((-2) + x0)) + 2*ks2 + 2*ks3 + ks2*ks3)) + 4*ks2 + 4*ks3 + 2*ks2*ks3, 3 + ((-1)*tl_math.abs(3 + ((-1)*tl_math.abs((-2) + x0)) + 2*ks2 + 2*ks3 + ks2*ks3)) + 2*ks2 + 2*ks3 + ks2*ks3)) % (2 + ks3)))
    tmp10 = tmp9 >= tmp1
    tmp11 = ks3
    tmp12 = tmp9 < tmp11
    tmp13 = tmp2 & tmp4
    tmp14 = tmp13 & tmp6
    tmp15 = tmp14 & tmp8
    tmp16 = tmp15 & tmp10
    tmp17 = tmp16 & tmp12
    tmp18 = tl.load(in_ptr0 + ((-1) + ((-1)*ks3) + ks3*((((tl.where(3 + ((-1)*tl_math.abs(3 + ((-1)*tl_math.abs((-2) + x0)) + 2*ks2 + 2*ks3 + ks2*ks3)) + 2*ks2 + 2*ks3 + ks2*ks3 < 0, 7 + ((-1)*tl_math.abs(3 + ((-1)*tl_math.abs((-2) + x0)) + 2*ks2 + 2*ks3 + ks2*ks3)) + 4*ks2 + 4*ks3 + 2*ks2*ks3, 3 + ((-1)*tl_math.abs(3 + ((-1)*tl_math.abs((-2) + x0)) + 2*ks2 + 2*ks3 + ks2*ks3)) + 2*ks2 + 2*ks3 + ks2*ks3)) // (2 + ks3)) % (2 + ks2))) + ((-1)*ks2*ks3) + ks2*ks3*(tl.where(1 + ks1 + ((-1)*tl_math.abs(1 + ks1 + ((-1)*tl_math.abs((-2) + x1)))) < 0, 3 + ((-1)*tl_math.abs(1 + ks1 + ((-1)*tl_math.abs((-2) + x1)))) + 2*ks1, 1 + ks1 + ((-1)*tl_math.abs(1 + ks1 + ((-1)*tl_math.abs((-2) + x1)))))) + (((tl.where(3 + ((-1)*tl_math.abs(3 + ((-1)*tl_math.abs((-2) + x0)) + 2*ks2 + 2*ks3 + ks2*ks3)) + 2*ks2 + 2*ks3 + ks2*ks3 < 0, 7 + ((-1)*tl_math.abs(3 + ((-1)*tl_math.abs((-2) + x0)) + 2*ks2 + 2*ks3 + ks2*ks3)) + 4*ks2 + 4*ks3 + 2*ks2*ks3, 3 + ((-1)*tl_math.abs(3 + ((-1)*tl_math.abs((-2) + x0)) + 2*ks2 + 2*ks3 + ks2*ks3)) + 2*ks2 + 2*ks3 + ks2*ks3)) % (2 + ks3)))), tmp17 & xmask, eviction_policy='evict_last', other=1.0)
    tl.store(out_ptr0 + (x2), tmp18, xmask)


def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, 1, s0, s1, s2), (s0*s1*s2, s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        8 + 2*s1 + 2*s2 + s1*s2
        buf0 = empty_strided_cuda((1, 1, 6 + s0, 8 + 2*s1 + 2*s2 + s1*s2), (48 + 8*s0 + 12*s1 + 12*s2 + 2*s0*s1 + 2*s0*s2 + 6*s1*s2 + s0*s1*s2, 48 + 8*s0 + 12*s1 + 12*s2 + 2*s0*s1 + 2*s0*s2 + 6*s1*s2 + s0*s1*s2, 8 + 2*s1 + 2*s2 + s1*s2, 1), torch.float32)

        triton_poi_fused_reflection_pad2d_0_xnumel = 48 + 8*s0 + 12*s1 + 12*s2 + 2*s0*s1 + 2*s0*s2 + 6*s1*s2 + s0*s1*s2
        get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_0[grid(triton_poi_fused_reflection_pad2d_0_xnumel)](arg3_1, buf0, 148, 10, 10, 10, 2368, XBLOCK=256, num_warps=4, num_stages=1)
        del arg3_1
    return (buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 10
    arg1_1 = 10
    arg2_1 = 10
    arg3_1 = rand_strided((1, 1, 10, 10, 10), (1000, 1000, 100, 10, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
