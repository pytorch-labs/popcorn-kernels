
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
def triton_poi_fused_elu_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = ((xindex // 16) % 36)
    x2 = xindex // 576
    x4 = xindex
    tmp0 = (-1) + ((((12*x1 + 432*x2 + ((11) * ((11) <= (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0))))) + (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) * ((((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) < (11)))) // 144) % 12))
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 10, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + ((((12*x1 + ((11) * ((11) <= (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0))))) + (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) * ((((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) < (11)))) // 12) % 12))
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = (-1) + ((11) * ((11) <= (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0))))) + (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) * ((((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) < (11)))
    tmp9 = tmp8 >= tmp1
    tmp10 = tmp8 < tmp3
    tmp11 = tmp2 & tmp4
    tmp12 = tmp11 & tmp6
    tmp13 = tmp12 & tmp7
    tmp14 = tmp13 & tmp9
    tmp15 = tmp14 & tmp10
    tmp16 = tl.load(in_ptr0 + ((-111) + 10*((((12*x1 + ((11) * ((11) <= (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0))))) + (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) * ((((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) < (11)))) // 12) % 12)) + 100*((((12*x1 + 432*x2 + ((11) * ((11) <= (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0))))) + (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) * ((((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) < (11)))) // 144) % 12)) + 1000*((12*x1 + 432*x2 + ((11) * ((11) <= (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0))))) + (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) * ((((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) < (11)))) // 1728) + ((11) * ((11) <= (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0))))) + (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) * ((((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) < (11)))), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = 0.0
    tmp18 = tmp16 > tmp17
    tmp19 = 1.0507009873554805
    tmp20 = tmp16 * tmp19
    tmp21 = 1.0
    tmp22 = tmp16 * tmp21
    tmp23 = libdevice.expm1(tmp22)
    tmp24 = 1.7580993408473766
    tmp25 = tmp23 * tmp24
    tmp26 = tl.where(tmp18, tmp20, tmp25)
    tl.store(out_ptr0 + (x4), tmp26, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (1, 3, 10, 10, 10), (3000, 1000, 100, 10, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 12, 12, 3, 16), (6912, 576, 48, 16, 1), torch.float32)

        get_raw_stream(0)
        triton_poi_fused_elu_0[grid(6912)](arg0_1, buf0, 6912, XBLOCK=256, num_warps=4, num_stages=1)
        del arg0_1
    return (buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 3, 10, 10, 10), (3000, 1000, 100, 10, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
