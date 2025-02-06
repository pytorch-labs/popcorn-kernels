
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
def triton_poi_fused_avg_pool2d_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 12)
    x1 = xindex // 12
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 25*x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 25*x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x2), tmp4, xmask)


def call(args):
    arg0_1, _arg1_1, _arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    assert_size_stride(arg3_1, (1, s0, 64, 64), (4096*s0, 4096, 64, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)

        buf0 = torch.ops.aten.adaptive_max_pool2d.default(arg3_1, [5, 5])
        del arg3_1
        buf1 = buf0[0]
        del buf0
        buf3 = empty_strided_cuda((1, s0, 1, 12), (12*s0, 12, 12, 1), torch.float32)

        triton_poi_fused_avg_pool2d_0_xnumel = 12*s0
        get_raw_stream(0)
        triton_poi_fused_avg_pool2d_0[grid(triton_poi_fused_avg_pool2d_0_xnumel)](buf1, buf3, 36, XBLOCK=64, num_warps=1, num_stages=1)
        del buf1
    return (reinterpret_tensor(buf3, (1, s0, 12), (12*s0, 12, 1), 0), )


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
