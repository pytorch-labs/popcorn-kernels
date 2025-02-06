
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
    xnumel = 3
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
def triton_poi_fused_tanh_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = libdevice.tanh(tmp0)
    tl.store(out_ptr0 + (x0), tmp1, None)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (1, 16, 16, 16), (4096, 256, 16, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, ), (1, ), torch.int64)

        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf0)
        buf1 = empty_strided_cuda((1, 1, 3), (3, 3, 1), torch.float32)

        get_raw_stream(0)
        triton_poi_fused_rand_0[grid(3)](buf0, buf1, 0, 3, XBLOCK=4, num_warps=1, num_stages=1)
        del buf0
        buf2 = empty_strided_cuda((1, 16, 16, 16), (4096, 256, 16, 1), torch.float32)

        get_raw_stream(0)
        triton_poi_fused_tanh_1[grid(4096)](arg0_1, buf2, 4096, XBLOCK=128, num_warps=4, num_stages=1)
        del arg0_1

        buf3 = torch.ops.aten.fractional_max_pool3d.default(reinterpret_tensor(buf2, (1, 1, 16, 16, 16), (0, 0, 256, 16, 1), 0), [2, 2, 2], [8, 8, 8], buf1)
        del buf1
        del buf2
        buf4 = buf3[0]
        del buf3
    return (reinterpret_tensor(buf4, (8, 8, 8), (64, 8, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 16, 16, 16), (4096, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
