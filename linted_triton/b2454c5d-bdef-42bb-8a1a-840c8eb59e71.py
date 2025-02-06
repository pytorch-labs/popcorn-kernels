
import torch
from torch._inductor.select_algorithm import extern_kernels
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
def triton_poi_fused_convolution_relu_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10150
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1015
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_relu_threshold_backward_1(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 106340
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 5317
    x0 = (xindex % 5317)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(out_ptr0 + (x0 + 5344*x1), tmp4, xmask)
    tl.store(out_ptr1 + (x0 + 5376*x1), tmp6, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_unsqueeze_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 106340
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (5344*(x0 // 5317) + ((x0 % 5317))), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_3(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 53170
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 > tmp0
    tmp3 = tl.full([1], 1, tl.int8)
    tmp4 = tl.full([1], 0, tl.int8)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(out_ptr0 + (x0), tmp5, xmask)
    tl.store(out_ptr1 + (x0), tmp6, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_randn_like_4(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 53170
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.randn(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (x0), tmp2, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_ones_like_5(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 53170
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 1.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (1, 100), (100, 1))
    assert_size_stride(primals_2, (1, 10, 5, 5), (250, 25, 5, 1))
    assert_size_stride(primals_3, (10, ), (1, ))
    assert_size_stride(primals_4, (10, 20, 5, 5), (500, 25, 5, 1))
    assert_size_stride(primals_5, (20, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)

        buf0 = extern_kernels.convolution(reinterpret_tensor(primals_1, (1, 1, 100, 1), (100, 100, 1, 1), 0), primals_2, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (1, 10, 203, 5), (10150, 1015, 5, 1))
        buf1 = buf0; del buf0

        get_raw_stream(0)
        triton_poi_fused_convolution_relu_0[grid(10150)](buf1, primals_3, 10150, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_3

        buf2 = extern_kernels.convolution(buf1, primals_4, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (1, 20, 409, 13), (106340, 5317, 13, 1))
        buf3 = empty_strided_cuda((1, 20, 409, 13), (106880, 5344, 13, 1), torch.float32)
        buf10 = empty_strided_cuda((1, 20, 409, 13), (107520, 5376, 13, 1), torch.bool)

        get_raw_stream(0)
        triton_poi_fused_convolution_relu_threshold_backward_1[grid(106340)](buf2, primals_5, buf3, buf10, 106340, XBLOCK=512, num_warps=8, num_stages=1)
        del primals_5
        buf4 = reinterpret_tensor(buf2, (1, 1, 1, 106340), (106368, 106368, 106368, 1), 0); del buf2

        get_raw_stream(0)
        triton_poi_fused_unsqueeze_2[grid(106340)](buf3, buf4, 106340, XBLOCK=1024, num_warps=4, num_stages=1)
        del buf3
        buf5 = empty_strided_cuda((1, 1, 1, 53170), (53248, 53248, 53248, 1), torch.int8)
        buf6 = empty_strided_cuda((1, 1, 1, 53170), (53170, 1, 53170, 1), torch.float32)

        get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_3[grid(53170)](buf4, buf5, buf6, 53170, XBLOCK=512, num_warps=4, num_stages=1)
        buf7 = empty_strided_cuda((1, ), (1, ), torch.int64)

        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf7)
        buf8 = empty_strided_cuda((1, 53170), (53170, 1), torch.float32)

        get_raw_stream(0)
        triton_poi_fused_randn_like_4[grid(53170)](buf7, buf8, 0, 53170, XBLOCK=256, num_warps=4, num_stages=1)
        del buf7
        buf9 = empty_strided_cuda((1, 53170), (53170, 1), torch.float32)

        get_raw_stream(0)
        triton_poi_fused_ones_like_5[grid(53170)](buf9, 53170, XBLOCK=512, num_warps=4, num_stages=1)
    return (reinterpret_tensor(buf6, (1, 53170), (53170, 1), 0), buf8, buf9, primals_2, primals_4, reinterpret_tensor(primals_1, (1, 1, 100, 1), (100, 100, 1, 1), 0), buf1, buf4, buf5, buf10, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 100), (100, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((1, 10, 5, 5), (250, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((10, 20, 5, 5), (500, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([primals_1, primals_2, primals_3, primals_4, primals_5])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
