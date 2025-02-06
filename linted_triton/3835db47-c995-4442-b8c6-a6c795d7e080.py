
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
from torch._inductor.runtime.triton_helpers import math as tl_math
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_relu_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 38440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 3844
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
def triton_poi_fused_convolution_relu_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 72000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 3600
    x0 = (xindex % 3600)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(out_ptr0 + (x0 + 3616*x1), tmp4, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_unpool3d_2(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 72000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_unpool3d_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp8 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([XBLOCK], 72000, tl.int32)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp2 < 0
    tmp6 = tl.where(tmp5, tmp4, tmp2)
    tl.device_assert(((0 <= tmp6) & (tmp6 < 72000)) | ~(xmask), "index out of bounds: 0 <= tmp6 < 72000")
    tl.store(out_ptr0 + (tl.broadcast_to(tmp6, [XBLOCK])), tmp8, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_arange_4(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.full([1], 0, tl.int64)
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp0, None)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 72000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (60*((((x0 // 120) % 120)) // 2) + 3600*((((x0 % 120)) % 2)) + 7200*(((((x0 // 120) % 120)) % 2)) + 14400*(x0 // 14400) + (((x0 % 120)) // 2)), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.int64)
    tl.store(out_ptr0 + (x0), tmp1, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__softmax_neg_6(in_out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 10
    R0_BLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_mask = r0_index < r0_numel
    r0_0 = r0_index
    tmp0 = tl.load(in_out_ptr0 + (r0_0), r0_mask, other=0.0)
    tmp1 = -tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.where(r0_mask, tmp2, float("-inf"))
    tmp5 = triton_helpers.max2(tmp4, 1)[:, None]
    tmp6 = tmp1 - tmp5
    tmp7 = tl_math.exp(tmp6)
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(r0_mask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tmp7 / tmp11
    tl.store(in_out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp12, r0_mask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6 = args
    args.clear()
    assert_size_stride(primals_1, (1, 1, 64, 64), (4096, 4096, 64, 1))
    assert_size_stride(primals_2, (10, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_3, (10, ), (1, ))
    assert_size_stride(primals_4, (20, 10, 3, 3), (90, 9, 3, 1))
    assert_size_stride(primals_5, (20, ), (1, ))
    assert_size_stride(primals_6, (100, 10), (10, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)

        buf0 = extern_kernels.convolution(primals_1, primals_2, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (1, 10, 62, 62), (38440, 3844, 62, 1))
        del primals_1
        del primals_2
        buf1 = buf0; del buf0

        get_raw_stream(0)
        triton_poi_fused_convolution_relu_0[grid(38440)](buf1, primals_3, 38440, XBLOCK=512, num_warps=4, num_stages=1)
        del primals_3

        buf2 = extern_kernels.convolution(buf1, primals_4, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (1, 20, 60, 60), (72000, 3600, 60, 1))
        del buf1
        del primals_4
        buf3 = empty_strided_cuda((1, 20, 60, 60), (72320, 3616, 60, 1), torch.float32)

        get_raw_stream(0)
        triton_poi_fused_convolution_relu_1[grid(72000)](buf2, primals_5, buf3, 72000, XBLOCK=512, num_warps=8, num_stages=1)
        del primals_5

        buf4 = torch.ops.aten.max_pool3d_with_indices.default(reinterpret_tensor(buf3, (1, 1, 20, 60, 60), (0, 0, 3616, 60, 1), 0), [2, 2, 2], [2, 2, 2])
        del buf3
        buf5 = buf4[0]
        buf6 = buf4[1]
        del buf4
        buf7 = reinterpret_tensor(buf2, (72000, ), (1, ), 0); del buf2

        get_raw_stream(0)
        triton_poi_fused_max_unpool3d_2[grid(72000)](buf7, 72000, XBLOCK=1024, num_warps=4, num_stages=1)

        get_raw_stream(0)
        triton_poi_fused_max_unpool3d_3[grid(9000)](buf6, buf5, buf7, 9000, XBLOCK=256, num_warps=4, num_stages=1)
        del buf5
        del buf6
        buf9 = empty_strided_cuda((1, ), (1, ), torch.int64)

        get_raw_stream(0)
        triton_poi_fused_arange_4[grid(1)](buf9, 1, XBLOCK=1, num_warps=1, num_stages=1)
        buf10 = empty_strided_cuda((1, 72000), (72000, 1), torch.int64)

        get_raw_stream(0)
        triton_poi_fused__to_copy_5[grid(72000)](buf7, buf10, 72000, XBLOCK=1024, num_warps=4, num_stages=1)
        del buf7

        buf11 = torch.ops.aten._embedding_bag.default(primals_6, reinterpret_tensor(buf10, (72000, ), (1, ), 0), buf9, False, 1)
        del primals_6
        buf12 = buf11[0]
        buf13 = buf11[1]
        buf14 = buf11[2]
        buf15 = buf11[3]
        del buf11
        buf18 = buf12; del buf12

        get_raw_stream(0)
        triton_per_fused__softmax_neg_6[grid(1)](buf18, 1, 10, XBLOCK=1, num_warps=2, num_stages=1)
    return (buf18, buf9, reinterpret_tensor(buf10, (72000, ), (1, ), 0), buf13, buf14, buf15, buf18, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 1, 64, 64), (4096, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((10, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((20, 10, 3, 3), (90, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((100, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
