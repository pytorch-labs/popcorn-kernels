
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
def triton_poi_fused_hardtanh_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = -1.0
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp3 = 1.0
    tmp4 = triton_helpers.minimum(tmp2, tmp3)
    tl.store(out_ptr0 + (x0), tmp4, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_relu_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_relu_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_add_addmm_bernoulli_mul_3(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp5 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.5
    tmp4 = tmp2 < tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4.to(tl.float32)
    tmp9 = 0.8864048946659319
    tmp10 = tmp8 * tmp9
    tmp11 = tmp7 * tmp10
    tmp12 = -1.0
    tmp13 = tmp8 + tmp12
    tmp14 = 1.558387861036063
    tmp15 = tmp13 * tmp14
    tmp16 = 0.7791939305180315
    tmp17 = tmp15 + tmp16
    tmp18 = tmp11 + tmp17
    tl.store(out_ptr1 + (x0), tmp4, xmask)
    tl.store(in_out_ptr0 + (x0), tmp18, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7 = args
    args.clear()
    assert_size_stride(primals_1, (1, 128), (128, 1))
    assert_size_stride(primals_2, (64, 128), (128, 1))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_4, (32, 64), (64, 1))
    assert_size_stride(primals_5, (32, ), (1, ))
    assert_size_stride(primals_6, (16, 32), (32, 1))
    assert_size_stride(primals_7, (16, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 128), (128, 1), torch.float32)

        get_raw_stream(0)
        triton_poi_fused_hardtanh_0[grid(128)](primals_1, buf0, 128, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_1
        buf1 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf0, reinterpret_tensor(primals_2, (128, 64), (1, 128), 0), out=buf1)
        del primals_2
        buf2 = buf1; del buf1

        get_raw_stream(0)
        triton_poi_fused_addmm_relu_1[grid(64)](buf2, primals_3, 64, XBLOCK=64, num_warps=1, num_stages=1)
        del primals_3
        buf3 = empty_strided_cuda((1, 32), (32, 1), torch.float32)

        extern_kernels.mm(buf2, reinterpret_tensor(primals_4, (64, 32), (1, 64), 0), out=buf3)
        buf4 = buf3; del buf3

        get_raw_stream(0)
        triton_poi_fused_addmm_relu_2[grid(32)](buf4, primals_5, 32, XBLOCK=32, num_warps=1, num_stages=1)
        del primals_5
        buf5 = empty_strided_cuda((1, 16), (16, 1), torch.float32)

        extern_kernels.mm(buf4, reinterpret_tensor(primals_6, (32, 16), (1, 32), 0), out=buf5)
        buf6 = empty_strided_cuda((1, ), (1, ), torch.int64)

        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf6)
        buf8 = empty_strided_cuda((1, 16), (16, 1), torch.bool)
        buf9 = buf5; del buf5

        get_raw_stream(0)
        triton_poi_fused__to_copy_add_addmm_bernoulli_mul_3[grid(16)](buf9, buf6, primals_7, buf8, 0, 16, XBLOCK=16, num_warps=1, num_stages=1)
        del buf6
        del primals_7
    return (buf9, buf0, buf2, buf4, buf8, primals_6, primals_4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((16, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
