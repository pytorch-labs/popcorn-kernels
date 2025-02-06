
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
from torch._inductor.runtime.triton_helpers import libdevice
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_group_norm_0(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 8
    R0_BLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 16*x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tl.where(xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 16, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = 16.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp21, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_group_norm_sub_tanh_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 // 16), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 // 16), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0), xmask)
    tmp7 = tl.load(in_ptr4 + (x0), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.tanh(tmp8)
    tmp10 = tmp8 - tmp9
    tmp11 = libdevice.tanh(tmp10)
    tl.store(in_out_ptr0 + (x0), tmp11, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_group_norm_2(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 4
    R0_BLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 16*x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tl.where(xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 16, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = 16.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp21, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_elu_native_group_norm_sigmoid_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 // 16), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 // 16), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0), xmask)
    tmp7 = tl.load(in_ptr4 + (x0), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = 0.0
    tmp11 = tmp9 > tmp10
    tmp12 = 1.0
    tmp13 = tmp9 * tmp12
    tmp14 = libdevice.expm1(tmp13)
    tmp15 = tmp14 * tmp12
    tmp16 = tl.where(tmp11, tmp13, tmp15)
    tl.store(in_out_ptr0 + (x0), tmp16, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_tanh_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = libdevice.tanh(tmp2)
    tl.store(in_out_ptr0 + (x0), tmp3, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15 = args
    args.clear()
    assert_size_stride(primals_1, (1, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_2, (128, 12288), (12288, 1))
    assert_size_stride(primals_3, (128, ), (1, ))
    assert_size_stride(primals_4, (128, ), (1, ))
    assert_size_stride(primals_5, (128, ), (1, ))
    assert_size_stride(primals_6, (64, 128), (128, 1))
    assert_size_stride(primals_7, (64, ), (1, ))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_10, (32, 64), (64, 1))
    assert_size_stride(primals_11, (32, ), (1, ))
    assert_size_stride(primals_12, (16, 32), (32, 1))
    assert_size_stride(primals_13, (16, ), (1, ))
    assert_size_stride(primals_14, (10, 16), (16, 1))
    assert_size_stride(primals_15, (10, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 128), (128, 1), torch.float32)

        extern_kernels.addmm(primals_3, reinterpret_tensor(primals_1, (1, 12288), (12288, 1), 0), reinterpret_tensor(primals_2, (12288, 128), (1, 12288), 0), alpha=1, beta=1, out=buf0)
        del primals_2
        del primals_3
        buf1 = empty_strided_cuda((1, 8, 1, 1), (8, 1, 1, 1), torch.float32)
        buf2 = empty_strided_cuda((1, 8, 1, 1), (8, 1, 8, 8), torch.float32)
        buf4 = reinterpret_tensor(buf2, (1, 8, 1, 1), (8, 1, 1, 1), 0); del buf2

        get_raw_stream(0)
        triton_per_fused_native_group_norm_0[grid(8)](buf4, buf0, buf1, 8, 16, XBLOCK=1, num_warps=2, num_stages=1)
        buf5 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        buf6 = buf5; del buf5

        get_raw_stream(0)
        triton_poi_fused_native_group_norm_sub_tanh_1[grid(128)](buf6, buf0, buf1, buf4, primals_4, primals_5, 128, XBLOCK=128, num_warps=4, num_stages=1)
        buf7 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.addmm(primals_7, buf6, reinterpret_tensor(primals_6, (128, 64), (1, 128), 0), alpha=1, beta=1, out=buf7)
        del primals_7
        buf8 = empty_strided_cuda((1, 4, 1, 1), (4, 1, 1, 1), torch.float32)
        buf9 = empty_strided_cuda((1, 4, 1, 1), (4, 1, 4, 4), torch.float32)
        buf11 = reinterpret_tensor(buf9, (1, 4, 1, 1), (4, 1, 1, 1), 0); del buf9

        get_raw_stream(0)
        triton_per_fused_native_group_norm_2[grid(4)](buf11, buf7, buf8, 4, 16, XBLOCK=1, num_warps=2, num_stages=1)
        buf12 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
        buf13 = buf12; del buf12

        get_raw_stream(0)
        triton_poi_fused_elu_native_group_norm_sigmoid_3[grid(64)](buf13, buf7, buf8, buf11, primals_8, primals_9, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf14 = empty_strided_cuda((1, 32), (32, 1), torch.float32)

        extern_kernels.addmm(primals_11, buf13, reinterpret_tensor(primals_10, (64, 32), (1, 64), 0), alpha=1, beta=1, out=buf14)
        del primals_11
        buf15 = empty_strided_cuda((1, 16), (16, 1), torch.float32)

        extern_kernels.mm(buf14, reinterpret_tensor(primals_12, (32, 16), (1, 32), 0), out=buf15)
        buf16 = buf15; del buf15

        get_raw_stream(0)
        triton_poi_fused_addmm_tanh_4[grid(16)](buf16, primals_13, 16, XBLOCK=16, num_warps=1, num_stages=1)
        del primals_13
        buf17 = empty_strided_cuda((1, 10), (10, 1), torch.float32)

        extern_kernels.addmm(primals_15, buf16, reinterpret_tensor(primals_14, (16, 10), (1, 16), 0), alpha=1, beta=1, out=buf17)
        del primals_15
    return (buf17, primals_4, primals_5, primals_8, primals_9, reinterpret_tensor(primals_1, (1, 12288), (12288, 1), 0), buf0, buf1, buf4, buf6, buf7, buf8, buf11, buf13, buf14, buf16, primals_14, primals_12, primals_10, primals_6, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((128, 12288), (12288, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((32, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((16, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((10, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
