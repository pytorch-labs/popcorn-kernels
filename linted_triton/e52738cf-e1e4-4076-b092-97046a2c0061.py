
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
def triton_per_fused__native_batch_norm_legit_functional_convolution_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, out_ptr3, out_ptr5, out_ptr7, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 10
    r0_numel = 96
    R0_BLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_mask = r0_index < r0_numel
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 96*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tl.where(r0_mask & xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp8 = tl.where(r0_mask & xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 96, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(r0_mask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp2 - tmp12
    tmp20 = 96.0
    tmp21 = tmp18 / tmp20
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tmp30 = 1.0105263157894737
    tmp31 = tmp21 * tmp30
    tmp32 = 0.1
    tmp33 = tmp31 * tmp32
    tmp35 = 0.9
    tmp36 = tmp34 * tmp35
    tmp37 = tmp33 + tmp36
    tmp38 = tmp12 * tmp32
    tmp40 = tmp39 * tmp35
    tmp41 = tmp38 + tmp40
    tl.store(in_out_ptr0 + (r0_1 + 96*x0), tmp2, r0_mask & xmask)
    tl.store(out_ptr2 + (r0_1 + 96*x0), tmp29, r0_mask & xmask)
    tl.store(out_ptr3 + (x0), tmp24, xmask)
    tl.store(out_ptr5 + (x0), tmp37, xmask)
    tl.store(out_ptr7 + (x0), tmp41, xmask)
    tl.store(out_ptr0 + (x0), tmp12, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_1(in_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp1 + tmp2
    tl.store(out_ptr1 + (tl.full([XBLOCK], 0, tl.int32)), tmp3, None)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8 = args
    args.clear()
    assert_size_stride(primals_1, (10, 1, 5), (5, 5, 1))
    assert_size_stride(primals_2, (10, ), (1, ))
    assert_size_stride(primals_3, (1, 1, 100), (100, 100, 1))
    assert_size_stride(primals_4, (), ())
    assert_size_stride(primals_5, (10, ), (1, ))
    assert_size_stride(primals_6, (10, ), (1, ))
    assert_size_stride(primals_7, (10, ), (1, ))
    assert_size_stride(primals_8, (10, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)

        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf0, (1, 10, 96), (960, 96, 1))
        buf1 = buf0; del buf0
        buf2 = empty_strided_cuda((1, 10, 1), (10, 1, 10), torch.float32)
        buf6 = empty_strided_cuda((1, 10, 96), (960, 96, 1), torch.float32)
        buf5 = empty_strided_cuda((1, 10, 1), (10, 1, 1), torch.float32)

        get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_convolution_0[grid(10)](buf1, primals_2, primals_7, primals_8, primals_6, primals_5, buf2, buf6, buf5, primals_6, primals_5, 10, 96, XBLOCK=8, num_warps=8, num_stages=1)
        del primals_2
        del primals_5
        del primals_6
        del primals_8

        get_raw_stream(0)
        triton_poi_fused_add_1[grid(1)](primals_4, primals_4, 1, XBLOCK=1, num_warps=1, num_stages=1)
        del primals_4
    return (reinterpret_tensor(buf6, (1, 96, 10), (960, 1, 96), 0), primals_1, primals_3, primals_7, buf1, reinterpret_tensor(buf5, (10, ), (1, ), 0), reinterpret_tensor(buf2, (1, 10, 1), (10, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((10, 1, 5), (5, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((1, 1, 100), (100, 100, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_5 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
