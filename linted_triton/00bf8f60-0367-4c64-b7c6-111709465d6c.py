
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
def triton_per_fused__native_batch_norm_legit_functional_squeeze_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, out_ptr3, out_ptr5, out_ptr7, xnumel, r0_numel):
    XBLOCK: tl.constexpr = 1
    R0_BLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    tl.full([R0_BLOCK], True, tl.int1)
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 1024*x0), None)
    tmp21 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [R0_BLOCK])
    tmp3 = tl.broadcast_to(tmp1, [R0_BLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tl.full([1], 1024, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 / tmp7
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [R0_BLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = tmp0 - tmp8
    tmp15 = 1024.0
    tmp16 = tmp13 / tmp15
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp14 * tmp19
    tmp22 = tmp20 * tmp21
    tmp24 = tmp22 + tmp23
    tmp25 = 1.0009775171065494
    tmp26 = tmp16 * tmp25
    tmp27 = 0.1
    tmp28 = tmp26 * tmp27
    tmp30 = 0.9
    tmp31 = tmp29 * tmp30
    tmp32 = tmp28 + tmp31
    tmp33 = tmp8 * tmp27
    tmp35 = tmp34 * tmp30
    tmp36 = tmp33 + tmp35
    tl.store(out_ptr2 + (r0_1 + 1024*x0), tmp24, None)
    tl.store(out_ptr3 + (x0), tmp19, None)
    tl.store(out_ptr5 + (x0), tmp32, None)
    tl.store(out_ptr7 + (x0), tmp36, None)
    tl.store(out_ptr0 + (x0), tmp8, None)


import triton

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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6 = args
    args.clear()
    assert_size_stride(primals_1, (1, 3, 32, 32), (3072, 1024, 32, 1))
    assert_size_stride(primals_2, (), ())
    assert_size_stride(primals_3, (3, ), (1, ))
    assert_size_stride(primals_4, (3, ), (1, ))
    assert_size_stride(primals_5, (3, ), (1, ))
    assert_size_stride(primals_6, (3, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 3, 1, 1), (3, 1, 3, 3), torch.float32)
        buf4 = empty_strided_cuda((1, 3, 32, 32), (3072, 1024, 32, 1), torch.float32)
        buf3 = empty_strided_cuda((1, 3, 1, 1), (3, 1, 3, 3), torch.float32)

        get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_squeeze_0[grid(3)](primals_1, primals_5, primals_6, primals_4, primals_3, buf0, buf4, buf3, primals_4, primals_3, 3, 1024, num_warps=8, num_stages=1)
        del primals_3
        del primals_4
        del primals_5
        del primals_6

        get_raw_stream(0)
        triton_poi_fused_add_1[grid(1)](primals_2, primals_2, 1, XBLOCK=1, num_warps=1, num_stages=1)
        del primals_2
    return (buf4, primals_1, reinterpret_tensor(buf3, (3, ), (1, ), 0), reinterpret_tensor(buf0, (1, 3, 1, 1), (3, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 3, 32, 32), (3072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_3 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
