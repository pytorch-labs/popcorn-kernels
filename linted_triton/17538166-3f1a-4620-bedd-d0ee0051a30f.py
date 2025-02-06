
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
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_hardtanh_log_sigmoid_forward_mean_mul_sub_var_view_0(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 10
    R0_BLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_mask = r0_index < r0_numel
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, other=0.0)
    tmp11 = tl.load(in_ptr1 + (r0_0), r0_mask, other=0.0)
    tmp13 = tl.load(in_ptr2 + (r0_0), r0_mask, other=0.0)
    tmp1 = 0.0
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp3 = 6.0
    tmp4 = triton_helpers.minimum(tmp2, tmp3)
    tmp5 = triton_helpers.minimum(tmp1, tmp4)
    tmp6 = tl_math.abs(tmp4)
    tmp7 = -tmp6
    tmp8 = tl_math.exp(tmp7)
    tmp9 = libdevice.log1p(tmp8)
    tmp10 = tmp5 - tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(r0_mask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp20 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
    tmp22 = tl.where(r0_mask, tmp20, 0)
    tmp23 = tl.sum(tmp22, 1)[:, None]
    tmp24 = tl.full([XBLOCK, 1], 10, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp15 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, R0_BLOCK])
    tmp31 = tl.where(r0_mask, tmp29, 0)
    tmp32 = tl.sum(tmp31, 1)[:, None]
    tmp33 = 10.0
    tmp34 = tmp18 / tmp33
    tmp35 = tmp14 - tmp34
    tmp36 = 9.0
    tmp37 = tmp32 / tmp36
    tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp35, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp34, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp37, None)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_randn_like_1(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tl.randn(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp2, None)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (1, 10), (10, 1))
    assert_size_stride(primals_2, (10, ), (1, ))
    assert_size_stride(primals_3, (10, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        buf2 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        buf7 = empty_strided_cuda((1, 10), (10, 1), torch.float32)
        buf8 = buf0; del buf0
        buf9 = buf2; del buf2

        get_raw_stream(0)
        triton_per_fused_add_hardtanh_log_sigmoid_forward_mean_mul_sub_var_view_0[grid(1)](buf8, buf9, primals_1, primals_2, primals_3, buf7, 1, 10, XBLOCK=1, num_warps=2, num_stages=1)
        del primals_2
        del primals_3
        buf4 = empty_strided_cuda((1, ), (1, ), torch.int64)

        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf4)
        buf5 = empty_strided_cuda((1, 1), (1, 1), torch.float32)

        get_raw_stream(0)
        triton_poi_fused_randn_like_1[grid(1)](buf4, buf5, 0, 1, XBLOCK=1, num_warps=1, num_stages=1)
        del buf4
    return (buf8, buf5, buf9, primals_1, buf7, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([primals_1, primals_2, primals_3])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
