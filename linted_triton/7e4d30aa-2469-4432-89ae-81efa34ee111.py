
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
def triton_poi_fused_rand_0(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9
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
def triton_poi_fused_max_unpool2d_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
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
def triton_poi_fused_max_unpool2d_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 768
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
    tmp7 = tl.full([1], 2, tl.int32)
    tmp8 = tl.where((tmp5 < 0) != (tmp7 < 0), tl.where(tmp5 % tmp7 != 0, tmp5 // tmp7 - 1, tmp5 // tmp7), tmp5 // tmp7)
    tmp9 = tmp8 * tmp7
    tmp10 = tmp5 - tmp9
    tmp11 = tl.full([1], 0, tl.int64)
    tmp12 = tmp11 + tmp8
    tmp13 = 2*((x0 % 4))
    tmp14 = tmp13 + tmp10
    tmp15 = tl.full([1], 8, tl.int64)
    tmp16 = tmp12 * tmp15
    tmp17 = tmp16 + tmp14
    tmp18 = 8*(x0 // 4)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.full([XBLOCK], 1536, tl.int32)
    tmp21 = tmp19 + tmp20
    tmp22 = tmp19 < 0
    tmp23 = tl.where(tmp22, tmp21, tmp19)
    tl.device_assert(((0 <= tmp23) & (tmp23 < 1536)) | ~(xmask), "index out of bounds: 0 <= tmp23 < 1536")
    tl.store(out_ptr0 + (tl.broadcast_to(tmp23, [XBLOCK])), tmp6, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_layer_norm_3(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel):
    XBLOCK: tl.constexpr = 1
    R0_BLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    tl.full([R0_BLOCK], True, tl.int1)
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 512*x0), None)
    tmp1 = tl.broadcast_to(tmp0, [R0_BLOCK])
    tmp3 = tl.broadcast_to(tmp1, [R0_BLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tl.full([1], 512, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 / tmp7
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [R0_BLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 512.0
    tmp15 = tmp13 / tmp14
    tmp16 = 1e-05
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp18, None)
    tl.store(out_ptr0 + (x0), tmp8, None)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_clamp_min_fill_mean_native_layer_norm_ne_soft_margin_loss_soft_margin_loss_backward_sub_where_zeros_like_4(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    _tmp13 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp25 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_2 = r0_index
        r0_1 = r0_index // 512
        r0_0 = (r0_index % 512)
        tmp0 = tl.load(in_ptr0 + (r0_2), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr3 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr4 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 - tmp1
        tmp4 = tmp2 * tmp3
        tmp6 = tmp4 * tmp5
        tmp8 = tmp6 + tmp7
        tmp9 = -tmp8
        tmp10 = tl_math.exp(tmp9)
        tmp11 = libdevice.log1p(tmp10)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(r0_mask, tmp14, _tmp13)
        tmp15 = 1.0
        tmp16 = tmp15 - tmp8
        tmp17 = 0.0
        tmp18 = triton_helpers.maximum(tmp16, tmp17)
        tmp19 = tl.full([1, 1], False, tl.int1)
        tmp20 = tl.where(tmp19, tmp18, tmp17)
        tmp21 = tl.full([1, 1], True, tl.int1)
        tmp22 = tl.where(tmp21, tmp8, tmp17)
        tmp23 = tmp20 + tmp22
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, R0_BLOCK])
        tmp26 = _tmp25 + tmp24
        _tmp25 = tl.where(r0_mask, tmp26, _tmp25)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    tmp27 = 1536.0
    tmp28 = tmp13 / tmp27
    tmp29 = tmp25 / tmp27
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp28, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp29, None)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (1, 3, 16, 16, 16), (12288, 4096, 256, 16, 1))
    assert_size_stride(primals_2, (8, 8, 8), (64, 8, 1))
    assert_size_stride(primals_3, (8, 8, 8), (64, 8, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, ), (1, ), torch.int64)

        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf0)
        buf1 = empty_strided_cuda((1, 3, 3), (9, 3, 1), torch.float32)

        get_raw_stream(0)
        triton_poi_fused_rand_0[grid(9)](buf0, buf1, 0, 9, XBLOCK=16, num_warps=1, num_stages=1)
        del buf0

        buf2 = torch.ops.aten.fractional_max_pool3d.default(primals_1, [2, 2, 2], [8, 8, 8], buf1)
        del buf1
        del primals_1
        buf3 = buf2[0]
        del buf2
        buf5 = empty_strided_cuda((1536, ), (1, ), torch.float32)

        get_raw_stream(0)
        triton_poi_fused_max_unpool2d_1[grid(1536)](buf5, 1536, XBLOCK=256, num_warps=4, num_stages=1)

        get_raw_stream(0)
        triton_poi_fused_max_unpool2d_2[grid(768)](buf3, buf5, 768, XBLOCK=128, num_warps=4, num_stages=1)
        del buf3
        buf7 = empty_strided_cuda((1, 3, 1, 1, 1), (3, 1, 1, 1, 1), torch.float32)
        buf8 = empty_strided_cuda((1, 3, 1, 1, 1), (3, 1, 3, 3, 3), torch.float32)
        buf10 = reinterpret_tensor(buf8, (1, 3, 1, 1, 1), (3, 1, 1, 1, 1), 0); del buf8

        get_raw_stream(0)
        triton_per_fused_native_layer_norm_3[grid(3)](buf10, buf5, buf7, 3, 512, num_warps=4, num_stages=1)
        buf12 = empty_strided_cuda((), (), torch.float32)
        buf13 = empty_strided_cuda((), (), torch.float32)
        buf14 = buf12; del buf12
        buf15 = buf13; del buf13

        get_raw_stream(0)
        triton_red_fused_add_clamp_min_fill_mean_native_layer_norm_ne_soft_margin_loss_soft_margin_loss_backward_sub_where_zeros_like_4[grid(1)](buf14, buf15, buf5, buf7, buf10, primals_2, primals_3, 1, 1536, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
    return (buf14, buf15, primals_2, primals_3, buf5, buf7, buf10, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 3, 16, 16, 16), (12288, 4096, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((8, 8, 8), (64, 8, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((8, 8, 8), (64, 8, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([primals_1, primals_2, primals_3])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
