
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
def triton_poi_fused_constant_pad_nd_glu_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 48
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 24)
    x1 = xindex // 24
    x2 = xindex
    tmp0 = (-2) + x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 20, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + ((-2) + x0 + 20*x1), tmp5 & xmask, other=0.0)
    tmp7 = tl.load(in_ptr0 + (38 + x0 + 20*x1), tmp5 & xmask, other=0.0)
    tmp8 = tl.sigmoid(tmp7)
    tmp9 = tmp6 * tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp5, tmp9, tmp10)
    tl.store(out_ptr0 + (x2), tmp11, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__adaptive_avg_pool2d_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 20
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 10)
    x1 = xindex // 10
    x2 = xindex
    tmp0 = tl.full([1], 0, tl.int64)
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = (12*x0) // 5
    tmp4 = (33 + 24*x0) // 10
    tmp5 = tmp3 < tmp4
    tmp6 = tmp2 & tmp5
    tmp7 = tl.load(in_ptr0 + (24*x1 + ((12*x0) // 5)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = 1 + ((12*x0) // 5)
    tmp9 = tmp8 < tmp4
    tmp10 = tmp2 & tmp9
    tmp11 = tl.load(in_ptr0 + (1 + 24*x1 + ((12*x0) // 5)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 + tmp7
    tmp13 = 2 + ((12*x0) // 5)
    tmp14 = tmp13 < tmp4
    tmp15 = tmp2 & tmp14
    tmp16 = tl.load(in_ptr0 + (2 + 24*x1 + ((12*x0) // 5)), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 + tmp12
    tmp18 = 3 + ((12*x0) // 5)
    tmp19 = tmp18 < tmp4
    tmp20 = tmp2 & tmp19
    tmp21 = tl.load(in_ptr0 + (3 + 24*x1 + ((12*x0) // 5)), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp21 + tmp17
    tmp23 = 1.0
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp6, tmp23, tmp24)
    tmp26 = 1.0
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp10, tmp26, tmp27)
    tmp29 = tmp28 + tmp25
    tmp30 = 1.0
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp15, tmp30, tmp31)
    tmp33 = tmp32 + tmp29
    tmp34 = 1.0
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp20, tmp34, tmp35)
    tmp37 = tmp36 + tmp33
    tmp38 = tmp22 / tmp37
    tl.store(out_ptr0 + (x2), tmp38, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_norm_roll_sub_2(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 2
    r0_numel = 10
    R0_BLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_mask = r0_index < r0_numel
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 10*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (10*x0 + (((9 + r0_1) % 10))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr0 + (10*x0 + (((8 + r0_1) % 10))), r0_mask & xmask, other=0.0)
    tmp2 = tmp0 - tmp1
    tmp3 = 1e-06
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp8 = tl.where(r0_mask & xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp11 = tmp0 - tmp10
    tmp12 = tmp11 + tmp3
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.where(r0_mask & xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp9, xmask)
    tl.store(out_ptr1 + (x0), tmp17, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_clamp_min_mean_norm_sub_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp5 = tl.load(in_ptr1 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp11 = tl.load(in_ptr0 + (1))
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK])
    tmp15 = tl.load(in_ptr1 + (1))
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp2 = libdevice.sqrt(tmp1)
    tmp3 = 1.0
    tmp4 = tmp2 + tmp3
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tmp4 - tmp7
    tmp9 = 0.0
    tmp10 = triton_helpers.maximum(tmp8, tmp9)
    tmp13 = libdevice.sqrt(tmp12)
    tmp14 = tmp13 + tmp3
    tmp17 = libdevice.sqrt(tmp16)
    tmp18 = tmp14 - tmp17
    tmp19 = triton_helpers.maximum(tmp18, tmp9)
    tmp20 = tmp10 + tmp19
    tmp21 = 2.0
    tmp22 = tmp20 / tmp21
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp22, None)


def call(args):
    _arg0_1, _arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg2_1, (1, 4, 20), (80, 20, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 2, 24), (48, 24, 1), torch.float32)

        get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_glu_0[grid(48)](arg2_1, buf0, 48, XBLOCK=64, num_warps=1, num_stages=1)
        del arg2_1
        buf1 = empty_strided_cuda((1, 2, 1, 10), (20, 10, 10, 1), torch.float32)

        get_raw_stream(0)
        triton_poi_fused__adaptive_avg_pool2d_1[grid(20)](buf0, buf1, 20, XBLOCK=32, num_warps=1, num_stages=1)
        del buf0
        buf2 = empty_strided_cuda((1, 2), (2, 1), torch.float32)
        buf3 = empty_strided_cuda((1, 2), (2, 1), torch.float32)

        get_raw_stream(0)
        triton_per_fused_add_norm_roll_sub_2[grid(2)](buf1, buf2, buf3, 2, 10, XBLOCK=1, num_warps=2, num_stages=1)
        buf4 = empty_strided_cuda((), (), torch.float32)

        get_raw_stream(0)
        triton_poi_fused_add_clamp_min_mean_norm_sub_3[grid(1)](buf2, buf3, buf4, 1, XBLOCK=1, num_warps=1, num_stages=1)
        del buf2
        del buf3
    return (reinterpret_tensor(buf1, (1, 2, 10), (20, 10, 1), 0), buf4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 4
    arg1_1 = 20
    arg2_1 = rand_strided((1, 4, 20), (80, 20, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
