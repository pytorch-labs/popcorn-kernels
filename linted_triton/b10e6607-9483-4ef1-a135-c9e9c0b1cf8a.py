
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
def triton_red_fused__log_softmax_0(in_ptr0, out_ptr0, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    _tmp35 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp0 = r0_1 + x0*((1 + ks0*ks1*ks2) // 2)
        tmp1 = ks0*ks1*ks2
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (((r0_1 + x0*((1 + ks0*ks1*ks2) // 2)) % (ks0*ks1*ks2))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 > tmp4
        tmp6 = libdevice.expm1(tmp3)
        tmp7 = tl.where(tmp5, tmp3, tmp6)
        tmp8 = tmp7 > tmp4
        tmp9 = libdevice.expm1(tmp7)
        tmp10 = tl.where(tmp8, tmp7, tmp9)
        tmp11 = 1.0
        tmp12 = tmp10 * tmp11
        tmp13 = 20.0
        tmp14 = tmp12 > tmp13
        tmp15 = tl_math.exp(tmp12)
        tmp16 = libdevice.log1p(tmp15)
        tmp17 = tmp16 * tmp11
        tmp18 = tl.where(tmp14, tmp10, tmp17)
        tmp19 = tmp18 * tmp11
        tmp20 = tmp19 > tmp13
        tmp21 = tl_math.exp(tmp19)
        tmp22 = libdevice.log1p(tmp21)
        tmp23 = tmp22 * tmp11
        tmp24 = tl.where(tmp20, tmp18, tmp23)
        tmp25 = tl_math.abs(tmp24)
        tmp26 = 0.5
        tmp27 = tmp25 <= tmp26
        tmp28 = tl.where(tmp27, tmp4, tmp24)
        tmp29 = tl_math.abs(tmp28)
        tmp30 = tmp29 <= tmp26
        tmp31 = tl.where(tmp30, tmp4, tmp28)
        tmp32 = tl.full(tmp31.shape, float("-inf"), tmp31.dtype)
        tmp33 = tl.where(tmp2, tmp31, tmp32)
        tmp34 = tl.broadcast_to(tmp33, [XBLOCK, R0_BLOCK])
        tmp36 = triton_helpers.maximum(_tmp35, tmp34)
        _tmp35 = tl.where(r0_mask & xmask, tmp36, _tmp35)
    tmp35 = triton_helpers.max2(_tmp35, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp35, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__log_softmax_1(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    R0_BLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = triton_helpers.max2(tmp1, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp3, None)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__log_softmax_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp29 = tl.load(in_ptr1 + (0))
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK])
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = libdevice.expm1(tmp0)
    tmp4 = tl.where(tmp2, tmp0, tmp3)
    tmp5 = tmp4 > tmp1
    tmp6 = libdevice.expm1(tmp4)
    tmp7 = tl.where(tmp5, tmp4, tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = 20.0
    tmp11 = tmp9 > tmp10
    tmp12 = tl_math.exp(tmp9)
    tmp13 = libdevice.log1p(tmp12)
    tmp14 = tmp13 * tmp8
    tmp15 = tl.where(tmp11, tmp7, tmp14)
    tmp16 = tmp15 * tmp8
    tmp17 = tmp16 > tmp10
    tmp18 = tl_math.exp(tmp16)
    tmp19 = libdevice.log1p(tmp18)
    tmp20 = tmp19 * tmp8
    tmp21 = tl.where(tmp17, tmp15, tmp20)
    tmp22 = tl_math.abs(tmp21)
    tmp23 = 0.5
    tmp24 = tmp22 <= tmp23
    tmp25 = tl.where(tmp24, tmp1, tmp21)
    tmp26 = tl_math.abs(tmp25)
    tmp27 = tmp26 <= tmp23
    tmp28 = tl.where(tmp27, tmp1, tmp25)
    tmp31 = tmp28 - tmp30
    tl.store(out_ptr0 + (x0), tmp31, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__log_softmax_3(in_ptr0, out_ptr0, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp0 = r0_1 + x0*((1 + ks0*ks1*ks2) // 2)
        tmp1 = ks0*ks1*ks2
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (r0_1 + x0*((1 + ks0*ks1*ks2) // 2)), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl_math.exp(tmp3)
        tmp5 = tl.full(tmp4.shape, 0, tmp4.dtype)
        tmp6 = tl.where(tmp2, tmp4, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__log_softmax_nll_loss_forward_4(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    R0_BLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), None)
    tmp4 = tl.load(in_ptr1 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, 1])
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None]
    tmp6 = tl_math.log(tmp3)
    tmp7 = tmp5 - tmp6
    tmp8 = -tmp7
    tmp9 = tl.full([1, 1], True, tl.int1)
    tmp10 = 0.0
    tmp11 = tl.where(tmp9, tmp8, tmp10)
    tmp12 = 1.0
    tmp13 = tmp11 / tmp12
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp13, None)


def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 1, 2), (2, 2, 1), torch.float32)

        (1 + s0*s1*s2) // 2
        get_raw_stream(0)
        triton_red_fused__log_softmax_0[grid(2)](arg3_1, buf0, 3, 64, 64, 2, 6144, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        buf1 = empty_strided_cuda((1, 1), (1, 1), torch.float32)

        get_raw_stream(0)
        triton_per_fused__log_softmax_1[grid(1)](buf0, buf1, 1, 2, XBLOCK=1, num_warps=2, num_stages=1)
        buf2 = empty_strided_cuda((1, s0*s1*s2), (s0*s1*s2, 1), torch.float32)

        triton_poi_fused__log_softmax_2_xnumel = s0*s1*s2
        get_raw_stream(0)
        triton_poi_fused__log_softmax_2[grid(triton_poi_fused__log_softmax_2_xnumel)](arg3_1, buf1, buf2, 12288, XBLOCK=256, num_warps=4, num_stages=1)
        del arg3_1
        buf3 = buf0; del buf0

        (1 + s0*s1*s2) // 2
        get_raw_stream(0)
        triton_red_fused__log_softmax_3[grid(2)](buf2, buf3, 3, 64, 64, 2, 6144, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        buf4 = buf1; del buf1
        buf5 = reinterpret_tensor(buf4, (), (), 0); del buf4

        get_raw_stream(0)
        triton_per_fused__log_softmax_nll_loss_forward_4[grid(1)](buf5, buf3, buf2, 1, 2, XBLOCK=1, num_warps=2, num_stages=1)
        del buf2
        del buf3
    return (buf5, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 3
    arg1_1 = 64
    arg2_1 = 64
    arg3_1 = rand_strided((1, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
