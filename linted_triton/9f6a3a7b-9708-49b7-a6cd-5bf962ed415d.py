
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
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_copy_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 70)
    x1 = xindex // 70
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = 68 + x0
    tmp4 = tl.full([1], 1, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tl.full([1], 69, tl.int64)
    tmp7 = tmp3 < tmp6
    tmp8 = tmp5 & tmp7
    tmp9 = tmp8 & tmp2
    tmp10 = x1
    tmp11 = tl.full([1], 1, tl.int64)
    tmp12 = tmp10 >= tmp11
    tmp13 = tl.full([1], 11, tl.int64)
    tmp14 = tmp10 < tmp13
    tmp15 = tmp12 & tmp14
    tmp16 = tmp15 & tmp9
    tmp17 = tl.load(in_ptr0 + ((-1) + 64*x1), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.load(in_ptr1 + (68 + x2), tmp9 & xmask, other=0.0)
    tmp19 = tl.where(tmp15, tmp17, tmp18)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp9, tmp19, tmp20)
    tmp22 = float("nan")
    tmp23 = tl.where(tmp8, tmp21, tmp22)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp2, tmp23, tmp24)
    tmp26 = tmp0 >= tmp1
    tmp27 = tl.full([1], 69, tl.int64)
    tmp28 = tmp0 < tmp27
    tmp29 = tmp26 & tmp28
    tmp30 = x1
    tmp31 = tl.full([1], 1, tl.int64)
    tmp32 = tmp30 >= tmp31
    tmp33 = tl.full([1], 11, tl.int64)
    tmp34 = tmp30 < tmp33
    tmp35 = tmp32 & tmp34
    tmp36 = tmp35 & tmp29
    tmp37 = tl.load(in_ptr0 + ((-64) + 64*x1 + ((63) * ((63) <= (((0) * ((0) >= ((-3) + x0)) + ((-3) + x0) * (((-3) + x0) > (0))))) + (((0) * ((0) >= ((-3) + x0)) + ((-3) + x0) * (((-3) + x0) > (0)))) * ((((0) * ((0) >= ((-3) + x0)) + ((-3) + x0) * (((-3) + x0) > (0)))) < (63)))), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.load(in_ptr1 + (x2), tmp29 & xmask, other=0.0)
    tmp39 = tl.where(tmp35, tmp37, tmp38)
    tmp40 = tl.full(tmp39.shape, 0.0, tmp39.dtype)
    tmp41 = tl.where(tmp29, tmp39, tmp40)
    tmp42 = float("nan")
    tmp43 = tl.where(tmp29, tmp41, tmp42)
    tmp44 = tl.where(tmp2, tmp25, tmp43)
    tl.store(out_ptr0 + (x2), tmp44, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 70
    x0 = (xindex % 70)
    x2 = xindex
    tmp41 = tl.load(in_ptr0 + (x2), xmask)
    tmp0 = x1
    tmp1 = tl.full([1], 11, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = (-10) + x1
    tmp4 = tl.full([1], 1, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = x0
    tmp8 = tl.full([1], 69, tl.int64)
    tmp9 = tmp7 >= tmp8
    tmp10 = tmp9 & tmp6
    tmp11 = tl.load(in_ptr0 + (1 + 70*x1), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr0 + (x2), tmp6 & xmask, other=0.0)
    tmp13 = tl.where(tmp9, tmp11, tmp12)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = x0
    tmp17 = tl.full([1], 69, tl.int64)
    tmp18 = tmp16 >= tmp17
    tmp19 = tmp18 & tmp2
    tmp20 = tl.load(in_ptr0 + ((-699) + 70*x1), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.load(in_ptr0 + ((-700) + x2), tmp2 & xmask, other=0.0)
    tmp22 = tl.where(tmp18, tmp20, tmp21)
    tmp23 = tl.where(tmp5, tmp15, tmp22)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp2, tmp23, tmp24)
    tmp26 = tl.full([1], 1, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = x0
    tmp29 = tl.full([1], 69, tl.int64)
    tmp30 = tmp28 >= tmp29
    tmp31 = tmp30 & tmp27
    tmp32 = tl.load(in_ptr0 + (701 + 70*x1), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tl.load(in_ptr0 + (700 + x2), tmp27 & xmask, other=0.0)
    tmp34 = tl.where(tmp30, tmp32, tmp33)
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp27, tmp34, tmp35)
    tmp37 = x0
    tmp38 = tl.full([1], 69, tl.int64)
    tmp39 = tmp37 >= tmp38
    tmp40 = tl.load(in_ptr0 + (1 + 70*x1), tmp39 & xmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.where(tmp39, tmp40, tmp41)
    tmp43 = tl.where(tmp27, tmp36, tmp42)
    tmp44 = tl.where(tmp2, tmp25, tmp43)
    tl.store(out_ptr0 + (x2), tmp44, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (1, 10, 64), (640, 64, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 1, 12, 70), (840, 840, 70, 1), torch.float32)
        buf1 = empty_strided_cuda((1, 1, 12, 70), (840, 840, 70, 1), torch.float32)

        get_raw_stream(0)
        triton_poi_fused_copy_0[grid(840)](arg0_1, buf0, buf1, 840, XBLOCK=128, num_warps=4, num_stages=1)
        del arg0_1
        buf2 = reinterpret_tensor(buf0, (1, 1, 12, 70), (840, 1, 70, 1), 0); del buf0

        get_raw_stream(0)
        triton_poi_fused_1[grid(840)](buf1, buf2, 840, XBLOCK=128, num_warps=4, num_stages=1)
        del buf1
    return (reinterpret_tensor(buf2, (1, 70, 12), (840, 1, 70), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 10, 64), (640, 64, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
