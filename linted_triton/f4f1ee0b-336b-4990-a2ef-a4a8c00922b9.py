
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
def triton_poi_fused_max_pool2d_with_indices_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (1 + 2*x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 1.0
    tmp4 = tmp0 * tmp3
    tmp5 = libdevice.expm1(tmp4)
    tmp6 = tmp5 * tmp3
    tmp7 = tl.where(tmp2, tmp4, tmp6)
    tmp8 = tl_math.abs(tmp7)
    tmp9 = 0.5
    tmp10 = tmp8 <= tmp9
    tmp11 = tl.where(tmp10, tmp1, tmp7)
    tmp12 = tl_math.abs(tmp11)
    tmp13 = tmp12 > tmp9
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = tmp14 < tmp11
    tmp16 = tmp15.to(tl.int8)
    tmp17 = tmp11 < tmp14
    tmp18 = tmp17.to(tl.int8)
    tmp19 = tmp16 - tmp18
    tmp20 = tmp19.to(tmp11.dtype)
    tmp21 = tmp20 * tmp9
    tmp22 = tmp11 - tmp21
    tmp23 = tmp11 * tmp1
    tmp24 = tl.where(tmp13, tmp22, tmp23)
    tmp26 = tmp25 > tmp1
    tmp27 = tmp25 * tmp3
    tmp28 = libdevice.expm1(tmp27)
    tmp29 = tmp28 * tmp3
    tmp30 = tl.where(tmp26, tmp27, tmp29)
    tmp31 = tl_math.abs(tmp30)
    tmp32 = tmp31 <= tmp9
    tmp33 = tl.where(tmp32, tmp1, tmp30)
    tmp34 = tl_math.abs(tmp33)
    tmp35 = tmp34 > tmp9
    tmp36 = tmp14 < tmp33
    tmp37 = tmp36.to(tl.int8)
    tmp38 = tmp33 < tmp14
    tmp39 = tmp38.to(tl.int8)
    tmp40 = tmp37 - tmp39
    tmp41 = tmp40.to(tmp33.dtype)
    tmp42 = tmp41 * tmp9
    tmp43 = tmp33 - tmp42
    tmp44 = tmp33 * tmp1
    tmp45 = tl.where(tmp35, tmp43, tmp44)
    tmp46 = triton_helpers.maximum(tmp45, tmp24)
    tl.store(out_ptr0 + (x0), tmp46, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (1 + 2*x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 1.0
    tmp4 = tmp0 * tmp3
    tmp5 = libdevice.expm1(tmp4)
    tmp6 = tmp5 * tmp3
    tmp7 = tl.where(tmp2, tmp4, tmp6)
    tmp8 = tl_math.abs(tmp7)
    tmp9 = 0.5
    tmp10 = tmp8 <= tmp9
    tmp11 = tl.where(tmp10, tmp1, tmp7)
    tmp12 = tl_math.abs(tmp11)
    tmp13 = tmp12 > tmp9
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = tmp14 < tmp11
    tmp16 = tmp15.to(tl.int8)
    tmp17 = tmp11 < tmp14
    tmp18 = tmp17.to(tl.int8)
    tmp19 = tmp16 - tmp18
    tmp20 = tmp19.to(tmp11.dtype)
    tmp21 = tmp20 * tmp9
    tmp22 = tmp11 - tmp21
    tmp23 = tmp11 * tmp1
    tmp24 = tl.where(tmp13, tmp22, tmp23)
    tmp26 = tmp25 > tmp1
    tmp27 = tmp25 * tmp3
    tmp28 = libdevice.expm1(tmp27)
    tmp29 = tmp28 * tmp3
    tmp30 = tl.where(tmp26, tmp27, tmp29)
    tmp31 = tl_math.abs(tmp30)
    tmp32 = tmp31 <= tmp9
    tmp33 = tl.where(tmp32, tmp1, tmp30)
    tmp34 = tl_math.abs(tmp33)
    tmp35 = tmp34 > tmp9
    tmp36 = tmp14 < tmp33
    tmp37 = tmp36.to(tl.int8)
    tmp38 = tmp33 < tmp14
    tmp39 = tmp38.to(tl.int8)
    tmp40 = tmp37 - tmp39
    tmp41 = tmp40.to(tmp33.dtype)
    tmp42 = tmp41 * tmp9
    tmp43 = tmp33 - tmp42
    tmp44 = tmp33 * tmp1
    tmp45 = tl.where(tmp35, tmp43, tmp44)
    tmp46 = triton_helpers.maximum(tmp45, tmp24)
    tl.store(out_ptr0 + (x0), tmp46, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (1, 64), (64, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 1, 1, 32), (32, 32, 32, 1), torch.float32)

        get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_0[grid(32)](arg0_1, buf0, 32, XBLOCK=32, num_warps=1, num_stages=1)
        del arg0_1
        buf1 = empty_strided_cuda((1, 1, 1, 16), (16, 16, 16, 1), torch.float32)

        get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_1[grid(16)](buf0, buf1, 16, XBLOCK=16, num_warps=1, num_stages=1)
        del buf0
    return (reinterpret_tensor(buf1, (1, 16), (16, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
