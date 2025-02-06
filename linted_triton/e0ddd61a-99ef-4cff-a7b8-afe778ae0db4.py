
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
def triton_poi_fused_copy_0(in_ptr0, out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    x2 = xindex
    tmp0 = x0
    tmp1 = 2 + ks1*ks2
    tmp2 = tmp0 >= tmp1
    tmp3 = x0 + ((-1)*ks1*ks2)
    tmp4 = tl.full([1], 2, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = x0
    tmp8 = tl.full([1], 2, tl.int64)
    tmp9 = tmp7 >= tmp8
    tmp10 = tl.broadcast_to(2 + ks1*ks2, [XBLOCK])
    tmp11 = tmp7 < tmp10
    tmp12 = tmp9 & tmp11
    tmp13 = tmp12 & tmp6
    tmp14 = tl.load(in_ptr0 + ((-2) + x0 + ks1*ks2*x1), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = float("nan")
    tmp16 = tl.where(tmp12, tmp14, tmp15)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp6, tmp16, tmp17)
    tmp19 = tmp3 >= tmp4
    tmp20 = tl.broadcast_to(2 + ks1*ks2, [XBLOCK])
    tmp21 = tmp3 < tmp20
    tmp22 = tmp19 & tmp21
    tmp23 = tmp22 & tmp2
    tmp24 = tl.load(in_ptr0 + ((-2) + x0 + ((-1)*ks1*ks2) + ks1*ks2*x1), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = float("nan")
    tmp26 = tl.where(tmp22, tmp24, tmp25)
    tmp27 = tl.where(tmp5, tmp18, tmp26)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp2, tmp27, tmp28)
    tmp30 = tl.full([1], 2, tl.int64)
    tmp31 = tmp0 < tmp30
    tmp32 = x0 + ks1*ks2
    tmp33 = tl.full([1], 2, tl.int64)
    tmp34 = tmp32 >= tmp33
    tmp35 = tl.broadcast_to(2 + ks1*ks2, [XBLOCK])
    tmp36 = tmp32 < tmp35
    tmp37 = tmp34 & tmp36
    tmp38 = tmp37 & tmp31
    tmp39 = tl.load(in_ptr0 + ((-2) + x0 + ks1*ks2 + ks1*ks2*x1), tmp38 & xmask, eviction_policy='evict_last', other=0.0)
    tmp40 = float("nan")
    tmp41 = tl.where(tmp37, tmp39, tmp40)
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp31, tmp41, tmp42)
    tmp44 = tmp0 >= tmp30
    tmp45 = tmp0 < tmp1
    tmp46 = tmp44 & tmp45
    tmp47 = tl.load(in_ptr0 + ((-2) + x0 + ks1*ks2*x1), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp48 = float("nan")
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tl.where(tmp31, tmp43, tmp49)
    tmp51 = tl.where(tmp2, tmp29, tmp50)
    tl.store(out_ptr0 + (x2), tmp51, xmask)


def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        4 + s0*s2
        buf1 = empty_strided_cuda((1, s1, 4 + s0*s2), (4*s1 + s0*s1*s2, 4 + s0*s2, 1), torch.float32)

        triton_poi_fused_copy_0_xnumel = 4*s1 + s0*s1*s2
        get_raw_stream(0)
        triton_poi_fused_copy_0[grid(triton_poi_fused_copy_0_xnumel)](arg3_1, buf1, 324, 10, 32, 10368, XBLOCK=256, num_warps=4, num_stages=1)
        del arg3_1
    return (buf1, s1, s2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 10
    arg1_1 = 32
    arg2_1 = 32
    arg3_1 = rand_strided((1, 10, 32, 32), (10240, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
