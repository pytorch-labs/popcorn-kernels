
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
def triton_poi_fused_max_pool2d_with_indices_0(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, ks6, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x5 = xindex // ks0
    x1 = ((xindex // ks2) % ks3)
    x0 = (xindex % ks2)
    x2 = xindex // ks6
    x7 = xindex
    tmp0 = (-1) + x5
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = ks1
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + 2*x1
    tmp6 = tmp5 >= tmp1
    tmp7 = ks4
    tmp8 = tmp5 < tmp7
    tmp9 = (-1) + 2*x0
    tmp10 = tmp9 >= tmp1
    tmp11 = ks5
    tmp12 = tmp9 < tmp11
    tmp13 = tmp2 & tmp4
    tmp14 = tmp13 & tmp6
    tmp15 = tmp14 & tmp8
    tmp16 = tmp15 & tmp10
    tmp17 = tmp16 & tmp12
    tmp18 = tl.load(in_ptr0 + ((-1) + ((-1)*ks5) + 2*x0 + ((-1)*ks4*ks5) + 2*ks5*x1 + ks4*ks5*x2), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = 2*x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp11
    tmp22 = tmp15 & tmp20
    tmp23 = tmp22 & tmp21
    tmp24 = tl.load(in_ptr0 + (((-1)*ks5) + 2*x0 + ((-1)*ks4*ks5) + 2*ks5*x1 + ks4*ks5*x2), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = 2*x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp7
    tmp29 = tmp13 & tmp27
    tmp30 = tmp29 & tmp28
    tmp31 = tmp30 & tmp10
    tmp32 = tmp31 & tmp12
    tmp33 = tl.load(in_ptr0 + ((-1) + 2*x0 + ((-1)*ks4*ks5) + 2*ks5*x1 + ks4*ks5*x2), tmp32 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = triton_helpers.maximum(tmp33, tmp25)
    tmp35 = tmp30 & tmp20
    tmp36 = tmp35 & tmp21
    tmp37 = tl.load(in_ptr0 + (2*x0 + ((-1)*ks4*ks5) + 2*ks5*x1 + ks4*ks5*x2), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp38 = triton_helpers.maximum(tmp37, tmp34)
    tl.store(out_ptr0 + (x7), tmp38, xmask)


def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        1 + (s1 // 2)*(s2 // 2) + (s1 // 2) + (s2 // 2)
        1 + (s2 // 2)
        1 + (s1 // 2)
        1 + (s1 // 2)*(s2 // 2) + (s1 // 2) + (s2 // 2)
        buf0 = empty_strided_cuda((1, 2 + s0, 1 + (s1 // 2), 1 + (s2 // 2)), (2 + s0 + 2*(s1 // 2) + 2*(s2 // 2) + s0*(s1 // 2) + s0*(s2 // 2) + 2*(s1 // 2)*(s2 // 2) + s0*(s1 // 2)*(s2 // 2), 1 + (s1 // 2)*(s2 // 2) + (s1 // 2) + (s2 // 2), 1 + (s2 // 2), 1), torch.float32)

        triton_poi_fused_max_pool2d_with_indices_0_xnumel = 2 + s0 + 2*(s1 // 2) + 2*(s2 // 2) + s0*(s1 // 2) + s0*(s2 // 2) + 2*(s1 // 2)*(s2 // 2) + s0*(s1 // 2)*(s2 // 2)
        get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_0[grid(triton_poi_fused_max_pool2d_with_indices_0_xnumel)](arg3_1, buf0, 289, 3, 17, 17, 32, 32, 289, 1445, XBLOCK=256, num_warps=4, num_stages=1)
        del arg3_1
    return (buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 3
    arg1_1 = 32
    arg2_1 = 32
    arg3_1 = rand_strided((1, 3, 32, 32), (3072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
