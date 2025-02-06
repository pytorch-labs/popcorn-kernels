# AOT ID: ['141_inference']
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


# kernel path: /tmp/torchinductor_sahanp/s6/cs63hy43hmhsdb5yyaxfmzidptgr442sknxf64c2jkfhnuaayt4h.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_1 => getitem
# Graph fragment:
#   %getitem : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 0), kwargs = {})

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_0(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // ks0) % ks1)
    x0 = (xindex % ks0)
    x2 = xindex // ks4
    x4 = xindex
    tmp0 = (-1) + 2*x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = ks2
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + 2*x0
    tmp6 = tmp5 >= tmp1
    tmp7 = ks3
    tmp8 = tmp5 < tmp7
    tmp9 = tmp2 & tmp4
    tmp10 = tmp9 & tmp6
    tmp11 = tmp10 & tmp8
    tmp12 = tl.load(in_ptr0 + ((-1) + ((-1)*ks3) + 2*x0 + 2*ks3*x1 + ks2*ks3*x2), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = 2*x0
    tmp14 = tmp13 >= tmp1
    tmp15 = tmp13 < tmp7
    tmp16 = tmp9 & tmp14
    tmp17 = tmp16 & tmp15
    tmp18 = tl.load(in_ptr0 + (((-1)*ks3) + 2*x0 + 2*ks3*x1 + ks2*ks3*x2), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = triton_helpers.maximum(tmp18, tmp12)
    tmp20 = 2*x1
    tmp21 = tmp20 >= tmp1
    tmp22 = tmp20 < tmp3
    tmp23 = tmp21 & tmp22
    tmp24 = tmp23 & tmp6
    tmp25 = tmp24 & tmp8
    tmp26 = tl.load(in_ptr0 + ((-1) + 2*x0 + 2*ks3*x1 + ks2*ks3*x2), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp27 = triton_helpers.maximum(tmp26, tmp19)
    tmp28 = tmp23 & tmp14
    tmp29 = tmp28 & tmp15
    tmp30 = tl.load(in_ptr0 + (2*x0 + 2*ks3*x1 + ks2*ks3*x2), tmp29 & xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = triton_helpers.maximum(tmp30, tmp27)
    tl.store(out_ptr0 + (x4), tmp31, xmask)


def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        1 + (s2 // 2)
        1 + (s1 // 2)
        1 + (s1 // 2)*(s2 // 2) + (s1 // 2) + (s2 // 2)
        buf0 = empty_strided_cuda((1, s0, 1 + (s1 // 2), 1 + (s2 // 2)), (s0 + s0*(s1 // 2) + s0*(s2 // 2) + s0*(s1 // 2)*(s2 // 2), 1 + (s1 // 2)*(s2 // 2) + (s1 // 2) + (s2 // 2), 1 + (s2 // 2), 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_0_xnumel = s0 + s0*(s1 // 2) + s0*(s2 // 2) + s0*(s1 // 2)*(s2 // 2)
        get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_0[grid(triton_poi_fused_max_pool2d_with_indices_0_xnumel)](arg3_1, buf0, 33, 33, 64, 64, 1089, 3267, XBLOCK=128, num_warps=4, num_stages=1)
        del arg3_1
    return (buf0, )


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
