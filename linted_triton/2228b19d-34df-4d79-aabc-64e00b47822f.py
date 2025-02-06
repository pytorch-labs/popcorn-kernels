# AOT ID: ['100_inference']
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


# kernel path: /tmp/torchinductor_sahanp/mb/cmbtfbiy6gbc43hobdmuuxxfvwn4e74xpbkthxdcde3cb6embbmg.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.avg_pool3d]
# Source node to ATen node mapping:
#   x_3 => avg_pool3d
# Graph fragment:
#   %avg_pool3d : [num_users=1] = call_function[target=torch.ops.aten.avg_pool3d.default](args = (%unsqueeze, [2, 2, 2], [2, 2, 2]), kwargs = {})

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_avg_pool3d_0(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks1)
    x2 = xindex // ks2
    tmp0 = tl.load(in_ptr0 + (2*(((-1) + (ks4 // 2)) * (((-1) + (ks4 // 2)) <= (((0) * ((0) >= ((-2) + 2*x0)) + ((-2) + 2*x0) * (((-2) + 2*x0) > (0))))) + (((0) * ((0) >= ((-2) + 2*x0)) + ((-2) + 2*x0) * (((-2) + 2*x0) > (0)))) * ((((0) * ((0) >= ((-2) + 2*x0)) + ((-2) + 2*x0) * (((-2) + 2*x0) > (0)))) < ((-1) + (ks4 // 2)))) + 2*(ks4 // 2)*((x1 % 2)) + 4*(ks4 // 2)*(((-1) + (ks3 // 2)) * (((-1) + (ks3 // 2)) <= (((0) * ((0) >= ((-2) + 2*x2)) + ((-2) + 2*x2) * (((-2) + 2*x2) > (0))))) + (((0) * ((0) >= ((-2) + 2*x2)) + ((-2) + 2*x2) * (((-2) + 2*x2) > (0)))) * ((((0) * ((0) >= ((-2) + 2*x2)) + ((-2) + 2*x2) * (((-2) + 2*x2) > (0)))) < ((-1) + (ks3 // 2)))) + 4*(ks3 // 2)*(ks4 // 2)*(x1 // 2) + (((2*x1) % 2))), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (2*(((-1) + (ks4 // 2)) * (((-1) + (ks4 // 2)) <= (((0) * ((0) >= ((-1) + 2*x0)) + ((-1) + 2*x0) * (((-1) + 2*x0) > (0))))) + (((0) * ((0) >= ((-1) + 2*x0)) + ((-1) + 2*x0) * (((-1) + 2*x0) > (0)))) * ((((0) * ((0) >= ((-1) + 2*x0)) + ((-1) + 2*x0) * (((-1) + 2*x0) > (0)))) < ((-1) + (ks4 // 2)))) + 2*(ks4 // 2)*((x1 % 2)) + 4*(ks4 // 2)*(((-1) + (ks3 // 2)) * (((-1) + (ks3 // 2)) <= (((0) * ((0) >= ((-2) + 2*x2)) + ((-2) + 2*x2) * (((-2) + 2*x2) > (0))))) + (((0) * ((0) >= ((-2) + 2*x2)) + ((-2) + 2*x2) * (((-2) + 2*x2) > (0)))) * ((((0) * ((0) >= ((-2) + 2*x2)) + ((-2) + 2*x2) * (((-2) + 2*x2) > (0)))) < ((-1) + (ks3 // 2)))) + 4*(ks3 // 2)*(ks4 // 2)*(x1 // 2) + (((2*x1) % 2))), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2*(((-1) + (ks4 // 2)) * (((-1) + (ks4 // 2)) <= (((0) * ((0) >= ((-2) + 2*x0)) + ((-2) + 2*x0) * (((-2) + 2*x0) > (0))))) + (((0) * ((0) >= ((-2) + 2*x0)) + ((-2) + 2*x0) * (((-2) + 2*x0) > (0)))) * ((((0) * ((0) >= ((-2) + 2*x0)) + ((-2) + 2*x0) * (((-2) + 2*x0) > (0)))) < ((-1) + (ks4 // 2)))) + 2*(ks4 // 2)*((x1 % 2)) + 4*(ks4 // 2)*(((-1) + (ks3 // 2)) * (((-1) + (ks3 // 2)) <= (((0) * ((0) >= ((-1) + 2*x2)) + ((-1) + 2*x2) * (((-1) + 2*x2) > (0))))) + (((0) * ((0) >= ((-1) + 2*x2)) + ((-1) + 2*x2) * (((-1) + 2*x2) > (0)))) * ((((0) * ((0) >= ((-1) + 2*x2)) + ((-1) + 2*x2) * (((-1) + 2*x2) > (0)))) < ((-1) + (ks3 // 2)))) + 4*(ks3 // 2)*(ks4 // 2)*(x1 // 2) + (((2*x1) % 2))), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (2*(((-1) + (ks4 // 2)) * (((-1) + (ks4 // 2)) <= (((0) * ((0) >= ((-1) + 2*x0)) + ((-1) + 2*x0) * (((-1) + 2*x0) > (0))))) + (((0) * ((0) >= ((-1) + 2*x0)) + ((-1) + 2*x0) * (((-1) + 2*x0) > (0)))) * ((((0) * ((0) >= ((-1) + 2*x0)) + ((-1) + 2*x0) * (((-1) + 2*x0) > (0)))) < ((-1) + (ks4 // 2)))) + 2*(ks4 // 2)*((x1 % 2)) + 4*(ks4 // 2)*(((-1) + (ks3 // 2)) * (((-1) + (ks3 // 2)) <= (((0) * ((0) >= ((-1) + 2*x2)) + ((-1) + 2*x2) * (((-1) + 2*x2) > (0))))) + (((0) * ((0) >= ((-1) + 2*x2)) + ((-1) + 2*x2) * (((-1) + 2*x2) > (0)))) * ((((0) * ((0) >= ((-1) + 2*x2)) + ((-1) + 2*x2) * (((-1) + 2*x2) > (0)))) < ((-1) + (ks3 // 2)))) + 4*(ks3 // 2)*(ks4 // 2)*(x1 // 2) + (((2*x1) % 2))), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (1 + 2*(((-1) + (ks4 // 2)) * (((-1) + (ks4 // 2)) <= (((0) * ((0) >= ((-2) + 2*x0)) + ((-2) + 2*x0) * (((-2) + 2*x0) > (0))))) + (((0) * ((0) >= ((-2) + 2*x0)) + ((-2) + 2*x0) * (((-2) + 2*x0) > (0)))) * ((((0) * ((0) >= ((-2) + 2*x0)) + ((-2) + 2*x0) * (((-2) + 2*x0) > (0)))) < ((-1) + (ks4 // 2)))) + 2*(ks4 // 2)*((((1 + 2*x1) // 2) % 2)) + 4*(ks4 // 2)*(((-1) + (ks3 // 2)) * (((-1) + (ks3 // 2)) <= (((0) * ((0) >= ((-2) + 2*x2)) + ((-2) + 2*x2) * (((-2) + 2*x2) > (0))))) + (((0) * ((0) >= ((-2) + 2*x2)) + ((-2) + 2*x2) * (((-2) + 2*x2) > (0)))) * ((((0) * ((0) >= ((-2) + 2*x2)) + ((-2) + 2*x2) * (((-2) + 2*x2) > (0)))) < ((-1) + (ks3 // 2)))) + 4*(ks3 // 2)*(ks4 // 2)*((((1 + 2*x1) // 4) % ks5))), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (1 + 2*(((-1) + (ks4 // 2)) * (((-1) + (ks4 // 2)) <= (((0) * ((0) >= ((-1) + 2*x0)) + ((-1) + 2*x0) * (((-1) + 2*x0) > (0))))) + (((0) * ((0) >= ((-1) + 2*x0)) + ((-1) + 2*x0) * (((-1) + 2*x0) > (0)))) * ((((0) * ((0) >= ((-1) + 2*x0)) + ((-1) + 2*x0) * (((-1) + 2*x0) > (0)))) < ((-1) + (ks4 // 2)))) + 2*(ks4 // 2)*((((1 + 2*x1) // 2) % 2)) + 4*(ks4 // 2)*(((-1) + (ks3 // 2)) * (((-1) + (ks3 // 2)) <= (((0) * ((0) >= ((-2) + 2*x2)) + ((-2) + 2*x2) * (((-2) + 2*x2) > (0))))) + (((0) * ((0) >= ((-2) + 2*x2)) + ((-2) + 2*x2) * (((-2) + 2*x2) > (0)))) * ((((0) * ((0) >= ((-2) + 2*x2)) + ((-2) + 2*x2) * (((-2) + 2*x2) > (0)))) < ((-1) + (ks3 // 2)))) + 4*(ks3 // 2)*(ks4 // 2)*((((1 + 2*x1) // 4) % ks5))), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (1 + 2*(((-1) + (ks4 // 2)) * (((-1) + (ks4 // 2)) <= (((0) * ((0) >= ((-2) + 2*x0)) + ((-2) + 2*x0) * (((-2) + 2*x0) > (0))))) + (((0) * ((0) >= ((-2) + 2*x0)) + ((-2) + 2*x0) * (((-2) + 2*x0) > (0)))) * ((((0) * ((0) >= ((-2) + 2*x0)) + ((-2) + 2*x0) * (((-2) + 2*x0) > (0)))) < ((-1) + (ks4 // 2)))) + 2*(ks4 // 2)*((((1 + 2*x1) // 2) % 2)) + 4*(ks4 // 2)*(((-1) + (ks3 // 2)) * (((-1) + (ks3 // 2)) <= (((0) * ((0) >= ((-1) + 2*x2)) + ((-1) + 2*x2) * (((-1) + 2*x2) > (0))))) + (((0) * ((0) >= ((-1) + 2*x2)) + ((-1) + 2*x2) * (((-1) + 2*x2) > (0)))) * ((((0) * ((0) >= ((-1) + 2*x2)) + ((-1) + 2*x2) * (((-1) + 2*x2) > (0)))) < ((-1) + (ks3 // 2)))) + 4*(ks3 // 2)*(ks4 // 2)*((((1 + 2*x1) // 4) % ks5))), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (1 + 2*(((-1) + (ks4 // 2)) * (((-1) + (ks4 // 2)) <= (((0) * ((0) >= ((-1) + 2*x0)) + ((-1) + 2*x0) * (((-1) + 2*x0) > (0))))) + (((0) * ((0) >= ((-1) + 2*x0)) + ((-1) + 2*x0) * (((-1) + 2*x0) > (0)))) * ((((0) * ((0) >= ((-1) + 2*x0)) + ((-1) + 2*x0) * (((-1) + 2*x0) > (0)))) < ((-1) + (ks4 // 2)))) + 2*(ks4 // 2)*((((1 + 2*x1) // 2) % 2)) + 4*(ks4 // 2)*(((-1) + (ks3 // 2)) * (((-1) + (ks3 // 2)) <= (((0) * ((0) >= ((-1) + 2*x2)) + ((-1) + 2*x2) * (((-1) + 2*x2) > (0))))) + (((0) * ((0) >= ((-1) + 2*x2)) + ((-1) + 2*x2) * (((-1) + 2*x2) > (0)))) * ((((0) * ((0) >= ((-1) + 2*x2)) + ((-1) + 2*x2) * (((-1) + 2*x2) > (0)))) < ((-1) + (ks3 // 2)))) + 4*(ks3 // 2)*(ks4 // 2)*((((1 + 2*x1) // 4) % ks5))), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp15 = 0.125
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (x0 + 2*x2 + 4*x1 + x2*(ks4 // 4) + 2*x1*(ks3 // 4) + 2*x1*(ks4 // 4) + x1*(ks3 // 4)*(ks4 // 4)), tmp16, xmask)


def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        2 + (s2 // 4)
        2*s0
        4*s0 + 2*s0*(s2 // 4)
        buf0 = empty_strided_cuda((1, 1, 2*s0, 2 + (s1 // 4), 2 + (s2 // 4)), (8*s0 + 4*s0*(s1 // 4) + 4*s0*(s2 // 4) + 2*s0*(s1 // 4)*(s2 // 4), 8*s0 + 4*s0*(s1 // 4) + 4*s0*(s2 // 4) + 2*s0*(s1 // 4)*(s2 // 4), 4 + 2*(s1 // 4) + 2*(s2 // 4) + (s1 // 4)*(s2 // 4), 2 + (s2 // 4), 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.avg_pool3d]
        triton_poi_fused_avg_pool3d_0_xnumel = 8*s0 + 4*s0*(s1 // 4) + 4*s0*(s2 // 4) + 2*s0*(s1 // 4)*(s2 // 4)
        get_raw_stream(0)
        triton_poi_fused_avg_pool3d_0[grid(triton_poi_fused_avg_pool3d_0_xnumel)](arg3_1, buf0, 18, 6, 108, 64, 64, 3, 1944, XBLOCK=128, num_warps=4, num_stages=1)
        del arg3_1
    return (reinterpret_tensor(buf0, (1, (8*s0 + 4*s0*(s1 // 4) + 4*s0*(s2 // 4) + 2*s0*(s1 // 4)*(s2 // 4)) // ((s0*(s1 // (s1 // 2))*(s2 // (s2 // 2))) // 2), (s0*(s1 // (s1 // 2))*(s2 // (s2 // 2))) // 2), (((8*s0 + 4*s0*(s1 // 4) + 4*s0*(s2 // 4) + 2*s0*(s1 // 4)*(s2 // 4)) // ((s0*(s1 // (s1 // 2))*(s2 // (s2 // 2))) // 2))*((s0*(s1 // (s1 // 2))*(s2 // (s2 // 2))) // 2), 1, (8*s0 + 4*s0*(s1 // 4) + 4*s0*(s2 // 4) + 2*s0*(s1 // 4)*(s2 // 4)) // ((s0*(s1 // (s1 // 2))*(s2 // (s2 // 2))) // 2)), 0), )


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
