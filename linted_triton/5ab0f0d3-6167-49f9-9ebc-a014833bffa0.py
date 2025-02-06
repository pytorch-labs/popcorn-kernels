# AOT ID: ['132_inference']
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


# kernel path: /tmp/torchinductor_sahanp/2t/c2ti4gyyqpzadzngumybtebqq43krfprlt3ocstezmuuuju5crtt.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.reflection_pad1d]
# Source node to ATen node mapping:
#   x => _unsafe_index
# Graph fragment:
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg2_1, [None, None, %sub_5]), kwargs = {})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_reflection_pad1d_0(in_ptr0, out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (ks1*x1 + (tl.where((-1) + ks1 + ((-1)*tl_math.abs(1 + ((-1)*ks1) + tl_math.abs((-2) + x0))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks1) + tl_math.abs((-2) + x0))) + 2*ks1, (-1) + ks1 + ((-1)*tl_math.abs(1 + ((-1)*ks1) + tl_math.abs((-2) + x0)))))), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp0, xmask)


def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    assert_size_stride(arg2_1, (1, s0, s1), (s0*s1, s1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        4 + s1
        buf0 = empty_strided_cuda((1, s0, 4 + s1), (4*s0 + s0*s1, 4 + s1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.reflection_pad1d]
        triton_poi_fused_reflection_pad1d_0_xnumel = 4*s0 + s0*s1
        get_raw_stream(0)
        triton_poi_fused_reflection_pad1d_0[grid(triton_poi_fused_reflection_pad1d_0_xnumel)](arg2_1, buf0, 20, 16, 200, XBLOCK=256, num_warps=4, num_stages=1)
        del arg2_1
    return (reinterpret_tensor(buf0, (1, s0, 1, 4 + s1), (4*s0 + s0*s1, 4 + s1, 4 + s1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 10
    arg1_1 = 16
    arg2_1 = rand_strided((1, 10, 16), (160, 16, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
