# AOT ID: ['172_inference']
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


# kernel path: /tmp/torchinductor_sahanp/pf/cpfq2wfozlcgwnqrwmmtzhbikg5unjz4lzw7zsaay5tfe3i5pbbr.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.reflection_pad3d]
# Source node to ATen node mapping:
#   x => _unsafe_index, _unsafe_index_1, _unsafe_index_2
# Graph fragment:
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg4_1, [None, None, %sub_5, None, None]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index, [None, None, None, %sub_11, None]), kwargs = {})
#   %_unsafe_index_2 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_1, [None, None, None, None, %sub_17]), kwargs = {})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_reflection_pad3d_0(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, ks6, ks7, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks1)
    x2 = ((xindex // ks2) % ks3)
    x3 = xindex // ks4
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (ks7*(tl.where((-1) + ks6 + ((-1)*tl_math.abs(1 + ((-1)*ks6) + tl_math.abs((-1) + x1))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks6) + tl_math.abs((-1) + x1))) + 2*ks6, (-1) + ks6 + ((-1)*tl_math.abs(1 + ((-1)*ks6) + tl_math.abs((-1) + x1))))) + ks6*ks7*(tl.where((-1) + ks5 + ((-1)*tl_math.abs(1 + ((-1)*ks5) + tl_math.abs((-1) + x2))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks5) + tl_math.abs((-1) + x2))) + 2*ks5, (-1) + ks5 + ((-1)*tl_math.abs(1 + ((-1)*ks5) + tl_math.abs((-1) + x2))))) + ks5*ks6*ks7*x3 + (tl.where((-1) + ks7 + ((-1)*tl_math.abs(1 + ((-1)*ks7) + tl_math.abs((-1) + x0))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks7) + tl_math.abs((-1) + x0))) + 2*ks7, (-1) + ks7 + ((-1)*tl_math.abs(1 + ((-1)*ks7) + tl_math.abs((-1) + x0)))))), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x4), tmp0, xmask)


def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    s3 = arg3_1
    assert_size_stride(arg4_1, (1, s0, s1, s2, s3), (s0*s1*s2*s3, s1*s2*s3, s2*s3, s3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        2 + s3
        2 + s2
        4 + 2*s2 + 2*s3 + s2*s3
        2 + s1
        8 + 4*s1 + 4*s2 + 4*s3 + 2*s1*s2 + 2*s1*s3 + 2*s2*s3 + s1*s2*s3
        buf0 = empty_strided_cuda((1, s0, 2 + s1, 2 + s2, 2 + s3), (8*s0 + 4*s0*s1 + 4*s0*s2 + 4*s0*s3 + 2*s0*s1*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + s0*s1*s2*s3, 8 + 4*s1 + 4*s2 + 4*s3 + 2*s1*s2 + 2*s1*s3 + 2*s2*s3 + s1*s2*s3, 4 + 2*s2 + 2*s3 + s2*s3, 2 + s3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.reflection_pad3d]
        triton_poi_fused_reflection_pad3d_0_xnumel = 8*s0 + 4*s0*s1 + 4*s0*s2 + 4*s0*s3 + 2*s0*s1*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + s0*s1*s2*s3
        get_raw_stream(0)
        triton_poi_fused_reflection_pad3d_0[grid(triton_poi_fused_reflection_pad3d_0_xnumel)](arg4_1, buf0, 34, 34, 1156, 34, 39304, 32, 32, 32, 117912, XBLOCK=512, num_warps=8, num_stages=1)
        del arg4_1
    return (buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 3
    arg1_1 = 32
    arg2_1 = 32
    arg3_1 = 32
    arg4_1 = rand_strided((1, 3, 32, 32, 32), (98304, 32768, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
