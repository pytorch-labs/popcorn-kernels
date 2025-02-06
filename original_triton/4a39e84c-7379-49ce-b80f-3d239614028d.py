# AOT ID: ['82_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import (
    grid,
    split_scan_grid,
    grid_combo_kernels,
    start_graph,
    end_graph,
    cooperative_reduction_grid,
)
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
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


# kernel path: /tmp/torchinductor_sahanp/gz/cgz55gnu73mdyou5lyex4x5onprde5muekl4cocryrfsmuuc6t5h.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.hardswish]
# Source node to ATen node mapping:
#   x_2 => add_14, clamp_max_2, clamp_min_2, div, mul_11
# Graph fragment:
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute, 3), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_14, 0), kwargs = {})
#   %clamp_max_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 6), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute, %clamp_max_2), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_11, 6), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_hardswish_0(in_ptr0, out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (ks2*(((-1) + ks1) * (((-1) + ks1) <= (((0) * ((0) >= ((-2) + (x0 // (4 + ks2)))) + ((-2) + (x0 // (4 + ks2))) * (((-2) + (x0 // (4 + ks2))) > (0))))) + (((0) * ((0) >= ((-2) + (x0 // (4 + ks2)))) + ((-2) + (x0 // (4 + ks2))) * (((-2) + (x0 // (4 + ks2))) > (0)))) * ((((0) * ((0) >= ((-2) + (x0 // (4 + ks2)))) + ((-2) + (x0 // (4 + ks2))) * (((-2) + (x0 // (4 + ks2))) > (0)))) < ((-1) + ks1))) + ks1*ks2*x1 + (((-1) + ks2) * (((-1) + ks2) <= (((0) * ((0) >= ((-2) + ((x0 % (4 + ks2))))) + ((-2) + ((x0 % (4 + ks2)))) * (((-2) + ((x0 % (4 + ks2)))) > (0))))) + (((0) * ((0) >= ((-2) + ((x0 % (4 + ks2))))) + ((-2) + ((x0 % (4 + ks2)))) * (((-2) + ((x0 % (4 + ks2)))) > (0)))) * ((((0) * ((0) >= ((-2) + ((x0 % (4 + ks2))))) + ((-2) + ((x0 % (4 + ks2)))) * (((-2) + ((x0 % (4 + ks2)))) > (0)))) < ((-1) + ks2)))), xmask, eviction_policy='evict_last')
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = 0.16666666666666666
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x2), tmp9, xmask)







def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        ps0 = 16 + 4*s1 + 4*s2 + s1*s2
        buf0 = empty_strided_cuda((1, 16 + 4*s1 + 4*s2 + s1*s2, s0), (16*s0 + 4*s0*s1 + 4*s0*s2 + s0*s1*s2, 1, 16 + 4*s1 + 4*s2 + s1*s2), torch.float32)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.hardswish]
        triton_poi_fused_hardswish_0_xnumel = 16*s0 + 4*s0*s1 + 4*s0*s2 + s0*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused_hardswish_0[grid(triton_poi_fused_hardswish_0_xnumel)](arg3_1, buf0, 1296, 32, 32, 3888, XBLOCK=256, num_warps=4, num_stages=1)
        del arg3_1
    return (buf0, 4 + s1, 4 + s2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 3
    arg1_1 = 32
    arg2_1 = 32
    arg3_1 = rand_strided((1, 3, 32, 32), (3072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
