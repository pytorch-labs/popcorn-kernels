# AOT ID: ['104_inference']
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


# kernel path: /tmp/torchinductor_sahanp/pd/cpdttccks6x3ldgdss5drnpva7sry6xjqcllkzoamkeydwiry3j2.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.im2col]
# Source node to ATen node mapping:
#   x => clone
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_im2col_0(in_ptr0, out_ptr0, ks0, ks1, ks2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x4 = xindex // ks0
    y1 = ((yindex // 3) % 3)
    x3 = (xindex % ks0)
    y0 = (yindex % 3)
    y2 = yindex // 9
    x7 = xindex
    y6 = yindex
    tl.device_assert((x4 + y1 < ks1) | ~(xmask & ymask), "index out of bounds: x4 + y1 < ks1")
    tl.device_assert((x3 + y0 < ks2) | ~(xmask & ymask), "index out of bounds: x3 + y0 < ks2")
    tmp2 = tl.load(in_ptr0 + (x3 + y0 + ks2*x4 + ks2*y1 + ks1*ks2*y2), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x7 + 4*y6 + ((-2)*ks1*y6) + ((-2)*ks2*y6) + ks1*ks2*y6), tmp2, xmask & ymask)




# kernel path: /tmp/torchinductor_sahanp/cv/ccvnobwwacmjsf355nsd6d2yigpymi577ouhfkqugjuzcgnug5ce.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten._adaptive_avg_pool2d]
# Source node to ATen node mapping:
#   x_2 => _adaptive_avg_pool2d
# Graph fragment:
#   %_adaptive_avg_pool2d : [num_users=1] = call_function[target=torch.ops.aten._adaptive_avg_pool2d.default](args = (%unsqueeze_6, [1, 10]), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__adaptive_avg_pool2d_1(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (((-2)*(x0 // ks1)) + 4*x1 + ks3*(x0 // ks1) + ((-2)*ks2*x1) + ((-2)*ks3*x1) + ks2*ks3*x1 + ((x0 % ks1))), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp0, xmask)







def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        ps0 = (-2) + s2
        buf0 = empty_strided_cuda((1, s0, 3, 3, (-2) + s1, (-2) + s2), (36*s0 + ((-18)*s0*s1) + ((-18)*s0*s2) + 9*s0*s1*s2, 36 + ((-18)*s1) + ((-18)*s2) + 9*s1*s2, 12 + ((-6)*s1) + ((-6)*s2) + 3*s1*s2, 4 + ((-2)*s1) + ((-2)*s2) + s1*s2, (-2) + s2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.im2col]
        triton_poi_fused_im2col_0_ynumel = 9*s0
        triton_poi_fused_im2col_0_xnumel = 4 + ((-2)*s1) + ((-2)*s2) + s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused_im2col_0[grid(triton_poi_fused_im2col_0_ynumel, triton_poi_fused_im2col_0_xnumel)](arg3_1, buf0, 30, 32, 32, 27, 900, XBLOCK=256, YBLOCK=1, num_warps=4, num_stages=1)
        del arg3_1
        ps1 = 4 + ((-2)*s1) + ((-2)*s2) + s1*s2
        buf1 = empty_strided_cuda((1, 9*s0, 1, 4 + ((-2)*s1) + ((-2)*s2) + s1*s2), (36*s0 + ((-18)*s0*s1) + ((-18)*s0*s2) + 9*s0*s1*s2, 4 + ((-2)*s1) + ((-2)*s2) + s1*s2, 36*s0 + ((-18)*s0*s1) + ((-18)*s0*s2) + 9*s0*s1*s2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten._adaptive_avg_pool2d]
        triton_poi_fused__adaptive_avg_pool2d_1_xnumel = 36*s0 + ((-18)*s0*s1) + ((-18)*s0*s2) + 9*s0*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused__adaptive_avg_pool2d_1[grid(triton_poi_fused__adaptive_avg_pool2d_1_xnumel)](buf0, buf1, 900, 30, 32, 32, 24300, XBLOCK=256, num_warps=4, num_stages=1)
        del buf0
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten._adaptive_avg_pool2d]
        buf2 = torch.ops.aten.avg_pool2d.default(buf1, [1, 90], [1, 90], [0, 0], False, True, None)
        del buf1
        buf3 = buf2
        del buf2
    return (reinterpret_tensor(buf3, (1, 1 + (((-86) + ((-2)*s1) + ((-2)*s2) + s1*s2) // 90), 9*s0), (9*s0 + 9*s0*(((-86) + ((-2)*s1) + ((-2)*s2) + s1*s2) // 90), 1, 1 + (((-86) + ((-2)*s1) + ((-2)*s2) + s1*s2) // 90)), 0), )


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
