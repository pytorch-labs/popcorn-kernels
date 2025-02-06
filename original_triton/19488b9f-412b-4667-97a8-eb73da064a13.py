# AOT ID: ['77_inference']
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


# kernel path: /tmp/torchinductor_sahanp/vc/cvco56l436v5xahgfp3cxi36fxqghrrmusau5i7tgbigkscfs52l.py
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
def triton_poi_fused_im2col_0(in_ptr0, out_ptr0, ks0, ks1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    x6 = xindex
    y2 = yindex // 9
    y7 = yindex
    tl.device_assert((x4 + y1 < 2 + ks1) | ~(xmask & ymask), "index out of bounds: x4 + y1 < 2 + ks1")
    tl.device_assert((x3 + y0 < 2 + ks0) | ~(xmask & ymask), "index out of bounds: x3 + y0 < 2 + ks0")
    tmp2 = (-1) + x4 + y1
    tmp3 = tl.full([1, 1], 0, tl.int64)
    tmp4 = tmp2 >= tmp3
    tmp5 = ks1
    tmp6 = tmp2 < tmp5
    tmp7 = (-1) + x3 + y0
    tmp8 = tmp7 >= tmp3
    tmp9 = ks0
    tmp10 = tmp7 < tmp9
    tmp11 = tmp4 & tmp6
    tmp12 = tmp11 & tmp8
    tmp13 = tmp12 & tmp10
    tmp14 = tl.load(in_ptr0 + ((-1) + x6 + y0 + ((-1)*ks0) + ks0*y1 + ks0*ks1*y2), tmp13 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tl.store(out_ptr0 + (x6 + ks0*ks1*y7), tmp14, xmask & ymask)







def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, s0, 3, 3, s1, s2), (9*s0*s1*s2, 9*s1*s2, 3*s1*s2, s1*s2, s2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.im2col]
        triton_poi_fused_im2col_0_ynumel = 9*s0
        triton_poi_fused_im2col_0_xnumel = s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused_im2col_0[grid(triton_poi_fused_im2col_0_ynumel, triton_poi_fused_im2col_0_xnumel)](arg3_1, buf0, 32, 32, 27, 1024, XBLOCK=256, YBLOCK=1, num_warps=4, num_stages=1)
        del arg3_1
    return (reinterpret_tensor(buf0, (1, s1*s2, 9*s0), (9*s0*s1*s2, 1, s1*s2), 0), )


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
