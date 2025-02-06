# AOT ID: ['106_inference']
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


# kernel path: /tmp/torchinductor_sahanp/vz/cvzplxjkreu22kawhuuoghzkb6cssizvvspn5bkmbjh63f64etfy.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten._unsafe_index]
# Source node to ATen node mapping:
#   x => _unsafe_index
# Graph fragment:
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg3_1, [None, None, %unsqueeze, %convert_element_type_3]), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__unsafe_index_0(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // ks1) % ks2)
    x0 = (xindex % ks1)
    x2 = xindex // ks4
    x4 = xindex
    tmp0 = tl.full([1], 2.0, tl.float64)
    tmp1 = ks0
    tmp2 = tmp1.to(tl.float64)
    tmp3 = tmp0 * tmp2
    tmp4 = tmp2 / tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = x1
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp7 * tmp5
    tmp9 = tmp8.to(tl.int64)
    tmp10 = tmp9 + tmp1
    tmp11 = tmp9 < 0
    tmp12 = tl.where(tmp11, tmp10, tmp9)
    tmp13 = ks3
    tmp14 = tmp13.to(tl.float64)
    tmp15 = tmp0 * tmp14
    tmp16 = tmp14 / tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = x0
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp19 * tmp17
    tmp21 = tmp20.to(tl.int64)
    tmp22 = tmp21 + tmp13
    tmp23 = tmp21 < 0
    tmp24 = tl.where(tmp23, tmp22, tmp21)
    tmp25 = tl.load(in_ptr0 + (tmp24 + ks3*tmp12 + ks0*ks3*x2), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x4), tmp25, xmask)







def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        ps0 = 2*s2
        ps1 = 2*s1
        ps2 = 4*s1*s2
        buf0 = empty_strided_cuda((1, s0, 2*s1, 2*s2), (4*s0*s1*s2, 4*s1*s2, 2*s2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten._unsafe_index]
        triton_poi_fused__unsafe_index_0_xnumel = 4*s0*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_0[grid(triton_poi_fused__unsafe_index_0_xnumel)](arg3_1, buf0, 32, 64, 64, 32, 4096, 12288, XBLOCK=256, num_warps=4, num_stages=1)
        del arg3_1
    return (reinterpret_tensor(buf0, (1, 4*s1*s2, s0), (4*s0*s1*s2, 1, 4*s1*s2), 0), 2*s1, 2*s2, )


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
