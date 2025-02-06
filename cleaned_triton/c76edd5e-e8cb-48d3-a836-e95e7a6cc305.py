# AOT ID: ['117_inference']
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


# kernel path: /tmp/torchinductor_sahanp/pf/cpfq2wfozlcgwnqrwmmtzhbikg5unjz4lzw7zsaay5tfe3i5pbbr.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.reflection_pad3d]
# Source node to ATen node mapping:
#   x => _unsafe_index, _unsafe_index_1, _unsafe_index_2
# Graph fragment:
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg4_1, [None, None, %sub_5, None, None]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index, [None, None, None, %sub_11, None]), kwargs = {})
#   %_unsafe_index_2 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_1, [None, None, None, None, %sub_17]), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
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




# kernel path: /tmp/torchinductor_sahanp/hg/chgebzfzzp7jzzt33lkkpbob5z24fz2f4usarupdcmyklqzamcaq.py
# Topologically Sorted Source Nodes: [target], Original ATen: [aten.randint]
# Source node to ATen node mapping:
#   target => inductor_lookup_seed_default, inductor_randint_default
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_randint_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_randint.default](args = (0, 10, [1], %inductor_lookup_seed_default), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_randint_1(in_out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_out_ptr0 + load_seed_offset)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 10, tl.int64)
    tmp4 = triton_helpers.randint64(tmp0, (tmp1).to(tl.uint32), tmp2, tmp3)
    tl.store(in_out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp4, None)







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
        ps0 = 2 + s3
        ps1 = 2 + s2
        ps2 = 4 + 2*s2 + 2*s3 + s2*s3
        ps3 = 2 + s1
        ps4 = 8 + 4*s1 + 4*s2 + 4*s3 + 2*s1*s2 + 2*s1*s3 + 2*s2*s3 + s1*s2*s3
        buf0 = empty_strided_cuda((1, s0, 2 + s1, 2 + s2, 2 + s3), (8*s0 + 4*s0*s1 + 4*s0*s2 + 4*s0*s3 + 2*s0*s1*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + s0*s1*s2*s3, 8 + 4*s1 + 4*s2 + 4*s3 + 2*s1*s2 + 2*s1*s3 + 2*s2*s3 + s1*s2*s3, 4 + 2*s2 + 2*s3 + s2*s3, 2 + s3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.reflection_pad3d]
        triton_poi_fused_reflection_pad3d_0_xnumel = 8*s0 + 4*s0*s1 + 4*s0*s2 + 4*s0*s3 + 2*s0*s1*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + s0*s1*s2*s3
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad3d_0[grid(triton_poi_fused_reflection_pad3d_0_xnumel)](arg4_1, buf0, 34, 34, 1156, 34, 39304, 32, 32, 32, 117912, XBLOCK=512, num_warps=8, num_stages=1)
        del arg4_1
        # Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.reflection_pad3d, aten.max_pool3d_with_indices]
        buf1 = torch.ops.aten.max_pool3d_with_indices.default(buf0, [2, 2, 2], [2, 2, 2])
        del buf0
        buf2 = buf1[0]
        del buf1
        buf4 = empty_strided_cuda((1, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf4)
        buf5 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [target], Original ATen: [aten.randint]
        stream0 = get_raw_stream(0)
        triton_poi_fused_randint_1[grid(1)](buf5, 0, 1, XBLOCK=1, num_warps=1, num_stages=1)
    return (reinterpret_tensor(buf2, (1, 128), (s0 + s0*(s1 // 2) + s0*(s2 // 2) + s0*(s3 // 2) + s0*(s1 // 2)*(s2 // 2) + s0*(s1 // 2)*(s3 // 2) + s0*(s2 // 2)*(s3 // 2) + s0*(s1 // 2)*(s2 // 2)*(s3 // 2), 1), 0), buf5, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 3
    arg1_1 = 32
    arg2_1 = 32
    arg3_1 = 32
    arg4_1 = rand_strided((1, 3, 32, 32, 32), (98304, 32768, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
