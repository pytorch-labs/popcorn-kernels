# AOT ID: ['62_inference']
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


# kernel path: /tmp/torchinductor_sahanp/mn/cmn6m4gnumgcmdporvbosorymwqs5z7sipqwmgurtwss2awtlvut.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.max_unpool3d]
# Source node to ATen node mapping:
#   x_1 => full
# Graph fragment:
#   %full : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, %arg0_1, %sub_9, %sub_11, %sub_13], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_unpool3d_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/tl/ctl7itaug3wfurx2kzfadfzfl6egiolommo4ajny6ccajktmve5g.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.max_unpool3d]
# Source node to ATen node mapping:
#   x_1 => index_put
# Graph fragment:
#   %index_put : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%view_2, [%view_1], %view_3), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_unpool3d_1(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp8 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = 8*(ks0 // 2)*(ks1 // 2)*(ks2 // 2)*(triton_helpers.div_floor_integer(x0,  (ks0 // 2)*(ks1 // 2)*(ks2 // 2)))
    tmp2 = tmp0 + tmp1
    tmp3 = 8*ks3*(ks0 // 2)*(ks1 // 2)*(ks2 // 2)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp2 < 0
    tmp6 = tl.where(tmp5, tmp4, tmp2)
    tl.device_assert(((0 <= tmp6) & (tmp6 < 8*ks3*(ks0 // 2)*(ks1 // 2)*(ks2 // 2))) | ~(xmask), "index out of bounds: 0 <= tmp6 < 8*ks3*(ks0 // 2)*(ks1 // 2)*(ks2 // 2)")
    tl.store(out_ptr0 + (tl.broadcast_to((tmp6 % (8*ks3*(ks0 // 2)*(ks1 // 2)*(ks2 // 2))), [XBLOCK])), tmp8, xmask)




# kernel path: /tmp/torchinductor_sahanp/ss/cssfdt4gktl5a3iglqf64rzysvzipazoqn6qkqkdpvovzcybi5cf.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.transpose]
# Source node to ATen node mapping:
#   x_2 => permute
# Graph fragment:
#   %permute : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_5, [0, 2, 1]), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_transpose_2(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*(ks4 // 2)*((((2*(ks4 // 2)*(triton_helpers.div_floor_integer(x0,  2*(ks4 // 2))) + ((x0 % (2*(ks4 // 2))))) // (2*(ks4 // 2))) % (2*(ks3 // 2)))) + 4*(ks3 // 2)*(ks4 // 2)*((((2*(ks4 // 2)*(triton_helpers.div_floor_integer(x0,  2*(ks4 // 2))) + 4*(ks3 // 2)*(ks4 // 2)*((x1 % (2*(ks2 // 2)))) + ((x0 % (2*(ks4 // 2))))) // (4*(ks3 // 2)*(ks4 // 2))) % (2*(ks2 // 2)))) + 8*(ks2 // 2)*(ks3 // 2)*(ks4 // 2)*((((2*(ks4 // 2)*(triton_helpers.div_floor_integer(x0,  2*(ks4 // 2))) + 4*(ks3 // 2)*(ks4 // 2)*((x1 % (2*(ks2 // 2)))) + 8*(ks2 // 2)*(ks3 // 2)*(ks4 // 2)*(triton_helpers.div_floor_integer(x1,  2*(ks2 // 2))) + ((x0 % (2*(ks4 // 2))))) // (8*(ks2 // 2)*(ks3 // 2)*(ks4 // 2))) % ks1)) + ((((x0 % (2*(ks4 // 2)))) % (2*(ks4 // 2))))), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp0, xmask)







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
        # Topologically Sorted Source Nodes: [max_pool3d_with_indices], Original ATen: [aten.max_pool3d_with_indices]
        buf0 = torch.ops.aten.max_pool3d_with_indices.default(arg4_1, [2, 2, 2], [2, 2, 2])
        del arg4_1
        buf1 = buf0[0]
        buf2 = buf0[1]
        del buf0
        buf3 = empty_strided_cuda((1, s0, 2*(s1 // 2), 2*(s2 // 2), 2*(s3 // 2)), (8*s0*(s1 // 2)*(s2 // 2)*(s3 // 2), 8*(s1 // 2)*(s2 // 2)*(s3 // 2), 4*(s2 // 2)*(s3 // 2), 2*(s3 // 2), 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.max_unpool3d]
        triton_poi_fused_max_unpool3d_0_xnumel = 8*s0*(s1 // 2)*(s2 // 2)*(s3 // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_unpool3d_0[grid(triton_poi_fused_max_unpool3d_0_xnumel)](buf3, 98304, XBLOCK=1024, num_warps=4, num_stages=1)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.max_unpool3d]
        triton_poi_fused_max_unpool3d_1_xnumel = s0*(s1 // 2)*(s2 // 2)*(s3 // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_unpool3d_1[grid(triton_poi_fused_max_unpool3d_1_xnumel)](buf2, buf1, buf3, 32, 32, 32, 3, 12288, XBLOCK=128, num_warps=4, num_stages=1)
        del buf1
        del buf2
        ps0 = 4*(s2 // 2)*(s3 // 2)
        buf5 = empty_strided_cuda((1, 4*(s2 // 2)*(s3 // 2), 2*s0*(s1 // 2)), (8*s0*(s1 // 2)*(s2 // 2)*(s3 // 2), 1, 4*(s2 // 2)*(s3 // 2)), torch.float32)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.transpose]
        triton_poi_fused_transpose_2_xnumel = 8*s0*(s1 // 2)*(s2 // 2)*(s3 // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_transpose_2[grid(triton_poi_fused_transpose_2_xnumel)](buf3, buf5, 1024, 3, 32, 32, 32, 98304, XBLOCK=512, num_warps=8, num_stages=1)
        del buf3
    return (buf5, 2*(s1 // 2), 2*(s2 // 2), 2*(s3 // 2), )


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
