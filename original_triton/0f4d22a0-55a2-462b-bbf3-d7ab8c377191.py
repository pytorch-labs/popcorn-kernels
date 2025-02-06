# AOT ID: ['45_inference']
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


# kernel path: /tmp/torchinductor_sahanp/mb/cmbqoumm7qgs44kcowdrw52yyduuezduu5gc4sjlj7yao4uzgj6c.py
# Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.constant_pad_nd, aten.abs, aten.add, aten.div, aten.le, aten.scalar_tensor, aten.where]
# Source node to ATen node mapping:
#   x => constant_pad_nd
#   x_1 => abs_1, add_8, div
#   x_2 => abs_2, full_default, le, where
# Graph fragment:
#   %constant_pad_nd : [num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%arg3_1, [2, 2, 2, 2], 3.0), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%constant_pad_nd,), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%abs_1, 1), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%constant_pad_nd, %add_8), kwargs = {})
#   %abs_2 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%div,), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%abs_2, 0.5), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le, %full_default, %div), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_abs_add_constant_pad_nd_div_le_scalar_tensor_where_0(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // ks0) % ks1)
    x0 = (xindex % ks0)
    x2 = xindex // ks4
    x4 = xindex
    tmp0 = (-2) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = ks2
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + x0
    tmp6 = tmp5 >= tmp1
    tmp7 = ks3
    tmp8 = tmp5 < tmp7
    tmp9 = tmp2 & tmp4
    tmp10 = tmp9 & tmp6
    tmp11 = tmp10 & tmp8
    tmp12 = tl.load(in_ptr0 + ((-2) + x0 + ((-2)*ks3) + ks3*x1 + ks2*ks3*x2), tmp11 & xmask, eviction_policy='evict_last', other=3.0)
    tmp13 = tl_math.abs(tmp12)
    tmp14 = 1.0
    tmp15 = tmp13 + tmp14
    tmp16 = tmp12 / tmp15
    tmp17 = tl_math.abs(tmp16)
    tmp18 = 0.5
    tmp19 = tmp17 <= tmp18
    tmp20 = 0.0
    tmp21 = tl.where(tmp19, tmp20, tmp16)
    tl.store(out_ptr0 + (x4), tmp21, xmask)




# kernel path: /tmp/torchinductor_sahanp/vj/cvjexqvdsmeuwa2mf6hadqyxoap6etbm53dgr47e75qhzp55kcm6.py
# Topologically Sorted Source Nodes: [x_4], Original ATen: [aten._adaptive_avg_pool2d]
# Source node to ATen node mapping:
#   x_4 => _adaptive_avg_pool2d
# Graph fragment:
#   %_adaptive_avg_pool2d : [num_users=1] = call_function[target=torch.ops.aten._adaptive_avg_pool2d.default](args = (%unsqueeze, [1, 10]), kwargs = {})
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
    tmp0 = tl.load(in_ptr0 + (4*(x0 // ks1) + 16*x1 + ks3*(x0 // ks1) + 4*ks2*x1 + 4*ks3*x1 + ks2*ks3*x1 + ((x0 % ks1))), xmask, eviction_policy='evict_last')
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
        ps0 = 4 + s2
        ps1 = 4 + s1
        ps2 = 16 + 4*s1 + 4*s2 + s1*s2
        buf0 = empty_strided_cuda((1, s0, 4 + s1, 4 + s2), (16*s0 + 4*s0*s1 + 4*s0*s2 + s0*s1*s2, 16 + 4*s1 + 4*s2 + s1*s2, 4 + s2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.constant_pad_nd, aten.abs, aten.add, aten.div, aten.le, aten.scalar_tensor, aten.where]
        triton_poi_fused_abs_add_constant_pad_nd_div_le_scalar_tensor_where_0_xnumel = 16*s0 + 4*s0*s1 + 4*s0*s2 + s0*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused_abs_add_constant_pad_nd_div_le_scalar_tensor_where_0[grid(triton_poi_fused_abs_add_constant_pad_nd_div_le_scalar_tensor_where_0_xnumel)](arg3_1, buf0, 32, 32, 28, 28, 1024, 3072, XBLOCK=256, num_warps=4, num_stages=1)
        del arg3_1
        ps3 = 16 + 4*s1 + 4*s2 + s1*s2
        buf1 = empty_strided_cuda((1, s0, 1, 16 + 4*s1 + 4*s2 + s1*s2), (16*s0 + 4*s0*s1 + 4*s0*s2 + s0*s1*s2, 16 + 4*s1 + 4*s2 + s1*s2, 16*s0 + 4*s0*s1 + 4*s0*s2 + s0*s1*s2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten._adaptive_avg_pool2d]
        triton_poi_fused__adaptive_avg_pool2d_1_xnumel = 16*s0 + 4*s0*s1 + 4*s0*s2 + s0*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused__adaptive_avg_pool2d_1[grid(triton_poi_fused__adaptive_avg_pool2d_1_xnumel)](buf0, buf1, 1024, 32, 28, 28, 3072, XBLOCK=128, num_warps=4, num_stages=1)
        del buf0
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten._adaptive_avg_pool2d]
        buf2 = torch.ops.aten._adaptive_avg_pool2d.default(buf1, [1, 10])
        del buf1
        buf3 = buf2
        del buf2
    return (reinterpret_tensor(buf3, (1, s0, 10), (10*s0, 10, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 3
    arg1_1 = 28
    arg2_1 = 28
    arg3_1 = rand_strided((1, 3, 28, 28), (2352, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
