# AOT ID: ['29_inference']
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


# kernel path: /tmp/torchinductor_sahanp/io/ciobeqqjhgbuputek3ocfvikybzv67p4od6tkzm3pnd2kamtjlwy.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.max_unpool2d]
# Source node to ATen node mapping:
#   x => full
# Graph fragment:
#   %full : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, %arg0_1, %sub_7, %sub_9], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_unpool2d_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/sp/cspceat5xy42wcs5bfvlgvfsag3eibj3zjru3woerinktrdpqrsp.py
# Topologically Sorted Source Nodes: [max_pool2d, x], Original ATen: [aten.max_pool2d_with_indices, aten.max_unpool2d]
# Source node to ATen node mapping:
#   max_pool2d => _low_memory_max_pool2d_offsets_to_indices, _low_memory_max_pool2d_with_offsets
#   x => add_11, index_put, mul_11
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%arg3_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %_low_memory_max_pool2d_offsets_to_indices : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_offsets_to_indices.default](args = (%getitem_1, 2, %arg2_1, [2, 2], [0, 0]), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, %mul_10), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_low_memory_max_pool2d_offsets_to_indices, %mul_11), kwargs = {})
#   %index_put : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%view_2, [%view_1], %view_3), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_max_unpool2d_1(in_ptr0, out_ptr1, ks0, ks1, ks2, ks3, ks4, ks5, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks1)
    x2 = xindex // ks2
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 2*ks4*x1 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 2*ks4*x1 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (ks4 + 2*x0 + 2*ks4*x1 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (1 + ks4 + 2*x0 + 2*ks4*x1 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr0 + (2*((x3 % ks0)) + 2*ks4*(((x3 // ks0) % ks1)) + ks3*ks4*(x3 // ks2)), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr0 + (1 + 2*((x3 % ks0)) + 2*ks4*(((x3 // ks0) % ks1)) + ks3*ks4*(x3 // ks2)), xmask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr0 + (ks4 + 2*((x3 % ks0)) + 2*ks4*(((x3 // ks0) % ks1)) + ks3*ks4*(x3 // ks2)), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr0 + (1 + ks4 + 2*((x3 % ks0)) + 2*ks4*(((x3 // ks0) % ks1)) + ks3*ks4*(x3 // ks2)), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 > tmp0
    tmp3 = tl.full([1], 1, tl.int8)
    tmp4 = tl.full([1], 0, tl.int8)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = triton_helpers.maximum(tmp1, tmp0)
    tmp8 = tmp7 > tmp6
    tmp9 = tl.full([1], 2, tl.int8)
    tmp10 = tl.where(tmp8, tmp9, tmp5)
    tmp11 = triton_helpers.maximum(tmp7, tmp6)
    tmp13 = tmp12 > tmp11
    tmp14 = tl.full([1], 3, tl.int8)
    tmp15 = tl.where(tmp13, tmp14, tmp10)
    tmp16 = triton_helpers.maximum(tmp12, tmp11)
    tmp17 = tl.full([1], 2, tl.int32)
    tmp18 = tl.where((tmp15 < 0) != (tmp17 < 0), tl.where(tmp15 % tmp17 != 0, tmp15 // tmp17 - 1, tmp15 // tmp17), tmp15 // tmp17)
    tmp19 = tmp18 * tmp17
    tmp20 = tmp15 - tmp19
    tmp21 = 2*x1
    tmp22 = tmp21 + tmp18
    tmp23 = 2*x0
    tmp24 = tmp23 + tmp20
    tmp25 = ks4
    tmp26 = tmp22 * tmp25
    tmp27 = tmp26 + tmp24
    tmp28 = 4*ks0*ks1*x2
    tmp29 = tmp27 + tmp28
    tmp30 = 4*ks0*ks1*ks5
    tmp31 = tmp29 + tmp30
    tmp32 = tmp29 < 0
    tmp33 = tl.where(tmp32, tmp31, tmp29)
    tl.device_assert(((0 <= tmp33) & (tmp33 < 4*ks5*(ks3 // 2)*(ks4 // 2))) | ~(xmask), "index out of bounds: 0 <= tmp33 < 4*ks5*(ks3 // 2)*(ks4 // 2)")
    tmp37 = triton_helpers.maximum(tmp36, tmp35)
    tmp39 = triton_helpers.maximum(tmp38, tmp37)
    tmp41 = triton_helpers.maximum(tmp40, tmp39)
    tl.store(out_ptr1 + (tl.broadcast_to((tmp33 % (4*ks0*ks1*ks5)), [XBLOCK])), tmp41, xmask)




# kernel path: /tmp/torchinductor_sahanp/p4/cp4ilzlyhrfssbox6rulz4b3f644x3ty7i6swozgql36wgzjfzog.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   x_1 => avg_pool2d
# Graph fragment:
#   %avg_pool2d : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%view_4, [2, 2], [2, 2]), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_avg_pool2d_2(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks1)
    x2 = xindex // ks2
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 2*ks0*((((x0 + 2*ks0*x1) // ks0) % (2*ks1))) + 4*ks0*ks1*((((x0 + 2*ks0*x1 + 2*ks0*ks1*x2) // (2*ks0*ks1)) % ks3))), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (2*ks0*((((1 + 2*x0 + 4*ks0*x1) // (2*ks0)) % (2*ks1))) + 4*ks0*ks1*((((1 + 2*x0 + 4*ks0*x1 + 4*ks0*ks1*x2) // (4*ks0*ks1)) % ks3)) + (((1 + 2*x0) % (2*ks0)))), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2*x0 + 2*ks0*((((ks0 + x0 + 2*ks0*x1) // ks0) % (2*ks1))) + 4*ks0*ks1*((((ks0 + x0 + 2*ks0*x1 + 2*ks0*ks1*x2) // (2*ks0*ks1)) % ks3))), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (2*ks0*((((1 + 2*ks0 + 2*x0 + 4*ks0*x1) // (2*ks0)) % (2*ks1))) + 4*ks0*ks1*((((1 + 2*ks0 + 2*x0 + 4*ks0*x1 + 4*ks0*ks1*x2) // (4*ks0*ks1)) % ks3)) + (((1 + 2*x0) % (2*ks0)))), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x3), tmp8, xmask)







def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((1, s0, 2*(s1 // 2), 2*(s2 // 2)), (4*s0*(s1 // 2)*(s2 // 2), 4*(s1 // 2)*(s2 // 2), 2*(s2 // 2), 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.max_unpool2d]
        triton_poi_fused_max_unpool2d_0_xnumel = 4*s0*(s1 // 2)*(s2 // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_unpool2d_0[grid(triton_poi_fused_max_unpool2d_0_xnumel)](buf1, 12288, XBLOCK=256, num_warps=4, num_stages=1)
        ps0 = s2 // 2
        ps1 = s1 // 2
        ps2 = (s1 // 2)*(s2 // 2)
        # Topologically Sorted Source Nodes: [max_pool2d, x], Original ATen: [aten.max_pool2d_with_indices, aten.max_unpool2d]
        triton_poi_fused_max_pool2d_with_indices_max_unpool2d_1_xnumel = s0*(s1 // 2)*(s2 // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_max_unpool2d_1[grid(triton_poi_fused_max_pool2d_with_indices_max_unpool2d_1_xnumel)](arg3_1, buf1, 32, 32, 1024, 64, 64, 3, 3072, XBLOCK=256, num_warps=4, num_stages=1)
        del arg3_1
        buf3 = empty_strided_cuda((1, s0, s1 // 2, s2 // 2), (s0*(s1 // 2)*(s2 // 2), (s1 // 2)*(s2 // 2), s2 // 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_2_xnumel = s0*(s1 // 2)*(s2 // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_2[grid(triton_poi_fused_avg_pool2d_2_xnumel)](buf1, buf3, 32, 32, 1024, 3, 3072, XBLOCK=128, num_warps=4, num_stages=1)
        del buf1
    return (reinterpret_tensor(buf3, (1, (s1 // 2)*(s2 // 2), s0), (s0*(s1 // 2)*(s2 // 2), 1, (s1 // 2)*(s2 // 2)), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 3
    arg1_1 = 64
    arg2_1 = 64
    arg3_1 = rand_strided((1, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
