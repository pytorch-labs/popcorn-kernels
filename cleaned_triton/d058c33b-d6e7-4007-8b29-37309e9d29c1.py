# AOT ID: ['149_inference']
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


# kernel path: /tmp/torchinductor_sahanp/5x/c5xv7is7efqicdttlumlmu7f6mmcvemjpyhxzkxxsdw7rr65uefs.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   x => var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_0(in_ptr0, out_ptr0, out_ptr1, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 3
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + ks0*ks1*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(r0_mask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(r0_mask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(r0_mask & xmask, tmp2_weight_next, tmp2_weight)
    tmp5, tmp6, tmp7 = triton_helpers.welford(tmp2_mean, tmp2_m2, tmp2_weight, 1)
    tmp2 = tmp5[:, None]
    tmp3 = tmp6[:, None]
    tmp4 = tmp7[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tl.store(out_ptr1 + (x0), tmp3, xmask)




# kernel path: /tmp/torchinductor_sahanp/bq/cbqdmqcmirv567ynw7luc7dm2nhkpvfgrvyqjus3opua7fa2gf2l.py
# Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten._native_batch_norm_legit, aten.view, aten.replication_pad2d]
# Source node to ATen node mapping:
#   x => add_4, mul_11, rsqrt, sub_2, var_mean, view_1
#   x_1 => _unsafe_index, _unsafe_index_1
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %getitem_1), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt), kwargs = {})
#   %view_1 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_11, [1, 3, %arg0_1, %arg1_1]), kwargs = {})
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%view_1, [None, None, %clamp_max, None]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index, [None, None, None, %clamp_max_1]), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_replication_pad2d_view_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks1)
    x2 = xindex // ks2
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (ks4*(((-1) + ks3) * (((-1) + ks3) <= (((0) * ((0) >= ((-2) + x1)) + ((-2) + x1) * (((-2) + x1) > (0))))) + (((0) * ((0) >= ((-2) + x1)) + ((-2) + x1) * (((-2) + x1) > (0)))) * ((((0) * ((0) >= ((-2) + x1)) + ((-2) + x1) * (((-2) + x1) > (0)))) < ((-1) + ks3))) + ks3*ks4*x2 + (((-1) + ks4) * (((-1) + ks4) <= (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0))))) + (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) * ((((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) < ((-1) + ks4)))), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = ks3*ks4
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp2 * tmp9
    tl.store(out_ptr0 + (x3), tmp10, xmask)




# kernel path: /tmp/torchinductor_sahanp/zh/czhktruck5ervyvrr6zwegxieil7ckfbbiqzp6zgoqtt4w5emwts.py
# Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.pixel_unshuffle]
# Source node to ATen node mapping:
#   x_4 => clone
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
def triton_poi_fused_pixel_unshuffle_2(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks1)
    x2 = xindex // ks2
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 8*x1 + 16*x2 + 2*ks4*x1 + 4*ks3*x2 + 4*ks4*x2 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 8*x1 + 16*x2 + 2*ks4*x1 + 4*ks3*x2 + 4*ks4*x2 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (4 + ks4 + 2*x0 + 8*x1 + 16*x2 + 2*ks4*x1 + 4*ks3*x2 + 4*ks4*x2 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (5 + ks4 + 2*x0 + 8*x1 + 16*x2 + 2*ks4*x1 + 4*ks3*x2 + 4*ks4*x2 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = 0.7071067811865476
    tmp10 = tmp6 * tmp9
    tmp11 = libdevice.erf(tmp10)
    tmp12 = 1.0
    tmp13 = tmp11 + tmp12
    tmp14 = tmp8 * tmp13
    tl.store(out_ptr0 + (x3), tmp14, xmask)




# kernel path: /tmp/torchinductor_sahanp/oa/coayvwkhlgoqd276b2ncuh5srjfdgandbi6k4fwtrcye33sn6e5f.py
# Topologically Sorted Source Nodes: [x_6], Original ATen: [aten._adaptive_avg_pool2d]
# Source node to ATen node mapping:
#   x_6 => _adaptive_avg_pool2d
# Graph fragment:
#   %_adaptive_avg_pool2d : [num_users=1] = call_function[target=torch.ops.aten._adaptive_avg_pool2d.default](args = (%unsqueeze, [1, 128]), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__adaptive_avg_pool2d_3(in_ptr0, out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 128)
    x1 = xindex // 128
    x2 = xindex
    tmp0 = tl.full([1], 0, tl.int64)
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = (17*x0) // 128
    tmp4 = (144 + 17*x0) // 128
    tmp5 = tmp3 < tmp4
    tmp6 = tmp2 & tmp5
    tmp7 = tl.load(in_ptr0 + (2*((17*x0) // 128) + 4*((x1 % (1 + (ks0 // 4)))) + 2*(ks1 // 4)*(triton_helpers.div_floor_integer(((x1 // (1 + (ks0 // 4))) % 12),  2)) + 4*(ks0 // 4)*(triton_helpers.div_floor_integer(((x1 // (1 + (ks0 // 4))) % 12),  4)) + 4*(ks1 // 4)*((x1 % (1 + (ks0 // 4)))) + 4*(ks0 // 4)*(ks1 // 4)*(triton_helpers.div_floor_integer(((x1 // (1 + (ks0 // 4))) % 12),  4)) + (((x1 // (1 + (ks0 // 4))) % 12))), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = 1 + ((17*x0) // 128)
    tmp9 = tmp8 < tmp4
    tmp10 = tmp2 & tmp9
    tmp11 = tl.load(in_ptr0 + (2 + 2*((17*x0) // 128) + 4*((x1 % (1 + (ks0 // 4)))) + 2*(ks1 // 4)*(triton_helpers.div_floor_integer(((x1 // (1 + (ks0 // 4))) % 12),  2)) + 4*(ks0 // 4)*(triton_helpers.div_floor_integer(((x1 // (1 + (ks0 // 4))) % 12),  4)) + 4*(ks1 // 4)*((x1 % (1 + (ks0 // 4)))) + 4*(ks0 // 4)*(ks1 // 4)*(triton_helpers.div_floor_integer(((x1 // (1 + (ks0 // 4))) % 12),  4)) + (((x1 // (1 + (ks0 // 4))) % 12))), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 + tmp7
    tmp13 = 1.0
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = 1.0
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp10, tmp16, tmp17)
    tmp19 = tmp18 + tmp15
    tmp20 = tmp12 / tmp19
    tl.store(out_ptr0 + (x2), tmp20, xmask)







def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    s1 = arg0_1
    s2 = arg1_1
    assert_size_stride(arg2_1, (1, 3, s1, s2), (3*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 3, 1, 1), (3, 1, 3, 3), torch.float32)
        buf1 = empty_strided_cuda((1, 3, 1, 1), (3, 1, 3, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten._native_batch_norm_legit]
        triton_red_fused__native_batch_norm_legit_0_r0_numel = s1*s2
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_0[grid(3)](arg2_1, buf0, buf1, 64, 64, 3, 4096, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        ps0 = 4 + s2
        ps1 = 4 + s1
        ps2 = 16 + 4*s1 + 4*s2 + s1*s2
        buf3 = empty_strided_cuda((1, 3, 4 + s1, 4 + s2), (48 + 12*s1 + 12*s2 + 3*s1*s2, 16 + 4*s1 + 4*s2 + s1*s2, 4 + s2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten._native_batch_norm_legit, aten.view, aten.replication_pad2d]
        triton_poi_fused__native_batch_norm_legit_replication_pad2d_view_1_xnumel = 48 + 12*s1 + 12*s2 + 3*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_replication_pad2d_view_1[grid(triton_poi_fused__native_batch_norm_legit_replication_pad2d_view_1_xnumel)](arg2_1, buf0, buf1, buf3, 68, 68, 4624, 64, 64, 13872, XBLOCK=256, num_warps=4, num_stages=1)
        del arg2_1
        del buf0
        del buf1
        ps3 = 2 + 2*(s2 // 4)
        ps4 = 2 + 2*(s1 // 4)
        ps5 = 4 + 4*(s1 // 4) + 4*(s2 // 4) + 4*(s1 // 4)*(s2 // 4)
        buf4 = empty_strided_cuda((1, 3, 2, 2, 1 + (s1 // 4), 1 + (s2 // 4)), (12 + 12*(s1 // 4) + 12*(s2 // 4) + 12*(s1 // 4)*(s2 // 4), 4 + 4*(s1 // 4) + 4*(s2 // 4) + 4*(s1 // 4)*(s2 // 4), 2 + 2*(s2 // 4), 1, 4 + 4*(s2 // 4), 2), torch.float32)
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.pixel_unshuffle]
        triton_poi_fused_pixel_unshuffle_2_xnumel = 12 + 12*(s1 // 4) + 12*(s2 // 4) + 12*(s1 // 4)*(s2 // 4)
        stream0 = get_raw_stream(0)
        triton_poi_fused_pixel_unshuffle_2[grid(triton_poi_fused_pixel_unshuffle_2_xnumel)](buf3, buf4, 34, 34, 1156, 64, 64, 3468, XBLOCK=256, num_warps=4, num_stages=1)
        del buf3
        buf5 = empty_strided_cuda((1, 3*((2 + (s1 // 2)) // (1 + (s1 // 4)))*((2 + (s2 // 2)) // (1 + (s2 // 4))) + 3*(s1 // 4)*((2 + (s1 // 2)) // (1 + (s1 // 4)))*((2 + (s2 // 2)) // (1 + (s2 // 4))), 1, 128), (384*((2 + (s1 // 2)) // (1 + (s1 // 4)))*((2 + (s2 // 2)) // (1 + (s2 // 4))) + 384*(s1 // 4)*((2 + (s1 // 2)) // (1 + (s1 // 4)))*((2 + (s2 // 2)) // (1 + (s2 // 4))), 128, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten._adaptive_avg_pool2d]
        triton_poi_fused__adaptive_avg_pool2d_3_xnumel = 384*((2 + (s1 // 2)) // (1 + (s1 // 4)))*((2 + (s2 // 2)) // (1 + (s2 // 4))) + 384*(s1 // 4)*((2 + (s1 // 2)) // (1 + (s1 // 4)))*((2 + (s2 // 2)) // (1 + (s2 // 4)))
        stream0 = get_raw_stream(0)
        triton_poi_fused__adaptive_avg_pool2d_3[grid(triton_poi_fused__adaptive_avg_pool2d_3_xnumel)](buf4, buf5, 64, 64, 26112, XBLOCK=128, num_warps=4, num_stages=1)
        del buf4
    return (reinterpret_tensor(buf5, (1, 3*((2 + (s1 // 2)) // (1 + (s1 // 4)))*((2 + (s2 // 2)) // (1 + (s2 // 4))), 1 + (s1 // 4), 128), (384*((2 + (s1 // 2)) // (1 + (s1 // 4)))*((2 + (s2 // 2)) // (1 + (s2 // 4))) + 384*(s1 // 4)*((2 + (s1 // 2)) // (1 + (s1 // 4)))*((2 + (s2 // 2)) // (1 + (s2 // 4))), 128 + 128*(s1 // 4), 128, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 64
    arg1_1 = 64
    arg2_1 = rand_strided((1, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
