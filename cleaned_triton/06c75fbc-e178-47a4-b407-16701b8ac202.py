# AOT ID: ['21_inference']
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


# kernel path: /tmp/torchinductor_sahanp/27/c27asyo4slytdszo2ufvx2ccsmmptzqmu7thcvoermdyswkcdhiz.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.copy]
# Source node to ATen node mapping:
#   x => copy
# Graph fragment:
#   %copy : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_5, %slice_6), kwargs = {})
#   %slice_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_tensor_1, %copy, 2, 1, %add), kwargs = {})
#   %slice_scatter_default_1 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_tensor, %slice_scatter_default, 3, 1, %add_2), kwargs = {})
#   %slice_scatter_default_2 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%empty, %slice_scatter_default_1, 4, 1, %add_4), kwargs = {})
#   %slice_scatter_default_3 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_2, %slice_15, 4, 0, 1), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_copy_0(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, ks6, ks7, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks2)
    x2 = ((xindex // ks4) % ks5)
    x3 = xindex // ks7
    x5 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = ks1 + x0
    tmp4 = tl.full([1], 1, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tl.broadcast_to(1 + ks1, [XBLOCK])
    tmp7 = tmp3 < tmp6
    tmp8 = tmp5 & tmp7
    tmp9 = tmp8 & tmp2
    tmp10 = x1
    tmp11 = tl.full([1], 1, tl.int64)
    tmp12 = tmp10 >= tmp11
    tmp13 = tl.broadcast_to(1 + ks3, [XBLOCK])
    tmp14 = tmp10 < tmp13
    tmp15 = tmp12 & tmp14
    tmp16 = tmp15 & tmp9
    tmp17 = x2
    tmp18 = tl.full([1], 1, tl.int64)
    tmp19 = tmp17 >= tmp18
    tmp20 = tl.broadcast_to(1 + ks6, [XBLOCK])
    tmp21 = tmp17 < tmp20
    tmp22 = tmp19 & tmp21
    tmp23 = tmp22 & tmp16
    tmp24 = tl.load(in_ptr0 + ((-1) + x0 + ks1*x1 + ((-1)*ks1*ks3) + ks1*ks3*x2 + ks1*ks3*ks6*x3), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tl.load(in_ptr1 + (ks1 + x5), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.where(tmp22, tmp24, tmp25)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp16, tmp26, tmp27)
    tmp29 = tl.load(in_ptr1 + (ks1 + x5), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.where(tmp15, tmp28, tmp29)
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp9, tmp30, tmp31)
    tmp33 = float("nan")
    tmp34 = tl.where(tmp8, tmp32, tmp33)
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp2, tmp34, tmp35)
    tmp37 = tmp0 >= tmp1
    tmp38 = 1 + ks1
    tmp39 = tmp0 < tmp38
    tmp40 = tmp37 & tmp39
    tmp41 = x1
    tmp42 = tl.full([1], 1, tl.int64)
    tmp43 = tmp41 >= tmp42
    tmp44 = tl.broadcast_to(1 + ks3, [XBLOCK])
    tmp45 = tmp41 < tmp44
    tmp46 = tmp43 & tmp45
    tmp47 = tmp46 & tmp40
    tmp48 = x2
    tmp49 = tl.full([1], 1, tl.int64)
    tmp50 = tmp48 >= tmp49
    tmp51 = tl.broadcast_to(1 + ks6, [XBLOCK])
    tmp52 = tmp48 < tmp51
    tmp53 = tmp50 & tmp52
    tmp54 = tmp53 & tmp47
    tmp55 = tl.load(in_ptr0 + ((-1) + x0 + ((-1)*ks1) + ks1*x1 + ((-1)*ks1*ks3) + ks1*ks3*x2 + ks1*ks3*ks6*x3), tmp54 & xmask, eviction_policy='evict_last', other=0.0)
    tmp56 = tl.load(in_ptr1 + (x5), tmp47 & xmask, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.where(tmp53, tmp55, tmp56)
    tmp58 = tl.full(tmp57.shape, 0.0, tmp57.dtype)
    tmp59 = tl.where(tmp47, tmp57, tmp58)
    tmp60 = tl.load(in_ptr1 + (x5), tmp40 & xmask, eviction_policy='evict_last', other=0.0)
    tmp61 = tl.where(tmp46, tmp59, tmp60)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp40, tmp61, tmp62)
    tmp64 = float("nan")
    tmp65 = tl.where(tmp40, tmp63, tmp64)
    tmp66 = tl.where(tmp2, tmp36, tmp65)
    tl.store(out_ptr0 + (x5), tmp66, xmask)




# kernel path: /tmp/torchinductor_sahanp/rq/crqgbwcmhj47pblbiwpip565md4kt2yigqjhghc5oqxcri7m7ka5.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %slice_scatter_default_4 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_3, %slice_20, 4, %add_4, %add_5), kwargs = {})
#   %slice_scatter_default_5 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_4, %slice_25, 3, 0, 1), kwargs = {})
#   %slice_scatter_default_6 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_5, %slice_30, 3, %add_2, %add_3), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // ks0) % ks1)
    x0 = (xindex % ks0)
    x4 = xindex // ks0
    x3 = xindex
    tmp41 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = 1 + ks2
    tmp2 = tmp0 >= tmp1
    tmp3 = x1 + ((-1)*ks2)
    tmp4 = tl.full([1], 1, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = x0
    tmp8 = tl.broadcast_to(1 + ks3, [XBLOCK])
    tmp9 = tmp7 >= tmp8
    tmp10 = tmp9 & tmp6
    tmp11 = tl.load(in_ptr0 + (1 + 2*x4 + ks3*x4), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr0 + (x3), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.where(tmp9, tmp11, tmp12)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = x0
    tmp17 = tl.broadcast_to(1 + ks3, [XBLOCK])
    tmp18 = tmp16 >= tmp17
    tmp19 = tmp18 & tmp2
    tmp20 = tl.load(in_ptr0 + (1 + ((-2)*ks2) + 2*x4 + ks3*x4 + ((-1)*ks2*ks3)), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.load(in_ptr0 + (x3 + ((-2)*ks2) + ((-1)*ks2*ks3)), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.where(tmp18, tmp20, tmp21)
    tmp23 = tl.where(tmp5, tmp15, tmp22)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp2, tmp23, tmp24)
    tmp26 = tl.full([1], 1, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = x0
    tmp29 = tl.broadcast_to(1 + ks3, [XBLOCK])
    tmp30 = tmp28 >= tmp29
    tmp31 = tmp30 & tmp27
    tmp32 = tl.load(in_ptr0 + (1 + 2*ks2 + 2*x4 + ks2*ks3 + ks3*x4), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tl.load(in_ptr0 + (x3 + 2*ks2 + ks2*ks3), tmp27 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.where(tmp30, tmp32, tmp33)
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp27, tmp34, tmp35)
    tmp37 = x0
    tmp38 = 1 + ks3
    tmp39 = tmp37 >= tmp38
    tmp40 = tl.load(in_ptr0 + (1 + 2*x4 + ks3*x4), tmp39 & xmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.where(tmp39, tmp40, tmp41)
    tmp43 = tl.where(tmp27, tmp36, tmp42)
    tmp44 = tl.where(tmp2, tmp25, tmp43)
    tl.store(out_ptr0 + (x3), tmp44, xmask)




# kernel path: /tmp/torchinductor_sahanp/yf/cyf5x3ftose5yv2fsjd6t27cfmcs74vzsf5aemnso5uroackkiyx.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %slice_scatter_default_7 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_6, %slice_35, 2, 0, 1), kwargs = {})
#   %slice_scatter_default_8 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_7, %slice_40, 2, %add, %add_1), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_2(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, ks6, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // ks0) % ks1)
    x5 = ((xindex // ks3) % ks1)
    x4 = (xindex % ks3)
    x6 = xindex // ks4
    x3 = xindex
    tmp15 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = 1 + ks2
    tmp2 = tmp0 >= tmp1
    tmp3 = x5 + ((-1)*ks2)
    tmp4 = tl.full([1], 1, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + (x4 + 4*ks2 + 8*x6 + 2*ks2*ks5 + 2*ks2*ks6 + 4*ks2*x6 + 4*ks5*x6 + 4*ks6*x6 + ks2*ks5*ks6 + 2*ks2*ks5*x6 + 2*ks2*ks6*x6 + 2*ks5*ks6*x6 + ks2*ks5*ks6*x6), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr0 + (x3 + ((-4)*ks2) + ((-2)*ks2*ks5) + ((-2)*ks2*ks6) + ((-1)*ks2*ks5*ks6)), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.where(tmp5, tmp7, tmp8)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = tl.full([1], 1, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr0 + (x4 + 4*ks2 + 8*x6 + 2*ks2*ks5 + 2*ks2*ks6 + 4*ks2*x6 + 4*ks5*x6 + 4*ks6*x6 + ks2*ks5*ks6 + 2*ks2*ks5*x6 + 2*ks2*ks6*x6 + 2*ks5*ks6*x6 + ks2*ks5*ks6*x6), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.where(tmp13, tmp14, tmp15)
    tmp17 = tl.where(tmp2, tmp11, tmp16)
    tl.store(out_ptr0 + (x3), tmp17, xmask)




# kernel path: /tmp/torchinductor_sahanp/ls/clsozzjaomye6eiaawp5sv5qfpc3buxl7qgonpzd2ih4ygfzetga.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %slice_scatter_default_7 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_6, %slice_35, 2, 0, 1), kwargs = {})
#   %slice_scatter_default_8 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_7, %slice_40, 2, %add, %add_1), kwargs = {})
#   %view_1 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%slice_scatter_default_8, [-1, 10]), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_3(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, ks6, ks7, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 10)
    x1 = xindex // 10
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*((((x0 + 10*x1) // ks0) % ks1)) + 4*((((x0 + 10*x1) // ks3) % ks2)) + 8*((((x0 + 10*x1) // (8 + 4*ks5 + 4*ks6 + 4*ks7 + 2*ks5*ks6 + 2*ks5*ks7 + 2*ks6*ks7 + ks5*ks6*ks7)) % ks4)) + ks7*((((x0 + 10*x1) // ks0) % ks1)) + 2*ks6*((((x0 + 10*x1) // ks3) % ks2)) + 2*ks7*((((x0 + 10*x1) // ks3) % ks2)) + 4*ks5*((((x0 + 10*x1) // (8 + 4*ks5 + 4*ks6 + 4*ks7 + 2*ks5*ks6 + 2*ks5*ks7 + 2*ks6*ks7 + ks5*ks6*ks7)) % ks4)) + 4*ks6*((((x0 + 10*x1) // (8 + 4*ks5 + 4*ks6 + 4*ks7 + 2*ks5*ks6 + 2*ks5*ks7 + 2*ks6*ks7 + ks5*ks6*ks7)) % ks4)) + 4*ks7*((((x0 + 10*x1) // (8 + 4*ks5 + 4*ks6 + 4*ks7 + 2*ks5*ks6 + 2*ks5*ks7 + 2*ks6*ks7 + ks5*ks6*ks7)) % ks4)) + ks6*ks7*((((x0 + 10*x1) // ks3) % ks2)) + 2*ks5*ks6*((((x0 + 10*x1) // (8 + 4*ks5 + 4*ks6 + 4*ks7 + 2*ks5*ks6 + 2*ks5*ks7 + 2*ks6*ks7 + ks5*ks6*ks7)) % ks4)) + 2*ks5*ks7*((((x0 + 10*x1) // (8 + 4*ks5 + 4*ks6 + 4*ks7 + 2*ks5*ks6 + 2*ks5*ks7 + 2*ks6*ks7 + ks5*ks6*ks7)) % ks4)) + 2*ks6*ks7*((((x0 + 10*x1) // (8 + 4*ks5 + 4*ks6 + 4*ks7 + 2*ks5*ks6 + 2*ks5*ks7 + 2*ks6*ks7 + ks5*ks6*ks7)) % ks4)) + ks5*ks6*ks7*((((x0 + 10*x1) // (8 + 4*ks5 + 4*ks6 + 4*ks7 + 2*ks5*ks6 + 2*ks5*ks7 + 2*ks6*ks7 + ks5*ks6*ks7)) % ks4)) + (((x0 + 10*x1) % ks0))), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp0, xmask)




#include "/tmp/torchinductor_sahanp/3b/c3bi5gk6mslf6u4iaqafhxm64z6u65e3eain4xlary5blqnvv6xx.h"
extern "C"  void kernel(const int64_t* in_ptr0,
                       int64_t* out_ptr0,
                       const int64_t ks0,
                       const int64_t ks1,
                       const int64_t ks2,
                       const int64_t ks3)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(c10::div_floor_integer(static_cast<int64_t>(8L*ks0 + 4L*ks0*ks1 + 4L*ks0*ks2 + 4L*ks0*ks3 + 2L*ks0*ks1*ks2 + 2L*ks0*ks1*ks3 + 2L*ks0*ks2*ks3 + ks0*ks1*ks2*ks3), static_cast<int64_t>(10L))); x0+=static_cast<int64_t>(1L))
        {
            {
                {
                    auto tmp0 = in_ptr0[static_cast<int64_t>(0L)];
                    auto tmp1 = x0;
                    auto tmp2 = c10::convert<int32_t>(tmp1);
                    auto tmp3 = static_cast<int64_t>(0);
                    auto tmp4 = static_cast<int64_t>(5);
                    auto tmp5 = randint64_cpu(tmp0, tmp2, tmp3, tmp4);
                    out_ptr0[static_cast<int64_t>(x0)] = tmp5;
                }
            }
        }
    }
}
''')





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
        buf0 = empty_strided_cuda((1, s0, 2 + s1, 2 + s2, 2 + s3), (8*s0 + 4*s0*s1 + 4*s0*s2 + 4*s0*s3 + 2*s0*s1*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + s0*s1*s2*s3, 8 + 4*s1 + 4*s2 + 4*s3 + 2*s1*s2 + 2*s1*s3 + 2*s2*s3 + s1*s2*s3, 4 + 2*s2 + 2*s3 + s2*s3, 2 + s3, 1), torch.float32)
        ps0 = 2 + s3
        ps1 = 2 + s2
        ps2 = 4 + 2*s2 + 2*s3 + s2*s3
        ps3 = 2 + s1
        ps4 = 8 + 4*s1 + 4*s2 + 4*s3 + 2*s1*s2 + 2*s1*s3 + 2*s2*s3 + s1*s2*s3
        buf1 = empty_strided_cuda((1, s0, 2 + s1, 2 + s2, 2 + s3), (8*s0 + 4*s0*s1 + 4*s0*s2 + 4*s0*s3 + 2*s0*s1*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + s0*s1*s2*s3, 8 + 4*s1 + 4*s2 + 4*s3 + 2*s1*s2 + 2*s1*s3 + 2*s2*s3 + s1*s2*s3, 4 + 2*s2 + 2*s3 + s2*s3, 2 + s3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.copy]
        triton_poi_fused_copy_0_xnumel = 8*s0 + 4*s0*s1 + 4*s0*s2 + 4*s0*s3 + 2*s0*s1*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + s0*s1*s2*s3
        stream0 = get_raw_stream(0)
        triton_poi_fused_copy_0[grid(triton_poi_fused_copy_0_xnumel)](arg4_1, buf0, buf1, 7, 5, 7, 5, 49, 7, 5, 343, 3430, XBLOCK=128, num_warps=4, num_stages=1)
        del arg4_1
        buf2 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1_xnumel = 8*s0 + 4*s0*s1 + 4*s0*s2 + 4*s0*s3 + 2*s0*s1*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + s0*s1*s2*s3
        stream0 = get_raw_stream(0)
        triton_poi_fused_1[grid(triton_poi_fused_1_xnumel)](buf1, buf2, 7, 7, 5, 5, 3430, XBLOCK=128, num_warps=4, num_stages=1)
        ps5 = 4 + 2*s2 + 2*s3 + s2*s3
        ps6 = 8 + 4*s1 + 4*s2 + 4*s3 + 2*s1*s2 + 2*s1*s3 + 2*s2*s3 + s1*s2*s3
        buf3 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2_xnumel = 8*s0 + 4*s0*s1 + 4*s0*s2 + 4*s0*s3 + 2*s0*s1*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + s0*s1*s2*s3
        stream0 = get_raw_stream(0)
        triton_poi_fused_2[grid(triton_poi_fused_2_xnumel)](buf2, buf3, 49, 7, 5, 49, 343, 5, 5, 3430, XBLOCK=128, num_warps=4, num_stages=1)
        del buf2
        buf4 = empty_strided_cuda(((8*s0 + 4*s0*s1 + 4*s0*s2 + 4*s0*s3 + 2*s0*s1*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + s0*s1*s2*s3) // 10, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_3_xnumel = 10*((8*s0 + 4*s0*s1 + 4*s0*s2 + 4*s0*s3 + 2*s0*s1*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + s0*s1*s2*s3) // 10)
        stream0 = get_raw_stream(0)
        triton_poi_fused_3[grid(triton_poi_fused_3_xnumel)](buf3, buf4, 7, 7, 7, 49, 10, 5, 5, 5, 3430, XBLOCK=128, num_warps=4, num_stages=1)
        del buf3
    buf5 = empty_strided_cpu((1, ), (1, ), torch.int64)
    # Topologically Sorted Source Nodes: [], Original ATen: []
    aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf5)
    buf6 = empty_strided_cpu(((8*s0 + 4*s0*s1 + 4*s0*s2 + 4*s0*s3 + 2*s0*s1*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + s0*s1*s2*s3) // 10, ), (1, ), torch.int64)
    cpp_fused_randint_4(buf5, buf6, s0, s1, s2, s3)
    return (buf4, buf6, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 10
    arg1_1 = 5
    arg2_1 = 5
    arg3_1 = 5
    arg4_1 = rand_strided((1, 10, 5, 5, 5), (1250, 125, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
