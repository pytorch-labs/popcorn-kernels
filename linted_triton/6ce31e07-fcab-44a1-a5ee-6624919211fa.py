# AOT ID: ['171_inference']
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


# kernel path: /tmp/torchinductor_sahanp/yk/cyk6gewv2oimc2qddmy5dixctnovv5tyj5dakevkcuhpn2pzjgdg.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.copy]
# Source node to ATen node mapping:
#   x_2 => copy
# Graph fragment:
#   %copy : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_3, %slice_4), kwargs = {})
#   %slice_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_tensor, %copy, 2, 1, %sub_17), kwargs = {})
#   %slice_scatter_default_1 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%empty, %slice_scatter_default, 3, 1, %sub_9), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_copy_0(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks2)
    x2 = xindex // ks4
    x3 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = 5 + ks1
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = x1
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp6 >= tmp7
    tmp9 = tl.broadcast_to(5 + ks3, [XBLOCK])
    tmp10 = tmp6 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tmp11 & tmp5
    tmp13 = (-3) + x1
    tmp14 = tl.full([1], 0, tl.int64)
    tmp15 = tmp13 >= tmp14
    tmp16 = tl.broadcast_to(ks3, [XBLOCK])
    tmp17 = tmp13 < tmp16
    tmp18 = (-3) + x0
    tmp19 = tmp18 >= tmp14
    tmp20 = tl.broadcast_to(ks1, [XBLOCK])
    tmp21 = tmp18 < tmp20
    tmp22 = tmp15 & tmp17
    tmp23 = tmp22 & tmp19
    tmp24 = tmp23 & tmp21
    tmp25 = tmp24 & tmp12
    tmp26 = tl.load(in_ptr0 + ((-3) + x0 + ((-3)*ks1) + ks1*x1 + ks1*ks3*x2), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp27 = -1.0
    tmp28 = triton_helpers.maximum(tmp26, tmp27)
    tmp29 = 1.0
    tmp30 = triton_helpers.minimum(tmp28, tmp29)
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp12, tmp30, tmp31)
    tmp33 = tl.load(in_ptr1 + (x3), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.where(tmp11, tmp32, tmp33)
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp5, tmp34, tmp35)
    tmp37 = float("nan")
    tmp38 = tl.where(tmp5, tmp36, tmp37)
    tl.store(out_ptr0 + (x3), tmp38, xmask)




# kernel path: /tmp/torchinductor_sahanp/2g/c2glemp3uxvf5x57vzr3lf5w5h2tjdtwmfntr63ifnyjfhqcdisz.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %slice_scatter_default_2 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_1, %slice_11, 3, 0, 1), kwargs = {})
#   %slice_scatter_default_3 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_2, %slice_16, 3, %sub_47, %add_11), kwargs = {})
#   %slice_scatter_default_4 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_3, %slice_21, 2, 0, 1), kwargs = {})
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
    tmp39 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0
    tmp4 = tl.broadcast_to(5 + ks2, [XBLOCK])
    tmp5 = tmp3 >= tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = (-4) + x0 + ((-1)*ks2)
    tmp8 = tl.full([1], 1, tl.int64)
    tmp9 = tmp7 < tmp8
    tmp10 = tmp9 & tmp6
    tmp11 = tl.load(in_ptr0 + (28 + 5*ks2 + 6*ks3 + 6*x4 + ks2*ks3 + ks2*x4), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr0 + (20 + x3 + 3*ks2 + 6*ks3 + ks2*ks3), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.where(tmp9, tmp11, tmp12)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.full([1], 1, tl.int64)
    tmp17 = tmp3 < tmp16
    tmp18 = tmp17 & tmp2
    tmp19 = tl.load(in_ptr0 + (28 + 5*ks2 + 6*ks3 + 6*x4 + ks2*ks3 + ks2*x4), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.load(in_ptr0 + (24 + x3 + 4*ks2 + 6*ks3 + ks2*ks3), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.where(tmp17, tmp19, tmp20)
    tmp22 = tl.where(tmp5, tmp15, tmp21)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp2, tmp22, tmp23)
    tmp25 = x0
    tmp26 = 5 + ks2
    tmp27 = tmp25 >= tmp26
    tmp28 = (-4) + x0 + ((-1)*ks2)
    tmp29 = tl.full([1], 1, tl.int64)
    tmp30 = tmp28 < tmp29
    tmp31 = tmp30 & tmp27
    tmp32 = tl.load(in_ptr0 + (4 + ks2 + 6*x4 + ks2*x4), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tl.load(in_ptr0 + ((-4) + x3 + ((-1)*ks2)), tmp27 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.where(tmp30, tmp32, tmp33)
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp27, tmp34, tmp35)
    tmp37 = tmp25 < tmp1
    tmp38 = tl.load(in_ptr0 + (4 + ks2 + 6*x4 + ks2*x4), tmp37 & xmask, eviction_policy='evict_last', other=0.0)
    tmp40 = tl.where(tmp37, tmp38, tmp39)
    tmp41 = tl.where(tmp27, tmp36, tmp40)
    tmp42 = tl.where(tmp2, tmp24, tmp41)
    tl.store(out_ptr0 + (x3), tmp42, xmask)




# kernel path: /tmp/torchinductor_sahanp/xp/cxp3lsqm7k75rzgxqb5k7tz3kl7b3lorlf3garkf3o7trbraqj4b.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.pow]
# Source node to ATen node mapping:
#   x_3 => pow_1
# Graph fragment:
#   %slice_scatter_default_5 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_4, %slice_26, 2, %sub_71, %add_9), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%slice_scatter_default_5, 2.0), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_pow_2(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // ks0) % ks1)
    x0 = (xindex % ks0)
    x2 = xindex // ks3
    x3 = xindex
    tmp4 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = 5 + ks2
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (6 + ks4 + x0 + 36*x2 + 6*ks2*x2 + 6*ks4*x2 + ks2*ks4*x2), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = tmp5 * tmp5
    tl.store(out_ptr0 + (x3), tmp6, xmask)




# kernel path: /tmp/torchinductor_sahanp/3l/c3lynp2wuomokqjiiaa4bsktro4qflfihq24wtn2pyegelrcjyfw.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.pow, aten.avg_pool2d, aten.sign, aten.abs, aten.relu, aten.mul]
# Source node to ATen node mapping:
#   x_3 => abs_1, avg_pool2d, mul_146, mul_151, pow_1, pow_2, relu, sign
# Graph fragment:
#   %slice_scatter_default_5 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_4, %slice_26, 2, %sub_71, %add_9), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%slice_scatter_default_5, 2.0), kwargs = {})
#   %avg_pool2d : [num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%pow_1, [2, 2], [2, 2]), kwargs = {})
#   %sign : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%avg_pool2d,), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%avg_pool2d,), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%abs_1,), kwargs = {})
#   %mul_146 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign, %relu), kwargs = {})
#   %mul_151 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_146, 4), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_151, 0.5), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_abs_avg_pool2d_mul_pow_relu_sign_3(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks1)
    x2 = xindex // ks2
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 12*x1 + 36*x2 + 2*ks4*x1 + 6*ks3*x2 + 6*ks4*x2 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 12*x1 + 36*x2 + 2*ks4*x1 + 6*ks3*x2 + 6*ks4*x2 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (6 + ks4 + 2*x0 + 12*x1 + 36*x2 + 2*ks4*x1 + 6*ks3*x2 + 6*ks4*x2 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (7 + ks4 + 2*x0 + 12*x1 + 36*x2 + 2*ks4*x1 + 6*ks3*x2 + 6*ks4*x2 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tmp9 = tl.full([1], 0, tl.int32)
    tmp10 = tmp9 < tmp8
    tmp11 = tmp10.to(tl.int8)
    tmp12 = tmp8 < tmp9
    tmp13 = tmp12.to(tl.int8)
    tmp14 = tmp11 - tmp13
    tmp15 = tmp14.to(tmp8.dtype)
    tmp16 = tl_math.abs(tmp8)
    tmp17 = triton_helpers.maximum(tmp9, tmp16)
    tmp18 = tmp15 * tmp17
    tmp19 = 4.0
    tmp20 = tmp18 * tmp19
    tmp21 = libdevice.sqrt(tmp20)
    tl.store(out_ptr0 + (x3), tmp21, xmask)




# kernel path: /tmp/torchinductor_sahanp/4r/c4ra46tjhfsflvlxddoauwqosqs3izguhukfy2e3gjenaoielhx6.py
# Topologically Sorted Source Nodes: [x_3, x_4], Original ATen: [aten.pow, aten.avg_pool2d, aten.sign, aten.abs, aten.relu, aten.mul, aten.view]
# Source node to ATen node mapping:
#   x_3 => abs_1, avg_pool2d, mul_146, mul_151, pow_1, pow_2, relu, sign
#   x_4 => view
# Graph fragment:
#   %slice_scatter_default_5 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_4, %slice_26, 2, %sub_71, %add_9), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%slice_scatter_default_5, 2.0), kwargs = {})
#   %avg_pool2d : [num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%pow_1, [2, 2], [2, 2]), kwargs = {})
#   %sign : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%avg_pool2d,), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%avg_pool2d,), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%abs_1,), kwargs = {})
#   %mul_146 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign, %relu), kwargs = {})
#   %mul_151 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_146, 4), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_151, 0.5), kwargs = {})
#   %view : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%pow_2, [1, -1]), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_abs_avg_pool2d_mul_pow_relu_sign_view_4(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (3*(((x0 // ks0) % ks1)) + 9*(triton_helpers.div_floor_integer(x0,  9 + 3*(ks2 // 2) + 3*(ks3 // 2) + (ks2 // 2)*(ks3 // 2))) + (ks3 // 2)*(((x0 // ks0) % ks1)) + 3*(ks2 // 2)*(triton_helpers.div_floor_integer(x0,  9 + 3*(ks2 // 2) + 3*(ks3 // 2) + (ks2 // 2)*(ks3 // 2))) + 3*(ks3 // 2)*(triton_helpers.div_floor_integer(x0,  9 + 3*(ks2 // 2) + 3*(ks3 // 2) + (ks2 // 2)*(ks3 // 2))) + (ks2 // 2)*(ks3 // 2)*(triton_helpers.div_floor_integer(x0,  9 + 3*(ks2 // 2) + 3*(ks3 // 2) + (ks2 // 2)*(ks3 // 2))) + ((x0 % ks0))), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




#include "/tmp/torchinductor_sahanp/3b/c3bi5gk6mslf6u4iaqafhxm64z6u65e3eain4xlary5blqnvv6xx.h"
extern "C"  void kernel(int64_t* in_out_ptr0)
{
    {
        {
            {
                auto tmp0 = in_out_ptr0[static_cast<int64_t>(0L)];
                auto tmp1 = static_cast<int32_t>(0);
                auto tmp2 = static_cast<int64_t>(0);
                auto tmp3 = static_cast<int64_t>(10);
                auto tmp4 = randint64_cpu(tmp0, tmp1, tmp2, tmp3);
                in_out_ptr0[static_cast<int64_t>(0L)] = tmp4;
            }
        }
    }
}
''')





def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, s0, 6 + s1, 6 + s2), (36*s0 + 6*s0*s1 + 6*s0*s2 + s0*s1*s2, 36 + 6*s1 + 6*s2 + s1*s2, 6 + s2, 1), torch.float32)
        ps0 = 6 + s2
        ps1 = 6 + s1
        ps2 = 36 + 6*s1 + 6*s2 + s1*s2
        buf1 = empty_strided_cuda((1, s0, 6 + s1, 6 + s2), (36*s0 + 6*s0*s1 + 6*s0*s2 + s0*s1*s2, 36 + 6*s1 + 6*s2 + s1*s2, 6 + s2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.copy]
        triton_poi_fused_copy_0_xnumel = 36*s0 + 6*s0*s1 + 6*s0*s2 + s0*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused_copy_0[grid(triton_poi_fused_copy_0_xnumel)](arg3_1, buf0, buf1, 38, 32, 38, 32, 1444, 4332, XBLOCK=128, num_warps=4, num_stages=1)
        del arg3_1
        buf2 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1_xnumel = 36*s0 + 6*s0*s1 + 6*s0*s2 + s0*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused_1[grid(triton_poi_fused_1_xnumel)](buf1, buf2, 38, 38, 32, 32, 4332, XBLOCK=128, num_warps=4, num_stages=1)
        buf3 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.pow]
        triton_poi_fused_pow_2_xnumel = 36*s0 + 6*s0*s1 + 6*s0*s2 + s0*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused_pow_2[grid(triton_poi_fused_pow_2_xnumel)](buf2, buf3, 38, 38, 32, 1444, 32, 4332, XBLOCK=256, num_warps=4, num_stages=1)
        del buf2
        ps3 = 3 + (s2 // 2)
        ps4 = 3 + (s1 // 2)
        ps5 = 9 + 3*(s1 // 2) + 3*(s2 // 2) + (s1 // 2)*(s2 // 2)
        buf4 = empty_strided_cuda((1, s0, 3 + (s1 // 2), 3 + (s2 // 2)), (9*s0 + 3*s0*(s1 // 2) + 3*s0*(s2 // 2) + s0*(s1 // 2)*(s2 // 2), 9 + 3*(s1 // 2) + 3*(s2 // 2) + (s1 // 2)*(s2 // 2), 3 + (s2 // 2), 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.pow, aten.avg_pool2d, aten.sign, aten.abs, aten.relu, aten.mul]
        triton_poi_fused_abs_avg_pool2d_mul_pow_relu_sign_3_xnumel = 9*s0 + 3*s0*(s1 // 2) + 3*s0*(s2 // 2) + s0*(s1 // 2)*(s2 // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_abs_avg_pool2d_mul_pow_relu_sign_3[grid(triton_poi_fused_abs_avg_pool2d_mul_pow_relu_sign_3_xnumel)](buf3, buf4, 19, 19, 361, 32, 32, 1083, XBLOCK=128, num_warps=4, num_stages=1)
        del buf3
        buf5 = empty_strided_cuda((1, 9*s0 + 3*s0*(s1 // 2) + 3*s0*(s2 // 2) + s0*(s1 // 2)*(s2 // 2)), (9*s0 + 3*s0*(s1 // 2) + 3*s0*(s2 // 2) + s0*(s1 // 2)*(s2 // 2), 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_3, x_4], Original ATen: [aten.pow, aten.avg_pool2d, aten.sign, aten.abs, aten.relu, aten.mul, aten.view]
        triton_poi_fused_abs_avg_pool2d_mul_pow_relu_sign_view_4_xnumel = 9*s0 + 3*s0*(s1 // 2) + 3*s0*(s2 // 2) + s0*(s1 // 2)*(s2 // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_abs_avg_pool2d_mul_pow_relu_sign_view_4[grid(triton_poi_fused_abs_avg_pool2d_mul_pow_relu_sign_view_4_xnumel)](buf4, buf5, 19, 19, 32, 32, 1083, XBLOCK=128, num_warps=4, num_stages=1)
        del buf4
    buf6 = empty_strided_cpu((1, ), (1, ), torch.int64)
    # Topologically Sorted Source Nodes: [], Original ATen: []
    aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf6)
    buf7 = buf6; del buf6  # reuse
    cpp_fused_randint_5(buf7)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf8 = empty_strided_cuda((1, ), (1, ), torch.int64)
        buf8.copy_(buf7, False)
        del buf7
    return (buf5, buf8, )


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
