# AOT ID: ['14_forward']
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


# kernel path: /tmp/torchinductor_sahanp/ql/cqlnl7x3govvfwrq4msekgf5ytlqex7ijoyx7s5i55nuituq6czg.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.max_unpool3d]
# Source node to ATen node mapping:
#   x_1 => full
# Graph fragment:
#   %full : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 1, %sub_7, %sub_9, %sub_11], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
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




# kernel path: /tmp/torchinductor_sahanp/7c/c7c3e5mwt2xddhjg6bqo6b2xbwmvdv32i2r2mvlvf3ixonl4sxum.py
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
def triton_poi_fused_max_unpool3d_1(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp8 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 + tmp1
    tmp3 = 8*(ks0 // 2)*(ks1 // 2)*(ks2 // 2)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp2 < 0
    tmp6 = tl.where(tmp5, tmp4, tmp2)
    tl.device_assert(((0 <= tmp6) & (tmp6 < 8*(ks0 // 2)*(ks1 // 2)*(ks2 // 2))) | ~(xmask), "index out of bounds: 0 <= tmp6 < 8*(ks0 // 2)*(ks1 // 2)*(ks2 // 2)")
    tl.store(out_ptr0 + (tl.broadcast_to((tmp6 % (8*(ks0 // 2)*(ks1 // 2)*(ks2 // 2))), [XBLOCK])), tmp8, xmask)




# kernel path: /tmp/torchinductor_sahanp/zw/czwcssz2fivqxohahwquypew53n73hummoe6ibcjzpp6gfybipkx.py
# Topologically Sorted Source Nodes: [x_3, x_4], Original ATen: [aten.constant_pad_nd, aten.reflection_pad2d]
# Source node to ATen node mapping:
#   x_3 => constant_pad_nd
#   x_4 => _unsafe_index, _unsafe_index_1
# Graph fragment:
#   %constant_pad_nd : [num_users=3] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_5, [1, 1, 1, 1], 0.0), kwargs = {})
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%constant_pad_nd, [None, None, %sub_24, None]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index, [None, None, None, %sub_30]), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_constant_pad_nd_reflection_pad2d_2(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // ks0
    x0 = (xindex % ks0)
    x2 = xindex
    tmp0 = (-1) + (tl.where(1 + ((-1)*tl_math.abs(1 + ((-1)*tl_math.abs((-1) + x1)) + 2*(ks1 // 2))) + 2*(ks1 // 2) < 0, 3 + ((-1)*tl_math.abs(1 + ((-1)*tl_math.abs((-1) + x1)) + 2*(ks1 // 2))) + 4*(ks1 // 2), 1 + ((-1)*tl_math.abs(1 + ((-1)*tl_math.abs((-1) + x1)) + 2*(ks1 // 2))) + 2*(ks1 // 2)))
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = 2*(ks1 // 2)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + (tl.where(1 + ((-1)*tl_math.abs(1 + ((-1)*tl_math.abs((-1) + x0)) + 4*(ks2 // 2)*(ks3 // 2))) + 4*(ks2 // 2)*(ks3 // 2) < 0, 3 + ((-1)*tl_math.abs(1 + ((-1)*tl_math.abs((-1) + x0)) + 4*(ks2 // 2)*(ks3 // 2))) + 8*(ks2 // 2)*(ks3 // 2), 1 + ((-1)*tl_math.abs(1 + ((-1)*tl_math.abs((-1) + x0)) + 4*(ks2 // 2)*(ks3 // 2))) + 4*(ks2 // 2)*(ks3 // 2)))
    tmp6 = tmp5 >= tmp1
    tmp7 = 4*(ks2 // 2)*(ks3 // 2)
    tmp8 = tmp5 < tmp7
    tmp9 = tmp2 & tmp4
    tmp10 = tmp9 & tmp6
    tmp11 = tmp10 & tmp8
    tmp12 = tl.load(in_ptr0 + (2*(ks3 // 2)*((((2*(ks3 // 2)*(((((-1) + (tl.where(1 + ((-1)*tl_math.abs(1 + ((-1)*tl_math.abs((-1) + x0)) + 4*(ks2 // 2)*(ks3 // 2))) + 4*(ks2 // 2)*(ks3 // 2) < 0, 3 + ((-1)*tl_math.abs(1 + ((-1)*tl_math.abs((-1) + x0)) + 4*(ks2 // 2)*(ks3 // 2))) + 8*(ks2 // 2)*(ks3 // 2), 1 + ((-1)*tl_math.abs(1 + ((-1)*tl_math.abs((-1) + x0)) + 4*(ks2 // 2)*(ks3 // 2))) + 4*(ks2 // 2)*(ks3 // 2)))) // (2*(ks3 // 2))) % (2*(ks2 // 2)))) + ((((-1) + (tl.where(1 + ((-1)*tl_math.abs(1 + ((-1)*tl_math.abs((-1) + x0)) + 4*(ks2 // 2)*(ks3 // 2))) + 4*(ks2 // 2)*(ks3 // 2) < 0, 3 + ((-1)*tl_math.abs(1 + ((-1)*tl_math.abs((-1) + x0)) + 4*(ks2 // 2)*(ks3 // 2))) + 8*(ks2 // 2)*(ks3 // 2), 1 + ((-1)*tl_math.abs(1 + ((-1)*tl_math.abs((-1) + x0)) + 4*(ks2 // 2)*(ks3 // 2))) + 4*(ks2 // 2)*(ks3 // 2)))) % (2*(ks3 // 2))))) // (2*(ks3 // 2))) % (2*(ks2 // 2)))) + 4*(ks2 // 2)*(ks3 // 2)*((((((-4)*(ks2 // 2)*(ks3 // 2)) + 2*(ks3 // 2)*(((((-1) + (tl.where(1 + ((-1)*tl_math.abs(1 + ((-1)*tl_math.abs((-1) + x0)) + 4*(ks2 // 2)*(ks3 // 2))) + 4*(ks2 // 2)*(ks3 // 2) < 0, 3 + ((-1)*tl_math.abs(1 + ((-1)*tl_math.abs((-1) + x0)) + 4*(ks2 // 2)*(ks3 // 2))) + 8*(ks2 // 2)*(ks3 // 2), 1 + ((-1)*tl_math.abs(1 + ((-1)*tl_math.abs((-1) + x0)) + 4*(ks2 // 2)*(ks3 // 2))) + 4*(ks2 // 2)*(ks3 // 2)))) // (2*(ks3 // 2))) % (2*(ks2 // 2)))) + 4*(ks2 // 2)*(ks3 // 2)*(tl.where(1 + ((-1)*tl_math.abs(1 + ((-1)*tl_math.abs((-1) + x1)) + 2*(ks1 // 2))) + 2*(ks1 // 2) < 0, 3 + ((-1)*tl_math.abs(1 + ((-1)*tl_math.abs((-1) + x1)) + 2*(ks1 // 2))) + 4*(ks1 // 2), 1 + ((-1)*tl_math.abs(1 + ((-1)*tl_math.abs((-1) + x1)) + 2*(ks1 // 2))) + 2*(ks1 // 2))) + ((((-1) + (tl.where(1 + ((-1)*tl_math.abs(1 + ((-1)*tl_math.abs((-1) + x0)) + 4*(ks2 // 2)*(ks3 // 2))) + 4*(ks2 // 2)*(ks3 // 2) < 0, 3 + ((-1)*tl_math.abs(1 + ((-1)*tl_math.abs((-1) + x0)) + 4*(ks2 // 2)*(ks3 // 2))) + 8*(ks2 // 2)*(ks3 // 2), 1 + ((-1)*tl_math.abs(1 + ((-1)*tl_math.abs((-1) + x0)) + 4*(ks2 // 2)*(ks3 // 2))) + 4*(ks2 // 2)*(ks3 // 2)))) % (2*(ks3 // 2))))) // (4*(ks2 // 2)*(ks3 // 2))) % (2*(ks1 // 2)))) + ((((((-1) + (tl.where(1 + ((-1)*tl_math.abs(1 + ((-1)*tl_math.abs((-1) + x0)) + 4*(ks2 // 2)*(ks3 // 2))) + 4*(ks2 // 2)*(ks3 // 2) < 0, 3 + ((-1)*tl_math.abs(1 + ((-1)*tl_math.abs((-1) + x0)) + 4*(ks2 // 2)*(ks3 // 2))) + 8*(ks2 // 2)*(ks3 // 2), 1 + ((-1)*tl_math.abs(1 + ((-1)*tl_math.abs((-1) + x0)) + 4*(ks2 // 2)*(ks3 // 2))) + 4*(ks2 // 2)*(ks3 // 2)))) % (2*(ks3 // 2)))) % (2*(ks3 // 2))))), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tl.store(out_ptr0 + (x2), tmp12, xmask)




# kernel path: /tmp/torchinductor_sahanp/a6/ca63cfg7dt4zpaijowix2umcwlpue6iniqvkntbcqk7jbusi7tcu.py
# Topologically Sorted Source Nodes: [x_7, x_8], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   x_7 => convolution
#   x_8 => relu
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%_adaptive_avg_pool3d, %primals_5, %primals_6, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_relu_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 216
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)




# kernel path: /tmp/torchinductor_sahanp/qo/cqokghypn3sra5fpsb76iccq3iig5svfpr6azfot27hnohybchch.py
# Topologically Sorted Source Nodes: [x_9, loss], Original ATen: [aten.convolution, aten.huber_loss_backward, aten.huber_loss]
# Source node to ATen node mapping:
#   loss => abs_5, lt_1, mean, mul_42, mul_43, mul_44, relu_1, sub_36, where
#   x_9 => convolution_1
# Graph fragment:
#   %convolution_1 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %primals_7, %primals_8, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %abs_5 : [num_users=4] = call_function[target=torch.ops.aten.abs.default](args = (%relu_1,), kwargs = {})
#   %lt_1 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%abs_5, 1.0), kwargs = {})
#   %mul_42 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%abs_5, 0.5), kwargs = {})
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_42, %abs_5), kwargs = {})
#   %sub_36 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%abs_5, 0.5), kwargs = {})
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_36, 1.0), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%lt_1, %mul_43, %mul_44), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%where,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_huber_loss_huber_loss_backward_4(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 1280
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp15 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        r0_1 = r0_index // 64
        tmp0 = tl.load(in_out_ptr0 + (r0_2), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.full([1, 1], 0, tl.int32)
        tmp4 = triton_helpers.maximum(tmp3, tmp2)
        tmp5 = tl_math.abs(tmp4)
        tmp6 = 1.0
        tmp7 = tmp5 < tmp6
        tmp8 = 0.5
        tmp9 = tmp5 * tmp8
        tmp10 = tmp9 * tmp5
        tmp11 = tmp5 - tmp8
        tmp12 = tmp11 * tmp6
        tmp13 = tl.where(tmp7, tmp10, tmp12)
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(r0_mask, tmp16, _tmp15)
        tl.store(in_out_ptr0 + (tl.broadcast_to(r0_2, [XBLOCK, R0_BLOCK])), tmp2, r0_mask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tmp17 = 1280.0
    tmp18 = tmp15 / tmp17
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp18, None)







def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8 = args
    args.clear()
    s0 = primals_1
    s1 = primals_2
    s2 = primals_3
    assert_size_stride(primals_4, (1, 1, s0, s1, s2), (s0*s1*s2, s0*s1*s2, s1*s2, s2, 1))
    assert_size_stride(primals_5, (10, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(primals_6, (10, ), (1, ))
    assert_size_stride(primals_7, (20, 10, 3, 3, 3), (270, 27, 9, 3, 1))
    assert_size_stride(primals_8, (20, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [max_pool3d_with_indices], Original ATen: [aten.max_pool3d_with_indices]
        buf0 = torch.ops.aten.max_pool3d_with_indices.default(primals_4, [2, 2, 2], [2, 2, 2])
        del primals_4
        buf1 = buf0[0]
        buf2 = buf0[1]
        del buf0
        buf3 = empty_strided_cuda((1, 1, 2*(s0 // 2), 2*(s1 // 2), 2*(s2 // 2)), (8*(s0 // 2)*(s1 // 2)*(s2 // 2), 8*(s0 // 2)*(s1 // 2)*(s2 // 2), 4*(s1 // 2)*(s2 // 2), 2*(s2 // 2), 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.max_unpool3d]
        triton_poi_fused_max_unpool3d_0_xnumel = 8*(s0 // 2)*(s1 // 2)*(s2 // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_unpool3d_0[grid(triton_poi_fused_max_unpool3d_0_xnumel)](buf3, 4096, XBLOCK=256, num_warps=4, num_stages=1)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.max_unpool3d]
        triton_poi_fused_max_unpool3d_1_xnumel = (s0 // 2)*(s1 // 2)*(s2 // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_unpool3d_1[grid(triton_poi_fused_max_unpool3d_1_xnumel)](buf2, buf1, buf3, 16, 16, 16, 512, XBLOCK=128, num_warps=4, num_stages=1)
        del buf1
        del buf2
        ps0 = 4 + 4*(s1 // 2)*(s2 // 2)
        buf5 = empty_strided_cuda((1, 1, 4 + 2*(s0 // 2), 4 + 4*(s1 // 2)*(s2 // 2)), (16 + 8*(s0 // 2) + 16*(s1 // 2)*(s2 // 2) + 8*(s0 // 2)*(s1 // 2)*(s2 // 2), 1, 4 + 4*(s1 // 2)*(s2 // 2), 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_3, x_4], Original ATen: [aten.constant_pad_nd, aten.reflection_pad2d]
        triton_poi_fused_constant_pad_nd_reflection_pad2d_2_xnumel = 16 + 8*(s0 // 2) + 16*(s1 // 2)*(s2 // 2) + 8*(s0 // 2)*(s1 // 2)*(s2 // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_reflection_pad2d_2[grid(triton_poi_fused_constant_pad_nd_reflection_pad2d_2_xnumel)](buf3, buf5, 260, 16, 16, 16, 5200, XBLOCK=256, num_warps=4, num_stages=1)
        del buf3
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten._adaptive_avg_pool3d]
        buf6 = torch.ops.aten._adaptive_avg_pool3d.default(reinterpret_tensor(buf5, (1, 1, 4 + 2*(s0 // 2), 4 + 4*(s1 // 2)*(s2 // 2), 1), (0, 0, 4 + 4*(s1 // 2)*(s2 // 2), 1, 0), 0), [8, 8, 8])
        del buf5
        buf7 = buf6
        del buf6
        # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf7, primals_5, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (1, 10, 6, 6, 6), (2160, 216, 36, 6, 1))
        buf9 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [x_7, x_8], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_3[grid(2160)](buf9, primals_6, 2160, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_6
        # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, primals_7, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (1, 20, 4, 4, 4), (1280, 64, 16, 4, 1))
        buf11 = buf10; del buf10  # reuse
        buf12 = empty_strided_cuda((), (), torch.float32)
        buf13 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [x_9, loss], Original ATen: [aten.convolution, aten.huber_loss_backward, aten.huber_loss]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_huber_loss_huber_loss_backward_4[grid(1)](buf11, buf13, primals_8, 1, 1280, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del primals_8
    return (buf13, primals_5, primals_7, buf7, buf9, buf11, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = 16
    primals_2 = 16
    primals_3 = 16
    primals_4 = rand_strided((1, 1, 16, 16, 16), (4096, 4096, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((10, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((20, 10, 3, 3, 3), (270, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
