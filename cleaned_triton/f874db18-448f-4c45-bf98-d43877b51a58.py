# AOT ID: ['47_forward']
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


# kernel path: /tmp/torchinductor_sahanp/k2/ck2kc2c25qbejprryz2tckgei5nnewsj66crxtimzwgkpbiuems5.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x => constant_pad_nd
# Graph fragment:
#   %constant_pad_nd : [num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%primals_4, [2, 2], 0.0), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_constant_pad_nd_0(in_ptr0, out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    x2 = xindex
    tmp0 = (-2) + x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = ks1
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + ((-2) + x0 + ks1*x1), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tl.store(out_ptr0 + (x2), tmp6, xmask)




# kernel path: /tmp/torchinductor_sahanp/jy/cjypi6lulgax3776ctowpw47ti2sgrkyxwssgxe2j2sbt4csh7ha.py
# Topologically Sorted Source Nodes: [conv_transpose3d, x_1], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   conv_transpose3d => convolution
#   x_1 => relu
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd, %primals_5, %primals_6, [2, 2, 2], [0, 0, 0], [1, 1, 1], True, [0, 0, 0], 1), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_relu_1(in_out_ptr0, in_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // ks0
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)




# kernel path: /tmp/torchinductor_sahanp/wp/cwpvaumgahgvxsxyhwtrddg7fbt5h6mwv7pxhew3ylszujshpmtk.py
# Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.sub, aten.add, aten.norm]
# Source node to ATen node mapping:
#   x_4 => add_46, pow_1, sub_26, sum_1
# Graph fragment:
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_2, %slice_4), kwargs = {})
#   %add_46 : [num_users=2] = call_function[target=torch.ops.aten.add.Scalar](args = (%sub_26, 1e-06), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_46, 2.0), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1]), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_norm_sub_2(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 134
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp19 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = r0_1 + x0*((1843 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // 134)
        tmp1 = 1710 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (19*((((r0_1 + x0*((1843 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // 134)) // (19 + 4*ks2)) % (3 + 4*ks1))) + 57*((((r0_1 + x0*((1843 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // 134)) // (57 + 12*ks2 + 76*ks1 + 16*ks1*ks2)) % (3 + 4*ks0))) + 171*((((r0_1 + x0*((1843 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // 134)) // (171 + 36*ks2 + 228*ks0 + 228*ks1 + 48*ks0*ks2 + 48*ks1*ks2 + 304*ks0*ks1 + 64*ks0*ks1*ks2)) % 20)) + 4*ks2*((((r0_1 + x0*((1843 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // 134)) // (19 + 4*ks2)) % (3 + 4*ks1))) + 12*ks2*((((r0_1 + x0*((1843 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // 134)) // (57 + 12*ks2 + 76*ks1 + 16*ks1*ks2)) % (3 + 4*ks0))) + 36*ks2*((((r0_1 + x0*((1843 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // 134)) // (171 + 36*ks2 + 228*ks0 + 228*ks1 + 48*ks0*ks2 + 48*ks1*ks2 + 304*ks0*ks1 + 64*ks0*ks1*ks2)) % 20)) + 76*ks1*((((r0_1 + x0*((1843 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // 134)) // (57 + 12*ks2 + 76*ks1 + 16*ks1*ks2)) % (3 + 4*ks0))) + 228*ks0*((((r0_1 + x0*((1843 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // 134)) // (171 + 36*ks2 + 228*ks0 + 228*ks1 + 48*ks0*ks2 + 48*ks1*ks2 + 304*ks0*ks1 + 64*ks0*ks1*ks2)) % 20)) + 228*ks1*((((r0_1 + x0*((1843 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // 134)) // (171 + 36*ks2 + 228*ks0 + 228*ks1 + 48*ks0*ks2 + 48*ks1*ks2 + 304*ks0*ks1 + 64*ks0*ks1*ks2)) % 20)) + 16*ks1*ks2*((((r0_1 + x0*((1843 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // 134)) // (57 + 12*ks2 + 76*ks1 + 16*ks1*ks2)) % (3 + 4*ks0))) + 48*ks0*ks2*((((r0_1 + x0*((1843 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // 134)) // (171 + 36*ks2 + 228*ks0 + 228*ks1 + 48*ks0*ks2 + 48*ks1*ks2 + 304*ks0*ks1 + 64*ks0*ks1*ks2)) % 20)) + 48*ks1*ks2*((((r0_1 + x0*((1843 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // 134)) // (171 + 36*ks2 + 228*ks0 + 228*ks1 + 48*ks0*ks2 + 48*ks1*ks2 + 304*ks0*ks1 + 64*ks0*ks1*ks2)) % 20)) + 304*ks0*ks1*((((r0_1 + x0*((1843 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // 134)) // (171 + 36*ks2 + 228*ks0 + 228*ks1 + 48*ks0*ks2 + 48*ks1*ks2 + 304*ks0*ks1 + 64*ks0*ks1*ks2)) % 20)) + 64*ks0*ks1*ks2*((((r0_1 + x0*((1843 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // 134)) // (171 + 36*ks2 + 228*ks0 + 228*ks1 + 48*ks0*ks2 + 48*ks1*ks2 + 304*ks0*ks1 + 64*ks0*ks1*ks2)) % 20)) + (((r0_1 + x0*((1843 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // 134)) % (19 + 4*ks2)))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + ((((r0_1 + x0*((1843 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // 134)) // (171 + 36*ks2 + 228*ks0 + 228*ks1 + 48*ks0*ks2 + 48*ks1*ks2 + 304*ks0*ks1 + 64*ks0*ks1*ks2)) % 20)), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.full([1, 1], 0, tl.int32)
        tmp7 = triton_helpers.maximum(tmp6, tmp5)
        tmp8 = tl.load(in_ptr0 + (19*((((1710 + r0_1 + 360*ks2 + 2280*ks0 + 2280*ks1 + x0*((1843 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // 134) + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // (19 + 4*ks2)) % (3 + 4*ks1))) + 57*((((1710 + r0_1 + 360*ks2 + 2280*ks0 + 2280*ks1 + x0*((1843 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // 134) + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // (57 + 12*ks2 + 76*ks1 + 16*ks1*ks2)) % (3 + 4*ks0))) + 171*((((1710 + r0_1 + 360*ks2 + 2280*ks0 + 2280*ks1 + x0*((1843 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // 134) + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // (171 + 36*ks2 + 228*ks0 + 228*ks1 + 48*ks0*ks2 + 48*ks1*ks2 + 304*ks0*ks1 + 64*ks0*ks1*ks2)) % 20)) + 4*ks2*((((1710 + r0_1 + 360*ks2 + 2280*ks0 + 2280*ks1 + x0*((1843 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // 134) + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // (19 + 4*ks2)) % (3 + 4*ks1))) + 12*ks2*((((1710 + r0_1 + 360*ks2 + 2280*ks0 + 2280*ks1 + x0*((1843 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // 134) + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // (57 + 12*ks2 + 76*ks1 + 16*ks1*ks2)) % (3 + 4*ks0))) + 36*ks2*((((1710 + r0_1 + 360*ks2 + 2280*ks0 + 2280*ks1 + x0*((1843 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // 134) + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // (171 + 36*ks2 + 228*ks0 + 228*ks1 + 48*ks0*ks2 + 48*ks1*ks2 + 304*ks0*ks1 + 64*ks0*ks1*ks2)) % 20)) + 76*ks1*((((1710 + r0_1 + 360*ks2 + 2280*ks0 + 2280*ks1 + x0*((1843 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // 134) + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // (57 + 12*ks2 + 76*ks1 + 16*ks1*ks2)) % (3 + 4*ks0))) + 228*ks0*((((1710 + r0_1 + 360*ks2 + 2280*ks0 + 2280*ks1 + x0*((1843 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // 134) + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // (171 + 36*ks2 + 228*ks0 + 228*ks1 + 48*ks0*ks2 + 48*ks1*ks2 + 304*ks0*ks1 + 64*ks0*ks1*ks2)) % 20)) + 228*ks1*((((1710 + r0_1 + 360*ks2 + 2280*ks0 + 2280*ks1 + x0*((1843 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // 134) + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // (171 + 36*ks2 + 228*ks0 + 228*ks1 + 48*ks0*ks2 + 48*ks1*ks2 + 304*ks0*ks1 + 64*ks0*ks1*ks2)) % 20)) + 16*ks1*ks2*((((1710 + r0_1 + 360*ks2 + 2280*ks0 + 2280*ks1 + x0*((1843 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // 134) + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // (57 + 12*ks2 + 76*ks1 + 16*ks1*ks2)) % (3 + 4*ks0))) + 48*ks0*ks2*((((1710 + r0_1 + 360*ks2 + 2280*ks0 + 2280*ks1 + x0*((1843 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // 134) + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // (171 + 36*ks2 + 228*ks0 + 228*ks1 + 48*ks0*ks2 + 48*ks1*ks2 + 304*ks0*ks1 + 64*ks0*ks1*ks2)) % 20)) + 48*ks1*ks2*((((1710 + r0_1 + 360*ks2 + 2280*ks0 + 2280*ks1 + x0*((1843 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // 134) + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // (171 + 36*ks2 + 228*ks0 + 228*ks1 + 48*ks0*ks2 + 48*ks1*ks2 + 304*ks0*ks1 + 64*ks0*ks1*ks2)) % 20)) + 304*ks0*ks1*((((1710 + r0_1 + 360*ks2 + 2280*ks0 + 2280*ks1 + x0*((1843 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // 134) + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // (171 + 36*ks2 + 228*ks0 + 228*ks1 + 48*ks0*ks2 + 48*ks1*ks2 + 304*ks0*ks1 + 64*ks0*ks1*ks2)) % 20)) + 64*ks0*ks1*ks2*((((1710 + r0_1 + 360*ks2 + 2280*ks0 + 2280*ks1 + x0*((1843 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // 134) + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // (171 + 36*ks2 + 228*ks0 + 228*ks1 + 48*ks0*ks2 + 48*ks1*ks2 + 304*ks0*ks1 + 64*ks0*ks1*ks2)) % 20)) + (((1710 + r0_1 + 360*ks2 + 2280*ks0 + 2280*ks1 + x0*((1843 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // 134) + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) % (19 + 4*ks2)))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr1 + ((((1710 + r0_1 + 360*ks2 + 2280*ks0 + 2280*ks1 + x0*((1843 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // 134) + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // (171 + 36*ks2 + 228*ks0 + 228*ks1 + 48*ks0*ks2 + 48*ks1*ks2 + 304*ks0*ks1 + 64*ks0*ks1*ks2)) % 20)), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tmp8 + tmp9
        tmp11 = triton_helpers.maximum(tmp6, tmp10)
        tmp12 = tmp7 - tmp11
        tmp13 = 1e-06
        tmp14 = tmp12 + tmp13
        tmp15 = tmp14 * tmp14
        tmp16 = tl.full(tmp15.shape, 0, tmp15.dtype)
        tmp17 = tl.where(tmp2, tmp15, tmp16)
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(r0_mask & xmask, tmp20, _tmp19)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp19, xmask)




# kernel path: /tmp/torchinductor_sahanp/ld/cld25q52c3hpif6el52xvm3cqz4uqkc7pahrwzv7phnfpcriewgh.py
# Topologically Sorted Source Nodes: [x_4, target, loss], Original ATen: [aten.sub, aten.add, aten.norm, aten.randint, aten.gather, aten.rsub, aten.clamp_min, aten.arange, aten.ne, aten.scalar_tensor, aten.where, aten.mean, aten.ge]
# Source node to ATen node mapping:
#   loss => add_49, clamp_min, full_default, gather, iota, mean, ne, sub_29, where
#   target => inductor_lookup_seed_default, inductor_randint_default
#   x_4 => add_46, pow_1, sub_26, sum_1
# Graph fragment:
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_2, %slice_4), kwargs = {})
#   %add_46 : [num_users=2] = call_function[target=torch.ops.aten.add.Scalar](args = (%sub_26, 1e-06), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_46, 2.0), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1]), kwargs = {})
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_randint_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_randint.default](args = (0, 1, [1], %inductor_lookup_seed_default), kwargs = {})
#   %gather : [num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%unsqueeze, 1, %unsqueeze_1), kwargs = {})
#   %sub_29 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %gather), kwargs = {})
#   %add_49 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub_29, %unsqueeze), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_49, 0), kwargs = {})
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %ne : [num_users=2] = call_function[target=torch.ops.aten.ne.Tensor](args = (%iota, %unsqueeze_1), kwargs = {})
#   %full_default : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne, %clamp_min, %full_default), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%where,), kwargs = {})
#   %ge_1 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_49, 0), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_arange_clamp_min_gather_ge_mean_ne_norm_randint_rsub_scalar_tensor_sub_where_3(in_out_ptr0, in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, load_seed_offset, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 134
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = tl.load(in_out_ptr0 + load_seed_offset)
    tmp6 = tl.full([1, 1], 0, tl.int32)
    tmp7 = tl.full([1, 1], 0, tl.int64)
    tmp8 = tl.full([1, 1], 1, tl.int64)
    tmp9 = triton_helpers.randint64(tmp5, (tmp6).to(tl.uint32), tmp7, tmp8)
    tmp10 = tmp7 != tmp9
    tmp11 = tl.full([XBLOCK, 1], 1, tl.int32)
    tmp12 = tmp9 + tmp11
    tmp13 = tmp9 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp9)
    tl.device_assert((0 <= tmp14) & (tmp14 < 1), "index out of bounds: 0 <= tmp14 < 1")
    tmp16 = libdevice.sqrt(tmp4)
    tmp17 = 1.0
    tmp18 = tmp17 - tmp16
    tmp19 = tmp18 + tmp16
    tmp20 = 0.0
    tmp21 = triton_helpers.maximum(tmp19, tmp20)
    tmp22 = tl.where(tmp10, tmp21, tmp20)
    tmp23 = tmp22 / tmp17
    tmp24 = tmp19 >= tmp20
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp9, None)
    tl.store(out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp10, None)
    tl.store(out_ptr2 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp23, None)
    tl.store(out_ptr3 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp24, None)
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp4, None)




# kernel path: /tmp/torchinductor_sahanp/g7/cg7cai5jlge3e6xw64njcc5rn3vlgunpdwqibo6mr62livhqckx3.py
# Topologically Sorted Source Nodes: [x_4, loss], Original ATen: [aten.sub, aten.add, aten.scalar_tensor, aten.div, aten.eq, aten.masked_fill]
# Source node to ATen node mapping:
#   loss => full_default
#   x_4 => add_46, sub_26
# Graph fragment:
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_2, %slice_4), kwargs = {})
#   %add_46 : [num_users=2] = call_function[target=torch.ops.aten.add.Scalar](args = (%sub_26, 1e-06), kwargs = {})
#   %full_default : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_46, %unsqueeze), kwargs = {})
#   %eq_31 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%unsqueeze, 0), kwargs = {})
#   %where_3 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_31, %full_default, %div_1), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_div_eq_masked_fill_scalar_tensor_sub_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp5 = tl.load(in_ptr1 + (19*(((x0 // (19 + 4*ks2)) % (3 + 4*ks1))) + 57*(((x0 // (57 + 12*ks2 + 76*ks1 + 16*ks1*ks2)) % (3 + 4*ks0))) + 171*(((x0 // (171 + 36*ks2 + 228*ks0 + 228*ks1 + 48*ks0*ks2 + 48*ks1*ks2 + 304*ks0*ks1 + 64*ks0*ks1*ks2)) % 20)) + 4*ks2*(((x0 // (19 + 4*ks2)) % (3 + 4*ks1))) + 12*ks2*(((x0 // (57 + 12*ks2 + 76*ks1 + 16*ks1*ks2)) % (3 + 4*ks0))) + 36*ks2*(((x0 // (171 + 36*ks2 + 228*ks0 + 228*ks1 + 48*ks0*ks2 + 48*ks1*ks2 + 304*ks0*ks1 + 64*ks0*ks1*ks2)) % 20)) + 76*ks1*(((x0 // (57 + 12*ks2 + 76*ks1 + 16*ks1*ks2)) % (3 + 4*ks0))) + 228*ks0*(((x0 // (171 + 36*ks2 + 228*ks0 + 228*ks1 + 48*ks0*ks2 + 48*ks1*ks2 + 304*ks0*ks1 + 64*ks0*ks1*ks2)) % 20)) + 228*ks1*(((x0 // (171 + 36*ks2 + 228*ks0 + 228*ks1 + 48*ks0*ks2 + 48*ks1*ks2 + 304*ks0*ks1 + 64*ks0*ks1*ks2)) % 20)) + 16*ks1*ks2*(((x0 // (57 + 12*ks2 + 76*ks1 + 16*ks1*ks2)) % (3 + 4*ks0))) + 48*ks0*ks2*(((x0 // (171 + 36*ks2 + 228*ks0 + 228*ks1 + 48*ks0*ks2 + 48*ks1*ks2 + 304*ks0*ks1 + 64*ks0*ks1*ks2)) % 20)) + 48*ks1*ks2*(((x0 // (171 + 36*ks2 + 228*ks0 + 228*ks1 + 48*ks0*ks2 + 48*ks1*ks2 + 304*ks0*ks1 + 64*ks0*ks1*ks2)) % 20)) + 304*ks0*ks1*(((x0 // (171 + 36*ks2 + 228*ks0 + 228*ks1 + 48*ks0*ks2 + 48*ks1*ks2 + 304*ks0*ks1 + 64*ks0*ks1*ks2)) % 20)) + 64*ks0*ks1*ks2*(((x0 // (171 + 36*ks2 + 228*ks0 + 228*ks1 + 48*ks0*ks2 + 48*ks1*ks2 + 304*ks0*ks1 + 64*ks0*ks1*ks2)) % 20)) + ((x0 % (19 + 4*ks2)))), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (((x0 // (171 + 36*ks2 + 228*ks0 + 228*ks1 + 48*ks0*ks2 + 48*ks1*ks2 + 304*ks0*ks1 + 64*ks0*ks1*ks2)) % 20)), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr1 + (19*((((1710 + x0 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // (19 + 4*ks2)) % (3 + 4*ks1))) + 57*((((1710 + x0 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // (57 + 12*ks2 + 76*ks1 + 16*ks1*ks2)) % (3 + 4*ks0))) + 171*((1710 + x0 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // (171 + 36*ks2 + 228*ks0 + 228*ks1 + 48*ks0*ks2 + 48*ks1*ks2 + 304*ks0*ks1 + 64*ks0*ks1*ks2)) + 4*ks2*((((1710 + x0 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // (19 + 4*ks2)) % (3 + 4*ks1))) + 12*ks2*((((1710 + x0 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // (57 + 12*ks2 + 76*ks1 + 16*ks1*ks2)) % (3 + 4*ks0))) + 36*ks2*((1710 + x0 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // (171 + 36*ks2 + 228*ks0 + 228*ks1 + 48*ks0*ks2 + 48*ks1*ks2 + 304*ks0*ks1 + 64*ks0*ks1*ks2)) + 76*ks1*((((1710 + x0 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // (57 + 12*ks2 + 76*ks1 + 16*ks1*ks2)) % (3 + 4*ks0))) + 228*ks0*((1710 + x0 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // (171 + 36*ks2 + 228*ks0 + 228*ks1 + 48*ks0*ks2 + 48*ks1*ks2 + 304*ks0*ks1 + 64*ks0*ks1*ks2)) + 228*ks1*((1710 + x0 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // (171 + 36*ks2 + 228*ks0 + 228*ks1 + 48*ks0*ks2 + 48*ks1*ks2 + 304*ks0*ks1 + 64*ks0*ks1*ks2)) + 16*ks1*ks2*((((1710 + x0 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // (57 + 12*ks2 + 76*ks1 + 16*ks1*ks2)) % (3 + 4*ks0))) + 48*ks0*ks2*((1710 + x0 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // (171 + 36*ks2 + 228*ks0 + 228*ks1 + 48*ks0*ks2 + 48*ks1*ks2 + 304*ks0*ks1 + 64*ks0*ks1*ks2)) + 48*ks1*ks2*((1710 + x0 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // (171 + 36*ks2 + 228*ks0 + 228*ks1 + 48*ks0*ks2 + 48*ks1*ks2 + 304*ks0*ks1 + 64*ks0*ks1*ks2)) + 304*ks0*ks1*((1710 + x0 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // (171 + 36*ks2 + 228*ks0 + 228*ks1 + 48*ks0*ks2 + 48*ks1*ks2 + 304*ks0*ks1 + 64*ks0*ks1*ks2)) + 64*ks0*ks1*ks2*((1710 + x0 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // (171 + 36*ks2 + 228*ks0 + 228*ks1 + 48*ks0*ks2 + 48*ks1*ks2 + 304*ks0*ks1 + 64*ks0*ks1*ks2)) + (((1710 + x0 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) % (19 + 4*ks2)))), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + ((1710 + x0 + 360*ks2 + 2280*ks0 + 2280*ks1 + 480*ks0*ks2 + 480*ks1*ks2 + 3040*ks0*ks1 + 640*ks0*ks1*ks2) // (171 + 36*ks2 + 228*ks0 + 228*ks1 + 48*ks0*ks2 + 48*ks1*ks2 + 304*ks0*ks1 + 64*ks0*ks1*ks2)), xmask, eviction_policy='evict_last')
    tmp2 = libdevice.sqrt(tmp1)
    tmp3 = 0.0
    tmp4 = tmp2 == tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full([1], 0, tl.int32)
    tmp9 = triton_helpers.maximum(tmp8, tmp7)
    tmp12 = tmp10 + tmp11
    tmp13 = triton_helpers.maximum(tmp8, tmp12)
    tmp14 = tmp9 - tmp13
    tmp15 = 1e-06
    tmp16 = tmp14 + tmp15
    tmp17 = tmp16 / tmp2
    tmp18 = tl.where(tmp4, tmp3, tmp17)
    tl.store(out_ptr0 + (x0), tmp18, xmask)




# kernel path: /tmp/torchinductor_sahanp/sd/csdoialou5nr2rnhl235znajts5ytqjcsoyevyojm2bvalnc6mm6.py
# Topologically Sorted Source Nodes: [conv_transpose3d_1, x_2], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   conv_transpose3d_1 => convolution_1
#   x_2 => relu_1
# Graph fragment:
#   %convolution_1 : [num_users=4] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %primals_7, %primals_8, [2, 2, 2], [0, 0, 0], [1, 1, 1], True, [0, 0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_1, 0), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_relu_threshold_backward_5(in_ptr0, in_ptr1, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // ks0
    tmp0 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(out_ptr0 + (x2), tmp6, xmask)







def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8 = args
    args.clear()
    s0 = primals_1
    s1 = primals_2
    s2 = primals_3
    assert_size_stride(primals_4, (1, 1, s0, s1, s2), (s0*s1*s2, s0*s1*s2, s1*s2, s2, 1))
    assert_size_stride(primals_5, (1, 10, 3, 3, 3), (270, 27, 9, 3, 1))
    assert_size_stride(primals_6, (10, ), (1, ))
    assert_size_stride(primals_7, (10, 20, 3, 3, 3), (540, 27, 9, 3, 1))
    assert_size_stride(primals_8, (20, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        ps0 = 4 + s2
        buf0 = empty_strided_cuda((1, 1, s0, s1, 4 + s2), (4*s0*s1 + s0*s1*s2, 4*s0*s1 + s0*s1*s2, 4*s1 + s1*s2, 4 + s2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_0_xnumel = 4*s0*s1 + s0*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_0[grid(triton_poi_fused_constant_pad_nd_0_xnumel)](primals_4, buf0, 14, 10, 1400, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_4
        # Topologically Sorted Source Nodes: [conv_transpose3d], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(buf0, primals_5, stride=(2, 2, 2), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=True, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (1, 10, 1 + 2*s0, 1 + 2*s1, 9 + 2*s2), (90 + 20*s2 + 180*s0 + 180*s1 + 40*s0*s2 + 40*s1*s2 + 360*s0*s1 + 80*s0*s1*s2, 9 + 2*s2 + 18*s0 + 18*s1 + 4*s0*s2 + 4*s1*s2 + 36*s0*s1 + 8*s0*s1*s2, 9 + 2*s2 + 18*s1 + 4*s1*s2, 9 + 2*s2, 1))
        ps1 = 9 + 2*s2 + 18*s0 + 18*s1 + 4*s0*s2 + 4*s1*s2 + 36*s0*s1 + 8*s0*s1*s2
        buf2 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [conv_transpose3d, x_1], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_1_xnumel = 90 + 20*s2 + 180*s0 + 180*s1 + 40*s0*s2 + 40*s1*s2 + 360*s0*s1 + 80*s0*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_1[grid(triton_poi_fused_convolution_relu_1_xnumel)](buf2, primals_6, 12789, 127890, XBLOCK=512, num_warps=8, num_stages=1)
        del primals_6
        # Topologically Sorted Source Nodes: [conv_transpose3d_1], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(buf2, primals_7, stride=(2, 2, 2), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=True, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (1, 20, 3 + 4*s0, 3 + 4*s1, 19 + 4*s2), (3420 + 720*s2 + 4560*s0 + 4560*s1 + 960*s0*s2 + 960*s1*s2 + 6080*s0*s1 + 1280*s0*s1*s2, 171 + 36*s2 + 228*s0 + 228*s1 + 48*s0*s2 + 48*s1*s2 + 304*s0*s1 + 64*s0*s1*s2, 57 + 12*s2 + 76*s1 + 16*s1*s2, 19 + 4*s2, 1))
        buf4 = empty_strided_cuda((1, 134), (134, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.sub, aten.add, aten.norm]
        triton_red_fused_add_norm_sub_2_r0_numel = (1843 + 360*s2 + 2280*s0 + 2280*s1 + 480*s0*s2 + 480*s1*s2 + 3040*s0*s1 + 640*s0*s1*s2) // 134
        stream0 = get_raw_stream(0)
        triton_red_fused_add_norm_sub_2[grid(134)](buf3, primals_8, buf4, 10, 10, 10, 134, 8142, XBLOCK=1, R0_BLOCK=1024, num_warps=16, num_stages=1)
        buf6 = empty_strided_cuda((1, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf6)
        buf5 = empty_strided_cuda((1, ), (1, ), torch.float32)
        buf7 = buf6; del buf6  # reuse
        buf8 = empty_strided_cuda((1, 1), (1, 1), torch.bool)
        buf11 = empty_strided_cuda((), (), torch.float32)
        buf12 = empty_strided_cuda((1, 1), (1, 1), torch.bool)
        # Topologically Sorted Source Nodes: [x_4, target, loss], Original ATen: [aten.sub, aten.add, aten.norm, aten.randint, aten.gather, aten.rsub, aten.clamp_min, aten.arange, aten.ne, aten.scalar_tensor, aten.where, aten.mean, aten.ge]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_arange_clamp_min_gather_ge_mean_ne_norm_randint_rsub_scalar_tensor_sub_where_3[grid(1)](buf7, buf4, buf5, buf8, buf11, buf12, 0, 1, 134, XBLOCK=1, num_warps=2, num_stages=1)
        del buf4
        buf9 = empty_strided_cuda((1, 1710 + 360*s2 + 2280*s0 + 2280*s1 + 480*s0*s2 + 480*s1*s2 + 3040*s0*s1 + 640*s0*s1*s2), (1710 + 360*s2 + 2280*s0 + 2280*s1 + 480*s0*s2 + 480*s1*s2 + 3040*s0*s1 + 640*s0*s1*s2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_4, loss], Original ATen: [aten.sub, aten.add, aten.scalar_tensor, aten.div, aten.eq, aten.masked_fill]
        triton_poi_fused_add_div_eq_masked_fill_scalar_tensor_sub_4_xnumel = 1710 + 360*s2 + 2280*s0 + 2280*s1 + 480*s0*s2 + 480*s1*s2 + 3040*s0*s1 + 640*s0*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_eq_masked_fill_scalar_tensor_sub_4[grid(triton_poi_fused_add_div_eq_masked_fill_scalar_tensor_sub_4_xnumel)](buf5, buf3, primals_8, buf9, 10, 10, 10, 1090910, XBLOCK=512, num_warps=8, num_stages=1)
        del buf5
        ps2 = 171 + 36*s2 + 228*s0 + 228*s1 + 48*s0*s2 + 48*s1*s2 + 304*s0*s1 + 64*s0*s1*s2
        buf10 = empty_strided_cuda((1, 20, 3 + 4*s0, 3 + 4*s1, 19 + 4*s2), (3420 + 720*s2 + 4560*s0 + 4560*s1 + 960*s0*s2 + 960*s1*s2 + 6080*s0*s1 + 1280*s0*s1*s2, 171 + 36*s2 + 228*s0 + 228*s1 + 48*s0*s2 + 48*s1*s2 + 304*s0*s1 + 64*s0*s1*s2, 57 + 12*s2 + 76*s1 + 16*s1*s2, 19 + 4*s2, 1), torch.bool)
        # Topologically Sorted Source Nodes: [conv_transpose3d_1, x_2], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
        triton_poi_fused_convolution_relu_threshold_backward_5_xnumel = 3420 + 720*s2 + 4560*s0 + 4560*s1 + 960*s0*s2 + 960*s1*s2 + 6080*s0*s1 + 1280*s0*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_threshold_backward_5[grid(triton_poi_fused_convolution_relu_threshold_backward_5_xnumel)](buf3, primals_8, buf10, 109091, 2181820, XBLOCK=1024, num_warps=4, num_stages=1)
        del buf3
        del primals_8
    return (buf11, primals_5, primals_7, buf0, buf2, reinterpret_tensor(buf7, (1, 1), (1, 1), 0), buf8, buf12, buf9, buf10, s0, s1, s2, 3 + 4*s0, 3 + 4*s1, 19 + 4*s2, 3420 + 720*s2 + 4560*s0 + 4560*s1 + 960*s0*s2 + 960*s1*s2 + 6080*s0*s1 + 1280*s0*s1*s2, 1710 + 360*s2 + 2280*s0 + 2280*s1 + 480*s0*s2 + 480*s1*s2 + 3040*s0*s1 + 640*s0*s1*s2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = 10
    primals_2 = 10
    primals_3 = 10
    primals_4 = rand_strided((1, 1, 10, 10, 10), (1000, 1000, 100, 10, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((1, 10, 3, 3, 3), (270, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((10, 20, 3, 3, 3), (540, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
