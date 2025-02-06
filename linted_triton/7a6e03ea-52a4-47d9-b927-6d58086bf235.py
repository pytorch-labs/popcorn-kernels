
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




















import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_convolution_mean_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, out_ptr3, out_ptr5, out_ptr7, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32
    r0_numel = 126
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 126*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(r0_mask & xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp8 = tl.where(r0_mask & xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 126, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(r0_mask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp2 - tmp12
    tmp20 = 126.0
    tmp21 = tmp18 / tmp20
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tmp30 = 0.1
    tmp31 = tmp12 * tmp30
    tmp33 = 0.9
    tmp34 = tmp32 * tmp33
    tmp35 = tmp31 + tmp34
    tmp36 = 1.0
    tmp37 = tmp35 / tmp36
    tmp38 = 1.008
    tmp39 = tmp21 * tmp38
    tmp40 = tmp39 * tmp30
    tmp42 = tmp41 * tmp33
    tmp43 = tmp40 + tmp42
    tmp44 = tmp43 / tmp36
    tl.store(in_out_ptr0 + (r0_1 + 126*x0), tmp2, r0_mask & xmask)
    tl.store(out_ptr2 + (r0_1 + 126*x0), tmp29, r0_mask & xmask)
    tl.store(out_ptr3 + (x0), tmp24, xmask)
    tl.store(out_ptr5 + (x0), tmp37, xmask)
    tl.store(out_ptr7 + (x0), tmp44, xmask)
    tl.store(out_ptr0 + (x0), tmp12, xmask)

















import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_mish_1(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2016
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 > tmp0
    tmp3 = tl.full([1], 1, tl.int8)
    tmp4 = tl.full([1], 0, tl.int8)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = triton_helpers.maximum(tmp1, tmp0)
    tmp7 = 20.0
    tmp8 = tmp6 > tmp7
    tmp9 = tl_math.exp(tmp6)
    tmp10 = libdevice.log1p(tmp9)
    tmp11 = tl.where(tmp8, tmp6, tmp10)
    tmp12 = libdevice.tanh(tmp11)
    tmp13 = tmp6 * tmp12
    tl.store(out_ptr0 + (x0), tmp5, xmask)
    tl.store(out_ptr1 + (x0), tmp13, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_2(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)















import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_addmm_stack_tanh_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp4 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = libdevice.tanh(tmp6)
    tl.store(in_out_ptr0 + (x0), tmp7, xmask)
    tl.store(out_ptr0 + (x0), tmp7, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (1 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (3 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (5 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (6 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_11(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (7 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (8 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_13(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (9 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_14(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (10 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_15(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (11 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_16(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (12 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_17(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (13 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_18(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (14 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_19(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (15 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_20(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (16 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_21(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (17 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_22(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (18 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_23(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (19 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_24(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (20 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_25(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (21 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_26(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (22 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_27(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (23 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_28(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (24 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_29(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (25 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_30(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (26 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_31(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (27 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_32(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (28 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_33(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (29 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_34(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (30 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_35(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (31 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_36(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (32 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_37(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (33 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_38(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (34 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_39(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (35 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_40(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (36 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_41(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (37 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_42(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (38 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_43(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (39 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_44(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (40 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_45(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (41 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_46(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (42 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_47(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (43 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_48(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (44 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_49(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (45 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_50(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (46 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_51(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (47 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_52(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (48 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_53(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (49 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_54(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (50 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_55(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (51 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_56(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (52 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_57(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (53 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_58(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (54 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_59(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (55 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_60(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (56 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_61(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (57 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_62(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (58 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_63(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (59 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_64(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (60 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_65(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (61 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_66(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (62 + 63*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)















import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_addmm_tanh_tanh_backward_67(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp3 = tl.load(in_ptr2 + (x0), xmask)
    tmp4 = tl.load(in_ptr3 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = libdevice.tanh(tmp6)
    tmp8 = tmp7 * tmp7
    tmp9 = 1.0
    tmp10 = tmp9 - tmp8
    tl.store(out_ptr0 + (x0), tmp7, xmask)
    tl.store(out_ptr1 + (x0), tmp10, xmask)





extern "C"  void kernel(const int64_t* in_ptr0,
                       int64_t* out_ptr0)
{
    {

        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(10L); x0+=static_cast<int64_t>(1L))
        {
            {
                {
                    auto tmp0 = in_ptr0[static_cast<int64_t>(0L)];
                    auto tmp1 = x0;
                    auto tmp2 = c10::convert<int32_t>(tmp1);
                    auto tmp3 = static_cast<int64_t>(1);
                    auto tmp4 = static_cast<int64_t>(10);
                    auto tmp5 = randint64_cpu(tmp0, tmp2, tmp3, tmp4);
                    out_ptr0[static_cast<int64_t>(x0)] = tmp5;
                }
            }
        }
    }
}
''')


#include "/tmp/torchinductor_sahanp/3b/c3bi5gk6mslf6u4iaqafhxm64z6u65e3eain4xlary5blqnvv6xx.h"
extern "C"  void kernel(const int64_t* in_ptr0,
                       int64_t* out_ptr0)
{
    {
        {
            {
                auto tmp0 = in_ptr0[static_cast<int64_t>(1L)];
                auto tmp1 = static_cast<int32_t>(0);
                auto tmp2 = static_cast<int64_t>(1);
                auto tmp3 = static_cast<int64_t>(10);
                auto tmp4 = randint64_cpu(tmp0, tmp1, tmp2, tmp3);
                out_ptr0[static_cast<int64_t>(0L)] = tmp4;
            }
        }
    }
}
''')











import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__log_softmax_70(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 64
    r0_numel = 63
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_1), r0_mask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (0))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp5 = tmp4 * tmp0
    tmp6 = tl.where(tmp2, tmp0, tmp5)
    tmp7 = tmp6 - tmp6
    tmp8 = tl_math.exp(tmp7)
    tmp9 = tmp8 / tmp8
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.where(r0_mask & xmask, tmp10, float("-inf"))
    tmp13 = triton_helpers.max2(tmp12, 1)[:, None]
    tmp14 = tmp9 - tmp13
    tmp15 = tl_math.exp(tmp14)
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
    tmp18 = tl.where(r0_mask & xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp19, xmask)












import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__log_softmax_71(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 63
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 64*y0), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (0))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, YBLOCK])
    tmp10 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp5 = tmp4 * tmp0
    tmp6 = tl.where(tmp2, tmp0, tmp5)
    tmp7 = tmp6 - tmp6
    tmp8 = tl_math.exp(tmp7)
    tmp9 = tmp8 / tmp8
    tmp11 = tmp9 - tmp10
    tmp13 = tl_math.log(tmp12)
    tmp14 = tmp11 - tmp13
    tl.store(out_ptr0 + (y0 + 63*x1), tmp14, xmask & ymask)





















import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_fill_mish_mul_sigmoid_sub_72(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2016
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp3 = 20.0
    tmp4 = tmp2 > tmp3
    tmp5 = tl_math.exp(tmp2)
    tmp6 = libdevice.log1p(tmp5)
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp8 = libdevice.tanh(tmp7)
    tmp9 = tl.sigmoid(tmp2)
    tmp10 = tmp2 * tmp9
    tmp11 = tmp8 * tmp8
    tmp12 = 1.0
    tmp13 = tmp12 - tmp11
    tmp14 = tmp10 * tmp13
    tmp15 = tmp8 + tmp14
    tl.store(out_ptr0 + (x0), tmp15, xmask)










import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_73(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.full([1], 63, tl.int64)
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp0, None)







def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12 = args
    args.clear()
    assert_size_stride(primals_1, (32, 1, 3), (3, 3, 1))
    assert_size_stride(primals_2, (32, ), (1, ))
    assert_size_stride(primals_3, (1, 1, 128), (128, 128, 1))
    assert_size_stride(primals_4, (32, ), (1, ))
    assert_size_stride(primals_5, (32, ), (1, ))
    assert_size_stride(primals_6, (32, ), (1, ))
    assert_size_stride(primals_7, (32, ), (1, ))
    assert_size_stride(primals_8, (64, 32), (32, 1))
    assert_size_stride(primals_9, (64, 64), (64, 1))
    assert_size_stride(primals_10, (64, ), (1, ))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_12, (1, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)

        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf0, (1, 32, 126), (4032, 126, 1))
        buf1 = buf0; del buf0
        buf2 = empty_strided_cuda((1, 32, 1, 1, 1), (32, 1, 32, 32, 32), torch.float32)
        buf6 = empty_strided_cuda((1, 32, 1, 1, 126), (4032, 126, 126, 126, 1), torch.float32)
        buf5 = empty_strided_cuda((1, 32, 1, 1, 1), (32, 1, 32, 32, 32), torch.float32)

        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_convolution_mean_0[grid(32)](buf1, primals_2, primals_6, primals_7, primals_4, primals_5, buf2, buf6, buf5, primals_4, primals_5, 32, 126, XBLOCK=1, num_warps=2, num_stages=1)
        del primals_2
        del primals_4
        del primals_5
        del primals_7
        buf7 = empty_strided_cuda((1, 32, 1, 63), (2048, 63, 63, 1), torch.int8)
        buf9 = empty_strided_cuda((1, 32, 63), (2016, 63, 1), torch.float32)

        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_mish_1[grid(2016)](buf6, buf7, buf9, 2016, XBLOCK=128, num_warps=4, num_stages=1)
        buf8 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2[grid(64)](buf8, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf10 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf8, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf10)
        buf11 = empty_strided_cuda((1, 32), (32, 1), torch.float32)

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_3[grid(32)](buf9, buf11, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf12 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf11, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf12)
        buf13 = buf10; del buf10
        buf324 = empty_strided_cuda((63, 64), (64, 1), torch.float32)
        buf261 = reinterpret_tensor(buf324, (1, 64), (64, 1), 0)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf13, primals_11, buf12, primals_10, buf261, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf14 = buf12; del buf12

        extern_kernels.mm(buf13, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf14)
        buf15 = buf11; del buf11

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_5[grid(32)](buf9, buf15, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf16 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf15, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf16)
        buf17 = buf14; del buf14
        buf262 = reinterpret_tensor(buf324, (1, 64), (64, 1), 64)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf17, primals_11, buf16, primals_10, buf262, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf18 = buf16; del buf16

        extern_kernels.mm(buf17, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf18)
        buf19 = buf15; del buf15

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_6[grid(32)](buf9, buf19, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf20 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf19, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf20)
        buf21 = buf18; del buf18
        buf263 = reinterpret_tensor(buf324, (1, 64), (64, 1), 128)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf21, primals_11, buf20, primals_10, buf263, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf22 = buf20; del buf20

        extern_kernels.mm(buf21, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf22)
        buf23 = buf19; del buf19

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_7[grid(32)](buf9, buf23, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf24 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf23, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf24)
        buf25 = buf22; del buf22
        buf264 = reinterpret_tensor(buf324, (1, 64), (64, 1), 192)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf25, primals_11, buf24, primals_10, buf264, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf26 = buf24; del buf24

        extern_kernels.mm(buf25, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf26)
        buf27 = buf23; del buf23

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_8[grid(32)](buf9, buf27, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf28 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf27, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf28)
        buf29 = buf26; del buf26
        buf265 = reinterpret_tensor(buf324, (1, 64), (64, 1), 256)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf29, primals_11, buf28, primals_10, buf265, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf30 = buf28; del buf28

        extern_kernels.mm(buf29, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf30)
        buf31 = buf27; del buf27

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_9[grid(32)](buf9, buf31, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf32 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf31, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf32)
        buf33 = buf30; del buf30
        buf266 = reinterpret_tensor(buf324, (1, 64), (64, 1), 320)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf33, primals_11, buf32, primals_10, buf266, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf34 = buf32; del buf32

        extern_kernels.mm(buf33, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf34)
        buf35 = buf31; del buf31

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_10[grid(32)](buf9, buf35, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf36 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf35, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf36)
        buf37 = buf34; del buf34
        buf267 = reinterpret_tensor(buf324, (1, 64), (64, 1), 384)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf37, primals_11, buf36, primals_10, buf267, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf38 = buf36; del buf36

        extern_kernels.mm(buf37, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf38)
        buf39 = buf35; del buf35

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_11[grid(32)](buf9, buf39, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf40 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf39, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf40)
        buf41 = buf38; del buf38
        buf268 = reinterpret_tensor(buf324, (1, 64), (64, 1), 448)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf41, primals_11, buf40, primals_10, buf268, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf42 = buf40; del buf40

        extern_kernels.mm(buf41, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf42)
        buf43 = buf39; del buf39

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_12[grid(32)](buf9, buf43, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf44 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf43, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf44)
        buf45 = buf42; del buf42
        buf269 = reinterpret_tensor(buf324, (1, 64), (64, 1), 512)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf45, primals_11, buf44, primals_10, buf269, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf46 = buf44; del buf44

        extern_kernels.mm(buf45, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf46)
        buf47 = buf43; del buf43

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_13[grid(32)](buf9, buf47, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf48 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf47, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf48)
        buf49 = buf46; del buf46
        buf270 = reinterpret_tensor(buf324, (1, 64), (64, 1), 576)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf49, primals_11, buf48, primals_10, buf270, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf50 = buf48; del buf48

        extern_kernels.mm(buf49, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf50)
        buf51 = buf47; del buf47

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_14[grid(32)](buf9, buf51, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf52 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf51, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf52)
        buf53 = buf50; del buf50
        buf271 = reinterpret_tensor(buf324, (1, 64), (64, 1), 640)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf53, primals_11, buf52, primals_10, buf271, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf54 = buf52; del buf52

        extern_kernels.mm(buf53, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf54)
        buf55 = buf51; del buf51

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_15[grid(32)](buf9, buf55, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf56 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf55, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf56)
        buf57 = buf54; del buf54
        buf272 = reinterpret_tensor(buf324, (1, 64), (64, 1), 704)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf57, primals_11, buf56, primals_10, buf272, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf58 = buf56; del buf56

        extern_kernels.mm(buf57, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf58)
        buf59 = buf55; del buf55

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_16[grid(32)](buf9, buf59, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf60 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf59, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf60)
        buf61 = buf58; del buf58
        buf273 = reinterpret_tensor(buf324, (1, 64), (64, 1), 768)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf61, primals_11, buf60, primals_10, buf273, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf62 = buf60; del buf60

        extern_kernels.mm(buf61, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf62)
        buf63 = buf59; del buf59

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_17[grid(32)](buf9, buf63, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf64 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf63, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf64)
        buf65 = buf62; del buf62
        buf274 = reinterpret_tensor(buf324, (1, 64), (64, 1), 832)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf65, primals_11, buf64, primals_10, buf274, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf66 = buf64; del buf64

        extern_kernels.mm(buf65, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf66)
        buf67 = buf63; del buf63

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_18[grid(32)](buf9, buf67, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf68 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf67, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf68)
        buf69 = buf66; del buf66
        buf275 = reinterpret_tensor(buf324, (1, 64), (64, 1), 896)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf69, primals_11, buf68, primals_10, buf275, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf70 = buf68; del buf68

        extern_kernels.mm(buf69, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf70)
        buf71 = buf67; del buf67

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_19[grid(32)](buf9, buf71, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf72 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf71, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf72)
        buf73 = buf70; del buf70
        buf276 = reinterpret_tensor(buf324, (1, 64), (64, 1), 960)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf73, primals_11, buf72, primals_10, buf276, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf74 = buf72; del buf72

        extern_kernels.mm(buf73, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf74)
        buf75 = buf71; del buf71

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_20[grid(32)](buf9, buf75, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf76 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf75, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf76)
        buf77 = buf74; del buf74
        buf277 = reinterpret_tensor(buf324, (1, 64), (64, 1), 1024)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf77, primals_11, buf76, primals_10, buf277, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf78 = buf76; del buf76

        extern_kernels.mm(buf77, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf78)
        buf79 = buf75; del buf75

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_21[grid(32)](buf9, buf79, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf80 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf79, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf80)
        buf81 = buf78; del buf78
        buf278 = reinterpret_tensor(buf324, (1, 64), (64, 1), 1088)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf81, primals_11, buf80, primals_10, buf278, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf82 = buf80; del buf80

        extern_kernels.mm(buf81, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf82)
        buf83 = buf79; del buf79

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_22[grid(32)](buf9, buf83, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf84 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf83, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf84)
        buf85 = buf82; del buf82
        buf279 = reinterpret_tensor(buf324, (1, 64), (64, 1), 1152)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf85, primals_11, buf84, primals_10, buf279, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf86 = buf84; del buf84

        extern_kernels.mm(buf85, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf86)
        buf87 = buf83; del buf83

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_23[grid(32)](buf9, buf87, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf88 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf87, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf88)
        buf89 = buf86; del buf86
        buf280 = reinterpret_tensor(buf324, (1, 64), (64, 1), 1216)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf89, primals_11, buf88, primals_10, buf280, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf90 = buf88; del buf88

        extern_kernels.mm(buf89, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf90)
        buf91 = buf87; del buf87

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_24[grid(32)](buf9, buf91, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf92 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf91, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf92)
        buf93 = buf90; del buf90
        buf281 = reinterpret_tensor(buf324, (1, 64), (64, 1), 1280)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf93, primals_11, buf92, primals_10, buf281, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf94 = buf92; del buf92

        extern_kernels.mm(buf93, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf94)
        buf95 = buf91; del buf91

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_25[grid(32)](buf9, buf95, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf96 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf95, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf96)
        buf97 = buf94; del buf94
        buf282 = reinterpret_tensor(buf324, (1, 64), (64, 1), 1344)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf97, primals_11, buf96, primals_10, buf282, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf98 = buf96; del buf96

        extern_kernels.mm(buf97, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf98)
        buf99 = buf95; del buf95

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_26[grid(32)](buf9, buf99, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf100 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf99, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf100)
        buf101 = buf98; del buf98
        buf283 = reinterpret_tensor(buf324, (1, 64), (64, 1), 1408)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf101, primals_11, buf100, primals_10, buf283, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf102 = buf100; del buf100

        extern_kernels.mm(buf101, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf102)
        buf103 = buf99; del buf99

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_27[grid(32)](buf9, buf103, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf104 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf103, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf104)
        buf105 = buf102; del buf102
        buf284 = reinterpret_tensor(buf324, (1, 64), (64, 1), 1472)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf105, primals_11, buf104, primals_10, buf284, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf106 = buf104; del buf104

        extern_kernels.mm(buf105, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf106)
        buf107 = buf103; del buf103

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_28[grid(32)](buf9, buf107, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf108 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf107, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf108)
        buf109 = buf106; del buf106
        buf285 = reinterpret_tensor(buf324, (1, 64), (64, 1), 1536)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf109, primals_11, buf108, primals_10, buf285, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf110 = buf108; del buf108

        extern_kernels.mm(buf109, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf110)
        buf111 = buf107; del buf107

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_29[grid(32)](buf9, buf111, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf112 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf111, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf112)
        buf113 = buf110; del buf110
        buf286 = reinterpret_tensor(buf324, (1, 64), (64, 1), 1600)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf113, primals_11, buf112, primals_10, buf286, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf114 = buf112; del buf112

        extern_kernels.mm(buf113, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf114)
        buf115 = buf111; del buf111

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_30[grid(32)](buf9, buf115, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf116 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf115, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf116)
        buf117 = buf114; del buf114
        buf287 = reinterpret_tensor(buf324, (1, 64), (64, 1), 1664)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf117, primals_11, buf116, primals_10, buf287, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf118 = buf116; del buf116

        extern_kernels.mm(buf117, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf118)
        buf119 = buf115; del buf115

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_31[grid(32)](buf9, buf119, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf120 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf119, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf120)
        buf121 = buf118; del buf118
        buf288 = reinterpret_tensor(buf324, (1, 64), (64, 1), 1728)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf121, primals_11, buf120, primals_10, buf288, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf122 = buf120; del buf120

        extern_kernels.mm(buf121, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf122)
        buf123 = buf119; del buf119

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_32[grid(32)](buf9, buf123, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf124 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf123, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf124)
        buf125 = buf122; del buf122
        buf289 = reinterpret_tensor(buf324, (1, 64), (64, 1), 1792)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf125, primals_11, buf124, primals_10, buf289, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf126 = buf124; del buf124

        extern_kernels.mm(buf125, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf126)
        buf127 = buf123; del buf123

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_33[grid(32)](buf9, buf127, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf128 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf127, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf128)
        buf129 = buf126; del buf126
        buf290 = reinterpret_tensor(buf324, (1, 64), (64, 1), 1856)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf129, primals_11, buf128, primals_10, buf290, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf130 = buf128; del buf128

        extern_kernels.mm(buf129, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf130)
        buf131 = buf127; del buf127

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_34[grid(32)](buf9, buf131, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf132 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf131, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf132)
        buf133 = buf130; del buf130
        buf291 = reinterpret_tensor(buf324, (1, 64), (64, 1), 1920)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf133, primals_11, buf132, primals_10, buf291, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf134 = buf132; del buf132

        extern_kernels.mm(buf133, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf134)
        buf135 = buf131; del buf131

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_35[grid(32)](buf9, buf135, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf136 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf135, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf136)
        buf137 = buf134; del buf134
        buf292 = reinterpret_tensor(buf324, (1, 64), (64, 1), 1984)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf137, primals_11, buf136, primals_10, buf292, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf138 = buf136; del buf136

        extern_kernels.mm(buf137, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf138)
        buf139 = buf135; del buf135

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_36[grid(32)](buf9, buf139, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf140 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf139, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf140)
        buf141 = buf138; del buf138
        buf293 = reinterpret_tensor(buf324, (1, 64), (64, 1), 2048)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf141, primals_11, buf140, primals_10, buf293, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf142 = buf140; del buf140

        extern_kernels.mm(buf141, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf142)
        buf143 = buf139; del buf139

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_37[grid(32)](buf9, buf143, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf144 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf143, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf144)
        buf145 = buf142; del buf142
        buf294 = reinterpret_tensor(buf324, (1, 64), (64, 1), 2112)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf145, primals_11, buf144, primals_10, buf294, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf146 = buf144; del buf144

        extern_kernels.mm(buf145, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf146)
        buf147 = buf143; del buf143

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_38[grid(32)](buf9, buf147, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf148 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf147, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf148)
        buf149 = buf146; del buf146
        buf295 = reinterpret_tensor(buf324, (1, 64), (64, 1), 2176)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf149, primals_11, buf148, primals_10, buf295, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf150 = buf148; del buf148

        extern_kernels.mm(buf149, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf150)
        buf151 = buf147; del buf147

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_39[grid(32)](buf9, buf151, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf152 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf151, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf152)
        buf153 = buf150; del buf150
        buf296 = reinterpret_tensor(buf324, (1, 64), (64, 1), 2240)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf153, primals_11, buf152, primals_10, buf296, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf154 = buf152; del buf152

        extern_kernels.mm(buf153, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf154)
        buf155 = buf151; del buf151

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_40[grid(32)](buf9, buf155, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf156 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf155, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf156)
        buf157 = buf154; del buf154
        buf297 = reinterpret_tensor(buf324, (1, 64), (64, 1), 2304)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf157, primals_11, buf156, primals_10, buf297, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf158 = buf156; del buf156

        extern_kernels.mm(buf157, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf158)
        buf159 = buf155; del buf155

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_41[grid(32)](buf9, buf159, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf160 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf159, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf160)
        buf161 = buf158; del buf158
        buf298 = reinterpret_tensor(buf324, (1, 64), (64, 1), 2368)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf161, primals_11, buf160, primals_10, buf298, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf162 = buf160; del buf160

        extern_kernels.mm(buf161, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf162)
        buf163 = buf159; del buf159

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_42[grid(32)](buf9, buf163, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf164 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf163, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf164)
        buf165 = buf162; del buf162
        buf299 = reinterpret_tensor(buf324, (1, 64), (64, 1), 2432)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf165, primals_11, buf164, primals_10, buf299, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf166 = buf164; del buf164

        extern_kernels.mm(buf165, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf166)
        buf167 = buf163; del buf163

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_43[grid(32)](buf9, buf167, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf168 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf167, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf168)
        buf169 = buf166; del buf166
        buf300 = reinterpret_tensor(buf324, (1, 64), (64, 1), 2496)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf169, primals_11, buf168, primals_10, buf300, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf170 = buf168; del buf168

        extern_kernels.mm(buf169, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf170)
        buf171 = buf167; del buf167

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_44[grid(32)](buf9, buf171, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf172 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf171, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf172)
        buf173 = buf170; del buf170
        buf301 = reinterpret_tensor(buf324, (1, 64), (64, 1), 2560)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf173, primals_11, buf172, primals_10, buf301, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf174 = buf172; del buf172

        extern_kernels.mm(buf173, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf174)
        buf175 = buf171; del buf171

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_45[grid(32)](buf9, buf175, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf176 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf175, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf176)
        buf177 = buf174; del buf174
        buf302 = reinterpret_tensor(buf324, (1, 64), (64, 1), 2624)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf177, primals_11, buf176, primals_10, buf302, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf178 = buf176; del buf176

        extern_kernels.mm(buf177, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf178)
        buf179 = buf175; del buf175

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_46[grid(32)](buf9, buf179, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf180 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf179, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf180)
        buf181 = buf178; del buf178
        buf303 = reinterpret_tensor(buf324, (1, 64), (64, 1), 2688)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf181, primals_11, buf180, primals_10, buf303, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf182 = buf180; del buf180

        extern_kernels.mm(buf181, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf182)
        buf183 = buf179; del buf179

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_47[grid(32)](buf9, buf183, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf184 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf183, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf184)
        buf185 = buf182; del buf182
        buf304 = reinterpret_tensor(buf324, (1, 64), (64, 1), 2752)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf185, primals_11, buf184, primals_10, buf304, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf186 = buf184; del buf184

        extern_kernels.mm(buf185, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf186)
        buf187 = buf183; del buf183

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_48[grid(32)](buf9, buf187, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf188 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf187, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf188)
        buf189 = buf186; del buf186
        buf305 = reinterpret_tensor(buf324, (1, 64), (64, 1), 2816)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf189, primals_11, buf188, primals_10, buf305, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf190 = buf188; del buf188

        extern_kernels.mm(buf189, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf190)
        buf191 = buf187; del buf187

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_49[grid(32)](buf9, buf191, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf192 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf191, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf192)
        buf193 = buf190; del buf190
        buf306 = reinterpret_tensor(buf324, (1, 64), (64, 1), 2880)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf193, primals_11, buf192, primals_10, buf306, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf194 = buf192; del buf192

        extern_kernels.mm(buf193, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf194)
        buf195 = buf191; del buf191

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_50[grid(32)](buf9, buf195, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf196 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf195, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf196)
        buf197 = buf194; del buf194
        buf307 = reinterpret_tensor(buf324, (1, 64), (64, 1), 2944)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf197, primals_11, buf196, primals_10, buf307, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf198 = buf196; del buf196

        extern_kernels.mm(buf197, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf198)
        buf199 = buf195; del buf195

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_51[grid(32)](buf9, buf199, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf200 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf199, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf200)
        buf201 = buf198; del buf198
        buf308 = reinterpret_tensor(buf324, (1, 64), (64, 1), 3008)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf201, primals_11, buf200, primals_10, buf308, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf202 = buf200; del buf200

        extern_kernels.mm(buf201, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf202)
        buf203 = buf199; del buf199

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_52[grid(32)](buf9, buf203, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf204 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf203, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf204)
        buf205 = buf202; del buf202
        buf309 = reinterpret_tensor(buf324, (1, 64), (64, 1), 3072)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf205, primals_11, buf204, primals_10, buf309, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf206 = buf204; del buf204

        extern_kernels.mm(buf205, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf206)
        buf207 = buf203; del buf203

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_53[grid(32)](buf9, buf207, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf208 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf207, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf208)
        buf209 = buf206; del buf206
        buf310 = reinterpret_tensor(buf324, (1, 64), (64, 1), 3136)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf209, primals_11, buf208, primals_10, buf310, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf210 = buf208; del buf208

        extern_kernels.mm(buf209, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf210)
        buf211 = buf207; del buf207

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_54[grid(32)](buf9, buf211, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf212 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf211, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf212)
        buf213 = buf210; del buf210
        buf311 = reinterpret_tensor(buf324, (1, 64), (64, 1), 3200)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf213, primals_11, buf212, primals_10, buf311, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf214 = buf212; del buf212

        extern_kernels.mm(buf213, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf214)
        buf215 = buf211; del buf211

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_55[grid(32)](buf9, buf215, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf216 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf215, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf216)
        buf217 = buf214; del buf214
        buf312 = reinterpret_tensor(buf324, (1, 64), (64, 1), 3264)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf217, primals_11, buf216, primals_10, buf312, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf218 = buf216; del buf216

        extern_kernels.mm(buf217, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf218)
        buf219 = buf215; del buf215

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_56[grid(32)](buf9, buf219, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf220 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf219, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf220)
        buf221 = buf218; del buf218
        buf313 = reinterpret_tensor(buf324, (1, 64), (64, 1), 3328)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf221, primals_11, buf220, primals_10, buf313, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf222 = buf220; del buf220

        extern_kernels.mm(buf221, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf222)
        buf223 = buf219; del buf219

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_57[grid(32)](buf9, buf223, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf224 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf223, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf224)
        buf225 = buf222; del buf222
        buf314 = reinterpret_tensor(buf324, (1, 64), (64, 1), 3392)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf225, primals_11, buf224, primals_10, buf314, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf226 = buf224; del buf224

        extern_kernels.mm(buf225, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf226)
        buf227 = buf223; del buf223

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_58[grid(32)](buf9, buf227, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf228 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf227, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf228)
        buf229 = buf226; del buf226
        buf315 = reinterpret_tensor(buf324, (1, 64), (64, 1), 3456)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf229, primals_11, buf228, primals_10, buf315, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf230 = buf228; del buf228

        extern_kernels.mm(buf229, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf230)
        buf231 = buf227; del buf227

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_59[grid(32)](buf9, buf231, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf232 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf231, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf232)
        buf233 = buf230; del buf230
        buf316 = reinterpret_tensor(buf324, (1, 64), (64, 1), 3520)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf233, primals_11, buf232, primals_10, buf316, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf234 = buf232; del buf232

        extern_kernels.mm(buf233, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf234)
        buf235 = buf231; del buf231

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_60[grid(32)](buf9, buf235, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf236 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf235, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf236)
        buf237 = buf234; del buf234
        buf317 = reinterpret_tensor(buf324, (1, 64), (64, 1), 3584)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf237, primals_11, buf236, primals_10, buf317, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf238 = buf236; del buf236

        extern_kernels.mm(buf237, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf238)
        buf239 = buf235; del buf235

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_61[grid(32)](buf9, buf239, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf240 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf239, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf240)
        buf241 = buf238; del buf238
        buf318 = reinterpret_tensor(buf324, (1, 64), (64, 1), 3648)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf241, primals_11, buf240, primals_10, buf318, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf242 = buf240; del buf240

        extern_kernels.mm(buf241, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf242)
        buf243 = buf239; del buf239

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_62[grid(32)](buf9, buf243, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf244 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf243, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf244)
        buf245 = buf242; del buf242
        buf319 = reinterpret_tensor(buf324, (1, 64), (64, 1), 3712)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf245, primals_11, buf244, primals_10, buf319, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf246 = buf244; del buf244

        extern_kernels.mm(buf245, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf246)
        buf247 = buf243; del buf243

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_63[grid(32)](buf9, buf247, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf248 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf247, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf248)
        buf249 = buf246; del buf246
        buf320 = reinterpret_tensor(buf324, (1, 64), (64, 1), 3776)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf249, primals_11, buf248, primals_10, buf320, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf250 = buf248; del buf248

        extern_kernels.mm(buf249, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf250)
        buf251 = buf247; del buf247

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_64[grid(32)](buf9, buf251, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf252 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf251, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf252)
        buf253 = buf250; del buf250
        buf321 = reinterpret_tensor(buf324, (1, 64), (64, 1), 3840)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf253, primals_11, buf252, primals_10, buf321, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf254 = buf252; del buf252

        extern_kernels.mm(buf253, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf254)
        buf255 = buf251; del buf251

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_65[grid(32)](buf9, buf255, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf256 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf255, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf256)
        buf257 = buf254; del buf254
        buf322 = reinterpret_tensor(buf324, (1, 64), (64, 1), 3904)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_4[grid(64)](buf257, primals_11, buf256, primals_10, buf322, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf258 = buf256; del buf256

        extern_kernels.mm(buf257, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), out=buf258)
        buf259 = buf255; del buf255

        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_66[grid(32)](buf9, buf259, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf260 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        extern_kernels.mm(buf259, reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf260)
        del buf259
        buf323 = reinterpret_tensor(buf324, (1, 64), (64, 1), 3968)
        buf333 = empty_strided_cuda((1, 64), (64, 1), torch.float32)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_tanh_backward_67[grid(64)](buf258, primals_11, buf260, primals_10, buf323, buf333, 64, XBLOCK=64, num_warps=1, num_stages=1)
        del primals_10
        del primals_11
    buf325 = empty_strided_cpu((2, ), (1, ), torch.int64)

    aten.randint.low_out(-9223372036854775808, 9223372036854775807, [2], out=buf325)
    buf326 = empty_strided_cpu((1, 10), (10, 1), torch.int64)
    cpp_fused_randint_68(buf325, buf326)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf327 = empty_strided_cuda((1, 10), (10, 1), torch.int64)
        buf327.copy_(buf326, False)
        del buf326
    buf328 = empty_strided_cpu((1, ), (1, ), torch.int64)
    cpp_fused_randint_69(buf325, buf328)
    del buf325
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf329 = empty_strided_cuda((1, ), (1, ), torch.int64)
        buf329.copy_(buf328, False)
        del buf328
        buf330 = reinterpret_tensor(buf260, (1, 64, 1), (64, 1, 64), 0); del buf260
        buf331 = reinterpret_tensor(buf258, (1, 64, 1), (64, 1, 64), 0); del buf258

        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax_70[grid(64)](buf324, primals_12, buf330, buf331, 64, 63, XBLOCK=32, num_warps=8, num_stages=1)
        buf332 = empty_strided_cuda((1, 64, 63), (4032, 63, 1), torch.float32)

        stream0 = get_raw_stream(0)
        triton_poi_fused__log_softmax_71[grid(63, 64)](buf324, primals_12, buf330, buf331, buf332, 63, 64, XBLOCK=64, YBLOCK=4, num_warps=4, num_stages=1)
        del buf330
        del buf331
        buf334 = empty_strided_cuda((1, 32, 63), (2016, 63, 1), torch.float32)

        stream0 = get_raw_stream(0)
        triton_poi_fused_add_fill_mish_mul_sigmoid_sub_72[grid(2016)](buf6, buf334, 2016, XBLOCK=128, num_warps=4, num_stages=1)
        buf343 = empty_strided_cuda((1, ), (1, ), torch.int64)

        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_73[grid(1)](buf343, 1, XBLOCK=1, num_warps=1, num_stages=1)
    return (buf332, buf327, buf343, buf329, primals_1, primals_3, primals_6, primals_12, buf1, reinterpret_tensor(buf5, (32, ), (1, ), 0), reinterpret_tensor(buf6, (1, 32, 1, 126), (4032, 126, 126, 1), 0), buf7, buf8, reinterpret_tensor(buf9, (1, 32), (2016, 63), 0), buf13, reinterpret_tensor(buf9, (1, 32), (2016, 63), 1), buf17, reinterpret_tensor(buf9, (1, 32), (2016, 63), 2), buf21, reinterpret_tensor(buf9, (1, 32), (2016, 63), 3), buf25, reinterpret_tensor(buf9, (1, 32), (2016, 63), 4), buf29, reinterpret_tensor(buf9, (1, 32), (2016, 63), 5), buf33, reinterpret_tensor(buf9, (1, 32), (2016, 63), 6), buf37, reinterpret_tensor(buf9, (1, 32), (2016, 63), 7), buf41, reinterpret_tensor(buf9, (1, 32), (2016, 63), 8), buf45, reinterpret_tensor(buf9, (1, 32), (2016, 63), 9), buf49, reinterpret_tensor(buf9, (1, 32), (2016, 63), 10), buf53, reinterpret_tensor(buf9, (1, 32), (2016, 63), 11), buf57, reinterpret_tensor(buf9, (1, 32), (2016, 63), 12), buf61, reinterpret_tensor(buf9, (1, 32), (2016, 63), 13), buf65, reinterpret_tensor(buf9, (1, 32), (2016, 63), 14), buf69, reinterpret_tensor(buf9, (1, 32), (2016, 63), 15), buf73, reinterpret_tensor(buf9, (1, 32), (2016, 63), 16), buf77, reinterpret_tensor(buf9, (1, 32), (2016, 63), 17), buf81, reinterpret_tensor(buf9, (1, 32), (2016, 63), 18), buf85, reinterpret_tensor(buf9, (1, 32), (2016, 63), 19), buf89, reinterpret_tensor(buf9, (1, 32), (2016, 63), 20), buf93, reinterpret_tensor(buf9, (1, 32), (2016, 63), 21), buf97, reinterpret_tensor(buf9, (1, 32), (2016, 63), 22), buf101, reinterpret_tensor(buf9, (1, 32), (2016, 63), 23), buf105, reinterpret_tensor(buf9, (1, 32), (2016, 63), 24), buf109, reinterpret_tensor(buf9, (1, 32), (2016, 63), 25), buf113, reinterpret_tensor(buf9, (1, 32), (2016, 63), 26), buf117, reinterpret_tensor(buf9, (1, 32), (2016, 63), 27), buf121, reinterpret_tensor(buf9, (1, 32), (2016, 63), 28), buf125, reinterpret_tensor(buf9, (1, 32), (2016, 63), 29), buf129, reinterpret_tensor(buf9, (1, 32), (2016, 63), 30), buf133, reinterpret_tensor(buf9, (1, 32), (2016, 63), 31), buf137, reinterpret_tensor(buf9, (1, 32), (2016, 63), 32), buf141, reinterpret_tensor(buf9, (1, 32), (2016, 63), 33), buf145, reinterpret_tensor(buf9, (1, 32), (2016, 63), 34), buf149, reinterpret_tensor(buf9, (1, 32), (2016, 63), 35), buf153, reinterpret_tensor(buf9, (1, 32), (2016, 63), 36), buf157, reinterpret_tensor(buf9, (1, 32), (2016, 63), 37), buf161, reinterpret_tensor(buf9, (1, 32), (2016, 63), 38), buf165, reinterpret_tensor(buf9, (1, 32), (2016, 63), 39), buf169, reinterpret_tensor(buf9, (1, 32), (2016, 63), 40), buf173, reinterpret_tensor(buf9, (1, 32), (2016, 63), 41), buf177, reinterpret_tensor(buf9, (1, 32), (2016, 63), 42), buf181, reinterpret_tensor(buf9, (1, 32), (2016, 63), 43), buf185, reinterpret_tensor(buf9, (1, 32), (2016, 63), 44), buf189, reinterpret_tensor(buf9, (1, 32), (2016, 63), 45), buf193, reinterpret_tensor(buf9, (1, 32), (2016, 63), 46), buf197, reinterpret_tensor(buf9, (1, 32), (2016, 63), 47), buf201, reinterpret_tensor(buf9, (1, 32), (2016, 63), 48), buf205, reinterpret_tensor(buf9, (1, 32), (2016, 63), 49), buf209, reinterpret_tensor(buf9, (1, 32), (2016, 63), 50), buf213, reinterpret_tensor(buf9, (1, 32), (2016, 63), 51), buf217, reinterpret_tensor(buf9, (1, 32), (2016, 63), 52), buf221, reinterpret_tensor(buf9, (1, 32), (2016, 63), 53), buf225, reinterpret_tensor(buf9, (1, 32), (2016, 63), 54), buf229, reinterpret_tensor(buf9, (1, 32), (2016, 63), 55), buf233, reinterpret_tensor(buf9, (1, 32), (2016, 63), 56), buf237, reinterpret_tensor(buf9, (1, 32), (2016, 63), 57), buf241, reinterpret_tensor(buf9, (1, 32), (2016, 63), 58), buf245, reinterpret_tensor(buf9, (1, 32), (2016, 63), 59), buf249, reinterpret_tensor(buf9, (1, 32), (2016, 63), 60), buf253, reinterpret_tensor(buf9, (1, 32), (2016, 63), 61), buf257, reinterpret_tensor(buf9, (1, 32), (2016, 63), 62), buf324, buf332, buf333, primals_8, primals_9, buf334, reinterpret_tensor(buf2, (1, 32, 1, 1, 1), (32, 1, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 1, 3), (3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((1, 1, 128), (128, 128, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
