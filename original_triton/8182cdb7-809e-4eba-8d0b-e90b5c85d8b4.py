# AOT ID: ['3_inference']
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


# kernel path: /tmp/torchinductor_sahanp/qq/cqqtc6qomm4rvf4acq7a4irmcpkw6p6jpxzpliriozvmdqzivudp.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.copy]
# Source node to ATen node mapping:
#   x => copy
# Graph fragment:
#   %copy : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_5, %slice_6), kwargs = {})
#   %slice_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_tensor_1, %copy, 2, 1, 11), kwargs = {})
#   %slice_scatter_default_1 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_tensor, %slice_scatter_default, 3, 1, 11), kwargs = {})
#   %slice_scatter_default_2 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%empty, %slice_scatter_default_1, 4, 1, 11), kwargs = {})
#   %slice_scatter_default_3 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_2, %slice_15, 4, 0, 1), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_copy_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 12)
    x1 = ((xindex // 12) % 12)
    x2 = ((xindex // 144) % 12)
    x3 = xindex // 1728
    x5 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = 10 + x0
    tmp4 = tl.full([1], 1, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tl.full([1], 11, tl.int64)
    tmp7 = tmp3 < tmp6
    tmp8 = tmp5 & tmp7
    tmp9 = tmp8 & tmp2
    tmp10 = x1
    tmp11 = tl.full([1], 1, tl.int64)
    tmp12 = tmp10 >= tmp11
    tmp13 = tl.full([1], 11, tl.int64)
    tmp14 = tmp10 < tmp13
    tmp15 = tmp12 & tmp14
    tmp16 = tmp15 & tmp9
    tmp17 = x2
    tmp18 = tl.full([1], 1, tl.int64)
    tmp19 = tmp17 >= tmp18
    tmp20 = tl.full([1], 11, tl.int64)
    tmp21 = tmp17 < tmp20
    tmp22 = tmp19 & tmp21
    tmp23 = tmp22 & tmp16
    tmp24 = tl.load(in_ptr0 + ((-101) + x0 + 10*x1 + 100*x2 + 1000*x3), tmp23 & xmask, other=0.0)
    tmp25 = tl.load(in_ptr1 + (10 + x5), tmp16 & xmask, other=0.0)
    tmp26 = tl.where(tmp22, tmp24, tmp25)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp16, tmp26, tmp27)
    tmp29 = tl.load(in_ptr1 + (10 + x5), tmp9 & xmask, other=0.0)
    tmp30 = tl.where(tmp15, tmp28, tmp29)
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp9, tmp30, tmp31)
    tmp33 = float("nan")
    tmp34 = tl.where(tmp8, tmp32, tmp33)
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp2, tmp34, tmp35)
    tmp37 = tmp0 >= tmp1
    tmp38 = tl.full([1], 11, tl.int64)
    tmp39 = tmp0 < tmp38
    tmp40 = tmp37 & tmp39
    tmp41 = x1
    tmp42 = tl.full([1], 1, tl.int64)
    tmp43 = tmp41 >= tmp42
    tmp44 = tl.full([1], 11, tl.int64)
    tmp45 = tmp41 < tmp44
    tmp46 = tmp43 & tmp45
    tmp47 = tmp46 & tmp40
    tmp48 = x2
    tmp49 = tl.full([1], 1, tl.int64)
    tmp50 = tmp48 >= tmp49
    tmp51 = tl.full([1], 11, tl.int64)
    tmp52 = tmp48 < tmp51
    tmp53 = tmp50 & tmp52
    tmp54 = tmp53 & tmp47
    tmp55 = tl.load(in_ptr0 + ((-111) + x0 + 10*x1 + 100*x2 + 1000*x3), tmp54 & xmask, other=0.0)
    tmp56 = tl.load(in_ptr1 + (x5), tmp47 & xmask, other=0.0)
    tmp57 = tl.where(tmp53, tmp55, tmp56)
    tmp58 = tl.full(tmp57.shape, 0.0, tmp57.dtype)
    tmp59 = tl.where(tmp47, tmp57, tmp58)
    tmp60 = tl.load(in_ptr1 + (x5), tmp40 & xmask, other=0.0)
    tmp61 = tl.where(tmp46, tmp59, tmp60)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp40, tmp61, tmp62)
    tmp64 = float("nan")
    tmp65 = tl.where(tmp40, tmp63, tmp64)
    tmp66 = tl.where(tmp2, tmp36, tmp65)
    tl.store(out_ptr0 + (x5), tmp66, xmask)




# kernel path: /tmp/torchinductor_sahanp/zl/czlnnfnydaublrkvnws7oth47yitrpto2oda52xkh4o6lyppknyh.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %slice_scatter_default_4 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_3, %slice_20, 4, 11, 12), kwargs = {})
#   %slice_scatter_default_5 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_4, %slice_25, 3, 0, 1), kwargs = {})
#   %slice_scatter_default_6 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_5, %slice_30, 3, 11, 12), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 12) % 12)
    x0 = (xindex % 12)
    x3 = xindex // 12
    x4 = xindex
    tmp40 = tl.load(in_ptr0 + (x4), xmask)
    tmp0 = x1
    tmp1 = tl.full([1], 11, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = (-10) + x1
    tmp4 = tl.full([1], 1, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = x0
    tmp8 = tl.full([1], 11, tl.int64)
    tmp9 = tmp7 >= tmp8
    tmp10 = tmp9 & tmp6
    tmp11 = tl.load(in_ptr0 + (1 + 12*x3), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr0 + (x4), tmp6 & xmask, other=0.0)
    tmp13 = tl.where(tmp9, tmp11, tmp12)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = x0
    tmp17 = tl.full([1], 11, tl.int64)
    tmp18 = tmp16 >= tmp17
    tmp19 = tmp18 & tmp2
    tmp20 = tl.load(in_ptr0 + ((-119) + 12*x3), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.load(in_ptr0 + ((-120) + x4), tmp2 & xmask, other=0.0)
    tmp22 = tl.where(tmp18, tmp20, tmp21)
    tmp23 = tl.where(tmp5, tmp15, tmp22)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp2, tmp23, tmp24)
    tmp26 = tl.full([1], 1, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = x0
    tmp29 = tl.full([1], 11, tl.int64)
    tmp30 = tmp28 >= tmp29
    tmp31 = tmp30 & tmp27
    tmp32 = tl.load(in_ptr0 + (121 + 12*x3), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tl.load(in_ptr0 + (120 + x4), tmp27 & xmask, other=0.0)
    tmp34 = tl.where(tmp30, tmp32, tmp33)
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp27, tmp34, tmp35)
    tmp37 = x0
    tmp38 = tmp37 >= tmp1
    tmp39 = tl.load(in_ptr0 + (1 + 12*x3), tmp38 & xmask, eviction_policy='evict_last', other=0.0)
    tmp41 = tl.where(tmp38, tmp39, tmp40)
    tmp42 = tl.where(tmp27, tmp36, tmp41)
    tmp43 = tl.where(tmp2, tmp25, tmp42)
    tl.store(out_ptr0 + (x4), tmp43, xmask)




# kernel path: /tmp/torchinductor_sahanp/ar/carmrg6plydzy6ywxhhciigflcrotloc5vdjmod7upxul72emgo6.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.copy]
# Source node to ATen node mapping:
#   x_1 => copy_7
# Graph fragment:
#   %copy_7 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_46, %slice_50), kwargs = {})
#   %slice_scatter_default_9 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_tensor_3, %copy_7, 2, 2, 14), kwargs = {})
#   %slice_scatter_default_10 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_tensor_2, %slice_scatter_default_9, 3, 2, 14), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_copy_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 12) % 16)
    x2 = ((xindex // 192) % 16)
    x3 = xindex // 3072
    x4 = (xindex % 192)
    x0 = (xindex % 12)
    x5 = xindex // 12
    x6 = xindex
    tmp39 = tl.load(in_ptr1 + (2 + x0 + 16*x5), xmask)
    tmp0 = x1
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 14, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = x2
    tmp7 = tl.full([1], 2, tl.int64)
    tmp8 = tmp6 >= tmp7
    tmp9 = tl.full([1], 14, tl.int64)
    tmp10 = tmp6 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tmp11 & tmp5
    tmp13 = (-2) + x2
    tmp14 = tl.full([1], 11, tl.int64)
    tmp15 = tmp13 >= tmp14
    tmp16 = tmp15 & tmp12
    tmp17 = (-12) + x2
    tmp18 = tl.full([1], 1, tl.int64)
    tmp19 = tmp17 < tmp18
    tmp20 = tmp19 & tmp16
    tmp21 = tl.load(in_ptr0 + (1416 + x4 + 1728*x3), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.load(in_ptr0 + ((-1752) + x4 + 144*x2 + 1728*x3), tmp16 & xmask, other=0.0)
    tmp23 = tl.where(tmp19, tmp21, tmp22)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp16, tmp23, tmp24)
    tmp26 = tl.full([1], 1, tl.int64)
    tmp27 = tmp13 < tmp26
    tmp28 = tmp27 & tmp12
    tmp29 = tl.load(in_ptr0 + (1416 + x4 + 1728*x3), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr0 + ((-312) + x4 + 144*x2 + 1728*x3), tmp12 & xmask, other=0.0)
    tmp31 = tl.where(tmp27, tmp29, tmp30)
    tmp32 = tl.where(tmp15, tmp25, tmp31)
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp12, tmp32, tmp33)
    tmp35 = tl.load(in_ptr1 + (2 + x0 + 16*x5), tmp5 & xmask, other=0.0)
    tmp36 = tl.where(tmp11, tmp34, tmp35)
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp5, tmp36, tmp37)
    tmp40 = tl.where(tmp5, tmp38, tmp39)
    tl.store(out_ptr0 + (x6), tmp40, xmask)




# kernel path: /tmp/torchinductor_sahanp/hd/chdc2u53i624ikxrd2ez7xrqp2c6pla6tkdhumhnwkqlj56rds7x.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %slice_scatter_default_11 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%empty_1, %slice_scatter_default_10, 4, 2, 14), kwargs = {})
#   %slice_scatter_default_12 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_11, %slice_59, 4, 0, 2), kwargs = {})
#   %slice_scatter_default_13 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_12, %slice_64, 4, 14, 16), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 14, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = (-12) + x0
    tmp4 = tl.full([1], 2, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = x0
    tmp8 = tl.full([1], 2, tl.int64)
    tmp9 = tmp7 >= tmp8
    tmp10 = tl.full([1], 14, tl.int64)
    tmp11 = tmp7 < tmp10
    tmp12 = tmp9 & tmp11
    tmp13 = tmp12 & tmp6
    tmp14 = tl.load(in_ptr0 + ((-2) + x0 + 12*x1), tmp13, other=0.0)
    tmp15 = float("nan")
    tmp16 = tl.where(tmp12, tmp14, tmp15)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp6, tmp16, tmp17)
    tmp19 = tmp3 >= tmp4
    tmp20 = tl.full([1], 14, tl.int64)
    tmp21 = tmp3 < tmp20
    tmp22 = tmp19 & tmp21
    tmp23 = tmp22 & tmp2
    tmp24 = tl.load(in_ptr0 + ((-14) + x0 + 12*x1), tmp23, other=0.0)
    tmp25 = float("nan")
    tmp26 = tl.where(tmp22, tmp24, tmp25)
    tmp27 = tl.where(tmp5, tmp18, tmp26)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp2, tmp27, tmp28)
    tmp30 = tl.full([1], 2, tl.int64)
    tmp31 = tmp0 < tmp30
    tmp32 = 12 + x0
    tmp33 = tl.full([1], 2, tl.int64)
    tmp34 = tmp32 >= tmp33
    tmp35 = tl.full([1], 14, tl.int64)
    tmp36 = tmp32 < tmp35
    tmp37 = tmp34 & tmp36
    tmp38 = tmp37 & tmp31
    tmp39 = tl.load(in_ptr0 + (10 + x0 + 12*x1), tmp38, other=0.0)
    tmp40 = float("nan")
    tmp41 = tl.where(tmp37, tmp39, tmp40)
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp31, tmp41, tmp42)
    tmp44 = tmp0 >= tmp30
    tmp45 = tmp0 < tmp1
    tmp46 = tmp44 & tmp45
    tmp47 = tl.load(in_ptr0 + ((-2) + x0 + 12*x1), tmp46, other=0.0)
    tmp48 = float("nan")
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tl.where(tmp31, tmp43, tmp49)
    tmp51 = tl.where(tmp2, tmp29, tmp50)
    tl.store(out_ptr0 + (x2), tmp51, None)




# kernel path: /tmp/torchinductor_sahanp/rw/crwfn2xm2awliknpw6fkzfgiayzcwkegsre6vtwptr57mx5p2vxg.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %slice_scatter_default_14 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_13, %slice_69, 3, 0, 2), kwargs = {})
#   %slice_scatter_default_15 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_14, %slice_74, 3, 14, 16), kwargs = {})
#   %slice_scatter_default_16 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_15, %slice_79, 2, 0, 2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 256) % 16)
    x1 = ((xindex // 16) % 16)
    x5 = xindex
    tmp39 = tl.load(in_ptr0 + (x5), None)
    tmp0 = x2
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x1
    tmp4 = tl.full([1], 14, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = (-12) + x1
    tmp8 = tl.full([1], 2, tl.int64)
    tmp9 = tmp7 < tmp8
    tmp10 = tmp9 & tmp6
    tmp11 = tl.load(in_ptr0 + (3072 + x5), tmp10, other=0.0)
    tmp12 = tl.load(in_ptr0 + (2880 + x5), tmp6, other=0.0)
    tmp13 = tl.where(tmp9, tmp11, tmp12)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.full([1], 2, tl.int64)
    tmp17 = tmp3 < tmp16
    tmp18 = tmp17 & tmp2
    tmp19 = tl.load(in_ptr0 + (3264 + x5), tmp18, other=0.0)
    tmp20 = tl.load(in_ptr0 + (3072 + x5), tmp2, other=0.0)
    tmp21 = tl.where(tmp17, tmp19, tmp20)
    tmp22 = tl.where(tmp5, tmp15, tmp21)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp2, tmp22, tmp23)
    tmp25 = x1
    tmp26 = tl.full([1], 14, tl.int64)
    tmp27 = tmp25 >= tmp26
    tmp28 = (-12) + x1
    tmp29 = tl.full([1], 2, tl.int64)
    tmp30 = tmp28 < tmp29
    tmp31 = tmp30 & tmp27
    tmp32 = tl.load(in_ptr0 + (x5), tmp31, other=0.0)
    tmp33 = tl.load(in_ptr0 + ((-192) + x5), tmp27, other=0.0)
    tmp34 = tl.where(tmp30, tmp32, tmp33)
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp27, tmp34, tmp35)
    tmp37 = tmp25 < tmp1
    tmp38 = tl.load(in_ptr0 + (192 + x5), tmp37, other=0.0)
    tmp40 = tl.where(tmp37, tmp38, tmp39)
    tmp41 = tl.where(tmp27, tmp36, tmp40)
    tmp42 = tl.where(tmp2, tmp24, tmp41)
    tl.store(out_ptr0 + (x5), tmp42, None)




# kernel path: /tmp/torchinductor_sahanp/f5/cf5x2qkzkfemzmtjop2dvvpenfowbmufgipikba5uqyepflwm2ev.py
# Topologically Sorted Source Nodes: [ones, target, loss], Original ATen: [aten.ones, aten._to_copy, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   loss => mul, mul_1, sum_1, sum_2
#   ones => full
#   target => device_put
# Graph fragment:
#   %full : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 12288], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cpu, pin_memory: False})
#   %device_put : [num_users=1] = call_function[target=torch.ops.prims.device_put.default](args = (%full, cuda:0), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, %device_put), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul, [1]), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, %view), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_1, [1]), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__to_copy_mul_ones_sum_5(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2
    r0_numel = 6144
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp11 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp15 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp4 = tl.load(in_ptr0 + (r0_1 + 6144*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp0 = (((r0_1 + 6144*x0) // 256) % 16)
        tmp1 = tl.full([1, 1], 14, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.load(in_ptr0 + ((-3072) + r0_1 + 6144*x0), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = libdevice.tanh(tmp5)
        tmp7 = tmp5 - tmp6
        tmp8 = 1.0
        tmp9 = tmp7 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(r0_mask & xmask, tmp12, _tmp11)
        tmp13 = tmp7 * tmp7
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(r0_mask & xmask, tmp16, _tmp15)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp11, xmask)
    tl.store(out_ptr1 + (x0), tmp15, xmask)




# kernel path: /tmp/torchinductor_sahanp/mr/cmr4y2aqto4jj7sizgj4sdahfrharjvp3lnjeg4y6vnij2kutaw5.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
#   loss => full_default, sum_3
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 12288], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%full_default, [1]), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_mul_sum_6(out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2
    r0_numel = 6144
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x0 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        tmp0 = 1.0
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)




# kernel path: /tmp/torchinductor_sahanp/kg/ckghadd7austnoo7ddiphzirg4dm3orvhswa7qc4jxkevv45267h.py
# Topologically Sorted Source Nodes: [loss, ones, target], Original ATen: [aten.eq, aten.fill, aten.ones, aten._to_copy, aten.mul, aten.sum, aten.add, aten.sqrt, aten.div, aten.sub, aten.zeros_like, aten.where, aten.clamp_min, aten.mean]
# Source node to ATen node mapping:
#   loss => add, add_1, add_2, clamp_min, div, full_default, full_default_1, full_default_2, full_default_3, full_default_4, mean, mul, mul_1, mul_3, sqrt, sub_1, sub_2, sum_1, sum_2, sum_3, where, where_1
#   ones => full
#   target => device_put
# Graph fragment:
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 12288], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cpu, pin_memory: False})
#   %device_put : [num_users=1] = call_function[target=torch.ops.prims.device_put.default](args = (%full, cuda:0), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, %device_put), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul, [1]), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, %view), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_1, [1]), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%sum_2, 9.999999960041972e-13), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 12288], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%full_default, [1]), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%sum_3, 9.999999960041972e-13), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %add_1), kwargs = {})
#   %sqrt : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%mul_3,), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_1, %sqrt), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default_2, %div), kwargs = {})
#   %full_default_1 : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([1], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%full_default_3, %sub_1, %full_default_1), kwargs = {})
#   %full_default_4 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], False), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Scalar](args = (%div, 0.0), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_2, 0), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%full_default_4, %clamp_min, %full_default_1), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where, %where_1), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%add_2,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__to_copy_add_clamp_min_div_eq_fill_mean_mul_ones_sqrt_sub_sum_where_zeros_like_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 2
    R0_BLOCK: tl.constexpr = 2
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), None)
    tmp4 = tl.load(in_ptr1 + (r0_0), None)
    tmp8 = tl.load(in_ptr2 + (r0_0), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None]
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.sum(tmp5, 1)[:, None]
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp11 = tl.sum(tmp9, 1)[:, None]
    tmp12 = 9.999999960041972e-13
    tmp13 = tmp7 + tmp12
    tmp14 = tmp11 + tmp12
    tmp15 = tmp13 * tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tmp3 / tmp16
    tmp18 = 1.0
    tmp19 = tmp18 - tmp17
    tmp20 = tl.full([1, 1], True, tl.int1)
    tmp21 = 0.0
    tmp22 = tl.where(tmp20, tmp19, tmp21)
    tmp23 = tmp17 - tmp21
    tmp24 = triton_helpers.maximum(tmp23, tmp21)
    tmp25 = tl.full([1, 1], False, tl.int1)
    tmp26 = tl.where(tmp25, tmp24, tmp21)
    tmp27 = tmp22 + tmp26
    tmp28 = tmp27 / tmp18
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp28, None)







def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (1, 3, 10, 10, 10), (3000, 1000, 100, 10, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 3, 12, 12, 12), (5184, 1728, 144, 12, 1), torch.float32)
        buf1 = empty_strided_cuda((1, 3, 12, 12, 12), (5184, 1728, 144, 12, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused_copy_0[grid(5184)](arg0_1, buf0, buf1, 5184, XBLOCK=128, num_warps=4, num_stages=1)
        del arg0_1
        buf2 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1[grid(5184)](buf1, buf2, 5184, XBLOCK=128, num_warps=4, num_stages=1)
        del buf1
        buf3 = empty_strided_cuda((1, 3, 16, 16, 16), (12288, 4096, 256, 16, 1), torch.float32)
        buf4 = empty_strided_cuda((1, 3, 16, 16, 12), (9216, 3072, 192, 12, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused_copy_2[grid(9216)](buf2, buf3, buf4, 9216, XBLOCK=256, num_warps=4, num_stages=1)
        del buf2
        buf5 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3[grid(12288)](buf4, buf5, 12288, XBLOCK=128, num_warps=4, num_stages=1)
        del buf4
        buf6 = empty_strided_cuda((1, 3, 16, 16, 16), (12288, 4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4[grid(12288)](buf5, buf6, 12288, XBLOCK=256, num_warps=4, num_stages=1)
        del buf5
        buf7 = empty_strided_cuda((1, 2), (2, 1), torch.float32)
        buf9 = empty_strided_cuda((1, 2), (2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ones, target, loss], Original ATen: [aten.ones, aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_mul_ones_sum_5[grid(2)](buf6, buf7, buf9, 2, 6144, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del buf6
        buf11 = empty_strided_cuda((1, 2), (2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [loss], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_6[grid(2)](buf11, 2, 6144, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        buf8 = empty_strided_cuda((1, ), (1, ), torch.float32)
        buf13 = reinterpret_tensor(buf8, (), (), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [loss, ones, target], Original ATen: [aten.eq, aten.fill, aten.ones, aten._to_copy, aten.mul, aten.sum, aten.add, aten.sqrt, aten.div, aten.sub, aten.zeros_like, aten.where, aten.clamp_min, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_clamp_min_div_eq_fill_mean_mul_ones_sqrt_sub_sum_where_zeros_like_7[grid(1)](buf13, buf7, buf9, buf11, 1, 2, XBLOCK=1, num_warps=2, num_stages=1)
        del buf11
        del buf7
        del buf9
    return (buf13, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 3, 10, 10, 10), (3000, 1000, 100, 10, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
