# AOT ID: ['59_inference']
import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import (
    grid,
)
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


# kernel path: /tmp/torchinductor_sahanp/ph/cphlrbc7oxqlrx55jxm5rvz4z4htouni3jmrpjv46lshwfe4tqsz.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.copy]
# Source node to ATen node mapping:
#   x => copy
# Graph fragment:
#   %copy : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_5, %slice_6), kwargs = {})
#   %slice_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_tensor_1, %copy, 2, 1, %add), kwargs = {})
#   %slice_scatter_default_1 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_tensor, %slice_scatter_default, 3, 1, %add_2), kwargs = {})
#   %slice_scatter_default_2 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%empty, %slice_scatter_default_1, 4, 1, %add_4), kwargs = {})
#   %slice_scatter_default_3 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_2, %slice_15, 4, 0, 1), kwargs = {})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
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


# kernel path: /tmp/torchinductor_sahanp/53/c53nrx45jznoztdgymteevluzb5vvqqcdbrsu5kkjqy25zecp2yd.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %slice_scatter_default_4 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_3, %slice_20, 4, %add_4, %add_5), kwargs = {})
#   %slice_scatter_default_5 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_4, %slice_25, 3, 0, 1), kwargs = {})
#   %slice_scatter_default_6 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_5, %slice_30, 3, %add_2, %add_3), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
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


# kernel path: /tmp/torchinductor_sahanp/2f/c2f5fxy6q3jkssf5j4myd76p3yaeguvzg74ch6py6m5c4wizc4ie.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.copy]
# Source node to ATen node mapping:
#   x_3 => copy_7
# Graph fragment:
#   %copy_7 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_44, %slice_45), kwargs = {})
#   %slice_scatter_default_9 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_tensor_2, %copy_7, 2, 1, %sub_166), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_copy_2(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, ks6, ks7, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // ks0) % ks1)
    x0 = (xindex % ks0)
    x2 = xindex // ks6
    x4 = xindex // ks0
    x3 = xindex
    tmp36 = tl.load(in_ptr1 + (1 + x0 + 4*x4 + ks7*x4), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = 5 + 2*ks2 + 2*ks3 + ks2*ks3
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = ((((-1) + x1) // ks4) % ks5)
    tmp7 = tl.broadcast_to(1 + ks2, [XBLOCK])
    tmp8 = tmp6 >= tmp7
    tmp9 = tmp8 & tmp5
    tmp10 = ((-1)*ks2) + (((((-1) + x1) // ks4) % ks5))
    tmp11 = tl.full([1], 1, tl.int64)
    tmp12 = tmp10 < tmp11
    tmp13 = tmp12 & tmp9
    tmp14 = tl.load(in_ptr0 + (x0 + 2*((((-1) + x1) % ks4)) + 4*ks2 + 8*x2 + ks7*((((-1) + x1) % ks4)) + 2*ks2*ks3 + 2*ks2*ks7 + 4*ks2*x2 + 4*ks3*x2 + 4*ks7*x2 + ks2*ks3*ks7 + 2*ks2*ks3*x2 + 2*ks2*ks7*x2 + 2*ks3*ks7*x2 + ks2*ks3*ks7*x2), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr0 + (x0 + ((-4)*ks2) + 2*((((-1) + x1) % ks4)) + 4*(((((-1) + x1) // ks4) % ks5)) + 8*x2 + ks7*((((-1) + x1) % ks4)) + ((-2)*ks2*ks3) + ((-2)*ks2*ks7) + 2*ks3*(((((-1) + x1) // ks4) % ks5)) + 2*ks7*(((((-1) + x1) // ks4) % ks5)) + 4*ks2*x2 + 4*ks3*x2 + 4*ks7*x2 + ks3*ks7*(((((-1) + x1) // ks4) % ks5)) + ((-1)*ks2*ks3*ks7) + 2*ks2*ks3*x2 + 2*ks2*ks7*x2 + 2*ks3*ks7*x2 + ks2*ks3*ks7*x2), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.where(tmp12, tmp14, tmp15)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp9, tmp16, tmp17)
    tmp19 = tl.full([1], 1, tl.int64)
    tmp20 = tmp6 < tmp19
    tmp21 = tmp20 & tmp5
    tmp22 = tl.load(in_ptr0 + (x0 + 2*((((-1) + x1) % ks4)) + 4*ks2 + 8*x2 + ks7*((((-1) + x1) % ks4)) + 2*ks2*ks3 + 2*ks2*ks7 + 4*ks2*x2 + 4*ks3*x2 + 4*ks7*x2 + ks2*ks3*ks7 + 2*ks2*ks3*x2 + 2*ks2*ks7*x2 + 2*ks3*ks7*x2 + ks2*ks3*ks7*x2), tmp21 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.load(in_ptr0 + (x0 + 2*((((-1) + x1) % ks4)) + 4*(((((-1) + x1) // ks4) % ks5)) + 8*x2 + ks7*((((-1) + x1) % ks4)) + 2*ks3*(((((-1) + x1) // ks4) % ks5)) + 2*ks7*(((((-1) + x1) // ks4) % ks5)) + 4*ks2*x2 + 4*ks3*x2 + 4*ks7*x2 + ks3*ks7*(((((-1) + x1) // ks4) % ks5)) + 2*ks2*ks3*x2 + 2*ks2*ks7*x2 + 2*ks3*ks7*x2 + ks2*ks3*ks7*x2), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.where(tmp20, tmp22, tmp23)
    tmp25 = tl.where(tmp8, tmp18, tmp24)
    tmp26 = 1.0
    tmp27 = tmp25 * tmp26
    tmp28 = 20.0
    tmp29 = tmp27 > tmp28
    tmp30 = tl_math.exp(tmp27)
    tmp31 = libdevice.log1p(tmp30)
    tmp32 = tmp31 * tmp26
    tmp33 = tl.where(tmp29, tmp25, tmp32)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp5, tmp33, tmp34)
    tmp37 = tl.where(tmp5, tmp35, tmp36)
    tl.store(out_ptr0 + (x3), tmp37, xmask)


# kernel path: /tmp/torchinductor_sahanp/ia/ciauvhqspw2oyjw5vndwc33uquagwmeefiezpat3tgsg3vndtoot.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %slice_scatter_default_10 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%empty_1, %slice_scatter_default_9, 3, 1, %add_217), kwargs = {})
#   %slice_scatter_default_11 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_10, %slice_52, 3, 0, 1), kwargs = {})
#   %slice_scatter_default_12 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_11, %slice_57, 3, %add_217, %add_218), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_3(in_ptr0, out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    x2 = xindex
    tmp0 = x0
    tmp1 = 3 + ks1
    tmp2 = tmp0 >= tmp1
    tmp3 = (-2) + x0 + ((-1)*ks1)
    tmp4 = tl.full([1], 1, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = x0
    tmp8 = tl.full([1], 1, tl.int64)
    tmp9 = tmp7 >= tmp8
    tmp10 = tl.broadcast_to(3 + ks1, [XBLOCK])
    tmp11 = tmp7 < tmp10
    tmp12 = tmp9 & tmp11
    tmp13 = tmp12 & tmp6
    tmp14 = tl.load(in_ptr0 + ((-1) + x0 + 2*x1 + ks1*x1), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = float("nan")
    tmp16 = tl.where(tmp12, tmp14, tmp15)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp6, tmp16, tmp17)
    tmp19 = tmp3 >= tmp4
    tmp20 = tl.broadcast_to(3 + ks1, [XBLOCK])
    tmp21 = tmp3 < tmp20
    tmp22 = tmp19 & tmp21
    tmp23 = tmp22 & tmp2
    tmp24 = tl.load(in_ptr0 + ((-3) + x0 + ((-1)*ks1) + 2*x1 + ks1*x1), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = float("nan")
    tmp26 = tl.where(tmp22, tmp24, tmp25)
    tmp27 = tl.where(tmp5, tmp18, tmp26)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp2, tmp27, tmp28)
    tmp30 = tl.full([1], 1, tl.int64)
    tmp31 = tmp0 < tmp30
    tmp32 = 2 + ks1 + x0
    tmp33 = tl.full([1], 1, tl.int64)
    tmp34 = tmp32 >= tmp33
    tmp35 = tl.broadcast_to(3 + ks1, [XBLOCK])
    tmp36 = tmp32 < tmp35
    tmp37 = tmp34 & tmp36
    tmp38 = tmp37 & tmp31
    tmp39 = tl.load(in_ptr0 + (1 + ks1 + x0 + 2*x1 + ks1*x1), tmp38 & xmask, eviction_policy='evict_last', other=0.0)
    tmp40 = float("nan")
    tmp41 = tl.where(tmp37, tmp39, tmp40)
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp31, tmp41, tmp42)
    tmp44 = tmp0 >= tmp30
    tmp45 = tmp0 < tmp1
    tmp46 = tmp44 & tmp45
    tmp47 = tl.load(in_ptr0 + ((-1) + x0 + 2*x1 + ks1*x1), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp48 = float("nan")
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tl.where(tmp31, tmp43, tmp49)
    tmp51 = tl.where(tmp2, tmp29, tmp50)
    tl.store(out_ptr0 + (x2), tmp51, xmask)


# kernel path: /tmp/torchinductor_sahanp/sj/csjimfiy4ivukdzb3njamxmo3wbwsklqpbwml4lkh7iq4tjrbcfu.py
# Topologically Sorted Source Nodes: [x_4, x_5], Original ATen: [aten.sigmoid, aten.view]
# Source node to ATen node mapping:
#   x_4 => sigmoid
#   x_5 => view_1
# Graph fragment:
#   %slice_scatter_default_13 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_12, %slice_62, 2, 0, 1), kwargs = {})
#   %slice_scatter_default_14 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_13, %slice_67, 2, %sub_220, %add_216), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%slice_scatter_default_14,), kwargs = {})
#   %view_1 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sigmoid, [1, %arg0_1, -1, %add_218]), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_sigmoid_view_4(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // ks1) % ks0)
    x0 = (xindex % ks1)
    x2 = xindex // ks4
    x3 = xindex
    tmp15 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = 5 + 2*ks2 + 2*ks3 + ks2*ks3
    tmp2 = tmp0 >= tmp1
    tmp3 = (-4) + x1 + ((-2)*ks2) + ((-2)*ks3) + ((-1)*ks2*ks3)
    tmp4 = tl.full([1], 1, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + (16 + x0 + 4*ks5 + 8*ks2 + 8*ks3 + 24*x2 + 2*ks2*ks5 + 2*ks3*ks5 + 4*ks2*ks3 + 6*ks5*x2 + 8*ks2*x2 + 8*ks3*x2 + ks2*ks3*ks5 + 2*ks2*ks5*x2 + 2*ks3*ks5*x2 + 4*ks2*ks3*x2 + ks2*ks3*ks5*x2), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr0 + ((-16) + x3 + ((-8)*ks2) + ((-8)*ks3) + ((-4)*ks5) + ((-4)*ks2*ks3) + ((-2)*ks2*ks5) + ((-2)*ks3*ks5) + ((-1)*ks2*ks3*ks5)), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.where(tmp5, tmp7, tmp8)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = tl.full([1], 1, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr0 + (16 + x0 + 4*ks5 + 8*ks2 + 8*ks3 + 24*x2 + 2*ks2*ks5 + 2*ks3*ks5 + 4*ks2*ks3 + 6*ks5*x2 + 8*ks2*x2 + 8*ks3*x2 + ks2*ks3*ks5 + 2*ks2*ks5*x2 + 2*ks3*ks5*x2 + 4*ks2*ks3*x2 + ks2*ks3*ks5*x2), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.where(tmp13, tmp14, tmp15)
    tmp17 = tl.where(tmp2, tmp11, tmp16)
    tmp18 = tl.sigmoid(tmp17)
    tl.store(out_ptr0 + (x3), tmp18, xmask)


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
        2 + s3
        2 + s2
        4 + 2*s2 + 2*s3 + s2*s3
        2 + s1
        8 + 4*s1 + 4*s2 + 4*s3 + 2*s1*s2 + 2*s1*s3 + 2*s2*s3 + s1*s2*s3
        buf1 = empty_strided_cuda((1, s0, 2 + s1, 2 + s2, 2 + s3), (8*s0 + 4*s0*s1 + 4*s0*s2 + 4*s0*s3 + 2*s0*s1*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + s0*s1*s2*s3, 8 + 4*s1 + 4*s2 + 4*s3 + 2*s1*s2 + 2*s1*s3 + 2*s2*s3 + s1*s2*s3, 4 + 2*s2 + 2*s3 + s2*s3, 2 + s3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.copy]
        triton_poi_fused_copy_0_xnumel = 8*s0 + 4*s0*s1 + 4*s0*s2 + 4*s0*s3 + 2*s0*s1*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + s0*s1*s2*s3
        get_raw_stream(0)
        triton_poi_fused_copy_0[grid(triton_poi_fused_copy_0_xnumel)](arg4_1, buf0, buf1, 66, 64, 66, 64, 4356, 66, 64, 287496, 862488, XBLOCK=512, num_warps=8, num_stages=1)
        del arg4_1
        buf2 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1_xnumel = 8*s0 + 4*s0*s1 + 4*s0*s2 + 4*s0*s3 + 2*s0*s1*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + s0*s1*s2*s3
        get_raw_stream(0)
        triton_poi_fused_1[grid(triton_poi_fused_1_xnumel)](buf1, buf2, 66, 66, 64, 64, 862488, XBLOCK=512, num_warps=8, num_stages=1)
        del buf1
        buf3 = empty_strided_cuda((1, s0, 6 + 2*s1 + 2*s2 + s1*s2, 4 + s3), (24*s0 + 6*s0*s3 + 8*s0*s1 + 8*s0*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + 4*s0*s1*s2 + s0*s1*s2*s3, 24 + 6*s3 + 8*s1 + 8*s2 + 2*s1*s3 + 2*s2*s3 + 4*s1*s2 + s1*s2*s3, 4 + s3, 1), torch.float32)
        6 + 2*s1 + 2*s2 + s1*s2
        12 + 4*s1 + 4*s2 + 6*s3 + 2*s1*s2 + 2*s1*s3 + 2*s2*s3 + s1*s2*s3
        buf4 = empty_strided_cuda((1, s0, 6 + 2*s1 + 2*s2 + s1*s2, 2 + s3), (12*s0 + 4*s0*s1 + 4*s0*s2 + 6*s0*s3 + 2*s0*s1*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + s0*s1*s2*s3, 12 + 4*s1 + 4*s2 + 6*s3 + 2*s1*s2 + 2*s1*s3 + 2*s2*s3 + s1*s2*s3, 2 + s3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.copy]
        triton_poi_fused_copy_2_xnumel = 12*s0 + 4*s0*s1 + 4*s0*s2 + 6*s0*s3 + 2*s0*s1*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + s0*s1*s2*s3
        get_raw_stream(0)
        triton_poi_fused_copy_2[grid(triton_poi_fused_copy_2_xnumel)](buf2, buf3, buf4, 66, 4358, 64, 64, 66, 66, 287628, 64, 862884, XBLOCK=512, num_warps=8, num_stages=1)
        del buf2
        4 + s3
        buf5 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_3_xnumel = 24*s0 + 6*s0*s3 + 8*s0*s1 + 8*s0*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + 4*s0*s1*s2 + s0*s1*s2*s3
        get_raw_stream(0)
        triton_poi_fused_3[grid(triton_poi_fused_3_xnumel)](buf4, buf5, 68, 64, 889032, XBLOCK=512, num_warps=8, num_stages=1)
        del buf4
        24 + 6*s3 + 8*s1 + 8*s2 + 2*s1*s3 + 2*s2*s3 + 4*s1*s2 + s1*s2*s3
        buf6 = empty_strided_cuda((1, s0, 6 + 2*s1 + 2*s2 + s1*s2, 4 + s3), (24*s0 + 6*s0*s3 + 8*s0*s1 + 8*s0*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + 4*s0*s1*s2 + s0*s1*s2*s3, 24 + 6*s3 + 8*s1 + 8*s2 + 2*s1*s3 + 2*s2*s3 + 4*s1*s2 + s1*s2*s3, 4 + s3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_4, x_5], Original ATen: [aten.sigmoid, aten.view]
        triton_poi_fused_sigmoid_view_4_xnumel = 24*s0 + 6*s0*s3 + 8*s0*s1 + 8*s0*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + 4*s0*s1*s2 + s0*s1*s2*s3
        get_raw_stream(0)
        triton_poi_fused_sigmoid_view_4[grid(triton_poi_fused_sigmoid_view_4_xnumel)](buf5, buf6, 4358, 68, 64, 64, 296344, 64, 889032, XBLOCK=512, num_warps=8, num_stages=1)
        del buf5
    return (buf6, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 3
    arg1_1 = 64
    arg2_1 = 64
    arg3_1 = 64
    arg4_1 = rand_strided((1, 3, 64, 64, 64), (786432, 262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
