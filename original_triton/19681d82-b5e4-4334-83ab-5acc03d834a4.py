# AOT ID: ['79_inference']
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


# kernel path: /tmp/torchinductor_sahanp/5r/c5rbjah6y5ob2jexek2eua5vfetiyuaxw6zwnumtvbmm7cku2ceb.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.bernoulli]
# Source node to ATen node mapping:
#   x => inductor_lookup_seed_default, inductor_random_default_4
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default_4 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, %arg0_1, 1], %inductor_lookup_seed_default, rand), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_bernoulli_0(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (x0), tmp2, xmask)




# kernel path: /tmp/torchinductor_sahanp/x6/cx6hk56bouowh5ir6iti5crly4rvyjx6thtf3h7lnpkcmlnnhz4y.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.bernoulli]
# Source node to ATen node mapping:
#   x_2 => inductor_lookup_seed_default_1, inductor_random_default_3
# Graph fragment:
#   %inductor_lookup_seed_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 1), kwargs = {})
#   %inductor_random_default_3 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, %arg0_1, 1], %inductor_lookup_seed_default_1, rand), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_bernoulli_1(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (x0), tmp2, xmask)




# kernel path: /tmp/torchinductor_sahanp/nv/cnvbw76goaf5yjbseefflrfnc2yfd2eqwsyqoaox2lrba32ugtsl.py
# Topologically Sorted Source Nodes: [x_1, x_2, x_3, x_4], Original ATen: [aten.copy, aten.bernoulli, aten._to_copy, aten.mul, aten.add]
# Source node to ATen node mapping:
#   x_1 => copy
#   x_2 => add_109, add_77, add_90, convert_element_type_1, lt_15, mul_77, mul_86, mul_89
#   x_3 => add_122, add_135, add_154, convert_element_type_2, lt_18, mul_112, mul_121, mul_124
#   x_4 => add_167, add_180, add_199, convert_element_type_3, lt_21, mul_147, mul_156, mul_159
# Graph fragment:
#   %copy : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_1, %slice_2), kwargs = {})
#   %slice_scatter_default : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%empty_1, %copy, 2, 2, %add_18), kwargs = {})
#   %slice_scatter_default_1 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default, %slice_7, 2, 0, 2), kwargs = {})
#   %slice_scatter_default_2 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_1, %slice_12, 2, %add_18, %add_19), kwargs = {})
#   %lt_15 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%inductor_random_default_3, 0.5), kwargs = {})
#   %convert_element_type_1 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt_15, torch.float32), kwargs = {})
#   %mul_86 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%convert_element_type_1, 0.8864048946659319), kwargs = {})
#   %mul_89 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_scatter_default_2, %mul_86), kwargs = {})
#   %add_77 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%convert_element_type_1, -1), kwargs = {})
#   %mul_77 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%add_77, 1.558387861036063), kwargs = {})
#   %add_90 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_77, 0.7791939305180315), kwargs = {})
#   %add_109 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_89, %add_90), kwargs = {})
#   %lt_18 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%inductor_random_default_2, 0.5), kwargs = {})
#   %convert_element_type_2 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt_18, torch.float32), kwargs = {})
#   %mul_121 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%convert_element_type_2, 0.8864048946659319), kwargs = {})
#   %mul_124 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_109, %mul_121), kwargs = {})
#   %add_122 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%convert_element_type_2, -1), kwargs = {})
#   %mul_112 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%add_122, 1.558387861036063), kwargs = {})
#   %add_135 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_112, 0.7791939305180315), kwargs = {})
#   %add_154 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_124, %add_135), kwargs = {})
#   %lt_21 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%inductor_random_default_1, 0.5), kwargs = {})
#   %convert_element_type_3 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt_21, torch.float32), kwargs = {})
#   %mul_156 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%convert_element_type_3, 0.8864048946659319), kwargs = {})
#   %mul_159 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_154, %mul_156), kwargs = {})
#   %add_167 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%convert_element_type_3, -1), kwargs = {})
#   %mul_147 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%add_167, 1.558387861036063), kwargs = {})
#   %add_180 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_147, 0.7791939305180315), kwargs = {})
#   %add_199 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_159, %add_180), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_add_bernoulli_copy_mul_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    x2 = xindex
    tmp88 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp102 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp111 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = 2 + ks1
    tmp2 = tmp0 >= tmp1
    tmp3 = x0 + ((-1)*ks1)
    tmp4 = tl.full([1], 2, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = x0
    tmp8 = tl.full([1], 2, tl.int64)
    tmp9 = tmp7 >= tmp8
    tmp10 = tl.broadcast_to(2 + ks1, [XBLOCK])
    tmp11 = tmp7 < tmp10
    tmp12 = tmp9 & tmp11
    tmp13 = tmp12 & tmp6
    tmp14 = tl.load(in_ptr0 + ((-2) + x0 + ks1*x1), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr1 + (x1), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = 0.5
    tmp17 = tmp15 < tmp16
    tmp18 = tmp17.to(tl.float32)
    tmp19 = 2.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp14 * tmp20
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp13, tmp21, tmp22)
    tmp24 = float("nan")
    tmp25 = tl.where(tmp12, tmp23, tmp24)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp6, tmp25, tmp26)
    tmp28 = tmp3 >= tmp4
    tmp29 = tl.broadcast_to(2 + ks1, [XBLOCK])
    tmp30 = tmp3 < tmp29
    tmp31 = tmp28 & tmp30
    tmp32 = tmp31 & tmp2
    tmp33 = tl.load(in_ptr0 + ((-2) + x0 + ((-1)*ks1) + ks1*x1), tmp32 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr1 + (x1), tmp32 & xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = 0.5
    tmp36 = tmp34 < tmp35
    tmp37 = tmp36.to(tl.float32)
    tmp38 = 2.0
    tmp39 = tmp37 * tmp38
    tmp40 = tmp33 * tmp39
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp32, tmp40, tmp41)
    tmp43 = float("nan")
    tmp44 = tl.where(tmp31, tmp42, tmp43)
    tmp45 = tl.where(tmp5, tmp27, tmp44)
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp2, tmp45, tmp46)
    tmp48 = tl.full([1], 2, tl.int64)
    tmp49 = tmp0 < tmp48
    tmp50 = ks1 + x0
    tmp51 = tl.full([1], 2, tl.int64)
    tmp52 = tmp50 >= tmp51
    tmp53 = tl.broadcast_to(2 + ks1, [XBLOCK])
    tmp54 = tmp50 < tmp53
    tmp55 = tmp52 & tmp54
    tmp56 = tmp55 & tmp49
    tmp57 = tl.load(in_ptr0 + ((-2) + ks1 + x0 + ks1*x1), tmp56 & xmask, eviction_policy='evict_last', other=0.0)
    tmp58 = tl.load(in_ptr1 + (x1), tmp56 & xmask, eviction_policy='evict_last', other=0.0)
    tmp59 = 0.5
    tmp60 = tmp58 < tmp59
    tmp61 = tmp60.to(tl.float32)
    tmp62 = 2.0
    tmp63 = tmp61 * tmp62
    tmp64 = tmp57 * tmp63
    tmp65 = tl.full(tmp64.shape, 0.0, tmp64.dtype)
    tmp66 = tl.where(tmp56, tmp64, tmp65)
    tmp67 = float("nan")
    tmp68 = tl.where(tmp55, tmp66, tmp67)
    tmp69 = tl.full(tmp68.shape, 0.0, tmp68.dtype)
    tmp70 = tl.where(tmp49, tmp68, tmp69)
    tmp71 = tmp0 >= tmp48
    tmp72 = tmp0 < tmp1
    tmp73 = tmp71 & tmp72
    tmp74 = tl.load(in_ptr0 + ((-2) + x0 + ks1*x1), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp75 = tl.load(in_ptr1 + (x1), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp76 = 0.5
    tmp77 = tmp75 < tmp76
    tmp78 = tmp77.to(tl.float32)
    tmp79 = 2.0
    tmp80 = tmp78 * tmp79
    tmp81 = tmp74 * tmp80
    tmp82 = tl.full(tmp81.shape, 0.0, tmp81.dtype)
    tmp83 = tl.where(tmp73, tmp81, tmp82)
    tmp84 = float("nan")
    tmp85 = tl.where(tmp73, tmp83, tmp84)
    tmp86 = tl.where(tmp49, tmp70, tmp85)
    tmp87 = tl.where(tmp2, tmp47, tmp86)
    tmp89 = 0.5
    tmp90 = tmp88 < tmp89
    tmp91 = tmp90.to(tl.float32)
    tmp92 = 0.8864048946659319
    tmp93 = tmp91 * tmp92
    tmp94 = tmp87 * tmp93
    tmp95 = -1.0
    tmp96 = tmp91 + tmp95
    tmp97 = 1.558387861036063
    tmp98 = tmp96 * tmp97
    tmp99 = 0.7791939305180315
    tmp100 = tmp98 + tmp99
    tmp101 = tmp94 + tmp100
    tmp103 = tmp102 < tmp89
    tmp104 = tmp103.to(tl.float32)
    tmp105 = tmp104 * tmp92
    tmp106 = tmp101 * tmp105
    tmp107 = tmp104 + tmp95
    tmp108 = tmp107 * tmp97
    tmp109 = tmp108 + tmp99
    tmp110 = tmp106 + tmp109
    tmp112 = tmp111 < tmp89
    tmp113 = tmp112.to(tl.float32)
    tmp114 = tmp113 * tmp92
    tmp115 = tmp110 * tmp114
    tmp116 = tmp113 + tmp95
    tmp117 = tmp116 * tmp97
    tmp118 = tmp117 + tmp99
    tmp119 = tmp115 + tmp118
    tl.store(in_out_ptr0 + (x2), tmp119, xmask)




# kernel path: /tmp/torchinductor_sahanp/lv/clvxbwcc34tc3vot3m6h76grb7lbzu2wuheg7v6s6uvqll37yqxz.py
# Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.copy]
# Source node to ATen node mapping:
#   x_6 => copy_3
# Graph fragment:
#   %copy_3 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_14, %slice_15), kwargs = {})
#   %slice_scatter_default_3 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%empty_6, %copy_3, 2, 2, %add_221), kwargs = {})
#   %slice_scatter_default_4 : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_3, %slice_20, 2, 0, 2), kwargs = {})
#   %slice_scatter_default_5 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_4, %slice_25, 2, %add_221, %add_222), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_copy_3(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    x2 = xindex
    tmp0 = x0
    tmp1 = 6 + ks1
    tmp2 = tmp0 >= tmp1
    tmp3 = (-4) + x0 + ((-1)*ks1)
    tmp4 = tl.full([1], 2, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = x0
    tmp8 = tl.full([1], 2, tl.int64)
    tmp9 = tmp7 >= tmp8
    tmp10 = tl.broadcast_to(6 + ks1, [XBLOCK])
    tmp11 = tmp7 < tmp10
    tmp12 = tmp9 & tmp11
    tmp13 = tmp12 & tmp6
    tmp14 = tl.load(in_ptr0 + ((-2) + x0 + 4*x1 + ks1*x1), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr1 + (x1), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = 0.5
    tmp17 = tmp15 < tmp16
    tmp18 = tmp17.to(tl.float32)
    tmp19 = 2.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp14 * tmp20
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp13, tmp21, tmp22)
    tmp24 = float("nan")
    tmp25 = tl.where(tmp12, tmp23, tmp24)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp6, tmp25, tmp26)
    tmp28 = tmp3 >= tmp4
    tmp29 = tl.broadcast_to(6 + ks1, [XBLOCK])
    tmp30 = tmp3 < tmp29
    tmp31 = tmp28 & tmp30
    tmp32 = tmp31 & tmp2
    tmp33 = tl.load(in_ptr0 + ((-6) + x0 + ((-1)*ks1) + 4*x1 + ks1*x1), tmp32 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr1 + (x1), tmp32 & xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = 0.5
    tmp36 = tmp34 < tmp35
    tmp37 = tmp36.to(tl.float32)
    tmp38 = 2.0
    tmp39 = tmp37 * tmp38
    tmp40 = tmp33 * tmp39
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp32, tmp40, tmp41)
    tmp43 = float("nan")
    tmp44 = tl.where(tmp31, tmp42, tmp43)
    tmp45 = tl.where(tmp5, tmp27, tmp44)
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp2, tmp45, tmp46)
    tmp48 = tl.full([1], 2, tl.int64)
    tmp49 = tmp0 < tmp48
    tmp50 = 4 + ks1 + x0
    tmp51 = tl.full([1], 2, tl.int64)
    tmp52 = tmp50 >= tmp51
    tmp53 = tl.broadcast_to(6 + ks1, [XBLOCK])
    tmp54 = tmp50 < tmp53
    tmp55 = tmp52 & tmp54
    tmp56 = tmp55 & tmp49
    tmp57 = tl.load(in_ptr0 + (2 + ks1 + x0 + 4*x1 + ks1*x1), tmp56 & xmask, eviction_policy='evict_last', other=0.0)
    tmp58 = tl.load(in_ptr1 + (x1), tmp56 & xmask, eviction_policy='evict_last', other=0.0)
    tmp59 = 0.5
    tmp60 = tmp58 < tmp59
    tmp61 = tmp60.to(tl.float32)
    tmp62 = 2.0
    tmp63 = tmp61 * tmp62
    tmp64 = tmp57 * tmp63
    tmp65 = tl.full(tmp64.shape, 0.0, tmp64.dtype)
    tmp66 = tl.where(tmp56, tmp64, tmp65)
    tmp67 = float("nan")
    tmp68 = tl.where(tmp55, tmp66, tmp67)
    tmp69 = tl.full(tmp68.shape, 0.0, tmp68.dtype)
    tmp70 = tl.where(tmp49, tmp68, tmp69)
    tmp71 = tmp0 >= tmp48
    tmp72 = tmp0 < tmp1
    tmp73 = tmp71 & tmp72
    tmp74 = tl.load(in_ptr0 + ((-2) + x0 + 4*x1 + ks1*x1), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp75 = tl.load(in_ptr1 + (x1), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp76 = 0.5
    tmp77 = tmp75 < tmp76
    tmp78 = tmp77.to(tl.float32)
    tmp79 = 2.0
    tmp80 = tmp78 * tmp79
    tmp81 = tmp74 * tmp80
    tmp82 = tl.full(tmp81.shape, 0.0, tmp81.dtype)
    tmp83 = tl.where(tmp73, tmp81, tmp82)
    tmp84 = float("nan")
    tmp85 = tl.where(tmp73, tmp83, tmp84)
    tmp86 = tl.where(tmp49, tmp70, tmp85)
    tmp87 = tl.where(tmp2, tmp47, tmp86)
    tl.store(out_ptr0 + (x2), tmp87, xmask)







def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    assert_size_stride(arg2_1, (1, s0, s1), (s0*s1, s1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((5, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [5], out=buf1)
        buf2 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.bernoulli]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bernoulli_0[grid(s0)](buf1, buf2, 4, 3, XBLOCK=4, num_warps=1, num_stages=1)
        buf5 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.bernoulli]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bernoulli_1[grid(s0)](buf1, buf5, 1, 3, XBLOCK=4, num_warps=1, num_stages=1)
        buf6 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.bernoulli]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bernoulli_0[grid(s0)](buf1, buf6, 4, 3, XBLOCK=4, num_warps=1, num_stages=1)
        buf7 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.bernoulli]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bernoulli_0[grid(s0)](buf1, buf7, 4, 3, XBLOCK=4, num_warps=1, num_stages=1)
        buf9 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.bernoulli]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bernoulli_0[grid(s0)](buf1, buf9, 4, 3, XBLOCK=4, num_warps=1, num_stages=1)
        del buf1
        ps0 = 4 + s1
        buf4 = empty_strided_cuda((1, s0, 4 + s1), (4*s0 + s0*s1, 4 + s1, 1), torch.float32)
        buf8 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [x_1, x_2, x_3, x_4], Original ATen: [aten.copy, aten.bernoulli, aten._to_copy, aten.mul, aten.add]
        triton_poi_fused__to_copy_add_bernoulli_copy_mul_2_xnumel = 4*s0 + s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_bernoulli_copy_mul_2[grid(triton_poi_fused__to_copy_add_bernoulli_copy_mul_2_xnumel)](buf8, arg2_1, buf2, buf5, buf6, buf7, 68, 64, 204, XBLOCK=128, num_warps=4, num_stages=1)
        del arg2_1
        del buf2
        del buf5
        del buf6
        del buf7
        ps1 = 8 + s1
        buf10 = empty_strided_cuda((1, s0, 8 + s1), (8*s0 + s0*s1, 8 + s1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.copy]
        triton_poi_fused_copy_3_xnumel = 8*s0 + s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused_copy_3[grid(triton_poi_fused_copy_3_xnumel)](buf8, buf9, buf10, 72, 64, 216, XBLOCK=256, num_warps=4, num_stages=1)
        del buf8
        del buf9
    return (buf10, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 3
    arg1_1 = 64
    arg2_1 = rand_strided((1, 3, 64), (192, 64, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
