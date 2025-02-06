# AOT ID: ['55_inference']
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
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.bernoulli]
# Source node to ATen node mapping:
#   x_2 => inductor_lookup_seed_default, inductor_random_default
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, %arg0_1, 1, 1, 1], %inductor_lookup_seed_default, rand), kwargs = {})
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




# kernel path: /tmp/torchinductor_sahanp/qf/cqfe3hgwfhmqwm6s6gprrmitp3qon44y7glfmcgi33bkxk6kb3ik.py
# Topologically Sorted Source Nodes: [pow_1, out, sign, abs_1, relu, mul, mul_1, x, x_1, x_2, pow_3], Original ATen: [aten.pow, aten.avg_pool3d, aten.sign, aten.abs, aten.relu, aten.mul, aten.mish, aten.bernoulli, aten._to_copy, aten.add]
# Source node to ATen node mapping:
#   abs_1 => abs_1
#   mul => mul_25
#   mul_1 => mul_31
#   out => avg_pool3d
#   pow_1 => pow_1
#   pow_3 => pow_3
#   relu => relu
#   sign => sign
#   x => pow_2
#   x_1 => exp, gt, log1p, mul_42, tanh, where
#   x_2 => add_112, add_60, add_81, convert_element_type, lt_2, mul_61, mul_70, mul_73
# Graph fragment:
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%arg4_1, 2.0), kwargs = {})
#   %avg_pool3d : [num_users=2] = call_function[target=torch.ops.aten.avg_pool3d.default](args = (%pow_1, [2, 2, 2], [2, 2, 2]), kwargs = {})
#   %sign : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%avg_pool3d,), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%avg_pool3d,), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%abs_1,), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign, %relu), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, 8), kwargs = {})
#   %pow_2 : [num_users=4] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_31, 0.5), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%pow_2, 20), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%pow_2,), kwargs = {})
#   %log1p : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp,), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %pow_2, %log1p), kwargs = {})
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%where,), kwargs = {})
#   %mul_42 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_2, %tanh), kwargs = {})
#   %lt_2 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%inductor_random_default, 0.5), kwargs = {})
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt_2, torch.float32), kwargs = {})
#   %mul_70 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%convert_element_type, 0.8864048946659319), kwargs = {})
#   %mul_73 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_42, %mul_70), kwargs = {})
#   %add_60 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%convert_element_type, -1), kwargs = {})
#   %mul_61 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%add_60, 1.558387861036063), kwargs = {})
#   %add_81 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_61, 0.7791939305180315), kwargs = {})
#   %add_112 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_73, %add_81), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_112, 2.0), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_abs_add_avg_pool3d_bernoulli_mish_mul_pow_relu_sign_1(in_out_ptr0, in_ptr0, in_ptr1, ks0, ks1, ks2, ks3, ks4, ks5, ks6, ks7, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks1)
    x2 = ((xindex // ks2) % ks3)
    x3 = xindex // ks4
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 2*ks7*x1 + 2*ks6*ks7*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 2*x0 + 2*ks7*x1 + 2*ks6*ks7*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (ks7 + 2*x0 + 2*ks7*x1 + 2*ks6*ks7*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (1 + ks7 + 2*x0 + 2*ks7*x1 + 2*ks6*ks7*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (2*x0 + ks6*ks7 + 2*ks7*x1 + 2*ks6*ks7*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (1 + 2*x0 + ks6*ks7 + 2*ks7*x1 + 2*ks6*ks7*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (ks7 + 2*x0 + ks6*ks7 + 2*ks7*x1 + 2*ks6*ks7*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr0 + (1 + ks7 + 2*x0 + ks6*ks7 + 2*ks7*x1 + 2*ks6*ks7*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 * tmp0
    tmp3 = tmp2 * tmp2
    tmp4 = tmp3 + tmp1
    tmp6 = tmp5 * tmp5
    tmp7 = tmp6 + tmp4
    tmp9 = tmp8 * tmp8
    tmp10 = tmp9 + tmp7
    tmp12 = tmp11 * tmp11
    tmp13 = tmp12 + tmp10
    tmp15 = tmp14 * tmp14
    tmp16 = tmp15 + tmp13
    tmp18 = tmp17 * tmp17
    tmp19 = tmp18 + tmp16
    tmp21 = tmp20 * tmp20
    tmp22 = tmp21 + tmp19
    tmp23 = 0.125
    tmp24 = tmp22 * tmp23
    tmp25 = tl.full([1], 0, tl.int32)
    tmp26 = tmp25 < tmp24
    tmp27 = tmp26.to(tl.int8)
    tmp28 = tmp24 < tmp25
    tmp29 = tmp28.to(tl.int8)
    tmp30 = tmp27 - tmp29
    tmp31 = tmp30.to(tmp24.dtype)
    tmp32 = tl_math.abs(tmp24)
    tmp33 = triton_helpers.maximum(tmp25, tmp32)
    tmp34 = tmp31 * tmp33
    tmp35 = 8.0
    tmp36 = tmp34 * tmp35
    tmp37 = libdevice.sqrt(tmp36)
    tmp38 = 20.0
    tmp39 = tmp37 > tmp38
    tmp40 = tl_math.exp(tmp37)
    tmp41 = libdevice.log1p(tmp40)
    tmp42 = tl.where(tmp39, tmp37, tmp41)
    tmp43 = libdevice.tanh(tmp42)
    tmp44 = tmp37 * tmp43
    tmp46 = 0.5
    tmp47 = tmp45 < tmp46
    tmp48 = tmp47.to(tl.float32)
    tmp49 = 0.8864048946659319
    tmp50 = tmp48 * tmp49
    tmp51 = tmp44 * tmp50
    tmp52 = -1.0
    tmp53 = tmp48 + tmp52
    tmp54 = 1.558387861036063
    tmp55 = tmp53 * tmp54
    tmp56 = 0.7791939305180315
    tmp57 = tmp55 + tmp56
    tmp58 = tmp51 + tmp57
    tmp59 = tmp58 * tmp58
    tl.store(in_out_ptr0 + (x5), tmp59, xmask)




# kernel path: /tmp/torchinductor_sahanp/bq/cbq3zdgagfwhazligif4crcfnki33itsit7mvo5x27k7v5yoexhz.py
# Topologically Sorted Source Nodes: [sign, abs_1, relu, mul, mul_1, x, x_1, x_2, pow_3, out_1], Original ATen: [aten.sign, aten.abs, aten.relu, aten.mul, aten.pow, aten.mish, aten.bernoulli, aten._to_copy, aten.add, aten.avg_pool3d]
# Source node to ATen node mapping:
#   abs_1 => abs_1
#   mul => mul_25
#   mul_1 => mul_31
#   out_1 => avg_pool3d_1
#   pow_3 => pow_3
#   relu => relu
#   sign => sign
#   x => pow_2
#   x_1 => exp, gt, log1p, mul_42, tanh, where
#   x_2 => add_112, add_60, add_81, convert_element_type, lt_2, mul_61, mul_70, mul_73
# Graph fragment:
#   %sign : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%avg_pool3d,), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%avg_pool3d,), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%abs_1,), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign, %relu), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, 8), kwargs = {})
#   %pow_2 : [num_users=4] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_31, 0.5), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%pow_2, 20), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%pow_2,), kwargs = {})
#   %log1p : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp,), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %pow_2, %log1p), kwargs = {})
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%where,), kwargs = {})
#   %mul_42 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_2, %tanh), kwargs = {})
#   %lt_2 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%inductor_random_default, 0.5), kwargs = {})
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt_2, torch.float32), kwargs = {})
#   %mul_70 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%convert_element_type, 0.8864048946659319), kwargs = {})
#   %mul_73 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_42, %mul_70), kwargs = {})
#   %add_60 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%convert_element_type, -1), kwargs = {})
#   %mul_61 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%add_60, 1.558387861036063), kwargs = {})
#   %add_81 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_61, 0.7791939305180315), kwargs = {})
#   %add_112 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_73, %add_81), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_112, 2.0), kwargs = {})
#   %avg_pool3d_1 : [num_users=2] = call_function[target=torch.ops.aten.avg_pool3d.default](args = (%pow_3, [2, 2, 2], [2, 2, 2]), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_abs_add_avg_pool3d_bernoulli_mish_mul_pow_relu_sign_2(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, ks6, ks7, ks8, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks1)
    x2 = ((xindex // ks2) % ks3)
    x3 = xindex // ks4
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 2*ks5*x1 + 2*ks5*ks6*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 2*ks5*x1 + 2*ks5*ks6*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (ks5 + 2*x0 + 2*ks5*x1 + 2*ks5*ks6*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (1 + ks5 + 2*x0 + 2*ks5*x1 + 2*ks5*ks6*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (ks8 + 2*x0 + 2*ks5*x1 + 2*ks5*ks6*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (1 + ks8 + 2*x0 + 2*ks5*x1 + 2*ks5*ks6*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (ks5 + ks8 + 2*x0 + 2*ks5*x1 + 2*ks5*ks6*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (1 + ks5 + ks8 + 2*x0 + 2*ks5*x1 + 2*ks5*ks6*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp15 = 0.125
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (x4), tmp16, xmask)




# kernel path: /tmp/torchinductor_sahanp/sm/csm57og6375c5k52haqkbcecbn6lkpxb2sptytagkh7rsuzlbjzx.py
# Topologically Sorted Source Nodes: [multi_margin_loss, target, soft_margin_loss, float_1, add, truediv], Original ATen: [aten.arange, aten.randint, aten.ne, aten.gather, aten.rsub, aten.add, aten.clamp_min, aten.scalar_tensor, aten.where, aten.mean, aten.soft_margin_loss, aten._to_copy, aten.div]
# Source node to ATen node mapping:
#   add => add_177
#   float_1 => convert_element_type_1
#   multi_margin_loss => add_165, clamp_min, full_default, gather, iota, mean, ne_3, sub_96, where_2
#   soft_margin_loss => exp_2, log1p_2, mean_1, mul_149, neg
#   target => inductor_lookup_seed_default_1, inductor_randint_default
#   truediv => div
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (%floordiv,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %inductor_lookup_seed_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 1), kwargs = {})
#   %inductor_randint_default : [num_users=2] = call_function[target=torch.ops.prims.inductor_randint.default](args = (0, 10, [1], %inductor_lookup_seed_default_1), kwargs = {})
#   %ne_3 : [num_users=1] = call_function[target=torch.ops.aten.ne.Tensor](args = (%iota, %unsqueeze), kwargs = {})
#   %gather : [num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%view, 1, %unsqueeze), kwargs = {})
#   %sub_96 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %gather), kwargs = {})
#   %add_165 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub_96, %view), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_165, 0), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_3, %clamp_min, %full_default), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%where_2,), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%view,), kwargs = {})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%inductor_randint_default, torch.float32), kwargs = {})
#   %mul_149 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg, %convert_element_type_1), kwargs = {})
#   %exp_2 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_149,), kwargs = {})
#   %log1p_2 : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_2,), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%log1p_2,), kwargs = {})
#   %add_177 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, %mean_1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_177, 2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__to_copy_add_arange_clamp_min_div_gather_mean_ne_randint_rsub_scalar_tensor_soft_margin_loss_where_3(in_out_ptr0, in_ptr0, in_ptr1, load_seed_offset, ks1, ks2, ks3, ks4, ks5, ks6, ks7, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp57 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp65 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp32 = tl.load(in_ptr1 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp0 = tl.load(in_ptr0 + load_seed_offset)
        tmp1 = tl.full([1, 1], 0, tl.int32)
        tmp2 = tl.full([1, 1], 0, tl.int64)
        tmp3 = tl.full([1, 1], 10, tl.int64)
        tmp4 = triton_helpers.randint64(tmp0, (tmp1).to(tl.uint32), tmp2, tmp3)
        tmp5 = ks1*ks2*ks3*ks4
        tmp6 = tmp4 + tmp5
        tmp7 = tmp4 < 0
        tmp8 = tl.where(tmp7, tmp6, tmp4)
        tl.device_assert((0 <= tmp8) & (tmp8 < ks4*(ks5 // 4)*(ks6 // 4)*(ks7 // 4)), "index out of bounds: 0 <= tmp8 < ks4*(ks5 // 4)*(ks6 // 4)*(ks7 // 4)")
        tmp10 = tl.load(in_ptr1 + ((tmp8 % (ks1*ks2*ks3*ks4))), None, eviction_policy='evict_last')
        tmp11 = tmp1 < tmp10
        tmp12 = tmp11.to(tl.int8)
        tmp13 = tmp10 < tmp1
        tmp14 = tmp13.to(tl.int8)
        tmp15 = tmp12 - tmp14
        tmp16 = tmp15.to(tmp10.dtype)
        tmp17 = tl_math.abs(tmp10)
        tmp18 = triton_helpers.maximum(tmp1, tmp17)
        tmp19 = tmp16 * tmp18
        tmp20 = 8.0
        tmp21 = tmp19 * tmp20
        tmp22 = libdevice.sqrt(tmp21)
        tmp23 = 20.0
        tmp24 = tmp22 > tmp23
        tmp25 = tl_math.exp(tmp22)
        tmp26 = libdevice.log1p(tmp25)
        tmp27 = tl.where(tmp24, tmp22, tmp26)
        tmp28 = libdevice.tanh(tmp27)
        tmp29 = tmp22 * tmp28
        tmp30 = 1.0
        tmp31 = tmp30 - tmp29
        tmp33 = tmp1 < tmp32
        tmp34 = tmp33.to(tl.int8)
        tmp35 = tmp32 < tmp1
        tmp36 = tmp35.to(tl.int8)
        tmp37 = tmp34 - tmp36
        tmp38 = tmp37.to(tmp32.dtype)
        tmp39 = tl_math.abs(tmp32)
        tmp40 = triton_helpers.maximum(tmp1, tmp39)
        tmp41 = tmp38 * tmp40
        tmp42 = tmp41 * tmp20
        tmp43 = libdevice.sqrt(tmp42)
        tmp44 = tmp43 > tmp23
        tmp45 = tl_math.exp(tmp43)
        tmp46 = libdevice.log1p(tmp45)
        tmp47 = tl.where(tmp44, tmp43, tmp46)
        tmp48 = libdevice.tanh(tmp47)
        tmp49 = tmp43 * tmp48
        tmp50 = tmp31 + tmp49
        tmp51 = r0_0
        tmp52 = tmp51 != tmp4
        tmp53 = 0.0
        tmp54 = triton_helpers.maximum(tmp50, tmp53)
        tmp55 = tl.where(tmp52, tmp54, tmp53)
        tmp56 = tl.broadcast_to(tmp55, [XBLOCK, R0_BLOCK])
        tmp58 = _tmp57 + tmp56
        _tmp57 = tl.where(r0_mask, tmp58, _tmp57)
        tmp59 = -tmp49
        tmp60 = tmp4.to(tl.float32)
        tmp61 = tmp59 * tmp60
        tmp62 = tl_math.exp(tmp61)
        tmp63 = libdevice.log1p(tmp62)
        tmp64 = tl.broadcast_to(tmp63, [XBLOCK, R0_BLOCK])
        tmp66 = _tmp65 + tmp64
        _tmp65 = tl.where(r0_mask, tmp66, _tmp65)
    tmp57 = tl.sum(_tmp57, 1)[:, None]
    tmp65 = tl.sum(_tmp65, 1)[:, None]
    tmp67 = ks1*ks2*ks3*ks4
    tmp68 = tmp67.to(tl.float32)
    tmp69 = tmp57 / tmp68
    tmp70 = tmp65 / tmp68
    tmp71 = tmp69 + tmp70
    tmp72 = 0.5
    tmp73 = tmp71 * tmp72
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp73, None)







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
        buf1 = empty_strided_cuda((2, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [2], out=buf1)
        buf2 = empty_strided_cuda((1, s0, 1, 1, 1), (s0, 1, s0, s0, s0), torch.float32)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.bernoulli]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bernoulli_0[grid(s0)](buf1, buf2, 0, 3, XBLOCK=4, num_warps=1, num_stages=1)
        ps0 = s3 // 2
        ps1 = s2 // 2
        ps2 = (s2 // 2)*(s3 // 2)
        ps3 = s1 // 2
        ps4 = (s1 // 2)*(s2 // 2)*(s3 // 2)
        buf0 = empty_strided_cuda((1, s0, s1 // 2, s2 // 2, s3 // 2), (s0*(s1 // 2)*(s2 // 2)*(s3 // 2), (s1 // 2)*(s2 // 2)*(s3 // 2), (s2 // 2)*(s3 // 2), s3 // 2, 1), torch.float32)
        buf3 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [pow_1, out, sign, abs_1, relu, mul, mul_1, x, x_1, x_2, pow_3], Original ATen: [aten.pow, aten.avg_pool3d, aten.sign, aten.abs, aten.relu, aten.mul, aten.mish, aten.bernoulli, aten._to_copy, aten.add]
        triton_poi_fused__to_copy_abs_add_avg_pool3d_bernoulli_mish_mul_pow_relu_sign_1_xnumel = s0*(s1 // 2)*(s2 // 2)*(s3 // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_abs_add_avg_pool3d_bernoulli_mish_mul_pow_relu_sign_1[grid(triton_poi_fused__to_copy_abs_add_avg_pool3d_bernoulli_mish_mul_pow_relu_sign_1_xnumel)](buf3, arg4_1, buf2, 16, 16, 256, 16, 4096, 32, 32, 32, 12288, XBLOCK=128, num_warps=4, num_stages=1)
        del arg4_1
        del buf2
        ps5 = s3 // 4
        ps6 = s2 // 4
        ps7 = (s2 // 4)*(s3 // 4)
        ps8 = s1 // 4
        ps9 = (s1 // 4)*(s2 // 4)*(s3 // 4)
        buf4 = empty_strided_cuda((1, s0, s1 // 4, s2 // 4, s3 // 4), (s0*(s1 // 4)*(s2 // 4)*(s3 // 4), (s1 // 4)*(s2 // 4)*(s3 // 4), (s2 // 4)*(s3 // 4), s3 // 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sign, abs_1, relu, mul, mul_1, x, x_1, x_2, pow_3, out_1], Original ATen: [aten.sign, aten.abs, aten.relu, aten.mul, aten.pow, aten.mish, aten.bernoulli, aten._to_copy, aten.add, aten.avg_pool3d]
        triton_poi_fused__to_copy_abs_add_avg_pool3d_bernoulli_mish_mul_pow_relu_sign_2_xnumel = s0*(s1 // 4)*(s2 // 4)*(s3 // 4)
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_abs_add_avg_pool3d_bernoulli_mish_mul_pow_relu_sign_2[grid(triton_poi_fused__to_copy_abs_add_avg_pool3d_bernoulli_mish_mul_pow_relu_sign_2_xnumel)](buf3, buf4, 8, 8, 64, 8, 512, 16, 16, 16, 256, 1536, XBLOCK=256, num_warps=4, num_stages=1)
        del buf3
        buf6 = empty_strided_cuda((), (), torch.float32)
        buf8 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [multi_margin_loss, target, soft_margin_loss, float_1, add, truediv], Original ATen: [aten.arange, aten.randint, aten.ne, aten.gather, aten.rsub, aten.add, aten.clamp_min, aten.scalar_tensor, aten.where, aten.mean, aten.soft_margin_loss, aten._to_copy, aten.div]
        triton_red_fused__to_copy_add_arange_clamp_min_div_gather_mean_ne_randint_rsub_scalar_tensor_soft_margin_loss_where_3_r0_numel = s0*(s1 // 4)*(s2 // 4)*(s3 // 4)
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_arange_clamp_min_div_gather_mean_ne_randint_rsub_scalar_tensor_soft_margin_loss_where_3[grid(1)](buf8, buf1, buf4, 1, 8, 8, 8, 3, 32, 32, 32, 1, 1536, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del buf1
        del buf4
    return (buf8, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 3
    arg1_1 = 32
    arg2_1 = 32
    arg3_1 = 32
    arg4_1 = rand_strided((1, 3, 32, 32, 32), (98304, 32768, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
