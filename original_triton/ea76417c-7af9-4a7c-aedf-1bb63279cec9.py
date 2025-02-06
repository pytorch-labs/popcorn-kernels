# AOT ID: ['8_inference']
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


#include "/tmp/torchinductor_sahanp/3b/c3bi5gk6mslf6u4iaqafhxm64z6u65e3eain4xlary5blqnvv6xx.h"
extern "C"  void kernel(int64_t* in_out_ptr0,
                       const int64_t ks0,
                       const int64_t ks1,
                       const int64_t ks2)
{
    {
        {
            {
                auto tmp0 = in_out_ptr0[static_cast<int64_t>(0L)];
                auto tmp1 = static_cast<int32_t>(0);
                auto tmp2 = static_cast<int64_t>(0);
                auto tmp3 = 64L*ks0 + 8L*ks0*ks1 + 8L*ks0*ks2 + ks0*ks1*ks2;
                auto tmp4 = c10::convert<int64_t>(tmp3);
                auto tmp5 = randint64_cpu(tmp0, tmp1, tmp2, tmp4);
                in_out_ptr0[static_cast<int64_t>(0L)] = tmp5;
            }
        }
    }
}
''')


# kernel path: /tmp/torchinductor_sahanp/e5/ce55e56eyvu77hd4el22fjckf5sild4jwi3qjpf5mzw3wahxigdk.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.bernoulli]
# Source node to ATen node mapping:
#   x_1 => inductor_lookup_seed_default, inductor_random_default_1
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, %arg0_1, 1, 1], %inductor_lookup_seed_default, rand), kwargs = {})
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




# kernel path: /tmp/torchinductor_sahanp/t4/ct475td7qdzdgtyxarakaj4fyrowgjxphn4jtgpyxejssmcqdbkg.py
# Topologically Sorted Source Nodes: [x, x_1, x_2, x_3], Original ATen: [aten.constant_pad_nd, aten.bernoulli, aten._to_copy, aten.div, aten.mul, aten.hardswish, aten.reflection_pad2d]
# Source node to ATen node mapping:
#   x => constant_pad_nd
#   x_1 => convert_element_type, div, lt_2, mul_16
#   x_2 => add_28, clamp_max, clamp_min, div_1, mul_21
#   x_3 => _unsafe_index
# Graph fragment:
#   %constant_pad_nd : [num_users=3] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%arg3_1, [2, 2, 2, 2], 3.0), kwargs = {})
#   %lt_2 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%inductor_random_default_1, 0.5), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt_2, torch.float32), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%convert_element_type, 0.5), kwargs = {})
#   %mul_16 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%constant_pad_nd, %div), kwargs = {})
#   %add_28 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_16, 3), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_28, 0), kwargs = {})
#   %clamp_max : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 6), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %clamp_max), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_21, 6), kwargs = {})
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%div_1, [None, None, %sub_19, None]), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_bernoulli_constant_pad_nd_div_hardswish_mul_reflection_pad2d_2(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // ks0) % ks1)
    x0 = (xindex % ks0)
    x2 = xindex // ks4
    x4 = xindex
    tmp13 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp0 = (-2) + (tl.where(3 + ks2 + ((-1)*tl_math.abs(3 + ks2 + ((-1)*tl_math.abs((-2) + x1)))) < 0, 7 + ((-1)*tl_math.abs(3 + ks2 + ((-1)*tl_math.abs((-2) + x1)))) + 2*ks2, 3 + ks2 + ((-1)*tl_math.abs(3 + ks2 + ((-1)*tl_math.abs((-2) + x1))))))
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = ks2
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + x0
    tmp6 = tmp5 >= tmp1
    tmp7 = ks3
    tmp8 = tmp5 < tmp7
    tmp9 = tmp2 & tmp4
    tmp10 = tmp9 & tmp6
    tmp11 = tmp10 & tmp8
    tmp12 = tl.load(in_ptr0 + ((-2) + x0 + ((-2)*ks3) + ks3*(tl.where(3 + ks2 + ((-1)*tl_math.abs(3 + ks2 + ((-1)*tl_math.abs((-2) + x1)))) < 0, 7 + ((-1)*tl_math.abs(3 + ks2 + ((-1)*tl_math.abs((-2) + x1)))) + 2*ks2, 3 + ks2 + ((-1)*tl_math.abs(3 + ks2 + ((-1)*tl_math.abs((-2) + x1)))))) + ks2*ks3*x2), tmp11 & xmask, eviction_policy='evict_last', other=3.0)
    tmp14 = 0.5
    tmp15 = tmp13 < tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = 2.0
    tmp18 = tmp16 * tmp17
    tmp19 = tmp12 * tmp18
    tmp20 = 3.0
    tmp21 = tmp19 + tmp20
    tmp22 = 0.0
    tmp23 = triton_helpers.maximum(tmp21, tmp22)
    tmp24 = 6.0
    tmp25 = triton_helpers.minimum(tmp23, tmp24)
    tmp26 = tmp19 * tmp25
    tmp27 = 0.16666666666666666
    tmp28 = tmp26 * tmp27
    tl.store(out_ptr0 + (x4), tmp28, xmask)




# kernel path: /tmp/torchinductor_sahanp/yy/cyydfvzwwke4n4hni2biwl4mkxzfxyprerx524iu5fva5b4o4xyh.py
# Topologically Sorted Source Nodes: [mm_loss, randn_like, softmax, kl_loss, log_prob, add], Original ATen: [aten.arange, aten.ne, aten.gather, aten.rsub, aten.add, aten.clamp_min, aten.scalar_tensor, aten.where, aten.mean, aten.randn_like, aten._softmax, aten.xlogy, aten._log_softmax, aten.mul, aten.sub]
# Source node to ATen node mapping:
#   add => add_67
#   kl_loss => eq_35, full_default_1, full_default_2, isnan, log_1, mean_1, mul_49, mul_52, sub_46, where_1, where_2
#   log_prob => amax, exp, log, sub_38, sub_39, sum_1
#   mm_loss => add_43, clamp_min_1, full_default, gather, iota_2, mean, ne_1, sub_30, where
#   randn_like => inductor_lookup_seed_default_1, inductor_random_default
#   softmax => amax_1, div_2, exp_1, sub_42, sum_2
# Graph fragment:
#   %iota_2 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (%mul_34,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %ne_1 : [num_users=1] = call_function[target=torch.ops.aten.ne.Tensor](args = (%iota_2, %unsqueeze), kwargs = {})
#   %gather : [num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%view, 1, %unsqueeze), kwargs = {})
#   %sub_30 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %gather), kwargs = {})
#   %add_43 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub_30, %view), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_43, 0), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_1, %clamp_min_1, %full_default), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%where,), kwargs = {})
#   %inductor_lookup_seed_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 1), kwargs = {})
#   %inductor_random_default : [num_users=2] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, %sym_size_int_3], %inductor_lookup_seed_default_1, randn), kwargs = {})
#   %amax_1 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%inductor_random_default, [1], True), kwargs = {})
#   %sub_42 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%inductor_random_default, %amax_1), kwargs = {})
#   %exp_1 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_42,), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_1, [1], True), kwargs = {})
#   %div_2 : [num_users=5] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_1, %sum_2), kwargs = {})
#   %isnan : [num_users=1] = call_function[target=torch.ops.aten.isnan.default](args = (%div_2,), kwargs = {})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], nan), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %eq_35 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%div_2, 0), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %log_1 : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%div_2,), kwargs = {})
#   %mul_52 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, %log_1), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_35, %full_default_1, %mul_52), kwargs = {})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%isnan, %full_default_2, %where_1), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view, [1], True), kwargs = {})
#   %sub_38 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %amax), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_38,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_1,), kwargs = {})
#   %sub_39 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_38, %log), kwargs = {})
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, %sub_39), kwargs = {})
#   %sub_46 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_2, %mul_49), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sub_46,), kwargs = {})
#   %add_67 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, %mean_1), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__log_softmax__softmax_add_arange_clamp_min_gather_mean_mul_ne_randn_like_rsub_scalar_tensor_sub_where_xlogy_3(in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, out_ptr0, load_seed_offset, ks1, ks2, ks3, ks4, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    tmp3 = tl.load(in_ptr1 + (0))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    _tmp20 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp23 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    _tmp26 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp14 = tl.load(in_ptr2 + (4*(((r0_0 // (8 + ks3)) % ks4)) + 32*(r0_0 // (64 + 8*ks2 + 8*ks3 + ks2*ks3)) + ks3*(((r0_0 // (8 + ks3)) % ks4)) + 4*ks2*(r0_0 // (64 + 8*ks2 + 8*ks3 + ks2*ks3)) + 8*ks3*(r0_0 // (64 + 8*ks2 + 8*ks3 + ks2*ks3)) + ks2*ks3*(r0_0 // (64 + 8*ks2 + 8*ks3 + ks2*ks3)) + (tl.where(3 + ks3 + ((-1)*tl_math.abs(3 + ks3 + ((-1)*tl_math.abs((-2) + ((r0_0 % (8 + ks3))))))) < 0, 7 + ((-1)*tl_math.abs(3 + ks3 + ((-1)*tl_math.abs((-2) + ((r0_0 % (8 + ks3))))))) + 2*ks3, 3 + ks3 + ((-1)*tl_math.abs(3 + ks3 + ((-1)*tl_math.abs((-2) + ((r0_0 % (8 + ks3)))))))))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp0 = tl.load(in_ptr0 + load_seed_offset)
        tmp1 = r0_0
        tmp2 = tl.randn(tmp0, (tmp1).to(tl.uint32))
        tmp5 = tmp1 != tmp4
        tmp6 = 64*ks1 + 8*ks1*ks2 + 8*ks1*ks3 + ks1*ks2*ks3
        tmp7 = tmp4 + tmp6
        tmp8 = tmp4 < 0
        tmp9 = tl.where(tmp8, tmp7, tmp4)
        tl.device_assert((0 <= tmp9) & (tmp9 < 64*ks1 + 8*ks1*ks2 + 8*ks1*ks3 + ks1*ks2*ks3), "index out of bounds: 0 <= tmp9 < 64*ks1 + 8*ks1*ks2 + 8*ks1*ks3 + ks1*ks2*ks3")
        tmp11 = tl.load(in_ptr2 + (4*(((tmp9 // (8 + ks3)) % ks4)) + 32*(((tmp9 // (64 + 8*ks2 + 8*ks3 + ks2*ks3)) % ks1)) + ks3*(((tmp9 // (8 + ks3)) % ks4)) + 4*ks2*(((tmp9 // (64 + 8*ks2 + 8*ks3 + ks2*ks3)) % ks1)) + 8*ks3*(((tmp9 // (64 + 8*ks2 + 8*ks3 + ks2*ks3)) % ks1)) + ks2*ks3*(((tmp9 // (64 + 8*ks2 + 8*ks3 + ks2*ks3)) % ks1)) + (tl.where(3 + ks3 + ((-1)*tl_math.abs(3 + ks3 + ((-1)*tl_math.abs((-2) + ((tmp9 % (8 + ks3))))))) < 0, 7 + ((-1)*tl_math.abs(3 + ks3 + ((-1)*tl_math.abs((-2) + ((tmp9 % (8 + ks3))))))) + 2*ks3, 3 + ks3 + ((-1)*tl_math.abs(3 + ks3 + ((-1)*tl_math.abs((-2) + ((tmp9 % (8 + ks3)))))))))), None, eviction_policy='evict_last')
        tmp12 = 1.0
        tmp13 = tmp12 - tmp11
        tmp15 = tmp13 + tmp14
        tmp16 = 0.0
        tmp17 = triton_helpers.maximum(tmp15, tmp16)
        tmp18 = tl.where(tmp5, tmp17, tmp16)
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(r0_mask, tmp21, _tmp20)
        tmp22 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp24 = triton_helpers.maximum(_tmp23, tmp22)
        _tmp23 = tl.where(r0_mask, tmp24, _tmp23)
        tmp25 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
        tmp27 = triton_helpers.maximum(_tmp26, tmp25)
        _tmp26 = tl.where(r0_mask, tmp27, _tmp26)
        tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp2, r0_mask)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tmp23 = triton_helpers.max2(_tmp23, 1)[:, None]
    tmp26 = triton_helpers.max2(_tmp26, 1)[:, None]
    _tmp32 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp28 = tl.load(out_ptr0 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp29 = tmp28 - tmp23
        tmp30 = tl_math.exp(tmp29)
        tmp31 = tl.broadcast_to(tmp30, [XBLOCK, R0_BLOCK])
        tmp33 = _tmp32 + tmp31
        _tmp32 = tl.where(r0_mask, tmp33, _tmp32)
    tmp32 = tl.sum(_tmp32, 1)[:, None]
    _tmp38 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp34 = tl.load(in_ptr2 + (4*(((r0_0 // (8 + ks3)) % ks4)) + 32*(r0_0 // (64 + 8*ks2 + 8*ks3 + ks2*ks3)) + ks3*(((r0_0 // (8 + ks3)) % ks4)) + 4*ks2*(r0_0 // (64 + 8*ks2 + 8*ks3 + ks2*ks3)) + 8*ks3*(r0_0 // (64 + 8*ks2 + 8*ks3 + ks2*ks3)) + ks2*ks3*(r0_0 // (64 + 8*ks2 + 8*ks3 + ks2*ks3)) + (tl.where(3 + ks3 + ((-1)*tl_math.abs(3 + ks3 + ((-1)*tl_math.abs((-2) + ((r0_0 % (8 + ks3))))))) < 0, 7 + ((-1)*tl_math.abs(3 + ks3 + ((-1)*tl_math.abs((-2) + ((r0_0 % (8 + ks3))))))) + 2*ks3, 3 + ks3 + ((-1)*tl_math.abs(3 + ks3 + ((-1)*tl_math.abs((-2) + ((r0_0 % (8 + ks3)))))))))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp35 = tmp34 - tmp26
        tmp36 = tl_math.exp(tmp35)
        tmp37 = tl.broadcast_to(tmp36, [XBLOCK, R0_BLOCK])
        tmp39 = _tmp38 + tmp37
        _tmp38 = tl.where(r0_mask, tmp39, _tmp38)
    tmp38 = tl.sum(_tmp38, 1)[:, None]
    _tmp59 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp40 = tl.load(out_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp52 = tl.load(in_ptr2 + (4*(((r0_0 // (8 + ks3)) % ks4)) + 32*(r0_0 // (64 + 8*ks2 + 8*ks3 + ks2*ks3)) + ks3*(((r0_0 // (8 + ks3)) % ks4)) + 4*ks2*(r0_0 // (64 + 8*ks2 + 8*ks3 + ks2*ks3)) + 8*ks3*(r0_0 // (64 + 8*ks2 + 8*ks3 + ks2*ks3)) + ks2*ks3*(r0_0 // (64 + 8*ks2 + 8*ks3 + ks2*ks3)) + (tl.where(3 + ks3 + ((-1)*tl_math.abs(3 + ks3 + ((-1)*tl_math.abs((-2) + ((r0_0 % (8 + ks3))))))) < 0, 7 + ((-1)*tl_math.abs(3 + ks3 + ((-1)*tl_math.abs((-2) + ((r0_0 % (8 + ks3))))))) + 2*ks3, 3 + ks3 + ((-1)*tl_math.abs(3 + ks3 + ((-1)*tl_math.abs((-2) + ((r0_0 % (8 + ks3)))))))))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp41 = tmp40 - tmp23
        tmp42 = tl_math.exp(tmp41)
        tmp43 = tmp42 / tmp32
        tmp44 = libdevice.isnan(tmp43).to(tl.int1)
        tmp45 = 0.0
        tmp46 = tmp43 == tmp45
        tmp47 = tl_math.log(tmp43)
        tmp48 = tmp43 * tmp47
        tmp49 = tl.where(tmp46, tmp45, tmp48)
        tmp50 = float("nan")
        tmp51 = tl.where(tmp44, tmp50, tmp49)
        tmp53 = tmp52 - tmp26
        tmp54 = tl_math.log(tmp38)
        tmp55 = tmp53 - tmp54
        tmp56 = tmp43 * tmp55
        tmp57 = tmp51 - tmp56
        tmp58 = tl.broadcast_to(tmp57, [XBLOCK, R0_BLOCK])
        tmp60 = _tmp59 + tmp58
        _tmp59 = tl.where(r0_mask, tmp60, _tmp59)
    tmp59 = tl.sum(_tmp59, 1)[:, None]
    tmp61 = 64*ks1 + 8*ks1*ks2 + 8*ks1*ks3 + ks1*ks2*ks3
    tmp62 = tmp61.to(tl.float32)
    tmp63 = tmp20 / tmp62
    tmp64 = tmp59 / tmp62
    tmp65 = tmp63 + tmp64
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp65, None)







def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    buf0 = empty_strided_cpu((1, ), (1, ), torch.int64)
    # Topologically Sorted Source Nodes: [], Original ATen: []
    aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf0)
    buf1 = buf0; del buf0  # reuse
    cpp_fused_randint_0(buf1, s0, s1, s2)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf2 = empty_strided_cuda((1, ), (1, ), torch.int64)
        buf2.copy_(buf1, False)
        del buf1
        buf3 = empty_strided_cuda((2, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [2], out=buf3)
        buf4 = empty_strided_cuda((1, s0, 1, 1), (s0, 1, s0, s0), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.bernoulli]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bernoulli_1[grid(s0)](buf3, buf4, 0, 3, XBLOCK=4, num_warps=1, num_stages=1)
        ps0 = 4 + s2
        ps1 = 8 + s1
        ps2 = 32 + 4*s1 + 8*s2 + s1*s2
        buf5 = empty_strided_cuda((1, s0, 8 + s1, 4 + s2), (32*s0 + 4*s0*s1 + 8*s0*s2 + s0*s1*s2, 32 + 4*s1 + 8*s2 + s1*s2, 4 + s2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_1, x_2, x_3], Original ATen: [aten.constant_pad_nd, aten.bernoulli, aten._to_copy, aten.div, aten.mul, aten.hardswish, aten.reflection_pad2d]
        triton_poi_fused__to_copy_bernoulli_constant_pad_nd_div_hardswish_mul_reflection_pad2d_2_xnumel = 32*s0 + 4*s0*s1 + 8*s0*s2 + s0*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_bernoulli_constant_pad_nd_div_hardswish_mul_reflection_pad2d_2[grid(triton_poi_fused__to_copy_bernoulli_constant_pad_nd_div_hardswish_mul_reflection_pad2d_2_xnumel)](arg3_1, buf4, buf5, 36, 40, 32, 32, 1440, 4320, XBLOCK=256, num_warps=4, num_stages=1)
        del arg3_1
        del buf4
        buf7 = empty_strided_cuda((1, 64*s0 + 8*s0*s1 + 8*s0*s2 + s0*s1*s2), (64*s0 + 8*s0*s1 + 8*s0*s2 + s0*s1*s2, 1), torch.float32)
        buf6 = empty_strided_cuda((), (), torch.float32)
        buf13 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [mm_loss, randn_like, softmax, kl_loss, log_prob, add], Original ATen: [aten.arange, aten.ne, aten.gather, aten.rsub, aten.add, aten.clamp_min, aten.scalar_tensor, aten.where, aten.mean, aten.randn_like, aten._softmax, aten.xlogy, aten._log_softmax, aten.mul, aten.sub]
        triton_red_fused__log_softmax__softmax_add_arange_clamp_min_gather_mean_mul_ne_randn_like_rsub_scalar_tensor_sub_where_xlogy_3_r0_numel = 64*s0 + 8*s0*s1 + 8*s0*s2 + s0*s1*s2
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax__softmax_add_arange_clamp_min_gather_mean_mul_ne_randn_like_rsub_scalar_tensor_sub_where_xlogy_3[grid(1)](buf13, buf3, buf2, buf5, buf7, 1, 3, 32, 32, 40, 1, 4800, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del buf2
        del buf3
        del buf5
        del buf7
    return (buf13, )


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
