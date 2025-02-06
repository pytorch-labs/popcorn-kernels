# AOT ID: ['77_inference']
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


# kernel path: /tmp/torchinductor_sahanp/mc/cmc3saeo3am3huwsmaqcf6yywnzamsi4njn6zezmowzptzisrudz.py
# Topologically Sorted Source Nodes: [x, result], Original ATen: [aten.constant_pad_nd, aten.threshold]
# Source node to ATen node mapping:
#   result => full_default, le, where
#   x => constant_pad_nd
# Graph fragment:
#   %constant_pad_nd : [num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%arg3_1, [2, 2, 2, 2], 0.0), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%constant_pad_nd, 0.5), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le, %full_default, %constant_pad_nd), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_constant_pad_nd_threshold_0(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // ks0) % ks1)
    x0 = (xindex % ks0)
    x2 = xindex // ks4
    x4 = xindex
    tmp0 = (-2) + x1
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
    tmp12 = tl.load(in_ptr0 + ((-2) + x0 + ((-2)*ks3) + ks3*x1 + ks2*ks3*x2), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = 0.5
    tmp14 = tmp12 <= tmp13
    tmp15 = 0.0
    tmp16 = tl.where(tmp14, tmp15, tmp12)
    tl.store(out_ptr0 + (x4), tmp16, xmask)




# kernel path: /tmp/torchinductor_sahanp/vj/cvjexqvdsmeuwa2mf6hadqyxoap6etbm53dgr47e75qhzp55kcm6.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten._adaptive_avg_pool2d]
# Source node to ATen node mapping:
#   x_2 => _adaptive_avg_pool2d
# Graph fragment:
#   %_adaptive_avg_pool2d : [num_users=1] = call_function[target=torch.ops.aten._adaptive_avg_pool2d.default](args = (%unsqueeze, [1, 10]), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__adaptive_avg_pool2d_1(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (4*(x0 // ks1) + 16*x1 + ks3*(x0 // ks1) + 4*ks2*x1 + 4*ks3*x1 + ks2*ks3*x1 + ((x0 % ks1))), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/gz/cgzxqukvuzltjb2hv3qgkjrqzhhhv3iuamqkshsexzki6xhogz3o.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.bernoulli]
# Source node to ATen node mapping:
#   x_3 => inductor_lookup_seed_default, inductor_random_default
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, %arg0_1, 1], %inductor_lookup_seed_default, rand), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_bernoulli_2(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (x0), tmp2, xmask)




# kernel path: /tmp/torchinductor_sahanp/oc/cocummrahx2gftt244gipf2rzp3wyh3oqhazwieokrss4tdyw32k.py
# Topologically Sorted Source Nodes: [x_3, x_4], Original ATen: [aten.bernoulli, aten._to_copy, aten.mul, aten.add, aten.neg, aten._softmax]
# Source node to ATen node mapping:
#   x_3 => add_31, add_44, add_63, convert_element_type, lt, mul_33, mul_42, mul_45
#   x_4 => amax, exp, neg, sub_27, sum_1
# Graph fragment:
#   %lt : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%inductor_random_default, 0.5), kwargs = {})
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt, torch.float32), kwargs = {})
#   %mul_42 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%convert_element_type, 0.8864048946659319), kwargs = {})
#   %mul_45 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze, %mul_42), kwargs = {})
#   %add_31 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%convert_element_type, -1), kwargs = {})
#   %mul_33 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%add_31, 1.558387861036063), kwargs = {})
#   %add_44 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_33, 0.7791939305180315), kwargs = {})
#   %add_63 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_45, %add_44), kwargs = {})
#   %neg : [num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%add_63,), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%neg, [1], True), kwargs = {})
#   %sub_27 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%neg, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_27,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__softmax__to_copy_add_bernoulli_mul_neg_3(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 10
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp17 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 10*r0_1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.5
        tmp3 = tmp1 < tmp2
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 0.8864048946659319
        tmp6 = tmp4 * tmp5
        tmp7 = tmp0 * tmp6
        tmp8 = -1.0
        tmp9 = tmp4 + tmp8
        tmp10 = 1.558387861036063
        tmp11 = tmp9 * tmp10
        tmp12 = 0.7791939305180315
        tmp13 = tmp11 + tmp12
        tmp14 = tmp7 + tmp13
        tmp15 = -tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
        tmp18 = triton_helpers.maximum(_tmp17, tmp16)
        _tmp17 = tl.where(r0_mask & xmask, tmp18, _tmp17)
    tmp17 = triton_helpers.max2(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp17, xmask)
    _tmp38 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp19 = tl.load(in_ptr0 + (x0 + 10*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp20 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp21 = 0.5
        tmp22 = tmp20 < tmp21
        tmp23 = tmp22.to(tl.float32)
        tmp24 = 0.8864048946659319
        tmp25 = tmp23 * tmp24
        tmp26 = tmp19 * tmp25
        tmp27 = -1.0
        tmp28 = tmp23 + tmp27
        tmp29 = 1.558387861036063
        tmp30 = tmp28 * tmp29
        tmp31 = 0.7791939305180315
        tmp32 = tmp30 + tmp31
        tmp33 = tmp26 + tmp32
        tmp34 = -tmp33
        tmp35 = tmp34 - tmp17
        tmp36 = tl_math.exp(tmp35)
        tmp37 = tl.broadcast_to(tmp36, [XBLOCK, R0_BLOCK])
        tmp39 = _tmp38 + tmp37
        _tmp38 = tl.where(r0_mask & xmask, tmp39, _tmp38)
    tmp38 = tl.sum(_tmp38, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp38, xmask)




# kernel path: /tmp/torchinductor_sahanp/hh/chh4ou7clqc72vgmq6lo4xcuz27wpjgvdocnsckznqalch5f6deo.py
# Topologically Sorted Source Nodes: [x_3, x_4], Original ATen: [aten.bernoulli, aten._to_copy, aten.mul, aten.add, aten.neg, aten._softmax]
# Source node to ATen node mapping:
#   x_3 => add_31, add_44, add_63, convert_element_type, lt, mul_33, mul_42, mul_45
#   x_4 => div, exp, neg, sub_27
# Graph fragment:
#   %lt : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%inductor_random_default, 0.5), kwargs = {})
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt, torch.float32), kwargs = {})
#   %mul_42 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%convert_element_type, 0.8864048946659319), kwargs = {})
#   %mul_45 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze, %mul_42), kwargs = {})
#   %add_31 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%convert_element_type, -1), kwargs = {})
#   %mul_33 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%add_31, 1.558387861036063), kwargs = {})
#   %add_44 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_33, 0.7791939305180315), kwargs = {})
#   %add_63 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_45, %add_44), kwargs = {})
#   %neg : [num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%add_63,), kwargs = {})
#   %sub_27 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%neg, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_27,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__softmax__to_copy_add_bernoulli_mul_neg_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 10
    x0 = (xindex % 10)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = 0.5
    tmp3 = tmp1 < tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 0.8864048946659319
    tmp6 = tmp4 * tmp5
    tmp7 = tmp0 * tmp6
    tmp8 = -1.0
    tmp9 = tmp4 + tmp8
    tmp10 = 1.558387861036063
    tmp11 = tmp9 * tmp10
    tmp12 = 0.7791939305180315
    tmp13 = tmp11 + tmp12
    tmp14 = tmp7 + tmp13
    tmp15 = -tmp14
    tmp17 = tmp15 - tmp16
    tmp18 = tl_math.exp(tmp17)
    tmp20 = tmp18 / tmp19
    tl.store(in_out_ptr0 + (x2), tmp20, xmask)




#include "/tmp/torchinductor_sahanp/3b/c3bi5gk6mslf6u4iaqafhxm64z6u65e3eain4xlary5blqnvv6xx.h"
extern "C"  void kernel(const int64_t* in_ptr0,
                       int64_t* out_ptr0,
                       int64_t* out_ptr1,
                       int64_t* out_ptr2,
                       const int64_t ks0)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(5L); x0+=static_cast<int64_t>(1L))
        {
            {
                {
                    auto tmp0 = in_ptr0[static_cast<int64_t>(0L)];
                    auto tmp1 = x0;
                    auto tmp2 = c10::convert<int32_t>(tmp1);
                    auto tmp3 = static_cast<int64_t>(0);
                    auto tmp4 = static_cast<int64_t>(10);
                    auto tmp5 = randint64_cpu(tmp0, tmp2, tmp3, tmp4);
                    out_ptr0[static_cast<int64_t>(x0)] = tmp5;
                }
            }
        }
    }
    {
        {
            {
                auto tmp0 = 10L*ks0;
                auto tmp1 = c10::convert<int64_t>(tmp0);
                out_ptr1[static_cast<int64_t>(0L)] = tmp1;
            }
        }
    }
    {
        {
            {
                auto tmp0 = in_ptr0[static_cast<int64_t>(1L)];
                auto tmp1 = static_cast<int32_t>(0);
                auto tmp2 = static_cast<int64_t>(1);
                auto tmp3 = static_cast<int64_t>(6);
                auto tmp4 = randint64_cpu(tmp0, tmp1, tmp2, tmp3);
                out_ptr2[static_cast<int64_t>(0L)] = tmp4;
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
        ps0 = 4 + s2
        ps1 = 4 + s1
        ps2 = 16 + 4*s1 + 4*s2 + s1*s2
        buf0 = empty_strided_cuda((1, s0, 4 + s1, 4 + s2), (16*s0 + 4*s0*s1 + 4*s0*s2 + s0*s1*s2, 16 + 4*s1 + 4*s2 + s1*s2, 4 + s2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, result], Original ATen: [aten.constant_pad_nd, aten.threshold]
        triton_poi_fused_constant_pad_nd_threshold_0_xnumel = 16*s0 + 4*s0*s1 + 4*s0*s2 + s0*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_threshold_0[grid(triton_poi_fused_constant_pad_nd_threshold_0_xnumel)](arg3_1, buf0, 36, 36, 32, 32, 1296, 3888, XBLOCK=256, num_warps=4, num_stages=1)
        del arg3_1
        ps3 = 16 + 4*s1 + 4*s2 + s1*s2
        buf1 = empty_strided_cuda((1, s0, 1, 16 + 4*s1 + 4*s2 + s1*s2), (16*s0 + 4*s0*s1 + 4*s0*s2 + s0*s1*s2, 16 + 4*s1 + 4*s2 + s1*s2, 16*s0 + 4*s0*s1 + 4*s0*s2 + s0*s1*s2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten._adaptive_avg_pool2d]
        triton_poi_fused__adaptive_avg_pool2d_1_xnumel = 16*s0 + 4*s0*s1 + 4*s0*s2 + s0*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused__adaptive_avg_pool2d_1[grid(triton_poi_fused__adaptive_avg_pool2d_1_xnumel)](buf0, buf1, 1296, 36, 32, 32, 3888, XBLOCK=128, num_warps=4, num_stages=1)
        del buf0
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten._adaptive_avg_pool2d]
        buf2 = torch.ops.aten._adaptive_avg_pool2d.default(buf1, [1, 10])
        del buf1
        buf3 = buf2
        del buf2
        buf4 = empty_strided_cuda((1, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf4)
        buf5 = empty_strided_cuda((1, s0, 1), (s0, 1, s0), torch.float32)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.bernoulli]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bernoulli_2[grid(s0)](buf4, buf5, 0, 3, XBLOCK=4, num_warps=1, num_stages=1)
        del buf4
        buf6 = empty_strided_cuda((1, 1, 10), (10, 10, 1), torch.float32)
        buf7 = empty_strided_cuda((1, 1, 10), (10, 10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_3, x_4], Original ATen: [aten.bernoulli, aten._to_copy, aten.mul, aten.add, aten.neg, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_red_fused__softmax__to_copy_add_bernoulli_mul_neg_3[grid(10)](buf3, buf5, buf6, buf7, 10, 3, XBLOCK=1, R0_BLOCK=4, num_warps=2, num_stages=1)
        buf8 = reinterpret_tensor(buf3, (1, s0, 10), (10*s0, 10, 1), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [x_3, x_4], Original ATen: [aten.bernoulli, aten._to_copy, aten.mul, aten.add, aten.neg, aten._softmax]
        triton_poi_fused__softmax__to_copy_add_bernoulli_mul_neg_4_xnumel = 10*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax__to_copy_add_bernoulli_mul_neg_4[grid(triton_poi_fused__softmax__to_copy_add_bernoulli_mul_neg_4_xnumel)](buf8, buf5, buf6, buf7, 30, XBLOCK=32, num_warps=1, num_stages=1)
        del buf5
        del buf6
        del buf7
    buf9 = empty_strided_cpu((2, ), (1, ), torch.int64)
    # Topologically Sorted Source Nodes: [], Original ATen: []
    aten.randint.low_out(-9223372036854775808, 9223372036854775807, [2], out=buf9)
    buf10 = empty_strided_cpu((1, 5), (5, 1), torch.int64)
    buf11 = empty_strided_cpu((1, ), (1, ), torch.int64)
    buf12 = empty_strided_cpu((1, ), (1, ), torch.int64)
    cpp_fused_full_randint_5(buf9, buf10, buf11, buf12, s0)
    return (reinterpret_tensor(buf8, (1, 10*s0), (10*s0, 1), 0), buf10, buf11, buf12, )


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
