# AOT ID: ['159_inference']
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


# kernel path: /tmp/torchinductor_sahanp/4a/c4avngit4peai5wyzxbzupzmlscvio7pwbdws2vz3lanrzskzb4n.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x => constant_pad_nd
# Graph fragment:
#   %constant_pad_nd : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%arg3_1, [2, 2, 2, 2], 3.0), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_constant_pad_nd_0(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
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
    tmp12 = tl.load(in_ptr0 + ((-2) + x0 + ((-2)*ks3) + ks3*x1 + ks2*ks3*x2), tmp11 & xmask, eviction_policy='evict_last', other=3.0)
    tl.store(out_ptr0 + (x4), tmp12, xmask)




# kernel path: /tmp/torchinductor_sahanp/mk/cmkkgp7ofkq5pn7tg4u4fkt634i7ukzs55qea2bgwx5wouse72ly.py
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
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*(((x0 // ks0) % ks1)) + 16*(x0 // (16 + 4*ks2 + 4*ks3 + ks2*ks3)) + ks3*(((x0 // ks0) % ks1)) + 4*ks2*(x0 // (16 + 4*ks2 + 4*ks3 + ks2*ks3)) + 4*ks3*(x0 // (16 + 4*ks2 + 4*ks3 + ks2*ks3)) + ks2*ks3*(x0 // (16 + 4*ks2 + 4*ks3 + ks2*ks3)) + ((x0 % ks0))), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/un/cunixvza2psh6tnjjzmhi5aquuspz5hwum37wcqyk667gsrupbp2.py
# Topologically Sorted Source Nodes: [x_3, x_4, x_5, x_6, x_7, randn_like, mul, positive, dist_pos, add_2, randn_like_1, mul_1, negative, dist_neg, sub, loss, loss_1], Original ATen: [aten.elu, aten.randn_like, aten.mul, aten.add, aten.sub, aten.norm, aten.clamp_min, aten.mean]
# Source node to ATen node mapping:
#   add_2 => add_12
#   dist_neg => add_11, pow_3, pow_4, sub_6, sum_2
#   dist_pos => add_10, pow_1, pow_2, sub_5, sum_1
#   loss => clamp_min
#   loss_1 => mean
#   mul => mul_24
#   mul_1 => mul_25
#   negative => add_9
#   positive => add_8
#   randn_like => inductor_lookup_seed_default, inductor_random_default_1
#   randn_like_1 => inductor_lookup_seed_default_1, inductor_random_default
#   sub => sub_7
#   x_3 => expm1, gt, mul_10, mul_11, mul_9, where
#   x_4 => expm1_1, gt_1, mul_12, mul_13, mul_14, where_1
#   x_5 => expm1_2, gt_2, mul_15, mul_16, mul_17, where_2
#   x_6 => expm1_3, gt_3, mul_18, mul_19, mul_20, where_3
#   x_7 => expm1_4, gt_4, mul_21, mul_22, mul_23, where_4
# Graph fragment:
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%squeeze, 0), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze, 1.0507009873554805), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze, 1.0), kwargs = {})
#   %expm1 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%mul_10,), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expm1, 1.7580993408473766), kwargs = {})
#   %where : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt, %mul_9, %mul_11), kwargs = {})
#   %gt_1 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%where, 0), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where, 1.0507009873554805), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where, 1.0), kwargs = {})
#   %expm1_1 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%mul_13,), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expm1_1, 1.7580993408473766), kwargs = {})
#   %where_1 : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %mul_12, %mul_14), kwargs = {})
#   %gt_2 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%where_1, 0), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_1, 1.0507009873554805), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_1, 1.0), kwargs = {})
#   %expm1_2 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%mul_16,), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expm1_2, 1.7580993408473766), kwargs = {})
#   %where_2 : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %mul_15, %mul_17), kwargs = {})
#   %gt_3 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%where_2, 0), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_2, 1.0507009873554805), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_2, 1.0), kwargs = {})
#   %expm1_3 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%mul_19,), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expm1_3, 1.7580993408473766), kwargs = {})
#   %where_3 : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt_3, %mul_18, %mul_20), kwargs = {})
#   %gt_4 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%where_3, 0), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_3, 1.0507009873554805), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_3, 1.0), kwargs = {})
#   %expm1_4 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%mul_22,), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expm1_4, 1.7580993408473766), kwargs = {})
#   %where_4 : [num_users=4] = call_function[target=torch.ops.aten.where.self](args = (%gt_4, %mul_21, %mul_23), kwargs = {})
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 10], %inductor_lookup_seed_default, randn), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%inductor_random_default_1, 0.1), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_4, %mul_24), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_4, %add_8), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%sub_5, 1e-06), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_10, 2.0), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1]), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_2, 1.0), kwargs = {})
#   %inductor_lookup_seed_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 1), kwargs = {})
#   %inductor_random_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 10], %inductor_lookup_seed_default_1, randn), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%inductor_random_default, 0.2), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_4, %mul_25), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_4, %add_9), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%sub_6, 1e-06), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_11, 2.0), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_3, [1]), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_2, 0.5), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_12, %pow_4), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_7, 0), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%clamp_min,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_clamp_min_elu_mean_mul_norm_randn_like_sub_2(in_out_ptr0, in_out_ptr1, in_ptr0, load_seed_offset, load_seed_offset1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 10
    R0_BLOCK: tl.constexpr = 16
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
    tmp0 = tl.load(in_out_ptr0 + (r0_0), r0_mask, other=0.0)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 1.0507009873554805
    tmp4 = tmp0 * tmp3
    tmp5 = 1.0
    tmp6 = tmp0 * tmp5
    tmp7 = libdevice.expm1(tmp6)
    tmp8 = 1.7580993408473766
    tmp9 = tmp7 * tmp8
    tmp10 = tl.where(tmp2, tmp4, tmp9)
    tmp11 = tmp10 > tmp1
    tmp12 = tmp10 * tmp3
    tmp13 = tmp10 * tmp5
    tmp14 = libdevice.expm1(tmp13)
    tmp15 = tmp14 * tmp8
    tmp16 = tl.where(tmp11, tmp12, tmp15)
    tmp17 = tmp16 > tmp1
    tmp18 = tmp16 * tmp3
    tmp19 = tmp16 * tmp5
    tmp20 = libdevice.expm1(tmp19)
    tmp21 = tmp20 * tmp8
    tmp22 = tl.where(tmp17, tmp18, tmp21)
    tmp23 = tmp22 > tmp1
    tmp24 = tmp22 * tmp3
    tmp25 = tmp22 * tmp5
    tmp26 = libdevice.expm1(tmp25)
    tmp27 = tmp26 * tmp8
    tmp28 = tl.where(tmp23, tmp24, tmp27)
    tmp29 = tmp28 > tmp1
    tmp30 = tmp28 * tmp3
    tmp31 = tmp28 * tmp5
    tmp32 = libdevice.expm1(tmp31)
    tmp33 = tmp32 * tmp8
    tmp34 = tl.where(tmp29, tmp30, tmp33)
    tmp35 = tl.load(in_ptr0 + load_seed_offset)
    tmp36 = r0_0
    tmp37 = tl.randn(tmp35, (tmp36).to(tl.uint32))
    tmp38 = tl.load(in_ptr0 + load_seed_offset1)
    tmp39 = tl.randn(tmp38, (tmp36).to(tl.uint32))
    tmp40 = 0.1
    tmp41 = tmp39 * tmp40
    tmp42 = tmp34 + tmp41
    tmp43 = tmp34 - tmp42
    tmp44 = 1e-06
    tmp45 = tmp43 + tmp44
    tmp46 = tmp45 * tmp45
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK, R0_BLOCK])
    tmp49 = tl.where(r0_mask, tmp47, 0)
    tmp50 = tl.sum(tmp49, 1)[:, None]
    tmp51 = 0.2
    tmp52 = tmp37 * tmp51
    tmp53 = tmp34 + tmp52
    tmp54 = tmp34 - tmp53
    tmp55 = tmp54 + tmp44
    tmp56 = tmp55 * tmp55
    tmp57 = tl.broadcast_to(tmp56, [XBLOCK, R0_BLOCK])
    tmp59 = tl.where(r0_mask, tmp57, 0)
    tmp60 = tl.sum(tmp59, 1)[:, None]
    tmp61 = libdevice.sqrt(tmp50)
    tmp62 = tmp61 + tmp5
    tmp63 = libdevice.sqrt(tmp60)
    tmp64 = tmp62 - tmp63
    tmp65 = triton_helpers.maximum(tmp64, tmp1)
    tmp66 = tmp65 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp66, None)







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
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_0_xnumel = 16*s0 + 4*s0*s1 + 4*s0*s2 + s0*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_0[grid(triton_poi_fused_constant_pad_nd_0_xnumel)](arg3_1, buf0, 36, 36, 32, 32, 1296, 3888, XBLOCK=256, num_warps=4, num_stages=1)
        del arg3_1
        buf1 = empty_strided_cuda((1, 1, 16*s0 + 4*s0*s1 + 4*s0*s2 + s0*s1*s2), (16*s0 + 4*s0*s1 + 4*s0*s2 + s0*s1*s2, 16*s0 + 4*s0*s1 + 4*s0*s2 + s0*s1*s2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten._adaptive_avg_pool2d]
        triton_poi_fused__adaptive_avg_pool2d_1_xnumel = 16*s0 + 4*s0*s1 + 4*s0*s2 + s0*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused__adaptive_avg_pool2d_1[grid(triton_poi_fused__adaptive_avg_pool2d_1_xnumel)](buf0, buf1, 36, 36, 32, 32, 3888, XBLOCK=256, num_warps=4, num_stages=1)
        del buf0
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten._adaptive_avg_pool2d]
        buf2 = torch.ops.aten._adaptive_avg_pool2d.default(buf1, [1, 10])
        del buf1
        buf3 = buf2
        del buf2
        buf5 = empty_strided_cuda((2, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [2], out=buf5)
        buf4 = reinterpret_tensor(buf3, (1, 10), (10, 1), 0); del buf3  # reuse
        buf7 = empty_strided_cuda((1, ), (1, ), torch.float32)
        buf10 = reinterpret_tensor(buf7, (), (), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [x_3, x_4, x_5, x_6, x_7, randn_like, mul, positive, dist_pos, add_2, randn_like_1, mul_1, negative, dist_neg, sub, loss, loss_1], Original ATen: [aten.elu, aten.randn_like, aten.mul, aten.add, aten.sub, aten.norm, aten.clamp_min, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clamp_min_elu_mean_mul_norm_randn_like_sub_2[grid(1)](buf4, buf10, buf5, 1, 0, 1, 10, XBLOCK=1, num_warps=2, num_stages=1)
        del buf4
        del buf5
    return (buf10, )


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
