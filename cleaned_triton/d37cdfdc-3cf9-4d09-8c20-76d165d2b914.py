# AOT ID: ['158_inference']
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


# kernel path: /tmp/torchinductor_sahanp/2f/c2f7ejo2yldfcuotbyhusswfltbglv2i2uzdqaqiiq3m5gcl36ok.py
# Topologically Sorted Source Nodes: [target_bce, bce_loss, sigmoid, target_cosine, cosine_loss, add, poisson_loss, target_poisson, add_1], Original ATen: [aten.randint, aten.binary_cross_entropy, aten.sigmoid, aten.eq, aten.fill, aten.mul, aten.sum, aten.add, aten.sqrt, aten.div, aten.sub, aten.zeros_like, aten.where, aten.clamp_min, aten.mean, aten.exp]
# Source node to ATen node mapping:
#   add => add_33
#   add_1 => add_34
#   bce_loss => full_default, full_default_1, log, log1p, maximum, maximum_1, mean, mul_15, mul_16, neg, sub_6, sub_7
#   cosine_loss => add_22, add_25, add_26, clamp_min, div, eq_10, eq_11, full_default_2, full_default_3, mean_1, mul_17, mul_20, mul_23, mul_26, sqrt, sub_11, sub_12, sum_1, sum_2, sum_3, where, where_1
#   poisson_loss => exp, mean_2, mul_27, sub_15
#   sigmoid => sigmoid
#   target_bce => convert_element_type_default_2, inductor_lookup_seed_default, inductor_randint_default_2
#   target_cosine => convert_element_type_default_1, inductor_lookup_seed_default_1, inductor_randint_default_1
#   target_poisson => convert_element_type_default, inductor_lookup_seed_default_2, inductor_randint_default
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_randint_default_2 : [num_users=1] = call_function[target=torch.ops.prims.inductor_randint.default](args = (0, 2, [1, %mul_8], %inductor_lookup_seed_default), kwargs = {})
#   %convert_element_type_default_2 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%inductor_randint_default_2, torch.float32), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_default_2, 1), kwargs = {})
#   %sigmoid : [num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view,), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%sigmoid,), kwargs = {})
#   %log1p : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%neg,), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -100), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %maximum : [num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%log1p, %full_default), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %maximum), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sigmoid,), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -100), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %maximum_1 : [num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%log, %full_default_1), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_2, %maximum_1), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_15, %mul_16), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sub_7,), kwargs = {})
#   %inductor_lookup_seed_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 1), kwargs = {})
#   %inductor_randint_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_randint.default](args = (0, 2, [1], %inductor_lookup_seed_default_1), kwargs = {})
#   %convert_element_type_default_1 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%inductor_randint_default_1, torch.float32), kwargs = {})
#   %eq_10 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%convert_element_type_default_1, 1), kwargs = {})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, %view), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_17, [1]), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, %view), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_20, [1]), kwargs = {})
#   %add_22 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%sum_2, 9.999999960041972e-13), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, %view), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_23, [1]), kwargs = {})
#   %add_25 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%sum_3, 9.999999960041972e-13), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_22, %add_25), kwargs = {})
#   %sqrt : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%mul_26,), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_1, %sqrt), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default_3, %div), kwargs = {})
#   %full_default_2 : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([1], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_10, %sub_11, %full_default_2), kwargs = {})
#   %eq_11 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%convert_element_type_default_1, -1), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Scalar](args = (%div, 0.0), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_12, 0), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_11, %clamp_min, %full_default_2), kwargs = {})
#   %add_26 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where, %where_1), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%add_26,), kwargs = {})
#   %add_33 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, %mean_1), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%view,), kwargs = {})
#   %inductor_lookup_seed_default_2 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 2), kwargs = {})
#   %inductor_randint_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_randint.default](args = (0, 10, [1, %mul_8], %inductor_lookup_seed_default_2), kwargs = {})
#   %convert_element_type_default : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%inductor_randint_default, torch.float32), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default, %view), kwargs = {})
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%exp, %mul_27), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sub_15,), kwargs = {})
#   %add_34 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_33, %mean_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_binary_cross_entropy_clamp_min_div_eq_exp_fill_mean_mul_randint_sigmoid_sqrt_sub_sum_where_zeros_like_0(in_out_ptr0, in_ptr0, in_ptr1, load_seed_offset, load_seed_offset1, ks2, load_seed_offset2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp20 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp24 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp34 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp8 = tl.load(in_ptr1 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp0 = tl.load(in_ptr0 + load_seed_offset)
        tmp1 = r0_0
        tmp2 = tl.full([1, 1], 0, tl.int64)
        tmp3 = tl.full([1, 1], 2, tl.int64)
        tmp4 = triton_helpers.randint64(tmp0, (tmp1).to(tl.uint32), tmp2, tmp3)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = 1.0
        tmp7 = tmp5 - tmp6
        tmp9 = tl.sigmoid(tmp8)
        tmp10 = -tmp9
        tmp11 = libdevice.log1p(tmp10)
        tmp12 = -100.0
        tmp13 = triton_helpers.maximum(tmp11, tmp12)
        tmp14 = tmp7 * tmp13
        tmp15 = tl_math.log(tmp9)
        tmp16 = triton_helpers.maximum(tmp15, tmp12)
        tmp17 = tmp5 * tmp16
        tmp18 = tmp14 - tmp17
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(r0_mask, tmp21, _tmp20)
        tmp22 = tmp8 * tmp8
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, R0_BLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(r0_mask, tmp25, _tmp24)
        tmp26 = tl_math.exp(tmp8)
        tmp27 = tl.load(in_ptr0 + load_seed_offset1)
        tmp28 = tl.full([1, 1], 10, tl.int64)
        tmp29 = triton_helpers.randint64(tmp27, (tmp1).to(tl.uint32), tmp2, tmp28)
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp30 * tmp8
        tmp32 = tmp26 - tmp31
        tmp33 = tl.broadcast_to(tmp32, [XBLOCK, R0_BLOCK])
        tmp35 = _tmp34 + tmp33
        _tmp34 = tl.where(r0_mask, tmp35, _tmp34)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tmp34 = tl.sum(_tmp34, 1)[:, None]
    tmp36 = 125*ks2
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tmp20 / tmp37
    tmp39 = tl.load(in_ptr0 + load_seed_offset2)
    tmp40 = tl.full([1, 1], 0, tl.int32)
    tmp41 = tl.full([1, 1], 0, tl.int64)
    tmp42 = tl.full([1, 1], 2, tl.int64)
    tmp43 = triton_helpers.randint64(tmp39, (tmp40).to(tl.uint32), tmp41, tmp42)
    tmp44 = tmp43.to(tl.float32)
    tmp45 = 1.0
    tmp46 = tmp44 == tmp45
    tmp47 = 9.999999960041972e-13
    tmp48 = tmp24 + tmp47
    tmp49 = tmp48 * tmp48
    tmp50 = libdevice.sqrt(tmp49)
    tmp51 = tmp24 / tmp50
    tmp52 = tmp45 - tmp51
    tmp53 = 0.0
    tmp54 = tl.where(tmp46, tmp52, tmp53)
    tmp55 = -1.0
    tmp56 = tmp44 == tmp55
    tmp57 = tmp51 - tmp53
    tmp58 = triton_helpers.maximum(tmp57, tmp53)
    tmp59 = tl.where(tmp56, tmp58, tmp53)
    tmp60 = tmp54 + tmp59
    tmp61 = tmp60 / tmp45
    tmp62 = tmp38 + tmp61
    tmp63 = tmp34 / tmp37
    tmp64 = tmp62 + tmp63
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp64, None)







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
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.adaptive_max_pool3d]
        buf0 = torch.ops.aten.adaptive_max_pool3d.default(arg4_1, [5, 5, 5])
        del arg4_1
        buf1 = buf0[0]
        del buf0
        buf3 = empty_strided_cuda((3, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [3], out=buf3)
        buf4 = empty_strided_cuda((), (), torch.float32)
        buf9 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [target_bce, bce_loss, sigmoid, target_cosine, cosine_loss, add, poisson_loss, target_poisson, add_1], Original ATen: [aten.randint, aten.binary_cross_entropy, aten.sigmoid, aten.eq, aten.fill, aten.mul, aten.sum, aten.add, aten.sqrt, aten.div, aten.sub, aten.zeros_like, aten.where, aten.clamp_min, aten.mean, aten.exp]
        triton_red_fused_add_binary_cross_entropy_clamp_min_div_eq_exp_fill_mean_mul_randint_sigmoid_sqrt_sub_sum_where_zeros_like_0_r0_numel = 125*s0
        stream0 = get_raw_stream(0)
        triton_red_fused_add_binary_cross_entropy_clamp_min_div_eq_exp_fill_mean_mul_randint_sigmoid_sqrt_sub_sum_where_zeros_like_0[grid(1)](buf9, buf3, buf1, 0, 2, 10, 1, 1, 1250, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del buf1
        del buf3
    return (buf9, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 10
    arg1_1 = 20
    arg2_1 = 20
    arg3_1 = 20
    arg4_1 = rand_strided((1, 10, 20, 20, 20), (80000, 8000, 400, 20, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
