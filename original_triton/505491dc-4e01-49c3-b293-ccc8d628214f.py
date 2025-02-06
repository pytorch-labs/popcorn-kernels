# AOT ID: ['98_inference']
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


# kernel path: /tmp/torchinductor_sahanp/vr/cvrjiwfec2bqnrk4e7egi5duul2vmjjaqaeqzhglidmkdypenwq7.py
# Topologically Sorted Source Nodes: [randn_like, target, kldiv_loss, log_softmax], Original ATen: [aten.randn_like, aten._softmax, aten.xlogy, aten._log_softmax, aten.mul, aten.sub, aten.sum, aten.div]
# Source node to ATen node mapping:
#   kldiv_loss => div_1, eq_80, full_default, full_default_1, isnan, log_1, mul_123, mul_126, sub_80, sum_5, where, where_1
#   log_softmax => amax_1, exp_1, log, sub_75, sub_76, sum_4
#   randn_like => inductor_lookup_seed_default, inductor_random_default
#   target => amax, div, exp, sub_73, sum_3
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default : [num_users=2] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, %sym_size_int_7], %inductor_lookup_seed_default, randn), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%inductor_random_default, [1], True), kwargs = {})
#   %sub_73 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%inductor_random_default, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_73,), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
#   %div : [num_users=5] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_3), kwargs = {})
#   %isnan : [num_users=1] = call_function[target=torch.ops.aten.isnan.default](args = (%div,), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], nan), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %eq_80 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%div, 0), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %log_1 : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%div,), kwargs = {})
#   %mul_126 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %log_1), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_80, %full_default, %mul_126), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%isnan, %full_default_1, %where), kwargs = {})
#   %amax_1 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view, [1], True), kwargs = {})
#   %sub_75 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %amax_1), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_75,), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_1, [1], True), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_4,), kwargs = {})
#   %sub_76 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_75, %log), kwargs = {})
#   %mul_123 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %sub_76), kwargs = {})
#   %sub_80 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_1, %mul_123), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%sub_80,), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_5, 1), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__log_softmax__softmax_div_mul_randn_like_sub_sum_xlogy_0(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, load_seed_offset, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + load_seed_offset)
        tmp1 = r0_0
        tmp2 = tl.randn(tmp0, (tmp1).to(tl.uint32))
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = triton_helpers.maximum(_tmp4, tmp3)
        _tmp4 = tl.where(r0_mask, tmp5, _tmp4)
        tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp2, r0_mask)
    tmp4 = triton_helpers.max2(_tmp4, 1)[:, None]
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp6 = tl.load(in_ptr1 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = triton_helpers.maximum(_tmp8, tmp7)
        _tmp8 = tl.where(r0_mask, tmp9, _tmp8)
    tmp8 = triton_helpers.max2(_tmp8, 1)[:, None]
    _tmp14 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp10 = tl.load(out_ptr0 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp10 - tmp4
        tmp12 = tl_math.exp(tmp11)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(r0_mask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    _tmp20 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp16 = tl.load(in_ptr1 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp17 = tmp16 - tmp8
        tmp18 = tl_math.exp(tmp17)
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(r0_mask, tmp21, _tmp20)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    _tmp41 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp22 = tl.load(out_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp34 = tl.load(in_ptr1 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp23 = tmp22 - tmp4
        tmp24 = tl_math.exp(tmp23)
        tmp25 = tmp24 / tmp14
        tmp26 = libdevice.isnan(tmp25).to(tl.int1)
        tmp27 = 0.0
        tmp28 = tmp25 == tmp27
        tmp29 = tl_math.log(tmp25)
        tmp30 = tmp25 * tmp29
        tmp31 = tl.where(tmp28, tmp27, tmp30)
        tmp32 = float("nan")
        tmp33 = tl.where(tmp26, tmp32, tmp31)
        tmp35 = tmp34 - tmp8
        tmp36 = tl_math.log(tmp20)
        tmp37 = tmp35 - tmp36
        tmp38 = tmp25 * tmp37
        tmp39 = tmp33 - tmp38
        tmp40 = tl.broadcast_to(tmp39, [XBLOCK, R0_BLOCK])
        tmp42 = _tmp41 + tmp40
        _tmp41 = tl.where(r0_mask, tmp42, _tmp41)
    tmp41 = tl.sum(_tmp41, 1)[:, None]
    tmp43 = 1.0
    tmp44 = tmp41 * tmp43
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp44, None)




# kernel path: /tmp/torchinductor_sahanp/wa/cwamvrssj3porsjmr5lgdwub45dt45y2kv72eulfdorbfek72c7b.py
# Topologically Sorted Source Nodes: [dist_pos, add, dist_neg, sub, loss, triplet_loss], Original ATen: [aten.sub, aten.add, aten.norm, aten.clamp_min, aten.mean]
# Source node to ATen node mapping:
#   add => add_133
#   dist_neg => add_123, pow_3, pow_4, sub_63, sum_2
#   dist_pos => add_108, pow_1, pow_2, sub_59, sum_1
#   loss => clamp_min
#   sub => sub_68
#   triplet_loss => mean
# Graph fragment:
#   %sub_59 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze_2, %unsqueeze_5), kwargs = {})
#   %add_108 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%sub_59, 1e-06), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_108, 2.0), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [4]), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %add_133 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_2, 1.0), kwargs = {})
#   %sub_63 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze_2, %unsqueeze_8), kwargs = {})
#   %add_123 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%sub_63, 1e-06), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_123, 2.0), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_3, [4]), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_2, 0.5), kwargs = {})
#   %sub_68 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_133, %pow_4), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_68, 0), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%clamp_min,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_clamp_min_mean_norm_sub_1(in_out_ptr0, in_ptr0, ks0, ks1, ks2, ks3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp18 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_0*(ks0 // 2)*(ks1 // 2)*(ks2 // 2)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr0 + (1 + (ks1 // 2)*(ks2 // 2) + r0_0*(ks0 // 2)*(ks1 // 2)*(ks2 // 2) + (ks2 // 2)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr0 + (2 + 2*(ks2 // 2) + 2*(ks1 // 2)*(ks2 // 2) + r0_0*(ks0 // 2)*(ks1 // 2)*(ks2 // 2)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 - tmp1
        tmp3 = 1e-06
        tmp4 = tmp2 + tmp3
        tmp5 = tmp4 * tmp4
        tmp6 = libdevice.sqrt(tmp5)
        tmp7 = 1.0
        tmp8 = tmp6 + tmp7
        tmp10 = tmp0 - tmp9
        tmp11 = tmp10 + tmp3
        tmp12 = tmp11 * tmp11
        tmp13 = libdevice.sqrt(tmp12)
        tmp14 = tmp8 - tmp13
        tmp15 = 0.0
        tmp16 = triton_helpers.maximum(tmp14, tmp15)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(r0_mask, tmp19, _tmp18)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tmp20 = ks3
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp18 / tmp21
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp22, None)







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
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.max_pool3d_with_indices]
        buf0 = torch.ops.aten.max_pool3d_with_indices.default(arg4_1, [2, 2, 2], [2, 2, 2])
        del arg4_1
        buf4 = empty_strided_cuda((1, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf4)
        buf1 = buf0[0]
        del buf0
        buf5 = empty_strided_cuda((1, s0*(s1 // 2)*(s2 // 2)*(s3 // 2)), (s0*(s1 // 2)*(s2 // 2)*(s3 // 2), 1), torch.float32)
        buf6 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        buf10 = reinterpret_tensor(buf6, (), (), 0); del buf6  # reuse
        buf12 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [randn_like, target, kldiv_loss, log_softmax], Original ATen: [aten.randn_like, aten._softmax, aten.xlogy, aten._log_softmax, aten.mul, aten.sub, aten.sum, aten.div]
        triton_red_fused__log_softmax__softmax_div_mul_randn_like_sub_sum_xlogy_0_r0_numel = s0*(s1 // 2)*(s2 // 2)*(s3 // 2)
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax__softmax_div_mul_randn_like_sub_sum_xlogy_0[grid(1)](buf12, buf4, buf1, buf5, 0, 1, 1536, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del buf4
        del buf5
        buf3 = empty_strided_cuda((), (), torch.float32)
        buf11 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [dist_pos, add, dist_neg, sub, loss, triplet_loss], Original ATen: [aten.sub, aten.add, aten.norm, aten.clamp_min, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_clamp_min_mean_norm_sub_1[grid(1)](buf11, buf1, 16, 16, 16, 3, 1, 3, XBLOCK=1, R0_BLOCK=4, num_warps=2, num_stages=1)
        del buf1
    return (buf11, buf12, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 3
    arg1_1 = 16
    arg2_1 = 16
    arg3_1 = 16
    arg4_1 = rand_strided((1, 3, 16, 16, 16), (12288, 4096, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
