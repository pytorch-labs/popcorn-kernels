# AOT ID: ['6_inference']
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


# kernel path: /tmp/torchinductor_sahanp/d4/cd4zdcdrajlik65ssjgq5qwkgq44kkvnhaapdf7rtf6yadh3p4gv.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   x => var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%arg0_1, [0, 2, 3, 4]), kwargs = {correction: 0, keepdim: True})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_0(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 10
    r0_numel = 125
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 125*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(r0_mask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 125, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(r0_mask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)




# kernel path: /tmp/torchinductor_sahanp/lx/clxcznbvqmibemgcje632tio67cyl7zcxwfenj7zdtjq775rrkde.py
# Topologically Sorted Source Nodes: [x_2, poisson_loss, exp], Original ATen: [aten._log_softmax, aten.exp, aten.mul, aten.sub, aten.mean]
# Source node to ATen node mapping:
#   exp => exp_1
#   poisson_loss => exp_2, mean_1, mul_1, sub_6
#   x_2 => amax, exp, log, sub_1, sub_2, sum_1
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_2, [1], True), kwargs = {})
#   %sub_1 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_2, %amax), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_1,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_1,), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_1, %log), kwargs = {})
#   %exp_2 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%view_3,), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%view_3,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_1, %view_3), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%exp_2, %mul_1), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sub_6,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__log_softmax_exp_mean_mul_sub_1(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 1250
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp11 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_0 // 125), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r0_0 // 125), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 - tmp1
        tmp4 = 125.0
        tmp5 = tmp3 / tmp4
        tmp6 = 1e-05
        tmp7 = tmp5 + tmp6
        tmp8 = libdevice.rsqrt(tmp7)
        tmp9 = tmp2 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp12 = triton_helpers.maximum(_tmp11, tmp10)
        _tmp11 = tl.where(r0_mask, tmp12, _tmp11)
    tmp11 = triton_helpers.max2(_tmp11, 1)[:, None]
    _tmp26 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp13 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr1 + (r0_0 // 125), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.load(in_ptr2 + (r0_0 // 125), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp15 = tmp13 - tmp14
        tmp17 = 125.0
        tmp18 = tmp16 / tmp17
        tmp19 = 1e-05
        tmp20 = tmp18 + tmp19
        tmp21 = libdevice.rsqrt(tmp20)
        tmp22 = tmp15 * tmp21
        tmp23 = tmp22 - tmp11
        tmp24 = tl_math.exp(tmp23)
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(r0_mask, tmp27, _tmp26)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    _tmp45 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp28 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp29 = tl.load(in_ptr1 + (r0_0 // 125), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp31 = tl.load(in_ptr2 + (r0_0 // 125), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp30 = tmp28 - tmp29
        tmp32 = 125.0
        tmp33 = tmp31 / tmp32
        tmp34 = 1e-05
        tmp35 = tmp33 + tmp34
        tmp36 = libdevice.rsqrt(tmp35)
        tmp37 = tmp30 * tmp36
        tmp38 = tmp37 - tmp11
        tmp39 = tl_math.log(tmp26)
        tmp40 = tmp38 - tmp39
        tmp41 = tl_math.exp(tmp40)
        tmp42 = tmp41 * tmp40
        tmp43 = tmp41 - tmp42
        tmp44 = tl.broadcast_to(tmp43, [XBLOCK, R0_BLOCK])
        tmp46 = _tmp45 + tmp44
        _tmp45 = tl.where(r0_mask, tmp46, _tmp45)
        tl.store(out_ptr2 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp40, r0_mask)
    tmp45 = tl.sum(_tmp45, 1)[:, None]
    tl.store(out_ptr3 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp45, None)




# kernel path: /tmp/torchinductor_sahanp/7n/c7nee6theefes6pr3hhmxasvmm3i3epbewbu2435iwvmqmkbbzoy.py
# Topologically Sorted Source Nodes: [dist_pos, add, dist_neg, sub, loss, triplet_loss, poisson_loss, exp, add_1], Original ATen: [aten.sub, aten.add, aten.norm, aten.clamp_min, aten.mean, aten.exp, aten.mul]
# Source node to ATen node mapping:
#   add => add_3
#   add_1 => add_4
#   dist_neg => add_2, pow_3, pow_4, sub_4, sum_3
#   dist_pos => add_1, pow_1, pow_2, sub_3, sum_2
#   exp => exp_1
#   loss => clamp_min
#   poisson_loss => exp_2, mean_1, mul_1, sub_6
#   sub => sub_5
#   triplet_loss => mean
# Graph fragment:
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select, %select_1), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%sub_3, 1e-06), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_1, 2.0), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1]), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_2, 0.5), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_2, 1.0), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select, %select_2), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%sub_4, 1e-06), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_2, 2.0), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_3, [1]), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_3, 0.5), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %pow_4), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_5, 0), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%clamp_min,), kwargs = {})
#   %exp_2 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%view_3,), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%view_3,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_1, %view_3), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%exp_2, %mul_1), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sub_6,), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, %mean_1), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_clamp_min_exp_mean_mul_norm_sub_2(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 125
    R0_BLOCK: tl.constexpr = 128
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
    tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (125 + r0_0), r0_mask, other=0.0)
    tmp10 = tl.load(in_ptr0 + (250 + r0_0), r0_mask, other=0.0)
    tmp26 = tl.load(in_ptr1 + (0))
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK, 1])
    tmp2 = tmp0 - tmp1
    tmp3 = 1e-06
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp8 = tl.where(r0_mask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp11 = tmp0 - tmp10
    tmp12 = tmp11 + tmp3
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.where(r0_mask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tmp18 = libdevice.sqrt(tmp9)
    tmp19 = 1.0
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.sqrt(tmp17)
    tmp22 = tmp20 - tmp21
    tmp23 = 0.0
    tmp24 = triton_helpers.maximum(tmp22, tmp23)
    tmp25 = tmp24 / tmp19
    tmp28 = 1250.0
    tmp29 = tmp27 / tmp28
    tmp30 = tmp25 + tmp29
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp30, None)







def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (1, 10, 5, 5, 5), (1250, 125, 25, 5, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 10, 1, 1, 1), (10, 1, 10, 10, 10), torch.float32)
        buf1 = empty_strided_cuda((1, 10, 1, 1, 1), (10, 1, 10, 10, 10), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten._native_batch_norm_legit]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_0[grid(10)](arg0_1, buf0, buf1, 10, 125, XBLOCK=1, num_warps=2, num_stages=1)
        buf5 = empty_strided_cuda((1, 1250), (1280, 1), torch.float32)
        buf8 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [x_2, poisson_loss, exp], Original ATen: [aten._log_softmax, aten.exp, aten.mul, aten.sub, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax_exp_mean_mul_sub_1[grid(1)](arg0_1, buf0, buf1, buf5, buf8, 1, 1250, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del arg0_1
        del buf0
        del buf1
        buf6 = empty_strided_cuda((1, ), (1, ), torch.float32)
        buf9 = reinterpret_tensor(buf6, (), (), 0); del buf6  # reuse
        # Topologically Sorted Source Nodes: [dist_pos, add, dist_neg, sub, loss, triplet_loss, poisson_loss, exp, add_1], Original ATen: [aten.sub, aten.add, aten.norm, aten.clamp_min, aten.mean, aten.exp, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clamp_min_exp_mean_mul_norm_sub_2[grid(1)](buf9, buf5, buf8, 1, 125, XBLOCK=1, num_warps=2, num_stages=1)
        del buf5
        del buf8
    return (buf9, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 10, 5, 5, 5), (1250, 125, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
