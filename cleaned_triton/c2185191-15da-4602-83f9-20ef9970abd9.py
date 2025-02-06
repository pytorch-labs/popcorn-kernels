# AOT ID: ['103_inference']
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


# kernel path: /tmp/torchinductor_sahanp/uk/cukr3hrisjtucu3uou6hllmcgd2ianjdfler6o4yfp22cq5f37rk.py
# Topologically Sorted Source Nodes: [target, hinge_loss, bce_loss, sigmoid, add], Original ATen: [aten.ones_like, aten.ne, aten.fill, aten.sub, aten.clamp_min, aten.zeros_like, aten.where, aten.add, aten.mean, aten.binary_cross_entropy, aten.sigmoid]
# Source node to ATen node mapping:
#   add => add_47
#   bce_loss => full_default, full_default_1, log, log1p, maximum, maximum_1, mean_1, mul_24, mul_25, neg, sub_24, sub_25
#   hinge_loss => add_42, clamp_min, full_1, full_2, mean, ne_10, ne_11, sub_14, where, where_1
#   sigmoid => sigmoid
#   target => full
# Graph fragment:
#   %full : [num_users=4] = call_function[target=torch.ops.aten.full.default](args = ([%sym_numel_default], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %ne_10 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%full, 1), kwargs = {})
#   %full_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([%sym_numel_default], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_2, %view), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_14, 0), kwargs = {})
#   %full_1 : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([%sym_numel_default], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_10, %clamp_min, %full_1), kwargs = {})
#   %ne_11 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%full, -1), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_11, %view, %full_1), kwargs = {})
#   %add_42 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where, %where_1), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%add_42,), kwargs = {})
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full, 1), kwargs = {})
#   %sigmoid : [num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view,), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%sigmoid,), kwargs = {})
#   %log1p : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%neg,), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -100), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %maximum : [num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%log1p, %full_default), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %maximum), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sigmoid,), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -100), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %maximum_1 : [num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%log, %full_default_1), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full, %maximum_1), kwargs = {})
#   %sub_25 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_24, %mul_25), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sub_25,), kwargs = {})
#   %add_47 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, %mean_1), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_binary_cross_entropy_clamp_min_fill_mean_ne_ones_like_sigmoid_sub_where_zeros_like_0(in_out_ptr0, in_ptr0, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp39 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp52 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp0 = 1.0
        tmp1 = tmp0 != tmp0
        tmp2 = (-1) + (((r0_0 // (6 + ks1)) % (6 + ks0)))
        tmp3 = tl.full([1, 1], 0, tl.int64)
        tmp4 = tmp2 >= tmp3
        tmp5 = 4 + ks0
        tmp6 = tmp2 < tmp5
        tmp7 = (-1) + ((r0_0 % (6 + ks1)))
        tmp8 = tmp7 >= tmp3
        tmp9 = 4 + ks1
        tmp10 = tmp7 < tmp9
        tmp11 = tmp4 & tmp6
        tmp12 = tmp11 & tmp8
        tmp13 = tmp12 & tmp10
        tmp14 = tl.broadcast_to((-3) + (((r0_0 // (6 + ks1)) % (6 + ks0))), [XBLOCK, R0_BLOCK])
        tmp15 = tl.full([1, 1], 0, tl.int64)
        tmp16 = tmp14 >= tmp15
        tmp17 = tl.broadcast_to(ks0, [XBLOCK, R0_BLOCK])
        tmp18 = tmp14 < tmp17
        tmp19 = tl.broadcast_to((-3) + ((r0_0 % (6 + ks1))), [XBLOCK, R0_BLOCK])
        tmp20 = tmp19 >= tmp15
        tmp21 = tl.broadcast_to(ks1, [XBLOCK, R0_BLOCK])
        tmp22 = tmp19 < tmp21
        tmp23 = tmp16 & tmp18
        tmp24 = tmp23 & tmp20
        tmp25 = tmp24 & tmp22
        tmp26 = tmp25 & tmp13
        tmp27 = tl.load(in_ptr0 + (tl.broadcast_to((-3) + ((-3)*ks1) + ks1*(((r0_0 // (6 + ks1)) % (6 + ks0))) + ks0*ks1*(r0_0 // (36 + 6*ks0 + 6*ks1 + ks0*ks1)) + ((r0_0 % (6 + ks1))), [XBLOCK, R0_BLOCK])), r0_mask & tmp26, eviction_policy='evict_last', other=0.0)
        tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
        tmp29 = tl.where(tmp13, tmp27, tmp28)
        tmp30 = tmp0 - tmp29
        tmp31 = 0.0
        tmp32 = triton_helpers.maximum(tmp30, tmp31)
        tmp33 = tl.where(tmp1, tmp32, tmp31)
        tmp34 = -1.0
        tmp35 = tmp0 != tmp34
        tmp36 = tl.where(tmp35, tmp29, tmp31)
        tmp37 = tmp33 + tmp36
        tmp38 = tl.broadcast_to(tmp37, [XBLOCK, R0_BLOCK])
        tmp40 = _tmp39 + tmp38
        _tmp39 = tl.where(r0_mask, tmp40, _tmp39)
        tmp41 = tl.sigmoid(tmp29)
        tmp42 = -tmp41
        tmp43 = libdevice.log1p(tmp42)
        tmp44 = -100.0
        tmp45 = triton_helpers.maximum(tmp43, tmp44)
        tmp46 = tmp31 * tmp45
        tmp47 = tl_math.log(tmp41)
        tmp48 = triton_helpers.maximum(tmp47, tmp44)
        tmp49 = tmp0 * tmp48
        tmp50 = tmp46 - tmp49
        tmp51 = tl.broadcast_to(tmp50, [XBLOCK, R0_BLOCK])
        tmp53 = _tmp52 + tmp51
        _tmp52 = tl.where(r0_mask, tmp53, _tmp52)
    tmp39 = tl.sum(_tmp39, 1)[:, None]
    tmp52 = tl.sum(_tmp52, 1)[:, None]
    tmp54 = 36*ks2 + 6*ks0*ks2 + 6*ks1*ks2 + ks0*ks1*ks2
    tmp55 = tmp54.to(tl.float32)
    tmp56 = tmp39 / tmp55
    tmp57 = tmp52 / tmp55
    tmp58 = tmp56 + tmp57
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp58, None)







def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((), (), torch.float32)
        buf2 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [target, hinge_loss, bce_loss, sigmoid, add], Original ATen: [aten.ones_like, aten.ne, aten.fill, aten.sub, aten.clamp_min, aten.zeros_like, aten.where, aten.add, aten.mean, aten.binary_cross_entropy, aten.sigmoid]
        triton_red_fused_add_binary_cross_entropy_clamp_min_fill_mean_ne_ones_like_sigmoid_sub_where_zeros_like_0_r0_numel = 36*s0 + 6*s0*s1 + 6*s0*s2 + s0*s1*s2
        stream0 = get_raw_stream(0)
        triton_red_fused_add_binary_cross_entropy_clamp_min_fill_mean_ne_ones_like_sigmoid_sub_where_zeros_like_0[grid(1)](buf2, arg3_1, 32, 32, 3, 1, 4332, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del arg3_1
    return (buf2, )


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
