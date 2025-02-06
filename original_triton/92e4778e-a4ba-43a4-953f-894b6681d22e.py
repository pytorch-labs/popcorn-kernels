# AOT ID: ['68_inference']
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


# kernel path: /tmp/torchinductor_sahanp/2p/c2psb2mivlvxgicgdwsled3ndqjxdrzvsfxkzk65jdd2zzivmgon.py
# Topologically Sorted Source Nodes: [x, target, loss, x_1, x_2, x_3], Original ATen: [aten.constant_pad_nd, aten.zeros_like, aten.binary_cross_entropy_with_logits, aten._unsafe_index, aten.softplus, aten.sigmoid]
# Source node to ATen node mapping:
#   loss => abs_1, exp_1, full_default_2, log1p_1, mean, minimum, mul_24, neg, sub_11, sub_12, sub_13
#   target => full
#   x => constant_pad_nd
#   x_1 => _unsafe_index
#   x_2 => div, exp, gt, log1p, mul_14, where
#   x_3 => sigmoid
# Graph fragment:
#   %constant_pad_nd : [num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%arg1_1, [2, 2], 0.0), kwargs = {})
#   %full : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 1, %floordiv], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %full), kwargs = {})
#   %_unsafe_index : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%constant_pad_nd, [None, None, %convert_element_type_1]), kwargs = {})
#   %mul_14 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index, 1.0), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%mul_14, 20.0), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_14,), kwargs = {})
#   %log1p : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%log1p, 1.0), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %_unsafe_index, %div), kwargs = {})
#   %sigmoid : [num_users=3] = call_function[target=torch.ops.aten.sigmoid.default](args = (%where,), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %sigmoid), kwargs = {})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %minimum : [num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%full_default_2, %sigmoid), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sigmoid,), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%abs_1,), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %log1p_1 : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_1,), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%minimum, %log1p_1), kwargs = {})
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_24, %sub_12), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sub_13,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__unsafe_index_binary_cross_entropy_with_logits_constant_pad_nd_sigmoid_softplus_zeros_like_0(in_out_ptr0, in_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp39 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp0 = 4.0
        tmp1 = ks0
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tmp0 + tmp2
        tmp4 = tmp3.to(tl.float64)
        tmp5 = tl.full([1, 1], 2.0, tl.float64)
        tmp6 = tmp5 * tmp4
        tmp7 = tmp4 / tmp6
        tmp8 = tmp7.to(tl.float32)
        tmp9 = r0_0
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tmp10 * tmp8
        tmp12 = tmp11.to(tl.int64)
        tmp13 = (-2) + tmp12
        tmp14 = tmp13.to(tl.int32)
        tmp15 = tl.full([1, 1], 0, tl.int64)
        tmp16 = tmp14 >= tmp15
        tmp17 = tmp14 < tmp1
        tmp18 = tmp16 & tmp17
        tmp19 = tl.load(in_ptr0 + (tl.broadcast_to((-2) + tmp12, [XBLOCK, R0_BLOCK])), r0_mask & tmp18, eviction_policy='evict_last', other=0.0)
        tmp20 = 1.0
        tmp21 = tmp19 * tmp20
        tmp22 = 20.0
        tmp23 = tmp21 > tmp22
        tmp24 = tl_math.exp(tmp21)
        tmp25 = libdevice.log1p(tmp24)
        tmp26 = tmp25 * tmp20
        tmp27 = tl.where(tmp23, tmp19, tmp26)
        tmp28 = tl.sigmoid(tmp27)
        tmp29 = tmp20 * tmp28
        tmp30 = 0.0
        tmp31 = triton_helpers.minimum(tmp30, tmp28)
        tmp32 = tl_math.abs(tmp28)
        tmp33 = -tmp32
        tmp34 = tl_math.exp(tmp33)
        tmp35 = libdevice.log1p(tmp34)
        tmp36 = tmp31 - tmp35
        tmp37 = tmp29 - tmp36
        tmp38 = tl.broadcast_to(tmp37, [XBLOCK, R0_BLOCK])
        tmp40 = _tmp39 + tmp38
        _tmp39 = tl.where(r0_mask, tmp40, _tmp39)
    tmp39 = tl.sum(_tmp39, 1)[:, None]
    tmp41 = 8 + 2*ks0
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tmp39 / tmp42
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp43, None)







def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    s0 = arg0_1
    assert_size_stride(arg1_1, (1, 1, s0), (s0, s0, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((), (), torch.float32)
        buf2 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [x, target, loss, x_1, x_2, x_3], Original ATen: [aten.constant_pad_nd, aten.zeros_like, aten.binary_cross_entropy_with_logits, aten._unsafe_index, aten.softplus, aten.sigmoid]
        triton_red_fused__unsafe_index_binary_cross_entropy_with_logits_constant_pad_nd_sigmoid_softplus_zeros_like_0_r0_numel = 8 + 2*s0
        stream0 = get_raw_stream(0)
        triton_red_fused__unsafe_index_binary_cross_entropy_with_logits_constant_pad_nd_sigmoid_softplus_zeros_like_0[grid(1)](buf2, arg1_1, 32, 1, 72, XBLOCK=1, R0_BLOCK=128, num_warps=2, num_stages=1)
        del arg1_1
    return (buf2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 32
    arg1_1 = rand_strided((1, 1, 32), (32, 32, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
