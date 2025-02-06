# AOT ID: ['2_forward']
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


# kernel path: /tmp/torchinductor_sahanp/mz/cmz2kc7n4mml3crx7cp4xbndyy7zpteolprkdv7jva23wslftybl.py
# Topologically Sorted Source Nodes: [x_1, x_2, x_4], Original ATen: [aten.constant_pad_nd, aten.bernoulli, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   x_1 => constant_pad_nd
#   x_2 => inductor_lookup_seed_default, inductor_random_default, lt
#   x_4 => add_5, add_6, add_7, mul_10, mul_11, mul_7, mul_8, mul_9, rsqrt, var_mean
# Graph fragment:
#   %constant_pad_nd : [num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%squeeze, [2, 2], 0.0), kwargs = {})
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 3, 1], %inductor_lookup_seed_default, rand), kwargs = {})
#   %lt : [num_users=2] = call_function[target=torch.ops.aten.lt.Scalar](args = (%inductor_random_default, 0.5), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%unsqueeze_2, [0, 2, 3, 4]), kwargs = {correction: 0, keepdim: True})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_5,), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_1, 0.1), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_4, 0.9), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %mul_8), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_3, 1.0149253731343284), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, 0.1), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_5, 0.9), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, %mul_11), kwargs = {})
#   %copy__1 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_4, %add_6), kwargs = {})
#   %copy__2 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_5, %add_7), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_bernoulli_constant_pad_nd_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, out_ptr3, out_ptr5, out_ptr7, out_ptr9, load_seed_offset, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 3
    r0_numel = 68
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.5
    tmp4 = tmp2 < tmp3
    tl.store(out_ptr1 + (x0), tmp4, xmask)
    tmp23_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp23_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp23_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp5 = (-2) + r0_1
        tmp6 = tl.full([1, 1], 0, tl.int64)
        tmp7 = tmp5 >= tmp6
        tmp8 = tl.full([1, 1], 64, tl.int64)
        tmp9 = tmp5 < tmp8
        tmp10 = tmp7 & tmp9
        tmp11 = tl.load(in_ptr1 + ((-4) + 2*r0_1 + 128*x0), r0_mask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr1 + ((-3) + 2*r0_1 + 128*x0), r0_mask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp12 + tmp11
        tmp14 = 0.5
        tmp15 = tmp13 * tmp14
        tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
        tmp17 = tl.where(tmp10, tmp15, tmp16)
        tmp18 = tmp4.to(tl.float32)
        tmp19 = 2.0
        tmp20 = tmp18 * tmp19
        tmp21 = tmp17 * tmp20
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, R0_BLOCK])
        tmp23_mean_next, tmp23_m2_next, tmp23_weight_next = triton_helpers.welford_reduce(
            tmp22, tmp23_mean, tmp23_m2, tmp23_weight, roffset == 0
        )
        tmp23_mean = tl.where(r0_mask & xmask, tmp23_mean_next, tmp23_mean)
        tmp23_m2 = tl.where(r0_mask & xmask, tmp23_m2_next, tmp23_m2)
        tmp23_weight = tl.where(r0_mask & xmask, tmp23_weight_next, tmp23_weight)
        tl.store(out_ptr2 + (r0_1 + 68*x0), tmp17, r0_mask & xmask)
    tmp26, tmp27, tmp28 = triton_helpers.welford(tmp23_mean, tmp23_m2, tmp23_weight, 1)
    tmp23 = tmp26[:, None]
    tmp24 = tmp27[:, None]
    tmp25 = tmp28[:, None]
    tl.store(out_ptr3 + (x0), tmp23, xmask)
    tmp38 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp29 = 68.0
    tmp30 = tmp24 / tmp29
    tmp31 = 1e-05
    tmp32 = tmp30 + tmp31
    tmp33 = libdevice.rsqrt(tmp32)
    tmp34 = 1.0149253731343284
    tmp35 = tmp30 * tmp34
    tmp36 = 0.1
    tmp37 = tmp35 * tmp36
    tmp39 = 0.9
    tmp40 = tmp38 * tmp39
    tmp41 = tmp37 + tmp40
    tmp42 = tmp23 * tmp36
    tmp44 = tmp43 * tmp39
    tmp45 = tmp42 + tmp44
    tl.store(out_ptr5 + (x0), tmp33, xmask)
    tl.store(out_ptr7 + (x0), tmp41, xmask)
    tl.store(out_ptr9 + (x0), tmp45, xmask)




# kernel path: /tmp/torchinductor_sahanp/oj/coj2wibkfr4pe7we53eu46ue7ii4wjqcslp2gx6jvrk5n4ijzkk6.py
# Topologically Sorted Source Nodes: [x_4, x_6, loss], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish, aten.soft_margin_loss_backward, aten.soft_margin_loss]
# Source node to ATen node mapping:
#   loss => exp, log1p, mean, neg
#   x_4 => add_8, mul_12, mul_6, sub_1
#   x_6 => add_9, clamp_max, clamp_min, div_1, mul_13
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze_2, %getitem_1), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %unsqueeze_5), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_12, %unsqueeze_8), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%squeeze_5, 3), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_9, 0), kwargs = {})
#   %clamp_max : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 6), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_5, %clamp_max), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_13, 6), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%div_1,), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %log1p : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp,), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%log1p,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_hardswish_soft_margin_loss_soft_margin_loss_backward_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 204
    R0_BLOCK: tl.constexpr = 256
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
    r0_2 = r0_index
    r0_1 = r0_index // 68
    tmp0 = tl.load(in_ptr0 + (r0_2), r0_mask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp6 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 2.0
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp5 - tmp6
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 3.0
    tmp15 = tmp13 + tmp14
    tmp16 = 0.0
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tmp20 = tmp13 * tmp19
    tmp21 = 0.16666666666666666
    tmp22 = tmp20 * tmp21
    tmp23 = -tmp22
    tmp24 = tl_math.exp(tmp23)
    tmp25 = libdevice.log1p(tmp24)
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
    tmp28 = tl.where(r0_mask, tmp26, 0)
    tmp29 = tl.sum(tmp28, 1)[:, None]
    tmp30 = 204.0
    tmp31 = tmp29 / tmp30
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp31, None)




# kernel path: /tmp/torchinductor_sahanp/ke/ckea67ksdr4pks2mcpodgvsybipmzvkeee2q7lfxwirdsmbe34x2.py
# Topologically Sorted Source Nodes: [add_], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   add_ => add_4
# Graph fragment:
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_3, 1), kwargs = {})
#   %copy_ : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_3, %add_4), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_2(in_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp1 + tmp2
    tl.store(out_ptr1 + (tl.full([XBLOCK], 0, tl.int32)), tmp3, None)







def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7 = args
    args.clear()
    s0 = primals_1
    assert_size_stride(primals_2, (1, 3, 128), (384, 128, 1))
    assert_size_stride(primals_3, (), ())
    assert_size_stride(primals_4, (3, ), (1, ))
    assert_size_stride(primals_5, (3, ), (1, ))
    assert_size_stride(primals_6, (3, ), (1, ))
    assert_size_stride(primals_7, (3, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((1, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf1)
        buf3 = empty_strided_cuda((1, 3, 1), (3, 1, 1), torch.bool)
        buf0 = empty_strided_cuda((1, 3, 68), (204, 68, 1), torch.float32)
        buf4 = empty_strided_cuda((1, 3, 1, 1, 1), (3, 1, 1, 1, 1), torch.float32)
        buf7 = empty_strided_cuda((1, 3, 1, 1, 1), (3, 1, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1, x_2, x_4], Original ATen: [aten.constant_pad_nd, aten.bernoulli, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_bernoulli_constant_pad_nd_0[grid(3)](buf1, primals_2, primals_5, primals_4, buf3, buf0, buf4, buf7, primals_5, primals_4, 0, 3, 68, XBLOCK=1, R0_BLOCK=128, num_warps=2, num_stages=1)
        del buf1
        del primals_2
        del primals_4
        del primals_5
        buf9 = empty_strided_cuda((), (), torch.float32)
        buf18 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [x_4, x_6, loss], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish, aten.soft_margin_loss_backward, aten.soft_margin_loss]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_hardswish_soft_margin_loss_soft_margin_loss_backward_1[grid(1)](buf18, buf0, buf3, buf4, buf7, primals_6, primals_7, 1, 204, XBLOCK=1, num_warps=2, num_stages=1)
        # Topologically Sorted Source Nodes: [add_], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_2[grid(1)](primals_3, primals_3, 1, XBLOCK=1, num_warps=1, num_stages=1)
        del primals_3
    return (buf18, primals_6, primals_7, buf0, buf3, buf4, buf7, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = 128
    primals_2 = rand_strided((1, 3, 128), (384, 128, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_4 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
