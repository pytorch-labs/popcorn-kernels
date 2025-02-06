# AOT ID: ['11_inference']
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


# kernel path: /tmp/torchinductor_sahanp/jg/cjg5zh72unfpryb4fcle72zczfzg6ypib52s2uigge5ho365zm3y.py
# Topologically Sorted Source Nodes: [target, margin_loss, input1, input2, hinge_loss, add, huber_loss, total_loss], Original ATen: [aten.randn_like, aten.neg, aten.sub, aten.mul, aten.add, aten.clamp_min, aten.mean, aten.ne, aten.fill, aten.zeros_like, aten.where, aten.huber_loss]
# Source node to ATen node mapping:
#   add => add_127
#   hinge_loss => add_123, clamp_min_1, full, full_1, mean_1, ne_44, ne_45, sub_37, where, where_1
#   huber_loss => abs_1, lt_65, mean_2, mul_186, mul_187, mul_188, sub_46, sub_47, where_2
#   input1 => inductor_lookup_seed_default_1, inductor_random_default_2
#   input2 => inductor_lookup_seed_default_2, inductor_random_default_1
#   margin_loss => add_66, clamp_min, mean, mul_71, neg, sub_24
#   target => inductor_lookup_seed_default_3, inductor_random_default
#   total_loss => add_128
# Graph fragment:
#   %inductor_lookup_seed_default_3 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 3), kwargs = {})
#   %inductor_random_default : [num_users=4] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 1, 10, %sym_size_int_2], %inductor_lookup_seed_default_3, randn), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%inductor_random_default,), kwargs = {})
#   %inductor_lookup_seed_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 1), kwargs = {})
#   %inductor_random_default_2 : [num_users=4] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 1, 10, %sym_size_int_2], %inductor_lookup_seed_default_1, randn), kwargs = {})
#   %inductor_lookup_seed_default_2 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 2), kwargs = {})
#   %inductor_random_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 1, 10, %sym_size_int_2], %inductor_lookup_seed_default_2, randn), kwargs = {})
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%inductor_random_default_2, %inductor_random_default_1), kwargs = {})
#   %mul_71 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg, %sub_24), kwargs = {})
#   %add_66 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_71, 0.0), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_66, 0), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%clamp_min,), kwargs = {})
#   %ne_44 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%inductor_random_default, 1), kwargs = {})
#   %full_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 1, 10, %sym_size_int_2], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sub_37 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_1, %inductor_random_default_2), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_37, 0), kwargs = {})
#   %full : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([1, 1, 10, %sym_size_int_2], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_44, %clamp_min_1, %full), kwargs = {})
#   %ne_45 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%inductor_random_default, -1), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_45, %inductor_random_default_2, %full), kwargs = {})
#   %add_123 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where, %where_1), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%add_123,), kwargs = {})
#   %add_127 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, %mean_1), kwargs = {})
#   %sub_46 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%inductor_random_default_2, %inductor_random_default), kwargs = {})
#   %abs_1 : [num_users=4] = call_function[target=torch.ops.aten.abs.default](args = (%sub_46,), kwargs = {})
#   %lt_65 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%abs_1, 1.0), kwargs = {})
#   %mul_186 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%abs_1, 0.5), kwargs = {})
#   %mul_187 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_186, %abs_1), kwargs = {})
#   %sub_47 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%abs_1, 0.5), kwargs = {})
#   %mul_188 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_47, 1.0), kwargs = {})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%lt_65, %mul_187, %mul_188), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%where_2,), kwargs = {})
#   %add_128 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_127, %mean_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_clamp_min_fill_huber_loss_mean_mul_ne_neg_randn_like_sub_where_zeros_like_0(in_out_ptr0, in_ptr0, load_seed_offset, load_seed_offset1, load_seed_offset2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 100
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
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = r0_0
    tmp2 = tl.randn(tmp0, (tmp1).to(tl.uint32))
    tmp3 = tl.load(in_ptr0 + load_seed_offset1)
    tmp4 = tl.randn(tmp3, (tmp1).to(tl.uint32))
    tmp5 = tl.load(in_ptr0 + load_seed_offset2)
    tmp6 = tl.randn(tmp5, (tmp1).to(tl.uint32))
    tmp7 = -tmp2
    tmp8 = tmp4 - tmp6
    tmp9 = tmp7 * tmp8
    tmp10 = 0.0
    tmp11 = tmp9 + tmp10
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(r0_mask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = 1.0
    tmp18 = tmp2 != tmp17
    tmp19 = tmp17 - tmp4
    tmp20 = triton_helpers.maximum(tmp19, tmp10)
    tmp21 = tl.where(tmp18, tmp20, tmp10)
    tmp22 = -1.0
    tmp23 = tmp2 != tmp22
    tmp24 = tl.where(tmp23, tmp4, tmp10)
    tmp25 = tmp21 + tmp24
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
    tmp28 = tl.where(r0_mask, tmp26, 0)
    tmp29 = tl.sum(tmp28, 1)[:, None]
    tmp30 = tmp4 - tmp2
    tmp31 = tl_math.abs(tmp30)
    tmp32 = tmp31 < tmp17
    tmp33 = 0.5
    tmp34 = tmp31 * tmp33
    tmp35 = tmp34 * tmp31
    tmp36 = tmp31 - tmp33
    tmp37 = tmp36 * tmp17
    tmp38 = tl.where(tmp32, tmp35, tmp37)
    tmp39 = tl.broadcast_to(tmp38, [XBLOCK, R0_BLOCK])
    tmp41 = tl.where(r0_mask, tmp39, 0)
    tmp42 = tl.sum(tmp41, 1)[:, None]
    tmp43 = 100.0
    tmp44 = tmp16 / tmp43
    tmp45 = tmp29 / tmp43
    tmp46 = tmp44 + tmp45
    tmp47 = tmp42 / tmp43
    tmp48 = tmp46 + tmp47
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp48, None)







def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    assert_size_stride(arg1_1, (1, 100), (100, 1))
    assert_size_stride(arg2_1, (10, 10), (10, 1))
    assert_size_stride(arg3_1, (10, 10), (10, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [4], out=buf0)
        buf5 = empty_strided_cuda((), (), torch.float32)
        buf8 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [target, margin_loss, input1, input2, hinge_loss, add, huber_loss, total_loss], Original ATen: [aten.randn_like, aten.neg, aten.sub, aten.mul, aten.add, aten.clamp_min, aten.mean, aten.ne, aten.fill, aten.zeros_like, aten.where, aten.huber_loss]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clamp_min_fill_huber_loss_mean_mul_ne_neg_randn_like_sub_where_zeros_like_0[grid(1)](buf8, buf0, 3, 1, 2, 1, 100, XBLOCK=1, num_warps=2, num_stages=1)
        del buf0
    return (buf8, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 100
    arg1_1 = rand_strided((1, 100), (100, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((10, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((10, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
