# AOT ID: ['13_inference']
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


# kernel path: /tmp/torchinductor_sahanp/2u/c2uio5agy6zf3faddgdmtmmahogsytx7pojobedoblrtu6wecsqf.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.bernoulli]
# Source node to ATen node mapping:
#   x_2 => inductor_lookup_seed_default, inductor_random_default_1
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 1, 1, 1, 1], %inductor_lookup_seed_default, rand), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_bernoulli_0(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp2, None)




# kernel path: /tmp/torchinductor_sahanp/ea/ceau4hpo774wiyt4zow74ajqy2xpawgg3lcc3awc6j2bfe4qcrs7.py
# Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.pow]
# Source node to ATen node mapping:
#   x_4 => pow_3
# Graph fragment:
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%squeeze, 2.0), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_pow_1(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks1)
    x2 = xindex // ks2
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 2*ks4*x1 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 2*x0 + 2*ks4*x1 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (ks4 + 2*x0 + 2*ks4*x1 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (1 + ks4 + 2*x0 + 2*ks4*x1 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr1 + (0))
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK])
    tmp1 = tmp0 * tmp0
    tmp3 = tmp2 * tmp2
    tmp4 = tmp3 + tmp1
    tmp6 = tmp5 * tmp5
    tmp7 = tmp6 + tmp4
    tmp9 = tmp8 * tmp8
    tmp10 = tmp9 + tmp7
    tmp11 = 0.25
    tmp12 = tmp10 * tmp11
    tmp13 = tl.full([1], 0, tl.int32)
    tmp14 = tmp13 < tmp12
    tmp15 = tmp14.to(tl.int8)
    tmp16 = tmp12 < tmp13
    tmp17 = tmp16.to(tl.int8)
    tmp18 = tmp15 - tmp17
    tmp19 = tmp18.to(tmp12.dtype)
    tmp20 = tl_math.abs(tmp12)
    tmp21 = triton_helpers.maximum(tmp13, tmp20)
    tmp22 = tmp19 * tmp21
    tmp23 = 4.0
    tmp24 = tmp22 * tmp23
    tmp25 = libdevice.sqrt(tmp24)
    tmp28 = 0.5
    tmp29 = tmp27 < tmp28
    tmp30 = tmp29.to(tl.float32)
    tmp31 = 2.0
    tmp32 = tmp30 * tmp31
    tmp33 = tmp25 * tmp32
    tmp34 = tmp33 * tmp33
    tl.store(out_ptr0 + (x3), tmp34, xmask)




# kernel path: /tmp/torchinductor_sahanp/fd/cfdzxj2xtvsm7ngspsl56gfyrlhnnvnesasq5qzxoofjyvmi2moc.py
# Topologically Sorted Source Nodes: [x_6, loss, target], Original ATen: [aten.bernoulli, aten.soft_margin_loss, aten.ones_like]
# Source node to ATen node mapping:
#   loss => exp, log1p, mean, mul_108, neg
#   target => full
#   x_6 => inductor_lookup_seed_default_1, inductor_random_default
# Graph fragment:
#   %inductor_lookup_seed_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 1), kwargs = {})
#   %inductor_random_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 1, 1, 1, 1], %inductor_lookup_seed_default_1, rand), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze_1,), kwargs = {})
#   %full : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 3, %sym_size_int_4, %sym_size_int_5], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_108 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg, %full), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_108,), kwargs = {})
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
def triton_red_fused_bernoulli_ones_like_soft_margin_loss_2(in_out_ptr0, in_ptr0, in_ptr1, load_seed_offset, ks1, ks2, ks3, ks4, ks5, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = tl.full([1, 1], 0, tl.int32)
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    _tmp36 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = (r0_index % ks1)
        r0_1 = ((r0_index // ks1) % ks2)
        r0_2 = r0_index // ks3
        tmp3 = tl.load(in_ptr1 + (2*r0_0 + 2*ks4*r0_1 + ks4*ks5*r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (1 + 2*r0_0 + 2*ks4*r0_1 + ks4*ks5*r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr1 + (ks4 + 2*r0_0 + 2*ks4*r0_1 + ks4*ks5*r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr1 + (1 + ks4 + 2*r0_0 + 2*ks4*r0_1 + ks4*ks5*r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp4 + tmp3
        tmp7 = tmp6 + tmp5
        tmp9 = tmp8 + tmp7
        tmp10 = 0.25
        tmp11 = tmp9 * tmp10
        tmp12 = tmp1 < tmp11
        tmp13 = tmp12.to(tl.int8)
        tmp14 = tmp11 < tmp1
        tmp15 = tmp14.to(tl.int8)
        tmp16 = tmp13 - tmp15
        tmp17 = tmp16.to(tmp11.dtype)
        tmp18 = tl_math.abs(tmp11)
        tmp19 = triton_helpers.maximum(tmp1, tmp18)
        tmp20 = tmp17 * tmp19
        tmp21 = 4.0
        tmp22 = tmp20 * tmp21
        tmp23 = libdevice.sqrt(tmp22)
        tmp24 = 0.5
        tmp25 = tmp2 < tmp24
        tmp26 = tmp25.to(tl.float32)
        tmp27 = 2.0
        tmp28 = tmp26 * tmp27
        tmp29 = tmp23 * tmp28
        tmp30 = -tmp29
        tmp31 = 1.0
        tmp32 = tmp30 * tmp31
        tmp33 = tl_math.exp(tmp32)
        tmp34 = libdevice.log1p(tmp33)
        tmp35 = tl.broadcast_to(tmp34, [XBLOCK, R0_BLOCK])
        tmp37 = _tmp36 + tmp35
        _tmp36 = tl.where(r0_mask, tmp37, _tmp36)
    tmp36 = tl.sum(_tmp36, 1)[:, None]
    tmp38 = 3*ks1*ks2
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tmp36 / tmp39
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp40, None)







def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    assert_size_stride(arg2_1, (1, 3, s0, s1), (3*s0*s1, s0*s1, s1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [2], out=buf0)
        buf1 = empty_strided_cuda((1, 1, 1, 1, 1), (1, 1, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.bernoulli]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bernoulli_0[grid(1)](buf0, buf1, 0, 1, XBLOCK=1, num_warps=1, num_stages=1)
        ps0 = s1 // 2
        ps1 = s0 // 2
        ps2 = (s0 // 2)*(s1 // 2)
        buf2 = empty_strided_cuda((1, 3, s0 // 2, s1 // 2), (3*(s0 // 2)*(s1 // 2), (s0 // 2)*(s1 // 2), s1 // 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.pow]
        triton_poi_fused_pow_1_xnumel = 3*(s0 // 2)*(s1 // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_pow_1[grid(triton_poi_fused_pow_1_xnumel)](arg2_1, buf1, buf2, 32, 32, 1024, 64, 64, 3072, XBLOCK=128, num_warps=4, num_stages=1)
        del arg2_1
        ps3 = s1 // 4
        ps4 = s0 // 4
        ps5 = (s0 // 4)*(s1 // 4)
        buf3 = buf1; del buf1  # reuse
        buf4 = reinterpret_tensor(buf3, (), (), 0); del buf3  # reuse
        buf5 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [x_6, loss, target], Original ATen: [aten.bernoulli, aten.soft_margin_loss, aten.ones_like]
        triton_red_fused_bernoulli_ones_like_soft_margin_loss_2_r0_numel = 3*(s0 // 4)*(s1 // 4)
        stream0 = get_raw_stream(0)
        triton_red_fused_bernoulli_ones_like_soft_margin_loss_2[grid(1)](buf5, buf0, buf2, 1, 16, 16, 256, 32, 32, 1, 768, XBLOCK=1, R0_BLOCK=1024, num_warps=8, num_stages=1)
        del buf0
        del buf2
    return (buf5, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 64
    arg1_1 = 64
    arg2_1 = rand_strided((1, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
