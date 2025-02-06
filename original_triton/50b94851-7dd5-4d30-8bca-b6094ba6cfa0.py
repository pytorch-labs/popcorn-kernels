# AOT ID: ['102_inference']
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


# kernel path: /tmp/torchinductor_sahanp/5r/c5rbjah6y5ob2jexek2eua5vfetiyuaxw6zwnumtvbmm7cku2ceb.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.bernoulli]
# Source node to ATen node mapping:
#   x_1 => inductor_lookup_seed_default, inductor_random_default_1
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, %arg0_1, 1, 1], %inductor_lookup_seed_default, rand), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_bernoulli_0(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (x0), tmp2, xmask)




# kernel path: /tmp/torchinductor_sahanp/ek/cekwuuyefql4u7kvs5p23jsljmjqhmgdoza2psjzngbeead4retx.py
# Topologically Sorted Source Nodes: [randn_like, target, loss, x_3], Original ATen: [aten.randn_like, aten._softmax, aten.xlogy, aten._log_softmax, aten.mul, aten.sub, aten.sum, aten.div]
# Source node to ATen node mapping:
#   loss => div_1, eq_37, full_default, full_default_1, isnan, log_1, mul_51, mul_54, sub_32, sum_3, where, where_1
#   randn_like => inductor_lookup_seed_default_1, inductor_random_default
#   target => amax_1, div, exp_1, sub_28, sum_2
#   x_3 => amax, exp, log, sub_24, sub_25, sum_1
# Graph fragment:
#   %inductor_lookup_seed_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 1), kwargs = {})
#   %inductor_random_default : [num_users=2] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, %sym_size_int_4], %inductor_lookup_seed_default_1, randn), kwargs = {})
#   %amax_1 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%inductor_random_default, [1], True), kwargs = {})
#   %sub_28 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%inductor_random_default, %amax_1), kwargs = {})
#   %exp_1 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_28,), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_1, [1], True), kwargs = {})
#   %div : [num_users=5] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_1, %sum_2), kwargs = {})
#   %isnan : [num_users=1] = call_function[target=torch.ops.aten.isnan.default](args = (%div,), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], nan), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %eq_37 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%div, 0), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %log_1 : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%div,), kwargs = {})
#   %mul_54 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %log_1), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_37, %full_default, %mul_54), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%isnan, %full_default_1, %where), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_1, [1], True), kwargs = {})
#   %sub_24 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_1, %amax), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_24,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_1,), kwargs = {})
#   %sub_25 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_24, %log), kwargs = {})
#   %mul_51 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %sub_25), kwargs = {})
#   %sub_32 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_1, %mul_51), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%sub_32,), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_3, 1), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__log_softmax__softmax_div_mul_randn_like_sub_sum_xlogy_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr3, out_ptr4, ks0, ks1, load_seed_offset, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp28 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp13 = tl.load(in_ptr1 + (r0_0 // (16 + 4*ks0 + 4*ks1 + ks0*ks1)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp0 = (-2) + (((r0_0 // (4 + ks1)) % (4 + ks0)))
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = ks0
        tmp4 = tmp0 < tmp3
        tmp5 = (-2) + ((r0_0 % (4 + ks1)))
        tmp6 = tmp5 >= tmp1
        tmp7 = ks1
        tmp8 = tmp5 < tmp7
        tmp9 = tmp2 & tmp4
        tmp10 = tmp9 & tmp6
        tmp11 = tmp10 & tmp8
        tmp12 = tl.load(in_ptr0 + (tl.broadcast_to((-2) + ((-2)*ks1) + ks1*(((r0_0 // (4 + ks1)) % (4 + ks0))) + ks0*ks1*(r0_0 // (16 + 4*ks0 + 4*ks1 + ks0*ks1)) + ((r0_0 % (4 + ks1))), [XBLOCK, R0_BLOCK])), r0_mask & tmp11, eviction_policy='evict_last', other=0.0)
        tmp14 = 0.5
        tmp15 = tmp13 < tmp14
        tmp16 = tmp15.to(tl.float32)
        tmp17 = 0.8864048946659319
        tmp18 = tmp16 * tmp17
        tmp19 = tmp12 * tmp18
        tmp20 = -1.0
        tmp21 = tmp16 + tmp20
        tmp22 = 1.558387861036063
        tmp23 = tmp21 * tmp22
        tmp24 = 0.7791939305180315
        tmp25 = tmp23 + tmp24
        tmp26 = tmp19 + tmp25
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, R0_BLOCK])
        tmp29 = triton_helpers.maximum(_tmp28, tmp27)
        _tmp28 = tl.where(r0_mask, tmp29, _tmp28)
    tmp28 = triton_helpers.max2(_tmp28, 1)[:, None]
    _tmp60 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp43 = tl.load(in_ptr1 + (r0_0 // (16 + 4*ks0 + 4*ks1 + ks0*ks1)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp30 = (-2) + (((r0_0 // (4 + ks1)) % (4 + ks0)))
        tmp31 = tl.full([1, 1], 0, tl.int64)
        tmp32 = tmp30 >= tmp31
        tmp33 = ks0
        tmp34 = tmp30 < tmp33
        tmp35 = (-2) + ((r0_0 % (4 + ks1)))
        tmp36 = tmp35 >= tmp31
        tmp37 = ks1
        tmp38 = tmp35 < tmp37
        tmp39 = tmp32 & tmp34
        tmp40 = tmp39 & tmp36
        tmp41 = tmp40 & tmp38
        tmp42 = tl.load(in_ptr0 + (tl.broadcast_to((-2) + ((-2)*ks1) + ks1*(((r0_0 // (4 + ks1)) % (4 + ks0))) + ks0*ks1*(r0_0 // (16 + 4*ks0 + 4*ks1 + ks0*ks1)) + ((r0_0 % (4 + ks1))), [XBLOCK, R0_BLOCK])), r0_mask & tmp41, eviction_policy='evict_last', other=0.0)
        tmp44 = 0.5
        tmp45 = tmp43 < tmp44
        tmp46 = tmp45.to(tl.float32)
        tmp47 = 0.8864048946659319
        tmp48 = tmp46 * tmp47
        tmp49 = tmp42 * tmp48
        tmp50 = -1.0
        tmp51 = tmp46 + tmp50
        tmp52 = 1.558387861036063
        tmp53 = tmp51 * tmp52
        tmp54 = 0.7791939305180315
        tmp55 = tmp53 + tmp54
        tmp56 = tmp49 + tmp55
        tmp57 = tmp56 - tmp28
        tmp58 = tl_math.exp(tmp57)
        tmp59 = tl.broadcast_to(tmp58, [XBLOCK, R0_BLOCK])
        tmp61 = _tmp60 + tmp59
        _tmp60 = tl.where(r0_mask, tmp61, _tmp60)
    tmp60 = tl.sum(_tmp60, 1)[:, None]
    _tmp96 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp75 = tl.load(in_ptr1 + (r0_0 // (16 + 4*ks0 + 4*ks1 + ks0*ks1)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp62 = (-2) + (((r0_0 // (4 + ks1)) % (4 + ks0)))
        tmp63 = tl.full([1, 1], 0, tl.int64)
        tmp64 = tmp62 >= tmp63
        tmp65 = ks0
        tmp66 = tmp62 < tmp65
        tmp67 = (-2) + ((r0_0 % (4 + ks1)))
        tmp68 = tmp67 >= tmp63
        tmp69 = ks1
        tmp70 = tmp67 < tmp69
        tmp71 = tmp64 & tmp66
        tmp72 = tmp71 & tmp68
        tmp73 = tmp72 & tmp70
        tmp74 = tl.load(in_ptr0 + (tl.broadcast_to((-2) + ((-2)*ks1) + ks1*(((r0_0 // (4 + ks1)) % (4 + ks0))) + ks0*ks1*(r0_0 // (16 + 4*ks0 + 4*ks1 + ks0*ks1)) + ((r0_0 % (4 + ks1))), [XBLOCK, R0_BLOCK])), r0_mask & tmp73, eviction_policy='evict_last', other=0.0)
        tmp76 = 0.5
        tmp77 = tmp75 < tmp76
        tmp78 = tmp77.to(tl.float32)
        tmp79 = 0.8864048946659319
        tmp80 = tmp78 * tmp79
        tmp81 = tmp74 * tmp80
        tmp82 = -1.0
        tmp83 = tmp78 + tmp82
        tmp84 = 1.558387861036063
        tmp85 = tmp83 * tmp84
        tmp86 = 0.7791939305180315
        tmp87 = tmp85 + tmp86
        tmp88 = tmp81 + tmp87
        tmp89 = tmp88 - tmp28
        tmp90 = tl_math.log(tmp60)
        tmp91 = tmp89 - tmp90
        tmp92 = tl.load(in_ptr2 + load_seed_offset)
        tmp93 = r0_0
        tmp94 = tl.randn(tmp92, (tmp93).to(tl.uint32))
        tmp95 = tl.broadcast_to(tmp94, [XBLOCK, R0_BLOCK])
        tmp97 = triton_helpers.maximum(_tmp96, tmp95)
        _tmp96 = tl.where(r0_mask, tmp97, _tmp96)
        tl.store(out_ptr3 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp91, r0_mask)
        tl.store(out_ptr4 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp94, r0_mask)
    tmp96 = triton_helpers.max2(_tmp96, 1)[:, None]
    _tmp102 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp98 = tl.load(out_ptr4 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp99 = tmp98 - tmp96
        tmp100 = tl_math.exp(tmp99)
        tmp101 = tl.broadcast_to(tmp100, [XBLOCK, R0_BLOCK])
        tmp103 = _tmp102 + tmp101
        _tmp102 = tl.where(r0_mask, tmp103, _tmp102)
    tmp102 = tl.sum(_tmp102, 1)[:, None]
    _tmp120 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp104 = tl.load(out_ptr4 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp116 = tl.load(out_ptr3 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp105 = tmp104 - tmp96
        tmp106 = tl_math.exp(tmp105)
        tmp107 = tmp106 / tmp102
        tmp108 = libdevice.isnan(tmp107).to(tl.int1)
        tmp109 = 0.0
        tmp110 = tmp107 == tmp109
        tmp111 = tl_math.log(tmp107)
        tmp112 = tmp107 * tmp111
        tmp113 = tl.where(tmp110, tmp109, tmp112)
        tmp114 = float("nan")
        tmp115 = tl.where(tmp108, tmp114, tmp113)
        tmp117 = tmp107 * tmp116
        tmp118 = tmp115 - tmp117
        tmp119 = tl.broadcast_to(tmp118, [XBLOCK, R0_BLOCK])
        tmp121 = _tmp120 + tmp119
        _tmp120 = tl.where(r0_mask, tmp121, _tmp120)
    tmp120 = tl.sum(_tmp120, 1)[:, None]
    tmp122 = 1.0
    tmp123 = tmp120 * tmp122
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp123, None)







def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [2], out=buf0)
        buf1 = empty_strided_cuda((1, s0, 1, 1), (s0, 1, s0, s0), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.bernoulli]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bernoulli_0[grid(s0)](buf0, buf1, 0, 3, XBLOCK=4, num_warps=1, num_stages=1)
        buf8 = empty_strided_cuda((1, 16*s0 + 4*s0*s1 + 4*s0*s2 + s0*s1*s2), (16*s0 + 4*s0*s1 + 4*s0*s2 + s0*s1*s2, 1), torch.float32)
        buf2 = empty_strided_cuda((1, 16*s0 + 4*s0*s1 + 4*s0*s2 + s0*s1*s2), (16*s0 + 4*s0*s1 + 4*s0*s2 + s0*s1*s2, 1), torch.float32)
        buf3 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        buf9 = reinterpret_tensor(buf3, (), (), 0); del buf3  # reuse
        buf10 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [randn_like, target, loss, x_3], Original ATen: [aten.randn_like, aten._softmax, aten.xlogy, aten._log_softmax, aten.mul, aten.sub, aten.sum, aten.div]
        triton_red_fused__log_softmax__softmax_div_mul_randn_like_sub_sum_xlogy_1_r0_numel = 16*s0 + 4*s0*s1 + 4*s0*s2 + s0*s1*s2
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax__softmax_div_mul_randn_like_sub_sum_xlogy_1[grid(1)](buf10, arg3_1, buf1, buf0, buf8, buf2, 32, 32, 1, 1, 3888, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del arg3_1
        del buf0
        del buf1
        del buf2
        del buf8
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
