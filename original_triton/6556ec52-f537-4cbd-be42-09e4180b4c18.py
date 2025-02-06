# AOT ID: ['161_inference']
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


# kernel path: /tmp/torchinductor_sahanp/pa/cpa3plu5eldksbmo27n2qkxv7ooxinj7vqd7gghkzxmfjyembm7f.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, x_2, target, loss], Original ATen: [aten.hardswish, aten.bernoulli, aten._to_copy, aten.mul, aten.add, aten._log_softmax, aten.zeros_like, aten.mse_loss]
# Source node to ATen node mapping:
#   input_1 => add_4, clamp_max, clamp_min, div, mul_3
#   input_2 => add_21, add_38, add_63, convert_element_type, inductor_lookup_seed_default, inductor_random_default_2, lt_1, mul_26, mul_39, mul_43
#   input_3 => add_68, clamp_max_1, clamp_min_1, div_1, mul_56
#   input_4 => add_102, add_127, add_85, convert_element_type_1, inductor_lookup_seed_default_1, inductor_random_default_1, lt_4, mul_79, mul_92, mul_96
#   input_5 => add_132, clamp_max_2, clamp_min_2, div_2, mul_109
#   input_6 => add_149, add_166, add_191, convert_element_type_2, inductor_lookup_seed_default_2, inductor_random_default, lt_7, mul_132, mul_145, mul_149
#   loss => mean, pow_1, sub_51
#   target => full
#   x_2 => amax, exp, log, sub_47, sub_48, sum_1
# Graph fragment:
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_adaptive_avg_pool2d, 3), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_4, 0), kwargs = {})
#   %clamp_max : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 6), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_adaptive_avg_pool2d, %clamp_max), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_3, 6), kwargs = {})
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default_2 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, %arg0_1, 7, 7], %inductor_lookup_seed_default, rand), kwargs = {})
#   %lt_1 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%inductor_random_default_2, 0.5), kwargs = {})
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt_1, torch.float32), kwargs = {})
#   %mul_39 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%convert_element_type, 0.8864048946659319), kwargs = {})
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %mul_39), kwargs = {})
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%convert_element_type, -1), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%add_21, 1.558387861036063), kwargs = {})
#   %add_38 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_26, 0.7791939305180315), kwargs = {})
#   %add_63 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_43, %add_38), kwargs = {})
#   %add_68 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_63, 3), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_68, 0), kwargs = {})
#   %clamp_max_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_1, 6), kwargs = {})
#   %mul_56 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_63, %clamp_max_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_56, 6), kwargs = {})
#   %inductor_lookup_seed_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 1), kwargs = {})
#   %inductor_random_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, %arg0_1, 7, 7], %inductor_lookup_seed_default_1, rand), kwargs = {})
#   %lt_4 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%inductor_random_default_1, 0.5), kwargs = {})
#   %convert_element_type_1 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt_4, torch.float32), kwargs = {})
#   %mul_92 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%convert_element_type_1, 0.8864048946659319), kwargs = {})
#   %mul_96 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %mul_92), kwargs = {})
#   %add_85 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%convert_element_type_1, -1), kwargs = {})
#   %mul_79 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%add_85, 1.558387861036063), kwargs = {})
#   %add_102 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_79, 0.7791939305180315), kwargs = {})
#   %add_127 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_96, %add_102), kwargs = {})
#   %add_132 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_127, 3), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_132, 0), kwargs = {})
#   %clamp_max_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 6), kwargs = {})
#   %mul_109 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_127, %clamp_max_2), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_109, 6), kwargs = {})
#   %inductor_lookup_seed_default_2 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 2), kwargs = {})
#   %inductor_random_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, %arg0_1, 7, 7], %inductor_lookup_seed_default_2, rand), kwargs = {})
#   %lt_7 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%inductor_random_default, 0.5), kwargs = {})
#   %convert_element_type_2 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt_7, torch.float32), kwargs = {})
#   %mul_145 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%convert_element_type_2, 0.8864048946659319), kwargs = {})
#   %mul_149 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, %mul_145), kwargs = {})
#   %add_149 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%convert_element_type_2, -1), kwargs = {})
#   %mul_132 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%add_149, 1.558387861036063), kwargs = {})
#   %add_166 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_132, 0.7791939305180315), kwargs = {})
#   %add_191 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_149, %add_166), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_1, [1], True), kwargs = {})
#   %sub_47 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_1, %amax), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_47,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_1,), kwargs = {})
#   %sub_48 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_47, %log), kwargs = {})
#   %full : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, %sym_size_int_2], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sub_51 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_48, %full), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_51, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_1,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__log_softmax__to_copy_add_bernoulli_hardswish_mse_loss_mul_zeros_like_0(in_out_ptr0, in_out_ptr1, in_ptr0, load_seed_offset, load_seed_offset1, load_seed_offset2, ks3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp57 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp7 = tl.load(in_out_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp0 = tl.load(in_ptr0 + load_seed_offset)
        tmp1 = r0_0
        tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
        tmp3 = tl.load(in_ptr0 + load_seed_offset1)
        tmp4 = tl.rand(tmp3, (tmp1).to(tl.uint32))
        tmp5 = tl.load(in_ptr0 + load_seed_offset2)
        tmp6 = tl.rand(tmp5, (tmp1).to(tl.uint32))
        tmp8 = 3.0
        tmp9 = tmp7 + tmp8
        tmp10 = 0.0
        tmp11 = triton_helpers.maximum(tmp9, tmp10)
        tmp12 = 6.0
        tmp13 = triton_helpers.minimum(tmp11, tmp12)
        tmp14 = tmp7 * tmp13
        tmp15 = 0.16666666666666666
        tmp16 = tmp14 * tmp15
        tmp17 = 0.5
        tmp18 = tmp2 < tmp17
        tmp19 = tmp18.to(tl.float32)
        tmp20 = 0.8864048946659319
        tmp21 = tmp19 * tmp20
        tmp22 = tmp16 * tmp21
        tmp23 = -1.0
        tmp24 = tmp19 + tmp23
        tmp25 = 1.558387861036063
        tmp26 = tmp24 * tmp25
        tmp27 = 0.7791939305180315
        tmp28 = tmp26 + tmp27
        tmp29 = tmp22 + tmp28
        tmp30 = tmp29 + tmp8
        tmp31 = triton_helpers.maximum(tmp30, tmp10)
        tmp32 = triton_helpers.minimum(tmp31, tmp12)
        tmp33 = tmp29 * tmp32
        tmp34 = tmp33 * tmp15
        tmp35 = tmp6 < tmp17
        tmp36 = tmp35.to(tl.float32)
        tmp37 = tmp36 * tmp20
        tmp38 = tmp34 * tmp37
        tmp39 = tmp36 + tmp23
        tmp40 = tmp39 * tmp25
        tmp41 = tmp40 + tmp27
        tmp42 = tmp38 + tmp41
        tmp43 = tmp42 + tmp8
        tmp44 = triton_helpers.maximum(tmp43, tmp10)
        tmp45 = triton_helpers.minimum(tmp44, tmp12)
        tmp46 = tmp42 * tmp45
        tmp47 = tmp46 * tmp15
        tmp48 = tmp4 < tmp17
        tmp49 = tmp48.to(tl.float32)
        tmp50 = tmp49 * tmp20
        tmp51 = tmp47 * tmp50
        tmp52 = tmp49 + tmp23
        tmp53 = tmp52 * tmp25
        tmp54 = tmp53 + tmp27
        tmp55 = tmp51 + tmp54
        tmp56 = tl.broadcast_to(tmp55, [XBLOCK, R0_BLOCK])
        tmp58 = triton_helpers.maximum(_tmp57, tmp56)
        _tmp57 = tl.where(r0_mask, tmp58, _tmp57)
        tl.store(in_out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp55, r0_mask)
    tmp57 = triton_helpers.max2(_tmp57, 1)[:, None]
    _tmp63 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp59 = tl.load(in_out_ptr0 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp60 = tmp59 - tmp57
        tmp61 = tl_math.exp(tmp60)
        tmp62 = tl.broadcast_to(tmp61, [XBLOCK, R0_BLOCK])
        tmp64 = _tmp63 + tmp62
        _tmp63 = tl.where(r0_mask, tmp64, _tmp63)
    tmp63 = tl.sum(_tmp63, 1)[:, None]
    _tmp73 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp65 = tl.load(in_out_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp66 = tmp65 - tmp57
        tmp67 = tl_math.log(tmp63)
        tmp68 = tmp66 - tmp67
        tmp69 = 0.0
        tmp70 = tmp68 - tmp69
        tmp71 = tmp70 * tmp70
        tmp72 = tl.broadcast_to(tmp71, [XBLOCK, R0_BLOCK])
        tmp74 = _tmp73 + tmp72
        _tmp73 = tl.where(r0_mask, tmp74, _tmp73)
    tmp73 = tl.sum(_tmp73, 1)[:, None]
    tmp75 = 49*ks3
    tmp76 = tmp75.to(tl.float32)
    tmp77 = tmp73 / tmp76
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp77, None)







def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, 64, 64), (4096*s0, 4096, 64, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten._adaptive_avg_pool2d]
        buf0 = torch.ops.aten._adaptive_avg_pool2d.default(arg3_1, [7, 7])
        del arg3_1
        buf1 = buf0
        del buf0
        buf2 = empty_strided_cuda((3, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [3], out=buf2)
        buf5 = buf1; del buf1  # reuse
        buf7 = buf5; del buf5  # reuse
        buf8 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        buf10 = reinterpret_tensor(buf8, (), (), 0); del buf8  # reuse
        buf11 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, x_2, target, loss], Original ATen: [aten.hardswish, aten.bernoulli, aten._to_copy, aten.mul, aten.add, aten._log_softmax, aten.zeros_like, aten.mse_loss]
        triton_red_fused__log_softmax__to_copy_add_bernoulli_hardswish_mse_loss_mul_zeros_like_0_r0_numel = 49*s0
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax__to_copy_add_bernoulli_hardswish_mse_loss_mul_zeros_like_0[grid(1)](buf7, buf11, buf2, 0, 2, 1, 3, 1, 147, XBLOCK=1, R0_BLOCK=256, num_warps=2, num_stages=1)
        del buf2
        del buf7
    return (buf11, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 3
    arg1_1 = 64
    arg2_1 = 64
    arg3_1 = rand_strided((1, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
