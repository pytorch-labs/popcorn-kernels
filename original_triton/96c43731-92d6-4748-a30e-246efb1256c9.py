# AOT ID: ['83_inference']
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


#include "/tmp/torchinductor_sahanp/3b/c3bi5gk6mslf6u4iaqafhxm64z6u65e3eain4xlary5blqnvv6xx.h"
extern "C"  void kernel(int64_t* in_out_ptr0,
                       const int64_t ks0,
                       const int64_t ks1)
{
    {
        {
            {
                auto tmp0 = in_out_ptr0[static_cast<int64_t>(0L)];
                auto tmp1 = static_cast<int32_t>(0);
                auto tmp2 = static_cast<int64_t>(0);
                auto tmp3 = ks0*(c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(2L)));
                auto tmp4 = c10::convert<int64_t>(tmp3);
                auto tmp5 = randint64_cpu(tmp0, tmp1, tmp2, tmp4);
                in_out_ptr0[static_cast<int64_t>(0L)] = tmp5;
            }
        }
    }
}
''')


# kernel path: /tmp/torchinductor_sahanp/ap/caperiw3pm7lhuaaxaktkgb5uinftda5teujyopsbd5jyzqp33jl.py
# Topologically Sorted Source Nodes: [loss, x_4], Original ATen: [aten.nll_loss_forward, aten._log_softmax]
# Source node to ATen node mapping:
#   loss => convert_element_type_1, div_1, full_default_1, ne_1, ne_2, neg, sum_2, sum_3, where_1
#   x_4 => amax, exp, sub_11, sum_1
# Graph fragment:
#   %ne_1 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%device_put, -100), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view, [1], True), kwargs = {})
#   %sub_11 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %amax), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_11,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze_1,), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_1, %neg, %full_default_1), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_1,), kwargs = {})
#   %ne_2 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%device_put, -100), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne_2,), kwargs = {})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_2, torch.float32), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_3, %convert_element_type_1), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__log_softmax_nll_loss_forward_1(in_out_ptr0, in_ptr0, in_ptr1, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp21 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (2*((r0_0 % (ks0 // 2))) + ks0*(triton_helpers.div_floor_integer(r0_0,  ks0 // 2))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr0 + (1 + 2*((r0_0 % (ks0 // 2))) + ks0*(triton_helpers.div_floor_integer(r0_0,  ks0 // 2))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp1 + tmp0
        tmp3 = 0.5
        tmp4 = tmp2 * tmp3
        tmp5 = tmp4 * tmp3
        tmp6 = 0.7071067811865476
        tmp7 = tmp4 * tmp6
        tmp8 = libdevice.erf(tmp7)
        tmp9 = 1.0
        tmp10 = tmp8 + tmp9
        tmp11 = tmp5 * tmp10
        tmp12 = 3.0
        tmp13 = tmp11 + tmp12
        tmp14 = 0.0
        tmp15 = triton_helpers.maximum(tmp13, tmp14)
        tmp16 = 6.0
        tmp17 = triton_helpers.minimum(tmp15, tmp16)
        tmp18 = 0.16666666666666666
        tmp19 = tmp17 * tmp18
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
        tmp22 = triton_helpers.maximum(_tmp21, tmp20)
        _tmp21 = tl.where(r0_mask, tmp22, _tmp21)
    tmp21 = triton_helpers.max2(_tmp21, 1)[:, None]
    _tmp46 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp23 = tl.load(in_ptr0 + (2*((r0_0 % (ks0 // 2))) + ks0*(triton_helpers.div_floor_integer(r0_0,  ks0 // 2))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp24 = tl.load(in_ptr0 + (1 + 2*((r0_0 % (ks0 // 2))) + ks0*(triton_helpers.div_floor_integer(r0_0,  ks0 // 2))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp25 = tmp24 + tmp23
        tmp26 = 0.5
        tmp27 = tmp25 * tmp26
        tmp28 = tmp27 * tmp26
        tmp29 = 0.7071067811865476
        tmp30 = tmp27 * tmp29
        tmp31 = libdevice.erf(tmp30)
        tmp32 = 1.0
        tmp33 = tmp31 + tmp32
        tmp34 = tmp28 * tmp33
        tmp35 = 3.0
        tmp36 = tmp34 + tmp35
        tmp37 = 0.0
        tmp38 = triton_helpers.maximum(tmp36, tmp37)
        tmp39 = 6.0
        tmp40 = triton_helpers.minimum(tmp38, tmp39)
        tmp41 = 0.16666666666666666
        tmp42 = tmp40 * tmp41
        tmp43 = tmp42 - tmp21
        tmp44 = tl_math.exp(tmp43)
        tmp45 = tl.broadcast_to(tmp44, [XBLOCK, R0_BLOCK])
        tmp47 = _tmp46 + tmp45
        _tmp46 = tl.where(r0_mask, tmp47, _tmp46)
    tmp46 = tl.sum(_tmp46, 1)[:, None]
    tmp48 = tl.load(in_ptr1 + (0))
    tmp49 = tl.broadcast_to(tmp48, [XBLOCK, 1])
    tmp50 = tl.full([1, 1], -100, tl.int64)
    tmp51 = tmp49 != tmp50
    tmp52 = tl.full([1, 1], 0, tl.int64)
    tmp53 = tl.where(tmp51, tmp49, tmp52)
    tmp54 = ks1*(ks0 // 2)
    tmp55 = tmp53 + tmp54
    tmp56 = tmp53 < 0
    tmp57 = tl.where(tmp56, tmp55, tmp53)
    tl.device_assert((0 <= tmp57) & (tmp57 < ks1*(ks0 // 2)), "index out of bounds: 0 <= tmp57 < ks1*(ks0 // 2)")
    tmp59 = tl.load(in_ptr0 + (2*((tmp57 % (ks0 // 2))) + ks0*(((tmp57 // (ks0 // 2)) % ks1))), None, eviction_policy='evict_last')
    tmp60 = tl.load(in_ptr0 + (1 + 2*((tmp57 % (ks0 // 2))) + ks0*(((tmp57 // (ks0 // 2)) % ks1))), None, eviction_policy='evict_last')
    tmp61 = tmp60 + tmp59
    tmp62 = 0.5
    tmp63 = tmp61 * tmp62
    tmp64 = tmp63 * tmp62
    tmp65 = 0.7071067811865476
    tmp66 = tmp63 * tmp65
    tmp67 = libdevice.erf(tmp66)
    tmp68 = 1.0
    tmp69 = tmp67 + tmp68
    tmp70 = tmp64 * tmp69
    tmp71 = 3.0
    tmp72 = tmp70 + tmp71
    tmp73 = 0.0
    tmp74 = triton_helpers.maximum(tmp72, tmp73)
    tmp75 = 6.0
    tmp76 = triton_helpers.minimum(tmp74, tmp75)
    tmp77 = 0.16666666666666666
    tmp78 = tmp76 * tmp77
    tmp79 = tmp78 - tmp21
    tmp80 = tl_math.log(tmp46)
    tmp81 = tmp79 - tmp80
    tmp82 = -tmp81
    tmp83 = tl.where(tmp51, tmp82, tmp73)
    tmp84 = tmp51.to(tl.int64)
    tmp85 = tmp84.to(tl.float32)
    tmp86 = tmp83 / tmp85
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp86, None)







def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    assert_size_stride(arg2_1, (1, s0, s1), (s0*s1, s1, 1))
    buf0 = empty_strided_cpu((1, ), (1, ), torch.int64)
    # Topologically Sorted Source Nodes: [], Original ATen: []
    aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf0)
    buf1 = buf0; del buf0  # reuse
    cpp_fused_randint_0(buf1, s0, s1)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf2 = empty_strided_cuda((1, ), (1, ), torch.int64)
        buf2.copy_(buf1, False)
        del buf1
        buf3 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        buf5 = reinterpret_tensor(buf3, (1, ), (1, ), 0); del buf3  # reuse
        buf6 = reinterpret_tensor(buf5, (), (), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [loss, x_4], Original ATen: [aten.nll_loss_forward, aten._log_softmax]
        triton_red_fused__log_softmax_nll_loss_forward_1_r0_numel = s0*(s1 // 2)
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax_nll_loss_forward_1[grid(1)](buf6, arg2_1, buf2, 64, 3, 1, 96, XBLOCK=1, R0_BLOCK=128, num_warps=2, num_stages=1)
        del arg2_1
        del buf2
    return (buf6, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 3
    arg1_1 = 64
    arg2_1 = rand_strided((1, 3, 64), (192, 64, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
