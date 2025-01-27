# AOT ID: ['180_inference']
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
                       const int64_t ks1,
                       const int64_t ks2)
{
    {
        {
            {
                auto tmp0 = in_out_ptr0[static_cast<int64_t>(0L)];
                auto tmp1 = static_cast<int32_t>(0);
                auto tmp2 = static_cast<int64_t>(0);
                auto tmp3 = ks0*ks1*ks2;
                auto tmp4 = c10::convert<int64_t>(tmp3);
                auto tmp5 = randint64_cpu(tmp0, tmp1, tmp2, tmp4);
                in_out_ptr0[static_cast<int64_t>(0L)] = tmp5;
            }
        }
    }
}
''')


# kernel path: /tmp/torchinductor_sahanp/ig/cig24ay5lx42iupoeui26qukjpe56wp7o56qq6d7jvmuz5wtrlgh.py
# Topologically Sorted Source Nodes: [log_softmax], Original ATen: [aten._log_softmax]
# Source node to ATen node mapping:
#   log_softmax => amax
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view, [1], True), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__log_softmax_1(in_ptr0, out_ptr0, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp30 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = r0_1 + x0*((1 + ks0*ks1*ks2) // 2)
        tmp1 = ks0*ks1*ks2
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (((r0_1 + x0*((1 + ks0*ks1*ks2) // 2)) % (ks0*ks1*ks2))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 3.0
        tmp5 = tmp3 + tmp4
        tmp6 = 0.0
        tmp7 = triton_helpers.maximum(tmp5, tmp6)
        tmp8 = 6.0
        tmp9 = triton_helpers.minimum(tmp7, tmp8)
        tmp10 = 0.16666666666666666
        tmp11 = tmp9 * tmp10
        tmp12 = tmp11 > tmp6
        tmp13 = 1.0
        tmp14 = tmp11 * tmp13
        tmp15 = libdevice.expm1(tmp14)
        tmp16 = tmp15 * tmp13
        tmp17 = tl.where(tmp12, tmp14, tmp16)
        tmp18 = tmp17 + tmp4
        tmp19 = triton_helpers.maximum(tmp18, tmp6)
        tmp20 = triton_helpers.minimum(tmp19, tmp8)
        tmp21 = tmp20 * tmp10
        tmp22 = tmp21 > tmp6
        tmp23 = tmp21 * tmp13
        tmp24 = libdevice.expm1(tmp23)
        tmp25 = tmp24 * tmp13
        tmp26 = tl.where(tmp22, tmp23, tmp25)
        tmp27 = tl.full(tmp26.shape, float("-inf"), tmp26.dtype)
        tmp28 = tl.where(tmp2, tmp26, tmp27)
        tmp29 = tl.broadcast_to(tmp28, [XBLOCK, R0_BLOCK])
        tmp31 = triton_helpers.maximum(_tmp30, tmp29)
        _tmp30 = tl.where(r0_mask & xmask, tmp31, _tmp30)
    tmp30 = triton_helpers.max2(_tmp30, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp30, xmask)




# kernel path: /tmp/torchinductor_sahanp/w6/cw6xxqvl7jurqwalstghbyddn2ys2zafui3upifmagznwy3g2q76.py
# Topologically Sorted Source Nodes: [log_softmax], Original ATen: [aten._log_softmax]
# Source node to ATen node mapping:
#   log_softmax => amax
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view, [1], True), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__log_softmax_2(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 2
    R0_BLOCK: tl.constexpr = 2
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = triton_helpers.max2(tmp1, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp3, None)




# kernel path: /tmp/torchinductor_sahanp/q7/cq7ule4mr4wzskbbwm5cdrbalinxzrr5dtvxbqtvo2ciwpx2y5sj.py
# Topologically Sorted Source Nodes: [log_softmax], Original ATen: [aten._log_softmax]
# Source node to ATen node mapping:
#   log_softmax => exp, sub_13, sum_1
# Graph fragment:
#   %sub_13 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %amax), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_13,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__log_softmax_3(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp33 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = r0_1 + x0*((1 + ks0*ks1*ks2) // 2)
        tmp1 = ks0*ks1*ks2
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (((r0_1 + x0*((1 + ks0*ks1*ks2) // 2)) % (ks0*ks1*ks2))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 3.0
        tmp5 = tmp3 + tmp4
        tmp6 = 0.0
        tmp7 = triton_helpers.maximum(tmp5, tmp6)
        tmp8 = 6.0
        tmp9 = triton_helpers.minimum(tmp7, tmp8)
        tmp10 = 0.16666666666666666
        tmp11 = tmp9 * tmp10
        tmp12 = tmp11 > tmp6
        tmp13 = 1.0
        tmp14 = tmp11 * tmp13
        tmp15 = libdevice.expm1(tmp14)
        tmp16 = tmp15 * tmp13
        tmp17 = tl.where(tmp12, tmp14, tmp16)
        tmp18 = tmp17 + tmp4
        tmp19 = triton_helpers.maximum(tmp18, tmp6)
        tmp20 = triton_helpers.minimum(tmp19, tmp8)
        tmp21 = tmp20 * tmp10
        tmp22 = tmp21 > tmp6
        tmp23 = tmp21 * tmp13
        tmp24 = libdevice.expm1(tmp23)
        tmp25 = tmp24 * tmp13
        tmp26 = tl.where(tmp22, tmp23, tmp25)
        tmp27 = tl.load(in_ptr1 + (tl.full([XBLOCK, R0_BLOCK], 0, tl.int32)), tmp2, eviction_policy='evict_last', other=0.0)
        tmp28 = tmp26 - tmp27
        tmp29 = tl_math.exp(tmp28)
        tmp30 = tl.full(tmp29.shape, 0, tmp29.dtype)
        tmp31 = tl.where(tmp2, tmp29, tmp30)
        tmp32 = tl.broadcast_to(tmp31, [XBLOCK, R0_BLOCK])
        tmp34 = _tmp33 + tmp32
        _tmp33 = tl.where(r0_mask & xmask, tmp34, _tmp33)
    tmp33 = tl.sum(_tmp33, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp33, xmask)




# kernel path: /tmp/torchinductor_sahanp/44/c443g22ci2v7pnmi5g7nv6wzw5u3w6icoxlllck65nt6osvkq4ds.py
# Topologically Sorted Source Nodes: [loss, log_softmax], Original ATen: [aten.nll_loss_forward, aten._log_softmax]
# Source node to ATen node mapping:
#   log_softmax => exp, sub_13, sum_1
#   loss => convert_element_type_1, div_2, full_default_1, ne_1, ne_2, neg, sum_2, sum_3, where_3
# Graph fragment:
#   %ne_1 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%device_put, -100), kwargs = {})
#   %sub_13 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %amax), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_13,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_3 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_1, %neg, %full_default_1), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_3,), kwargs = {})
#   %ne_2 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%device_put, -100), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne_2,), kwargs = {})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_2, torch.float32), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_3, %convert_element_type_1), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__log_softmax_nll_loss_forward_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 2
    R0_BLOCK: tl.constexpr = 2
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), None)
    tmp4 = tl.load(in_ptr1 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, 1])
    tmp39 = tl.load(in_out_ptr0 + (0))
    tmp40 = tl.broadcast_to(tmp39, [XBLOCK, 1])
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None]
    tmp6 = tl.full([1, 1], -100, tl.int64)
    tmp7 = tmp5 != tmp6
    tmp8 = tl.full([1, 1], 0, tl.int64)
    tmp9 = tl.where(tmp7, tmp5, tmp8)
    tmp10 = ks0*ks1*ks2
    tmp11 = tmp9 + tmp10
    tmp12 = tmp9 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp9)
    tl.device_assert((0 <= tmp13) & (tmp13 < ks0*ks1*ks2), "index out of bounds: 0 <= tmp13 < ks0*ks1*ks2")
    tmp15 = tl.load(in_ptr2 + ((tmp13 % (ks0*ks1*ks2))), None, eviction_policy='evict_last')
    tmp16 = 3.0
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp22 = 0.16666666666666666
    tmp23 = tmp21 * tmp22
    tmp24 = tmp23 > tmp18
    tmp25 = 1.0
    tmp26 = tmp23 * tmp25
    tmp27 = libdevice.expm1(tmp26)
    tmp28 = tmp27 * tmp25
    tmp29 = tl.where(tmp24, tmp26, tmp28)
    tmp30 = tmp29 + tmp16
    tmp31 = triton_helpers.maximum(tmp30, tmp18)
    tmp32 = triton_helpers.minimum(tmp31, tmp20)
    tmp33 = tmp32 * tmp22
    tmp34 = tmp33 > tmp18
    tmp35 = tmp33 * tmp25
    tmp36 = libdevice.expm1(tmp35)
    tmp37 = tmp36 * tmp25
    tmp38 = tl.where(tmp34, tmp35, tmp37)
    tmp41 = tmp38 - tmp40
    tmp42 = tl_math.log(tmp3)
    tmp43 = tmp41 - tmp42
    tmp44 = -tmp43
    tmp45 = tl.where(tmp7, tmp44, tmp18)
    tmp46 = tmp7.to(tl.int64)
    tmp47 = tmp46.to(tl.float32)
    tmp48 = tmp45 / tmp47
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp48, None)







def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    buf0 = empty_strided_cpu((1, ), (1, ), torch.int64)
    # Topologically Sorted Source Nodes: [], Original ATen: []
    aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf0)
    buf1 = buf0; del buf0  # reuse
    cpp_fused_randint_0(buf1, s0, s1, s2)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf2 = empty_strided_cuda((1, ), (1, ), torch.int64)
        buf2.copy_(buf1, False)
        del buf1
        buf3 = empty_strided_cuda((1, 1, 2), (2, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [log_softmax], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_1_r0_numel = (1 + s0*s1*s2) // 2
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax_1[grid(2)](arg3_1, buf3, 3, 64, 64, 2, 6144, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        buf4 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [log_softmax], Original ATen: [aten._log_softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax_2[grid(1)](buf3, buf4, 1, 2, XBLOCK=1, num_warps=2, num_stages=1)
        buf5 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [log_softmax], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_3_r0_numel = (1 + s0*s1*s2) // 2
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax_3[grid(2)](arg3_1, buf4, buf5, 3, 64, 64, 2, 6144, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        buf7 = reinterpret_tensor(buf4, (), (), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [loss, log_softmax], Original ATen: [aten.nll_loss_forward, aten._log_softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax_nll_loss_forward_4[grid(1)](buf7, buf5, buf2, arg3_1, 3, 64, 64, 1, 2, XBLOCK=1, num_warps=2, num_stages=1)
        del arg3_1
        del buf2
        del buf5
    return (buf7, )


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
