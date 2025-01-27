# AOT ID: ['59_inference']
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


# kernel path: /tmp/torchinductor_sahanp/3n/c3npxm6psy5nzyq6dvqbm2nizb5vw4u4f6jz24pu3fm44e7qrrez.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.neg, aten._softmax]
# Source node to ATen node mapping:
#   x => amax, exp, neg, sub_3, sum_1
# Graph fragment:
#   %neg : [num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%arg3_1,), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%neg, [1], True), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%neg, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_3,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__softmax_neg_1(in_ptr0, out_ptr0, out_ptr1, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + ks0*ks1*r0_1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = -tmp0
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = triton_helpers.maximum(_tmp3, tmp2)
        _tmp3 = tl.where(r0_mask & xmask, tmp4, _tmp3)
    tmp3 = triton_helpers.max2(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp3, xmask)
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp5 = tl.load(in_ptr0 + (x0 + ks0*ks1*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = -tmp5
        tmp7 = tmp6 - tmp3
        tmp8 = tl_math.exp(tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp10, xmask)




# kernel path: /tmp/torchinductor_sahanp/mf/cmfzldoyjgmpos5pd36rtau2zamwcqgapkrxkzcphipyb6puvr5j.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
# Source node to ATen node mapping:
#   loss => convert_element_type_1, div_1, full_default_1, ne_1, ne_2, neg_1, sum_2, sum_3, where_1
# Graph fragment:
#   %ne_1 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%device_put, -100), kwargs = {})
#   %neg_1 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_1, %neg_1, %full_default_1), kwargs = {})
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
def triton_poi_fused_nll_loss_forward_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.full([1], -100, tl.int64)
    tmp3 = tmp1 != tmp2
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tl.where(tmp3, tmp1, tmp4)
    tmp6 = ks0*ks1*ks2
    tmp7 = tmp5 + tmp6
    tmp8 = tmp5 < 0
    tmp9 = tl.where(tmp8, tmp7, tmp5)
    tl.device_assert((0 <= tmp9) & (tmp9 < ks0*ks1*ks2), "index out of bounds: 0 <= tmp9 < ks0*ks1*ks2")
    tmp11 = tl.load(in_ptr1 + ((tmp9 % (ks0*ks1*ks2))), None, eviction_policy='evict_last')
    tmp12 = -tmp11
    tmp13 = tl.load(in_ptr2 + ((tmp9 % (ks1*ks2))), None, eviction_policy='evict_last')
    tmp14 = tmp12 - tmp13
    tmp15 = tl_math.exp(tmp14)
    tmp16 = tl.load(in_ptr3 + ((tmp9 % (ks1*ks2))), None, eviction_policy='evict_last')
    tmp17 = tmp15 / tmp16
    tmp18 = 0.5
    tmp19 = tmp17 * tmp18
    tmp20 = 0.7071067811865476
    tmp21 = tmp17 * tmp20
    tmp22 = libdevice.erf(tmp21)
    tmp23 = 1.0
    tmp24 = tmp22 + tmp23
    tmp25 = tmp19 * tmp24
    tmp26 = 1e-09
    tmp27 = tmp25 + tmp26
    tmp28 = tl_math.log(tmp27)
    tmp29 = -tmp28
    tmp30 = 0.0
    tmp31 = tl.where(tmp3, tmp29, tmp30)
    tmp32 = tmp3.to(tl.int64)
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp31 / tmp33
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp34, None)







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
        buf3 = empty_strided_cuda((1, 1, s1, s2), (s1*s2, s1*s2, s2, 1), torch.float32)
        buf4 = empty_strided_cuda((1, 1, s1, s2), (s1*s2, s1*s2, s2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.neg, aten._softmax]
        triton_red_fused__softmax_neg_1_xnumel = s1*s2
        stream0 = get_raw_stream(0)
        triton_red_fused__softmax_neg_1[grid(triton_red_fused__softmax_neg_1_xnumel)](arg3_1, buf3, buf4, 32, 32, 1024, 10, XBLOCK=32, R0_BLOCK=16, num_warps=4, num_stages=1)
        buf5 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_forward_2[grid(1)](buf2, arg3_1, buf3, buf4, buf5, 10, 32, 32, 1, XBLOCK=1, num_warps=1, num_stages=1)
        del arg3_1
        del buf2
        del buf3
        del buf4
    return (buf5, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 10
    arg1_1 = 32
    arg2_1 = 32
    arg3_1 = rand_strided((1, 10, 32, 32), (10240, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
