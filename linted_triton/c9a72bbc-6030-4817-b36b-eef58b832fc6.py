# AOT ID: ['123_inference']
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
extern "C"  void kernel(const int64_t* in_ptr0,
                       int64_t* out_ptr0,
                       const int64_t ks0,
                       const int64_t ks1)
{
    {
        {
            {
                auto tmp0 = in_ptr0[static_cast<int64_t>(0L)];
                auto tmp1 = static_cast<int32_t>(0);
                auto tmp2 = static_cast<int64_t>(0);
                auto tmp3 = 4L*ks0*ks1;
                auto tmp4 = c10::convert<int64_t>(tmp3);
                auto tmp5 = randint64_cpu(tmp0, tmp1, tmp2, tmp4);
                out_ptr0[static_cast<int64_t>(0L)] = tmp5;
            }
        }
    }
}
''')


# kernel path: /tmp/torchinductor_sahanp/5y/c5yxdby7r456cj3cg23ku4vvrtahhdyoakg3bdl55hg3fbcrkkfy.py
# Topologically Sorted Source Nodes: [loss1], Original ATen: [aten.arange, aten.ne, aten.gather, aten.rsub, aten.add, aten.clamp_min, aten.scalar_tensor, aten.where, aten.mean]
# Source node to ATen node mapping:
#   loss1 => add_14, clamp_min_1, full_default, gather, iota, mean, ne_4, sub_9, where_1
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (%floordiv_1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %ne_4 : [num_users=1] = call_function[target=torch.ops.aten.ne.Tensor](args = (%iota, %unsqueeze), kwargs = {})
#   %gather : [num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%view_2, 1, %unsqueeze), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %gather), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub_9, %view_2), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_14, 0), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_4, %clamp_min_1, %full_default), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%where_1,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_arange_clamp_min_gather_mean_ne_rsub_scalar_tensor_where_1(in_ptr0, in_ptr1, out_ptr1, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    _tmp38 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp25 = tl.load(in_ptr1 + (ks2*((((((r0_0 // (2*ks2)) % (2*ks1))) // 2) % ks1)) + ks1*ks2*((((r0_0 % (2*ks2))) % 2)) + 2*ks1*ks2*(((((r0_0 // (2*ks2)) % (2*ks1))) % 2)) + (((((r0_0 % (2*ks2))) // 2) % ks2))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = r0_0
        tmp3 = tmp2 != tmp1
        tmp4 = 4*ks1*ks2*(ks0 // 4)
        tmp5 = tmp1 + tmp4
        tmp6 = tmp1 < 0
        tmp7 = tl.where(tmp6, tmp5, tmp1)
        tl.device_assert((0 <= tmp7) & (tmp7 < 4*ks1*ks2*(ks0 // 4)), "index out of bounds: 0 <= tmp7 < 4*ks1*ks2*(ks0 // 4)")
        tmp9 = tl.load(in_ptr1 + (ks2*((((((tmp7 // (2*ks2)) % (2*ks1))) // 2) % ks1)) + ks1*ks2*((((tmp7 % (2*ks2))) % 2)) + 2*ks1*ks2*(((((tmp7 // (2*ks2)) % (2*ks1))) % 2)) + (((((tmp7 % (2*ks2))) // 2) % ks2))), None, eviction_policy='evict_last')
        tmp10 = 0.0
        tmp11 = tmp9 > tmp10
        tmp12 = 0.1
        tmp13 = tmp9 * tmp12
        tmp14 = tl.where(tmp11, tmp9, tmp13)
        tmp15 = 3.0
        tmp16 = tmp14 + tmp15
        tmp17 = triton_helpers.maximum(tmp16, tmp10)
        tmp18 = 6.0
        tmp19 = triton_helpers.minimum(tmp17, tmp18)
        tmp20 = tmp14 * tmp19
        tmp21 = 0.16666666666666666
        tmp22 = tmp20 * tmp21
        tmp23 = 1.0
        tmp24 = tmp23 - tmp22
        tmp26 = tmp25 > tmp10
        tmp27 = tmp25 * tmp12
        tmp28 = tl.where(tmp26, tmp25, tmp27)
        tmp29 = tmp28 + tmp15
        tmp30 = triton_helpers.maximum(tmp29, tmp10)
        tmp31 = triton_helpers.minimum(tmp30, tmp18)
        tmp32 = tmp28 * tmp31
        tmp33 = tmp32 * tmp21
        tmp34 = tmp24 + tmp33
        tmp35 = triton_helpers.maximum(tmp34, tmp10)
        tmp36 = tl.where(tmp3, tmp35, tmp10)
        tmp37 = tl.broadcast_to(tmp36, [XBLOCK, R0_BLOCK])
        tmp39 = _tmp38 + tmp37
        _tmp38 = tl.where(r0_mask, tmp39, _tmp38)
    tmp38 = tl.sum(_tmp38, 1)[:, None]
    tl.store(out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp38, None)




#include "/tmp/torchinductor_sahanp/3b/c3bi5gk6mslf6u4iaqafhxm64z6u65e3eain4xlary5blqnvv6xx.h"
extern "C"  void kernel(const int64_t* in_ptr0,
                       int64_t* out_ptr0,
                       const int64_t ks0,
                       const int64_t ks1,
                       const int64_t ks2)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>((c10::div_floor_integer(static_cast<int64_t>(2L*ks1*ks2*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(2L*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L))))))), static_cast<int64_t>(static_cast<int64_t>(std::trunc(std::pow(static_cast<double>(4L*ks1*ks2), 0.5))))))*static_cast<int64_t>(std::trunc(std::pow(static_cast<double>(4L*ks1*ks2), 0.5)))); x0+=static_cast<int64_t>(1L))
        {
            {
                {
                    auto tmp0 = in_ptr0[static_cast<int64_t>(1L)];
                    auto tmp1 = x0;
                    auto tmp2 = c10::convert<int32_t>(tmp1);
                    auto tmp3 = static_cast<int64_t>(0);
                    auto tmp4 = static_cast<int64_t>(1);
                    auto tmp5 = randint64_cpu(tmp0, tmp2, tmp3, tmp4);
                    out_ptr0[static_cast<int64_t>(x0)] = tmp5;
                }
            }
        }
    }
}
''')


# kernel path: /tmp/torchinductor_sahanp/sc/csc2k4b4wsteluq3fz3ghayywzyiriflexierf4h3qd5t2idsen2.py
# Topologically Sorted Source Nodes: [loss1, loss2, add], Original ATen: [aten.mean, aten.nll_loss2d_forward, aten.add]
# Source node to ATen node mapping:
#   add => add_35
#   loss1 => mean
#   loss2 => convert_element_type_2, div_1, full_default_2, ne_6, ne_7, neg, sum_1, sum_2, where_3
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%where_1,), kwargs = {})
#   %ne_6 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%device_put_1, -100), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_3 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_6, %neg, %full_default_2), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_3,), kwargs = {})
#   %ne_7 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%device_put_1, -100), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne_7,), kwargs = {})
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_1, torch.float32), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_2, %convert_element_type_2), kwargs = {})
#   %add_35 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, %div_1), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_mean_nll_loss2d_forward_3(in_out_ptr0, in_ptr0, in_ptr1, ks0, ks1, ks2, ks3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp27 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp33 = tl.full([XBLOCK, R0_BLOCK], 0, tl.int64)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        r0_0 = (r0_index % ks0)
        r0_1 = r0_index // ks0
        tmp0 = tl.load(in_ptr0 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.load(in_ptr1 + (ks2*(((((((r0_0 + r0_1*libdevice.trunc(libdevice.pow((4*ks1*ks2).to(tl.float64), tl.full([], 0.500000000000000, tl.float64))).to(tl.int32)) // (2*ks2)) % (2*ks1))) // 2) % ks1)) + ks1*ks2*(((((r0_0 + r0_1*libdevice.trunc(libdevice.pow((4*ks1*ks2).to(tl.float64), tl.full([], 0.500000000000000, tl.float64))).to(tl.int32)) % (2*ks2))) % 2)) + 2*ks1*ks2*((((((r0_0 + r0_1*libdevice.trunc(libdevice.pow((4*ks1*ks2).to(tl.float64), tl.full([], 0.500000000000000, tl.float64))).to(tl.int32)) // (2*ks2)) % (2*ks1))) % 2)) + ((((((r0_0 + r0_1*libdevice.trunc(libdevice.pow((4*ks1*ks2).to(tl.float64), tl.full([], 0.500000000000000, tl.float64))).to(tl.int32)) % (2*ks2))) // 2) % ks2))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp29 = tl.load(in_ptr0 + (r0_2), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.full([1, 1], -100, tl.int64)
        tmp2 = tmp0 != tmp1
        tmp3 = tl.full([1, 1], 0, tl.int64)
        tmp4 = tl.where(tmp2, tmp0, tmp3)
        tmp5 = tl.full([XBLOCK, R0_BLOCK], 1, tl.int32)
        tmp6 = tmp4 + tmp5
        tmp7 = tmp4 < 0
        tmp8 = tl.where(tmp7, tmp6, tmp4)
        tl.device_assert(((0 <= tmp8) & (tmp8 < 1)) | ~(r0_mask), "index out of bounds: 0 <= tmp8 < 1")
        tmp11 = 0.0
        tmp12 = tmp10 > tmp11
        tmp13 = 0.1
        tmp14 = tmp10 * tmp13
        tmp15 = tl.where(tmp12, tmp10, tmp14)
        tmp16 = 3.0
        tmp17 = tmp15 + tmp16
        tmp18 = triton_helpers.maximum(tmp17, tmp11)
        tmp19 = 6.0
        tmp20 = triton_helpers.minimum(tmp18, tmp19)
        tmp21 = tmp15 * tmp20
        tmp22 = 0.16666666666666666
        tmp23 = tmp21 * tmp22
        tmp24 = -tmp23
        tmp25 = tl.where(tmp2, tmp24, tmp11)
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(r0_mask, tmp28, _tmp27)
        tmp30 = tmp29 != tmp1
        tmp31 = tmp30.to(tl.int64)
        tmp32 = tl.broadcast_to(tmp31, [XBLOCK, R0_BLOCK])
        tmp34 = _tmp33 + tmp32
        _tmp33 = tl.where(r0_mask, tmp34, _tmp33)
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tmp33 = tl.sum(_tmp33, 1)[:, None]
    tmp35 = tl.load(in_out_ptr0 + (0))
    tmp36 = tl.broadcast_to(tmp35, [XBLOCK, 1])
    tmp37 = 2*ks1*ks2*(triton_helpers.div_floor_integer(ks3,  2*(ks3 // 4)))
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tmp36 / tmp38
    tmp40 = tmp33.to(tl.float32)
    tmp41 = tmp27 / tmp40
    tmp42 = tmp39 + tmp41
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp42, None)







def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    buf0 = empty_strided_cpu((2, ), (1, ), torch.int64)
    # Topologically Sorted Source Nodes: [], Original ATen: []
    aten.randint.low_out(-9223372036854775808, 9223372036854775807, [2], out=buf0)
    buf1 = empty_strided_cpu((1, ), (1, ), torch.int64)
    cpp_fused_randint_0(buf0, buf1, s1, s2)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf2 = empty_strided_cuda((1, ), (1, ), torch.int64)
        buf2.copy_(buf1, False)
        del buf1
        buf4 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [loss1], Original ATen: [aten.arange, aten.ne, aten.gather, aten.rsub, aten.add, aten.clamp_min, aten.scalar_tensor, aten.where, aten.mean]
        triton_red_fused_add_arange_clamp_min_gather_mean_ne_rsub_scalar_tensor_where_1_r0_numel = 2*s1*s2*(s0 // (2*(s0 // 4)))
        stream0 = get_raw_stream(0)
        triton_red_fused_add_arange_clamp_min_gather_mean_ne_rsub_scalar_tensor_where_1[grid(1)](buf2, arg3_1, buf4, 4, 16, 16, 1, 1024, XBLOCK=1, R0_BLOCK=1024, num_warps=8, num_stages=1)
        del buf2
    buf5 = empty_strided_cpu((1, math.trunc(torch.sym_float(4*s1*s2) ** 0.5), (2*s1*s2*(s0 // (2*(s0 // 4)))) // (math.trunc(torch.sym_float(4*s1*s2) ** 0.5))), (((2*s1*s2*(s0 // (2*(s0 // 4)))) // (math.trunc(torch.sym_float(4*s1*s2) ** 0.5)))*math.trunc(torch.sym_float(4*s1*s2) ** 0.5), (2*s1*s2*(s0 // (2*(s0 // 4)))) // (math.trunc(torch.sym_float(4*s1*s2) ** 0.5)), 1), torch.int64)
    cpp_fused_randint_2(buf0, buf5, s0, s1, s2)
    del buf0
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf6 = empty_strided_cuda((1, math.trunc(torch.sym_float(4*s1*s2) ** 0.5), (2*s1*s2*(s0 // (2*(s0 // 4)))) // (math.trunc(torch.sym_float(4*s1*s2) ** 0.5))), (((2*s1*s2*(s0 // (2*(s0 // 4)))) // (math.trunc(torch.sym_float(4*s1*s2) ** 0.5)))*math.trunc(torch.sym_float(4*s1*s2) ** 0.5), (2*s1*s2*(s0 // (2*(s0 // 4)))) // (math.trunc(torch.sym_float(4*s1*s2) ** 0.5)), 1), torch.int64)
        buf6.copy_(buf5, False)
        del buf5
        ps0 = (2*s1*s2*(s0 // (2*(s0 // 4)))) // (math.trunc(torch.sym_float(4*s1*s2) ** 0.5))
        buf9 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [loss1, loss2, add], Original ATen: [aten.mean, aten.nll_loss2d_forward, aten.add]
        triton_red_fused_add_mean_nll_loss2d_forward_3_r0_numel = ((2*s1*s2*(s0 // (2*(s0 // 4)))) // (math.trunc(torch.sym_float(4*s1*s2) ** 0.5)))*math.trunc(torch.sym_float(4*s1*s2) ** 0.5)
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mean_nll_loss2d_forward_3[grid(1)](buf9, buf6, arg3_1, 32, 16, 16, 4, 1, 1024, XBLOCK=1, R0_BLOCK=1024, num_warps=8, num_stages=1)
        del arg3_1
        del buf6
    return (buf9, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 4
    arg1_1 = 16
    arg2_1 = 16
    arg3_1 = rand_strided((1, 4, 16, 16), (1024, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
