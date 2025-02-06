
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



extern "C"  void kernel(const int64_t* in_ptr0,
                       int64_t* out_ptr0,
                       const int64_t ks0,
                       const int64_t ks1,
                       const int64_t ks2)
{
    {
        {
            {
                auto tmp0 = in_ptr0[static_cast<int64_t>(0L)];
                auto tmp1 = static_cast<int32_t>(0);
                auto tmp2 = static_cast<int64_t>(0);
                auto tmp3 = (c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L)))*(c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(4L)))*(c10::div_floor_integer(static_cast<int64_t>(ks2), static_cast<int64_t>(4L)));
                auto tmp4 = c10::convert<int64_t>(tmp3);
                auto tmp5 = randint64_cpu(tmp0, tmp1, tmp2, tmp4);
                out_ptr0[static_cast<int64_t>(0L)] = tmp5;
            }
        }
    }
}
''')


#include "/tmp/torchinductor_sahanp/3b/c3bi5gk6mslf6u4iaqafhxm64z6u65e3eain4xlary5blqnvv6xx.h"
extern "C"  void kernel(const int64_t* in_ptr0,
                       float* out_ptr0,
                       const int64_t ks0,
                       const int64_t ks1,
                       const int64_t ks2)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>((c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L)))*(c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(4L)))*(c10::div_floor_integer(static_cast<int64_t>(ks2), static_cast<int64_t>(4L)))); x0+=static_cast<int64_t>(1L))
        {
            {
                {
                    auto tmp0 = in_ptr0[static_cast<int64_t>(1L)];
                    auto tmp1 = x0;
                    auto tmp2 = c10::convert<int32_t>(tmp1);
                    auto tmp3 = static_cast<int64_t>(0);
                    auto tmp4 = static_cast<int64_t>(2);
                    auto tmp5 = randint64_cpu(tmp0, tmp2, tmp3, tmp4);
                    auto tmp6 = c10::convert<float>(tmp5);
                    out_ptr0[static_cast<int64_t>(x0)] = tmp6;
                }
            }
        }
    }
}
''')



































import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__log_softmax_add_binary_cross_entropy_with_logits_nll_loss_forward_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = triton_helpers.maximum(_tmp2, tmp1)
        _tmp2 = tl.where(r0_mask, tmp3, _tmp2)
    tmp2 = triton_helpers.max2(_tmp2, 1)[:, None]
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp23 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp4 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.load(in_ptr1 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp4 - tmp2
        tmp6 = tl_math.exp(tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask, tmp9, _tmp8)
        tmp11 = 1.0
        tmp12 = tmp11 - tmp10
        tmp13 = tmp12 * tmp4
        tmp14 = 0.0
        tmp15 = triton_helpers.minimum(tmp14, tmp4)
        tmp16 = tl_math.abs(tmp4)
        tmp17 = -tmp16
        tmp18 = tl_math.exp(tmp17)
        tmp19 = libdevice.log1p(tmp18)
        tmp20 = tmp15 - tmp19
        tmp21 = tmp13 - tmp20
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, R0_BLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(r0_mask, tmp24, _tmp23)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tmp25 = tl.load(in_ptr2 + (0))
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK, 1])
    tmp27 = tl.full([1, 1], -100, tl.int64)
    tmp28 = tmp26 != tmp27
    tmp29 = tl.full([1, 1], 0, tl.int64)
    tmp30 = tl.where(tmp28, tmp26, tmp29)
    tmp31 = (ks0 // 4)*(ks1 // 4)*(ks2 // 4)
    tmp32 = tmp30 + tmp31
    tmp33 = tmp30 < 0
    tmp34 = tl.where(tmp33, tmp32, tmp30)
    tl.device_assert((0 <= tmp34) & (tmp34 < (ks0 // 4)*(ks1 // 4)*(ks2 // 4)), "index out of bounds: 0 <= tmp34 < (ks0 // 4)*(ks1 // 4)*(ks2 // 4)")
    tmp36 = tl.load(in_ptr0 + (tmp34), None, eviction_policy='evict_last')
    tmp37 = tmp36 - tmp2
    tmp38 = tl_math.log(tmp8)
    tmp39 = tmp37 - tmp38
    tmp40 = -tmp39
    tmp41 = 0.0
    tmp42 = tl.where(tmp28, tmp40, tmp41)
    tmp43 = tmp28.to(tl.int64)
    tmp44 = tmp43.to(tl.float32)
    tmp45 = tmp42 / tmp44
    tmp46 = tmp31.to(tl.float32)
    tmp47 = tmp23 / tmp46
    tmp48 = tmp45 + tmp47
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp48, None)







def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, 1, s0, s1, s2), (s0*s1*s2, s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)

        buf0 = torch.ops.aten.max_pool3d_with_indices.default(arg3_1, [2, 2, 2], [2, 2, 2])
        del arg3_1
        buf1 = buf0[0]
        del buf0

        buf3 = torch.ops.aten.max_pool3d_with_indices.default(buf1, [2, 2, 2], [2, 2, 2])
        del buf1
        buf4 = buf3[0]
        del buf3
    buf6 = empty_strided_cpu((2, ), (1, ), torch.int64)

    aten.randint.low_out(-9223372036854775808, 9223372036854775807, [2], out=buf6)
    buf7 = empty_strided_cpu((1, ), (1, ), torch.int64)
    cpp_fused_randint_0(buf6, buf7, s0, s1, s2)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf8 = empty_strided_cuda((1, ), (1, ), torch.int64)
        buf8.copy_(buf7, False)
        del buf7
    buf11 = empty_strided_cpu((1, (s0 // 4)*(s1 // 4)*(s2 // 4)), ((s0 // 4)*(s1 // 4)*(s2 // 4), 1), torch.float32)
    cpp_fused__to_copy_randint_1(buf6, buf11, s0, s1, s2)
    del buf6
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf12 = empty_strided_cuda((1, (s0 // 4)*(s1 // 4)*(s2 // 4)), ((s0 // 4)*(s1 // 4)*(s2 // 4), 1), torch.float32)
        buf12.copy_(buf11, False)
        del buf11
        buf9 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        buf14 = reinterpret_tensor(buf9, (), (), 0); del buf9

        triton_red_fused__log_softmax_add_binary_cross_entropy_with_logits_nll_loss_forward_2_r0_numel = (s0 // 4)*(s1 // 4)*(s2 // 4)
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax_add_binary_cross_entropy_with_logits_nll_loss_forward_2[grid(1)](buf14, buf4, buf12, buf8, 32, 32, 32, 1, 512, XBLOCK=1, R0_BLOCK=512, num_warps=4, num_stages=1)
        del buf12
        del buf4
        del buf8
    return (buf14, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 32
    arg1_1 = 32
    arg2_1 = 32
    arg3_1 = rand_strided((1, 1, 32, 32, 32), (32768, 32768, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
