
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












import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_linalg_vector_norm_mul_sum_0(in_ptr0, out_ptr0, out_ptr1, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp13 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_0 + ((ks0*ks1*ks2) // 2)), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = libdevice.tanh(tmp0)
        tmp2 = tmp0 - tmp1
        tmp3 = 3.0
        tmp4 = tmp2 + tmp3
        tmp5 = 0.0
        tmp6 = triton_helpers.maximum(tmp4, tmp5)
        tmp7 = 6.0
        tmp8 = triton_helpers.minimum(tmp6, tmp7)
        tmp9 = 0.16666666666666666
        tmp10 = tmp8 * tmp9
        tmp11 = tmp10 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(r0_mask, tmp14, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp13, None)
    tl.store(out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp13, None)








































import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_clamp_min_div_eq_fill_linalg_vector_norm_mean_mul_sqrt_sub_sum_where_zeros_like_1(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp13 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + ((r0_0 % (ks0*ks1*ks2))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = libdevice.tanh(tmp0)
        tmp2 = tmp0 - tmp1
        tmp3 = 3.0
        tmp4 = tmp2 + tmp3
        tmp5 = 0.0
        tmp6 = triton_helpers.maximum(tmp4, tmp5)
        tmp7 = 6.0
        tmp8 = triton_helpers.minimum(tmp6, tmp7)
        tmp9 = 0.16666666666666666
        tmp10 = tmp8 * tmp9
        tmp11 = tmp10 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(r0_mask, tmp14, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tmp37 = tl.load(in_ptr1 + (0))
    tmp38 = tl.broadcast_to(tmp37, [XBLOCK, R0_BLOCK])
    _tmp44 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp48 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp52 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp15 = tl.load(in_ptr0 + ((r0_0 % (ks0*ks1*ks2))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp30 = tl.load(in_ptr0 + (((r0_0 + ((ks0*ks1*ks2) // 2)) % (ks0*ks1*ks2))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp16 = libdevice.tanh(tmp15)
        tmp17 = tmp15 - tmp16
        tmp18 = 3.0
        tmp19 = tmp17 + tmp18
        tmp20 = 0.0
        tmp21 = triton_helpers.maximum(tmp19, tmp20)
        tmp22 = 6.0
        tmp23 = triton_helpers.minimum(tmp21, tmp22)
        tmp24 = 0.16666666666666666
        tmp25 = tmp23 * tmp24
        tmp26 = libdevice.sqrt(tmp13)
        tmp27 = 1e-08
        tmp28 = triton_helpers.maximum(tmp26, tmp27)
        tmp29 = tmp25 / tmp28
        tmp31 = libdevice.tanh(tmp30)
        tmp32 = tmp30 - tmp31
        tmp33 = tmp32 + tmp18
        tmp34 = triton_helpers.maximum(tmp33, tmp20)
        tmp35 = triton_helpers.minimum(tmp34, tmp22)
        tmp36 = tmp35 * tmp24
        tmp39 = libdevice.sqrt(tmp38)
        tmp40 = triton_helpers.maximum(tmp39, tmp27)
        tmp41 = tmp36 / tmp40
        tmp42 = tmp29 * tmp41
        tmp43 = tl.broadcast_to(tmp42, [XBLOCK, R0_BLOCK])
        tmp45 = _tmp44 + tmp43
        _tmp44 = tl.where(r0_mask, tmp45, _tmp44)
        tmp46 = tmp25 * tmp36
        tmp47 = tl.broadcast_to(tmp46, [XBLOCK, R0_BLOCK])
        tmp49 = _tmp48 + tmp47
        _tmp48 = tl.where(r0_mask, tmp49, _tmp48)
        tmp50 = tmp25 * tmp25
        tmp51 = tl.broadcast_to(tmp50, [XBLOCK, R0_BLOCK])
        tmp53 = _tmp52 + tmp51
        _tmp52 = tl.where(r0_mask, tmp53, _tmp52)
    tmp44 = tl.sum(_tmp44, 1)[:, None]
    tmp48 = tl.sum(_tmp48, 1)[:, None]
    tmp52 = tl.sum(_tmp52, 1)[:, None]
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp44, None)
    tmp56 = tl.load(in_ptr2 + (0))
    tmp57 = tl.broadcast_to(tmp56, [XBLOCK, 1])
    tmp54 = 9.999999960041972e-13
    tmp55 = tmp52 + tmp54
    tmp58 = tmp57 + tmp54
    tmp59 = tmp55 * tmp58
    tmp60 = libdevice.sqrt(tmp59)
    tmp61 = tmp48 / tmp60
    tmp62 = 1.0
    tmp63 = tmp62 - tmp61
    tmp64 = tl.full([1, 1], True, tl.int1)
    tmp65 = 0.0
    tmp66 = tl.where(tmp64, tmp63, tmp65)
    tmp67 = tmp61 - tmp65
    tmp68 = triton_helpers.maximum(tmp67, tmp65)
    tmp69 = tl.full([1, 1], False, tl.int1)
    tmp70 = tl.where(tmp69, tmp68, tmp65)
    tmp71 = tmp66 + tmp70
    tmp72 = tmp71 / tmp62
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp72, None)





extern "C"  void kernel(const int64_t* in_ptr0,
                       int64_t* out_ptr0)
{
    {
        {
            {
                auto tmp0 = in_ptr0[static_cast<int64_t>(1L)];
                auto tmp1 = static_cast<int32_t>(0);
                auto tmp2 = static_cast<int64_t>(0);
                auto tmp3 = static_cast<int64_t>(10);
                auto tmp4 = randint64_cpu(tmp0, tmp1, tmp2, tmp3);
                out_ptr0[static_cast<int64_t>(0L)] = tmp4;
            }
        }
    }
}
''')


#include "/tmp/torchinductor_sahanp/3b/c3bi5gk6mslf6u4iaqafhxm64z6u65e3eain4xlary5blqnvv6xx.h"
extern "C"  void kernel(const int64_t* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(10L); x0+=static_cast<int64_t>(1L))
        {
            {
                {
                    auto tmp0 = in_ptr0[static_cast<int64_t>(0L)];
                    auto tmp1 = x0;
                    auto tmp2 = c10::convert<int32_t>(tmp1);
                    auto tmp3 = randn_cpu(tmp0, tmp2);
                    out_ptr0[static_cast<int64_t>(x0)] = tmp3;
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
def triton_per_fused__log_softmax_nll_loss_forward_4(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 10
    R0_BLOCK: tl.constexpr = 16
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
    tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, other=0.0)
    tmp11 = tl.load(in_ptr1 + (0))
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, 1])
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tmp5 = tmp0 - tmp4
    tmp6 = tl_math.exp(tmp5)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(r0_mask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp13 = tl.full([1, 1], -100, tl.int64)
    tmp14 = tmp12 != tmp13
    tmp15 = tl.full([1, 1], 0, tl.int64)
    tmp16 = tl.where(tmp14, tmp12, tmp15)
    tmp17 = tl.full([XBLOCK, 1], 10, tl.int32)
    tmp18 = tmp16 + tmp17
    tmp19 = tmp16 < 0
    tmp20 = tl.where(tmp19, tmp18, tmp16)
    tl.device_assert((0 <= tmp20) & (tmp20 < 10), "index out of bounds: 0 <= tmp20 < 10")
    tmp22 = tl.load(in_ptr0 + (tmp20), None, eviction_policy='evict_last')
    tmp23 = tmp22 - tmp4
    tmp24 = tl_math.log(tmp10)
    tmp25 = tmp23 - tmp24
    tmp26 = -tmp25
    tmp27 = 0.0
    tmp28 = tl.where(tmp14, tmp26, tmp27)
    tmp29 = tmp14.to(tl.int64)
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp28 / tmp30
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp31, None)







def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        buf5 = empty_strided_cuda((1, ), (1, ), torch.float32)

        triton_red_fused_linalg_vector_norm_mul_sum_0_r0_numel = ((-1)*((s0*s1*s2) // 2)) + s0*s1*s2
        stream0 = get_raw_stream(0)
        triton_red_fused_linalg_vector_norm_mul_sum_0[grid(1)](arg3_1, buf1, buf5, 3, 64, 64, 1, 6144, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        buf0 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        buf2 = reinterpret_tensor(buf0, (1, ), (1, ), 0); del buf0
        buf3 = empty_strided_cuda((1, ), (1, ), torch.float32)
        buf13 = reinterpret_tensor(buf3, (), (), 0); del buf3

        triton_red_fused_add_clamp_min_div_eq_fill_linalg_vector_norm_mean_mul_sqrt_sub_sum_where_zeros_like_1_r0_numel = (s0*s1*s2) // 2
        stream0 = get_raw_stream(0)
        triton_red_fused_add_clamp_min_div_eq_fill_linalg_vector_norm_mean_mul_sqrt_sub_sum_where_zeros_like_1[grid(1)](buf2, buf13, arg3_1, buf1, buf5, 3, 64, 64, 1, 6144, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del arg3_1
        del buf1
    buf6 = empty_strided_cpu((2, ), (1, ), torch.int64)

    aten.randint.low_out(-9223372036854775808, 9223372036854775807, [2], out=buf6)
    buf7 = empty_strided_cpu((1, ), (1, ), torch.int64)
    cpp_fused_randint_2(buf6, buf7)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf8 = empty_strided_cuda((1, ), (1, ), torch.int64)
        buf8.copy_(buf7, False)
        del buf7
    buf9 = empty_strided_cpu((1, 10), (10, 1), torch.float32)
    cpp_fused_randn_3(buf6, buf9)
    del buf6
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf10 = empty_strided_cuda((1, 10), (10, 1), torch.float32)
        buf10.copy_(buf9, False)
        del buf9
        buf11 = reinterpret_tensor(buf5, (1, 1), (1, 1), 0); del buf5
        buf14 = reinterpret_tensor(buf11, (), (), 0); del buf11

        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax_nll_loss_forward_4[grid(1)](buf14, buf10, buf8, 1, 10, XBLOCK=1, num_warps=2, num_stages=1)
        del buf10
        del buf8
    return (buf2, buf13, buf14, )


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
