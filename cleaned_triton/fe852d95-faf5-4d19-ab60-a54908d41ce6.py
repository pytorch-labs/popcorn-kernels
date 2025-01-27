# AOT ID: ['58_inference']
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


# kernel path: /tmp/torchinductor_sahanp/ja/cjaksrnveguifsw7hosc4dbahl5kf5552j7owvla6hh5wibxcy5q.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.bernoulli]
# Source node to ATen node mapping:
#   x_2 => inductor_lookup_seed_default, inductor_random_default_1
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, %arg0_1, 1, 1, 1], %inductor_lookup_seed_default, rand), kwargs = {})
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




# kernel path: /tmp/torchinductor_sahanp/qt/cqtdnap5nala2y6b6ml3u75vhlciskjsb5gkhm6mmg5yngaqo5wx.py
# Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.bernoulli]
# Source node to ATen node mapping:
#   x_6 => inductor_lookup_seed_default_1, inductor_random_default
# Graph fragment:
#   %inductor_lookup_seed_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 1), kwargs = {})
#   %inductor_random_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, %arg0_1, 1, 1, 1], %inductor_lookup_seed_default_1, rand), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_bernoulli_1(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (x0), tmp2, xmask)




# kernel path: /tmp/torchinductor_sahanp/sg/csgkerzdqxkby3siufsgu4ffv5aklm35aw6bhe54todfjpifjlch.py
# Topologically Sorted Source Nodes: [log_probs], Original ATen: [aten._log_softmax]
# Source node to ATen node mapping:
#   log_probs => amax, exp, sub_46, sum_1
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%squeeze_3, [1], True), kwargs = {})
#   %sub_46 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%squeeze_3, %amax), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_46,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__log_softmax_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp14 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (ks0*r0_1 + (tl.where((-1) + ks0 + ((-1)*tl_math.abs(1 + ((-1)*ks0) + tl_math.abs((-2) + (tl.where(3 + ks0 + ((-1)*tl_math.abs(3 + ks0 + ((-1)*tl_math.abs((-1) + x0)))) < 0, 7 + ((-1)*tl_math.abs(3 + ks0 + ((-1)*tl_math.abs((-1) + x0)))) + 2*ks0, 3 + ks0 + ((-1)*tl_math.abs(3 + ks0 + ((-1)*tl_math.abs((-1) + x0))))))))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks0) + tl_math.abs((-2) + (tl.where(3 + ks0 + ((-1)*tl_math.abs(3 + ks0 + ((-1)*tl_math.abs((-1) + x0)))) < 0, 7 + ((-1)*tl_math.abs(3 + ks0 + ((-1)*tl_math.abs((-1) + x0)))) + 2*ks0, 3 + ks0 + ((-1)*tl_math.abs(3 + ks0 + ((-1)*tl_math.abs((-1) + x0))))))))) + 2*ks0, (-1) + ks0 + ((-1)*tl_math.abs(1 + ((-1)*ks0) + tl_math.abs((-2) + (tl.where(3 + ks0 + ((-1)*tl_math.abs(3 + ks0 + ((-1)*tl_math.abs((-1) + x0)))) < 0, 7 + ((-1)*tl_math.abs(3 + ks0 + ((-1)*tl_math.abs((-1) + x0)))) + 2*ks0, 3 + ks0 + ((-1)*tl_math.abs(3 + ks0 + ((-1)*tl_math.abs((-1) + x0)))))))))))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.5
        tmp3 = tmp1 < tmp2
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 2.0
        tmp6 = tmp4 * tmp5
        tmp7 = tmp0 * tmp6
        tmp9 = tmp8 < tmp2
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tmp10 * tmp5
        tmp12 = tmp7 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
        tmp15 = triton_helpers.maximum(_tmp14, tmp13)
        _tmp14 = tl.where(r0_mask & xmask, tmp15, _tmp14)
    tmp14 = triton_helpers.max2(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp14, xmask)
    _tmp32 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp16 = tl.load(in_ptr0 + (ks0*r0_1 + (tl.where((-1) + ks0 + ((-1)*tl_math.abs(1 + ((-1)*ks0) + tl_math.abs((-2) + (tl.where(3 + ks0 + ((-1)*tl_math.abs(3 + ks0 + ((-1)*tl_math.abs((-1) + x0)))) < 0, 7 + ((-1)*tl_math.abs(3 + ks0 + ((-1)*tl_math.abs((-1) + x0)))) + 2*ks0, 3 + ks0 + ((-1)*tl_math.abs(3 + ks0 + ((-1)*tl_math.abs((-1) + x0))))))))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks0) + tl_math.abs((-2) + (tl.where(3 + ks0 + ((-1)*tl_math.abs(3 + ks0 + ((-1)*tl_math.abs((-1) + x0)))) < 0, 7 + ((-1)*tl_math.abs(3 + ks0 + ((-1)*tl_math.abs((-1) + x0)))) + 2*ks0, 3 + ks0 + ((-1)*tl_math.abs(3 + ks0 + ((-1)*tl_math.abs((-1) + x0))))))))) + 2*ks0, (-1) + ks0 + ((-1)*tl_math.abs(1 + ((-1)*ks0) + tl_math.abs((-2) + (tl.where(3 + ks0 + ((-1)*tl_math.abs(3 + ks0 + ((-1)*tl_math.abs((-1) + x0)))) < 0, 7 + ((-1)*tl_math.abs(3 + ks0 + ((-1)*tl_math.abs((-1) + x0)))) + 2*ks0, 3 + ks0 + ((-1)*tl_math.abs(3 + ks0 + ((-1)*tl_math.abs((-1) + x0)))))))))))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp24 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp18 = 0.5
        tmp19 = tmp17 < tmp18
        tmp20 = tmp19.to(tl.float32)
        tmp21 = 2.0
        tmp22 = tmp20 * tmp21
        tmp23 = tmp16 * tmp22
        tmp25 = tmp24 < tmp18
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp26 * tmp21
        tmp28 = tmp23 * tmp27
        tmp29 = tmp28 - tmp14
        tmp30 = tl_math.exp(tmp29)
        tmp31 = tl.broadcast_to(tmp30, [XBLOCK, R0_BLOCK])
        tmp33 = _tmp32 + tmp31
        _tmp32 = tl.where(r0_mask & xmask, tmp33, _tmp32)
    tmp32 = tl.sum(_tmp32, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp32, xmask)




# kernel path: /tmp/torchinductor_sahanp/4u/c4ulkgqsuiow6x7mpyyqt3sq4rtirorlbs7knzg3u6mjvp5nq3ik.py
# Topologically Sorted Source Nodes: [log_probs], Original ATen: [aten._log_softmax]
# Source node to ATen node mapping:
#   log_probs => log, sub_46, sub_47
# Graph fragment:
#   %sub_46 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%squeeze_3, %amax), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_1,), kwargs = {})
#   %sub_47 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_46, %log), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__log_softmax_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (ks1*x1 + (tl.where((-1) + ks1 + ((-1)*tl_math.abs(1 + ((-1)*ks1) + tl_math.abs((-2) + (tl.where(3 + ks1 + ((-1)*tl_math.abs(3 + ks1 + ((-1)*tl_math.abs((-1) + x0)))) < 0, 7 + ((-1)*tl_math.abs(3 + ks1 + ((-1)*tl_math.abs((-1) + x0)))) + 2*ks1, 3 + ks1 + ((-1)*tl_math.abs(3 + ks1 + ((-1)*tl_math.abs((-1) + x0))))))))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks1) + tl_math.abs((-2) + (tl.where(3 + ks1 + ((-1)*tl_math.abs(3 + ks1 + ((-1)*tl_math.abs((-1) + x0)))) < 0, 7 + ((-1)*tl_math.abs(3 + ks1 + ((-1)*tl_math.abs((-1) + x0)))) + 2*ks1, 3 + ks1 + ((-1)*tl_math.abs(3 + ks1 + ((-1)*tl_math.abs((-1) + x0))))))))) + 2*ks1, (-1) + ks1 + ((-1)*tl_math.abs(1 + ((-1)*ks1) + tl_math.abs((-2) + (tl.where(3 + ks1 + ((-1)*tl_math.abs(3 + ks1 + ((-1)*tl_math.abs((-1) + x0)))) < 0, 7 + ((-1)*tl_math.abs(3 + ks1 + ((-1)*tl_math.abs((-1) + x0)))) + 2*ks1, 3 + ks1 + ((-1)*tl_math.abs(3 + ks1 + ((-1)*tl_math.abs((-1) + x0)))))))))))), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = 0.5
    tmp3 = tmp1 < tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 2.0
    tmp6 = tmp4 * tmp5
    tmp7 = tmp0 * tmp6
    tmp9 = tmp8 < tmp2
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp10 * tmp5
    tmp12 = tmp7 * tmp11
    tmp14 = tmp12 - tmp13
    tmp16 = tl_math.log(tmp15)
    tmp17 = tmp14 - tmp16
    tl.store(out_ptr0 + (x2), tmp17, xmask)




#include "/tmp/torchinductor_sahanp/3b/c3bi5gk6mslf6u4iaqafhxm64z6u65e3eain4xlary5blqnvv6xx.h"
extern "C"  void kernel(const int64_t* in_ptr0,
                       int64_t* out_ptr0,
                       int64_t* out_ptr1,
                       int64_t* out_ptr2,
                       const int64_t ks0)
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
                    auto tmp3 = static_cast<int64_t>(0);
                    auto tmp4 = static_cast<int64_t>(10);
                    auto tmp5 = randint64_cpu(tmp0, tmp2, tmp3, tmp4);
                    out_ptr0[static_cast<int64_t>(x0)] = tmp5;
                }
            }
        }
    }
    {
        {
            {
                auto tmp0 = 6L + ks0;
                auto tmp1 = c10::convert<int64_t>(tmp0);
                out_ptr1[static_cast<int64_t>(0L)] = tmp1;
            }
        }
    }
    {
        {
            {
                auto tmp0 = in_ptr0[static_cast<int64_t>(1L)];
                auto tmp1 = static_cast<int32_t>(0);
                auto tmp2 = static_cast<int64_t>(1);
                auto tmp3 = static_cast<int64_t>(11);
                auto tmp4 = randint64_cpu(tmp0, tmp1, tmp2, tmp3);
                out_ptr2[static_cast<int64_t>(0L)] = tmp4;
            }
        }
    }
}
''')





def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    assert_size_stride(arg2_1, (1, s0, s1), (s0*s1, s1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [2], out=buf0)
        buf1 = empty_strided_cuda((1, s0, 1, 1, 1), (s0, 1, s0, s0, s0), torch.float32)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.bernoulli]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bernoulli_0[grid(s0)](buf0, buf1, 0, 10, XBLOCK=16, num_warps=1, num_stages=1)
        buf2 = empty_strided_cuda((1, s0, 1, 1, 1), (s0, 1, s0, s0, s0), torch.float32)
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.bernoulli]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bernoulli_1[grid(s0)](buf0, buf2, 1, 10, XBLOCK=16, num_warps=1, num_stages=1)
        del buf0
        buf3 = empty_strided_cuda((1, 1, 6 + s1), (6 + s1, 6 + s1, 1), torch.float32)
        buf4 = empty_strided_cuda((1, 1, 6 + s1), (6 + s1, 6 + s1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [log_probs], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_2_xnumel = 6 + s1
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax_2[grid(triton_red_fused__log_softmax_2_xnumel)](arg2_1, buf1, buf2, buf3, buf4, 20, 26, 10, XBLOCK=1, R0_BLOCK=16, num_warps=2, num_stages=1)
        ps0 = 6 + s1
        buf5 = empty_strided_cuda((1, s0, 6 + s1), (6*s0 + s0*s1, 6 + s1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [log_probs], Original ATen: [aten._log_softmax]
        triton_poi_fused__log_softmax_3_xnumel = 6*s0 + s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused__log_softmax_3[grid(triton_poi_fused__log_softmax_3_xnumel)](arg2_1, buf1, buf2, buf3, buf4, buf5, 26, 20, 260, XBLOCK=128, num_warps=4, num_stages=1)
        del arg2_1
        del buf1
        del buf2
        del buf3
        del buf4
    buf6 = empty_strided_cpu((2, ), (1, ), torch.int64)
    # Topologically Sorted Source Nodes: [], Original ATen: []
    aten.randint.low_out(-9223372036854775808, 9223372036854775807, [2], out=buf6)
    buf7 = empty_strided_cpu((1, 10), (10, 1), torch.int64)
    buf8 = empty_strided_cpu((1, ), (1, ), torch.int64)
    buf9 = empty_strided_cpu((1, ), (1, ), torch.int64)
    cpp_fused_full_randint_4(buf6, buf7, buf8, buf9, s1)
    return (buf5, buf7, buf8, buf9, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 10
    arg1_1 = 20
    arg2_1 = rand_strided((1, 10, 20), (200, 20, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
