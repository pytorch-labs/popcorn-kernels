# AOT ID: ['12_inference']
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


# kernel path: /tmp/torchinductor_sahanp/aj/cajgq75regztxxkjkmstev6ecfscb4jacugltalggkdmxwicerpq.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   x_2 => amax, exp, sub_12, sum_1
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%unsqueeze_1, [-3], True), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze_1, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_12,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-3], True), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__softmax_0(in_ptr0, out_ptr0, out_ptr1, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (2*x0 + ks0*r0_1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + ks0*r0_1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = triton_helpers.maximum(tmp1, tmp0)
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = triton_helpers.maximum(_tmp4, tmp3)
        _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
    tmp4 = triton_helpers.max2(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp6 = tl.load(in_ptr0 + (2*x0 + ks0*r0_1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr0 + (1 + 2*x0 + ks0*r0_1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = triton_helpers.maximum(tmp7, tmp6)
        tmp9 = tmp8 - tmp4
        tmp10 = tl_math.exp(tmp9)
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp12, xmask)




# kernel path: /tmp/torchinductor_sahanp/3l/c3l6bi2n73cyux534bdingb5wfashjf5k5bqtsq4dohn5rj7xifd.py
# Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.max_unpool2d]
# Source node to ATen node mapping:
#   x_4 => full
# Graph fragment:
#   %full : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, %arg0_1, %sub_18, 1], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_unpool2d_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/3s/c3s7gzywvua4nm7jrtc7mgrrn42foh7vka3youehlvo4u3x3bwz3.py
# Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.max_unpool2d]
# Source node to ATen node mapping:
#   x_4 => index_put
# Graph fragment:
#   %index_put : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%view_2, [%view_1], %view_3), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_unpool2d_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2*((x0 % (ks0 // 2))) + ks0*(triton_helpers.div_floor_integer(x0,  ks0 // 2))), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*((x0 % (ks0 // 2))) + ks0*(triton_helpers.div_floor_integer(x0,  ks0 // 2))), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr1 + ((x0 % (ks0 // 2))), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr2 + ((x0 % (ks0 // 2))), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 > tmp0
    tmp3 = tl.full([1], 1, tl.int8)
    tmp4 = tl.full([1], 0, tl.int8)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = triton_helpers.maximum(tmp1, tmp0)
    tmp7 = tl.full([1], 2, tl.int32)
    tmp8 = tl.where((tmp5 < 0) != (tmp7 < 0), tl.where(tmp5 % tmp7 != 0, tmp5 // tmp7 - 1, tmp5 // tmp7), tmp5 // tmp7)
    tmp9 = tmp8 * tmp7
    tmp10 = tmp5 - tmp9
    tmp11 = tl.full([1], 0, tl.int64)
    tmp12 = tmp11 + tmp8
    tmp13 = 2*((x0 % (ks0 // 2)))
    tmp14 = tmp13 + tmp10
    tmp15 = ks0
    tmp16 = tmp12 * tmp15
    tmp17 = tmp16 + tmp14
    tmp18 = 2*(ks0 // 2)*(triton_helpers.div_floor_integer(x0,  ks0 // 2))
    tmp19 = tmp17 + tmp18
    tmp20 = 2*ks1*(ks0 // 2)
    tmp21 = tmp19 + tmp20
    tmp22 = tmp19 < 0
    tmp23 = tl.where(tmp22, tmp21, tmp19)
    tl.device_assert(((0 <= tmp23) & (tmp23 < 2*ks1*(ks0 // 2))) | ~(xmask), "index out of bounds: 0 <= tmp23 < 2*ks1*(ks0 // 2)")
    tmp26 = tmp6 - tmp25
    tmp27 = tl_math.exp(tmp26)
    tmp29 = tmp27 / tmp28
    tl.store(out_ptr0 + (tl.broadcast_to((tmp23 % (2*ks1*(ks0 // 2))), [XBLOCK])), tmp29, xmask)




# kernel path: /tmp/torchinductor_sahanp/lq/clqm2b5hiuvmj2ndc3ekxpkuunkdvke2zuwkccikrn4snmq6rqxh.py
# Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.permute]
# Source node to ATen node mapping:
#   x_5 => permute
# Graph fragment:
#   %permute : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%squeeze_3, [2, 0, 1]), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_permute_3(in_ptr0, out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 2*(ks2 // 2)*((((x0 + 2*x1*(ks2 // 2)) // (2*(ks2 // 2))) % ks1))), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp0, xmask)




#include "/tmp/torchinductor_sahanp/3b/c3bi5gk6mslf6u4iaqafhxm64z6u65e3eain4xlary5blqnvv6xx.h"
extern "C"  void kernel(const int64_t* in_ptr0,
                       int64_t* out_ptr0,
                       int64_t* out_ptr1,
                       int64_t* out_ptr2,
                       const int64_t ks0,
                       const int64_t ks1)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(ks0); x0+=static_cast<int64_t>(1L))
        {
            {
                {
                    auto tmp0 = in_ptr0[static_cast<int64_t>(0L)];
                    auto tmp1 = x0;
                    auto tmp2 = c10::convert<int32_t>(tmp1);
                    auto tmp3 = static_cast<int64_t>(1);
                    auto tmp4 = ks1;
                    auto tmp5 = c10::convert<int64_t>(tmp4);
                    auto tmp6 = randint64_cpu(tmp0, tmp2, tmp3, tmp5);
                    out_ptr0[static_cast<int64_t>(x0)] = tmp6;
                }
            }
        }
    }
    {
        {
            {
                auto tmp0 = ks0;
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
                auto tmp3 = ks0;
                auto tmp4 = c10::convert<int64_t>(tmp3);
                auto tmp5 = randint64_cpu(tmp0, tmp1, tmp2, tmp4);
                out_ptr2[static_cast<int64_t>(0L)] = tmp5;
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
        buf0 = empty_strided_cuda((1, 1, 1, s1 // 2), (s1 // 2, s1 // 2, s1 // 2, 1), torch.float32)
        buf1 = empty_strided_cuda((1, 1, 1, s1 // 2), (s1 // 2, s1 // 2, s1 // 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten._softmax]
        triton_red_fused__softmax_0_xnumel = s1 // 2
        stream0 = get_raw_stream(0)
        triton_red_fused__softmax_0[grid(triton_red_fused__softmax_0_xnumel)](arg2_1, buf0, buf1, 64, 32, 10, XBLOCK=1, R0_BLOCK=16, num_warps=2, num_stages=1)
        buf2 = empty_strided_cuda((1, s0, 2*(s1 // 2), 1), (2*s0*(s1 // 2), 2*(s1 // 2), 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.max_unpool2d]
        triton_poi_fused_max_unpool2d_1_xnumel = 2*s0*(s1 // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_unpool2d_1[grid(triton_poi_fused_max_unpool2d_1_xnumel)](buf2, 640, XBLOCK=256, num_warps=4, num_stages=1)
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.max_unpool2d]
        triton_poi_fused_max_unpool2d_2_xnumel = s0*(s1 // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_unpool2d_2[grid(triton_poi_fused_max_unpool2d_2_xnumel)](arg2_1, buf0, buf1, buf2, 64, 10, 320, XBLOCK=128, num_warps=4, num_stages=1)
        del arg2_1
        del buf0
        del buf1
        ps0 = 2*(s1 // 2)
        buf4 = empty_strided_cuda((2*(s1 // 2), 1, s0), (1, 2*s0*(s1 // 2), 2*(s1 // 2)), torch.float32)
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.permute]
        triton_poi_fused_permute_3_xnumel = 2*s0*(s1 // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_permute_3[grid(triton_poi_fused_permute_3_xnumel)](buf2, buf4, 64, 10, 64, 640, XBLOCK=128, num_warps=4, num_stages=1)
        del buf2
    buf5 = empty_strided_cpu((2, ), (1, ), torch.int64)
    # Topologically Sorted Source Nodes: [], Original ATen: []
    aten.randint.low_out(-9223372036854775808, 9223372036854775807, [2], out=buf5)
    buf6 = empty_strided_cpu((1, s1), (s1, 1), torch.int64)
    buf7 = empty_strided_cpu((1, ), (1, ), torch.int64)
    buf8 = empty_strided_cpu((1, ), (1, ), torch.int64)
    cpp_fused_full_randint_4(buf5, buf6, buf7, buf8, s1, s0)
    return (buf4, buf6, buf7, buf8, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 10
    arg1_1 = 64
    arg2_1 = rand_strided((1, 10, 64), (640, 64, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
