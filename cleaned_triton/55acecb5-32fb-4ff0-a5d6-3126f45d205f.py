# AOT ID: ['117_inference']
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


# kernel path: /tmp/torchinductor_sahanp/ro/cro5mb3is6yiy4xpefrmvgkpn77xd4mkmesvmmbjaorefvoop2n2.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.rand]
# Source node to ATen node mapping:
#   x_1 => inductor_lookup_seed_default, inductor_random_default
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, %arg0_1, 2], %inductor_lookup_seed_default, rand), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_rand_0(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (x0), tmp2, xmask)




# kernel path: /tmp/torchinductor_sahanp/i6/ci6l3tloiagcfamsjar7zczwh7konqm3tv3mhnet7foin5uxqehg.py
# Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.replication_pad2d, aten.fractional_max_pool2d, aten.sigmoid]
# Source node to ATen node mapping:
#   x => _unsafe_index, _unsafe_index_1
#   x_1 => fractional_max_pool2d
#   x_2 => sigmoid
# Graph fragment:
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg3_1, [None, None, %clamp_max, None]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index, [None, None, None, %clamp_max_1]), kwargs = {})
#   %fractional_max_pool2d : [num_users=1] = call_function[target=torch.ops.aten.fractional_max_pool2d.default](args = (%_unsafe_index_1, [2, 2], [14, 14], %inductor_random_default), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%getitem,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_fractional_max_pool2d_replication_pad2d_sigmoid_1(in_out_ptr0, in_ptr0, in_ptr1, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex // 196
    x1 = ((xindex // 14) % 14)
    x0 = (xindex % 14)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr0 + (1 + 2*x2), xmask, eviction_policy='evict_last')
    tmp1 = (2 + ks0) / 13
    tmp2 = tmp1.to(tl.float32)
    tmp3 = x1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp4 + tmp0
    tmp6 = tmp5 * tmp2
    tmp7 = libdevice.floor(tmp6)
    tmp8 = tmp0 * tmp2
    tmp9 = libdevice.floor(tmp8)
    tmp10 = tmp7 - tmp9
    tmp11 = tmp10.to(tl.int64)
    tmp12 = tl.full([1], 13, tl.int64)
    tmp13 = tmp4 < tmp12
    tmp14 = 2 + ks0
    tmp15 = tl.where(tmp13, tmp11, tmp14)
    tmp16 = 4 + ks0
    tmp17 = tmp15 + tmp16
    tmp18 = tmp15 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp15)
    tl.device_assert(((0 <= tmp19) & (tmp19 < 4 + ks0)) | ~(xmask), "index out of bounds: 0 <= tmp19 < 4 + ks0")
    tmp22 = (2 + ks1) / 13
    tmp23 = tmp22.to(tl.float32)
    tmp24 = x0
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp25 + tmp21
    tmp27 = tmp26 * tmp23
    tmp28 = libdevice.floor(tmp27)
    tmp29 = tmp21 * tmp23
    tmp30 = libdevice.floor(tmp29)
    tmp31 = tmp28 - tmp30
    tmp32 = tmp31.to(tl.int64)
    tmp33 = tmp25 < tmp12
    tmp34 = 2 + ks1
    tmp35 = tl.where(tmp33, tmp32, tmp34)
    tmp36 = 4 + ks1
    tmp37 = tmp35 + tmp36
    tmp38 = tmp35 < 0
    tmp39 = tl.where(tmp38, tmp37, tmp35)
    tl.device_assert(((0 <= tmp39) & (tmp39 < 4 + ks1)) | ~(xmask), "index out of bounds: 0 <= tmp39 < 4 + ks1")
    tmp41 = tl.load(in_ptr1 + (ks1*(((-1) + ks0) * (((-1) + ks0) <= (((0) * ((0) >= ((-2) + tmp19)) + ((-2) + tmp19) * (((-2) + tmp19) > (0))))) + (((0) * ((0) >= ((-2) + tmp19)) + ((-2) + tmp19) * (((-2) + tmp19) > (0)))) * ((((0) * ((0) >= ((-2) + tmp19)) + ((-2) + tmp19) * (((-2) + tmp19) > (0)))) < ((-1) + ks0))) + ks0*ks1*x2 + (((-1) + ks1) * (((-1) + ks1) <= (((0) * ((0) >= ((-2) + tmp39)) + ((-2) + tmp39) * (((-2) + tmp39) > (0))))) + (((0) * ((0) >= ((-2) + tmp39)) + ((-2) + tmp39) * (((-2) + tmp39) > (0)))) * ((((0) * ((0) >= ((-2) + tmp39)) + ((-2) + tmp39) * (((-2) + tmp39) > (0)))) < ((-1) + ks1)))), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr1 + (ks1*(((-1) + ks0) * (((-1) + ks0) <= (((0) * ((0) >= ((-2) + tmp19)) + ((-2) + tmp19) * (((-2) + tmp19) > (0))))) + (((0) * ((0) >= ((-2) + tmp19)) + ((-2) + tmp19) * (((-2) + tmp19) > (0)))) * ((((0) * ((0) >= ((-2) + tmp19)) + ((-2) + tmp19) * (((-2) + tmp19) > (0)))) < ((-1) + ks0))) + ks0*ks1*x2 + (((-1) + ks1) * (((-1) + ks1) <= (((0) * ((0) >= ((-1) + tmp39)) + ((-1) + tmp39) * (((-1) + tmp39) > (0))))) + (((0) * ((0) >= ((-1) + tmp39)) + ((-1) + tmp39) * (((-1) + tmp39) > (0)))) * ((((0) * ((0) >= ((-1) + tmp39)) + ((-1) + tmp39) * (((-1) + tmp39) > (0)))) < ((-1) + ks1)))), xmask, eviction_policy='evict_last')
    tmp43 = triton_helpers.maximum(tmp42, tmp41)
    tmp44 = tl.load(in_ptr1 + (ks1*(((-1) + ks0) * (((-1) + ks0) <= (((0) * ((0) >= ((-1) + tmp19)) + ((-1) + tmp19) * (((-1) + tmp19) > (0))))) + (((0) * ((0) >= ((-1) + tmp19)) + ((-1) + tmp19) * (((-1) + tmp19) > (0)))) * ((((0) * ((0) >= ((-1) + tmp19)) + ((-1) + tmp19) * (((-1) + tmp19) > (0)))) < ((-1) + ks0))) + ks0*ks1*x2 + (((-1) + ks1) * (((-1) + ks1) <= (((0) * ((0) >= ((-2) + tmp39)) + ((-2) + tmp39) * (((-2) + tmp39) > (0))))) + (((0) * ((0) >= ((-2) + tmp39)) + ((-2) + tmp39) * (((-2) + tmp39) > (0)))) * ((((0) * ((0) >= ((-2) + tmp39)) + ((-2) + tmp39) * (((-2) + tmp39) > (0)))) < ((-1) + ks1)))), xmask, eviction_policy='evict_last')
    tmp45 = triton_helpers.maximum(tmp44, tmp43)
    tmp46 = tl.load(in_ptr1 + (ks1*(((-1) + ks0) * (((-1) + ks0) <= (((0) * ((0) >= ((-1) + tmp19)) + ((-1) + tmp19) * (((-1) + tmp19) > (0))))) + (((0) * ((0) >= ((-1) + tmp19)) + ((-1) + tmp19) * (((-1) + tmp19) > (0)))) * ((((0) * ((0) >= ((-1) + tmp19)) + ((-1) + tmp19) * (((-1) + tmp19) > (0)))) < ((-1) + ks0))) + ks0*ks1*x2 + (((-1) + ks1) * (((-1) + ks1) <= (((0) * ((0) >= ((-1) + tmp39)) + ((-1) + tmp39) * (((-1) + tmp39) > (0))))) + (((0) * ((0) >= ((-1) + tmp39)) + ((-1) + tmp39) * (((-1) + tmp39) > (0)))) * ((((0) * ((0) >= ((-1) + tmp39)) + ((-1) + tmp39) * (((-1) + tmp39) > (0)))) < ((-1) + ks1)))), xmask, eviction_policy='evict_last')
    tmp47 = triton_helpers.maximum(tmp46, tmp45)
    tmp48 = tl.sigmoid(tmp47)
    tl.store(in_out_ptr0 + (x3), tmp48, xmask)




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
                    auto tmp3 = static_cast<int64_t>(1);
                    auto tmp4 = ks0;
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
                auto tmp0 = static_cast<int64_t>(196);
                out_ptr1[static_cast<int64_t>(0L)] = tmp0;
            }
        }
    }
    {
        {
            {
                auto tmp0 = in_ptr0[static_cast<int64_t>(1L)];
                auto tmp1 = static_cast<int32_t>(0);
                auto tmp2 = static_cast<int64_t>(1);
                auto tmp3 = static_cast<int64_t>(10);
                auto tmp4 = randint64_cpu(tmp0, tmp1, tmp2, tmp3);
                out_ptr2[static_cast<int64_t>(0L)] = tmp4;
            }
        }
    }
}
''')





def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf0)
        buf1 = empty_strided_cuda((1, s0, 2), (2*s0, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.rand]
        triton_poi_fused_rand_0_xnumel = 2*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_rand_0[grid(triton_poi_fused_rand_0_xnumel)](buf0, buf1, 0, 6, XBLOCK=8, num_warps=1, num_stages=1)
        del buf0
        buf2 = empty_strided_cuda((1, s0, 14, 14), (196*s0, 196, 14, 1), torch.float32)
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.replication_pad2d, aten.fractional_max_pool2d, aten.sigmoid]
        triton_poi_fused_fractional_max_pool2d_replication_pad2d_sigmoid_1_xnumel = 196*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_fractional_max_pool2d_replication_pad2d_sigmoid_1[grid(triton_poi_fused_fractional_max_pool2d_replication_pad2d_sigmoid_1_xnumel)](buf3, buf1, arg3_1, 32, 32, 588, XBLOCK=128, num_warps=4, num_stages=1)
        del arg3_1
        del buf1
    buf4 = empty_strided_cpu((2, ), (1, ), torch.int64)
    # Topologically Sorted Source Nodes: [], Original ATen: []
    aten.randint.low_out(-9223372036854775808, 9223372036854775807, [2], out=buf4)
    buf5 = empty_strided_cpu((1, 10), (10, 1), torch.int64)
    buf6 = empty_strided_cpu((1, ), (1, ), torch.int64)
    buf7 = empty_strided_cpu((1, ), (1, ), torch.int64)
    cpp_fused_full_randint_2(buf4, buf5, buf6, buf7, s0)
    return (reinterpret_tensor(buf3, (196, 1, s0), (1, 196*s0, 196), 0), buf5, buf6, buf7, )


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
