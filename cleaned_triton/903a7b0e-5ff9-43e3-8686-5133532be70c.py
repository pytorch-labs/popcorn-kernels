# AOT ID: ['64_inference']
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
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.bernoulli]
# Source node to ATen node mapping:
#   x => inductor_lookup_seed_default, inductor_random_default
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, %arg0_1, 1, 1, 1], %inductor_lookup_seed_default, rand), kwargs = {})
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




# kernel path: /tmp/torchinductor_sahanp/n5/cn5ik53hy6xgf6ohthu3qg4urohcljlroq7sjkyvbtxxgfyv3pbq.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.bernoulli, aten._to_copy, aten.div, aten.mul]
# Source node to ATen node mapping:
#   x => convert_element_type, div, lt_2, mul_13
# Graph fragment:
#   %lt_2 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%inductor_random_default, 0.5), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt_2, torch.float32), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%convert_element_type, 0.5), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg4_1, %div), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_bernoulli_div_mul_1(in_ptr0, in_ptr1, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // ks0
    tmp0 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = 0.5
    tmp3 = tmp1 < tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 2.0
    tmp6 = tmp4 * tmp5
    tmp7 = tmp0 * tmp6
    tl.store(out_ptr0 + (x2), tmp7, xmask)




# kernel path: /tmp/torchinductor_sahanp/rv/crvfegtgujjhiybpstafpexfkm6zfqrqpellnoiwdcnidl3gdhdm.py
# Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.bernoulli, aten._to_copy, aten.div, aten.mul, aten.avg_pool3d]
# Source node to ATen node mapping:
#   x => convert_element_type, div, lt_2, mul_13
#   x_1 => avg_pool3d
# Graph fragment:
#   %lt_2 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%inductor_random_default, 0.5), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt_2, torch.float32), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%convert_element_type, 0.5), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg4_1, %div), kwargs = {})
#   %avg_pool3d : [num_users=4] = call_function[target=torch.ops.aten.avg_pool3d.default](args = (%mul_13, [2, 2, 2], [2, 2, 2]), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_avg_pool3d_bernoulli_div_mul_2(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, ks6, ks7, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks1)
    x2 = ((xindex // ks2) % ks3)
    x3 = xindex // ks4
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 2*ks7*x1 + 2*ks6*ks7*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 2*ks7*x1 + 2*ks6*ks7*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (ks7 + 2*x0 + 2*ks7*x1 + 2*ks6*ks7*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (1 + ks7 + 2*x0 + 2*ks7*x1 + 2*ks6*ks7*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (2*x0 + ks6*ks7 + 2*ks7*x1 + 2*ks6*ks7*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (1 + 2*x0 + ks6*ks7 + 2*ks7*x1 + 2*ks6*ks7*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (ks7 + 2*x0 + ks6*ks7 + 2*ks7*x1 + 2*ks6*ks7*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (1 + ks7 + 2*x0 + ks6*ks7 + 2*ks7*x1 + 2*ks6*ks7*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp15 = 0.125
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (x4), tmp16, xmask)




# kernel path: /tmp/torchinductor_sahanp/fd/cfdq6hptazosdr5vpxhqie7y3famqicslk6kjcruawfmzudxmhtp.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_3 => clone
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_3(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks1)
    x2 = xindex // ks2
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + ks3*ks4*x2 + ks3*ks4*ks5*x1), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3), tmp0, xmask)




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
                auto tmp0 = c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(2L));
                auto tmp1 = c10::convert<int64_t>(tmp0);
                out_ptr1[static_cast<int64_t>(0L)] = tmp1;
            }
        }
    }
    {
        {
            {
                auto tmp0 = static_cast<int64_t>(10);
                out_ptr2[static_cast<int64_t>(0L)] = tmp0;
            }
        }
    }
}
''')





def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    s3 = arg3_1
    assert_size_stride(arg4_1, (1, s0, s1, s2, s3), (s0*s1*s2*s3, s1*s2*s3, s2*s3, s3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf0)
        buf1 = empty_strided_cuda((1, s0, 1, 1, 1), (s0, 1, s0, s0, s0), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.bernoulli]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bernoulli_0[grid(s0)](buf0, buf1, 0, 10, XBLOCK=16, num_warps=1, num_stages=1)
        del buf0
        ps0 = s1*s2*s3
        buf2 = empty_strided_cuda((1, s0, s1, s2, s3), (s0*s1*s2*s3, s1*s2*s3, s2*s3, s3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.bernoulli, aten._to_copy, aten.div, aten.mul]
        triton_poi_fused__to_copy_bernoulli_div_mul_1_xnumel = s0*s1*s2*s3
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_bernoulli_div_mul_1[grid(triton_poi_fused__to_copy_bernoulli_div_mul_1_xnumel)](arg4_1, buf1, buf2, 4096, 40960, XBLOCK=512, num_warps=4, num_stages=1)
        del arg4_1
        del buf1
        ps1 = s3 // 2
        ps2 = s2 // 2
        ps3 = (s2 // 2)*(s3 // 2)
        ps4 = s1 // 2
        ps5 = (s1 // 2)*(s2 // 2)*(s3 // 2)
        buf3 = empty_strided_cuda((1, s0, s1 // 2, s2 // 2, s3 // 2), (s0*(s1 // 2)*(s2 // 2)*(s3 // 2), (s1 // 2)*(s2 // 2)*(s3 // 2), (s2 // 2)*(s3 // 2), s3 // 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.bernoulli, aten._to_copy, aten.div, aten.mul, aten.avg_pool3d]
        triton_poi_fused__to_copy_avg_pool3d_bernoulli_div_mul_2_xnumel = s0*(s1 // 2)*(s2 // 2)*(s3 // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool3d_bernoulli_div_mul_2[grid(triton_poi_fused__to_copy_avg_pool3d_bernoulli_div_mul_2_xnumel)](buf2, buf3, 8, 8, 64, 8, 512, 16, 16, 16, 5120, XBLOCK=128, num_warps=4, num_stages=1)
        del buf2
        ps6 = s0*(s2 // 2)*(s3 // 2)
        buf4 = empty_strided_cuda((s1 // 2, 1, s0, s2 // 2, s3 // 2), (s0*(s2 // 2)*(s3 // 2), 1, (s2 // 2)*(s3 // 2), s3 // 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_3_xnumel = s0*(s1 // 2)*(s2 // 2)*(s3 // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3[grid(triton_poi_fused_clone_3_xnumel)](buf3, buf4, 64, 10, 640, 8, 8, 8, 5120, XBLOCK=128, num_warps=4, num_stages=1)
        del buf3
    buf5 = empty_strided_cpu((1, ), (1, ), torch.int64)
    # Topologically Sorted Source Nodes: [], Original ATen: []
    aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf5)
    buf6 = empty_strided_cpu((1, 10), (10, 1), torch.int64)
    buf7 = empty_strided_cpu((1, ), (1, ), torch.int64)
    buf8 = empty_strided_cpu((1, ), (1, ), torch.int64)
    cpp_fused_full_randint_4(buf5, buf6, buf7, buf8, s0, s1)
    return (reinterpret_tensor(buf4, (s1 // 2, 1, s0*(s2 // 2)*(s3 // 2)), (s0*(s2 // 2)*(s3 // 2), s0*(s2 // 2)*(s3 // 2), 1), 0), buf6, buf7, buf8, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 10
    arg1_1 = 16
    arg2_1 = 16
    arg3_1 = 16
    arg4_1 = rand_strided((1, 10, 16, 16, 16), (40960, 4096, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
