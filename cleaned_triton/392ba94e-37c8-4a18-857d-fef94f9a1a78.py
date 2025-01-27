# AOT ID: ['37_inference']
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


# kernel path: /tmp/torchinductor_sahanp/bb/cbbyvpqws6xi6eu76uwplmqymygere3uqcnpam473m6m6abfxfnh.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.reflection_pad2d]
# Source node to ATen node mapping:
#   x_1 => _unsafe_index, _unsafe_index_1
# Graph fragment:
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%view_1, [None, None, %sub_8, None]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index, [None, None, None, %sub_14]), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_reflection_pad2d_0(in_ptr0, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 36)
    x1 = ((xindex // 36) % 36)
    x2 = xindex // 1296
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (1023 + ((-1)*tl_math.abs((-31) + tl_math.abs((-2) + x0))) + ((-32)*tl_math.abs((-31) + tl_math.abs((-2) + x1))) + 1024*(((x2 // 2) % (ks0 // 2))) + 1024*(ks0 // 2)*((x2 % 2))), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3), tmp0, xmask)




#include "/tmp/torchinductor_sahanp/3b/c3bi5gk6mslf6u4iaqafhxm64z6u65e3eain4xlary5blqnvv6xx.h"
extern "C"  void kernel(const int64_t* in_ptr0,
                       int64_t* out_ptr0,
                       int64_t* out_ptr1,
                       int64_t* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(5L); x0+=static_cast<int64_t>(1L))
        {
            {
                {
                    auto tmp0 = in_ptr0[static_cast<int64_t>(0L)];
                    auto tmp1 = x0;
                    auto tmp2 = c10::convert<int32_t>(tmp1);
                    auto tmp3 = static_cast<int64_t>(1);
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
                auto tmp0 = static_cast<int64_t>(5);
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
                auto tmp3 = static_cast<int64_t>(6);
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
    assert_size_stride(arg3_1, (1, s0, 32, 32), (1024*s0, 1024, 32, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, s0, 36, 36), (1296*s0, 1296, 36, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.reflection_pad2d]
        triton_poi_fused_reflection_pad2d_0_xnumel = 1296*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_0[grid(triton_poi_fused_reflection_pad2d_0_xnumel)](arg3_1, buf0, 4, 5184, XBLOCK=256, num_warps=4, num_stages=1)
        del arg3_1
        # Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten.reflection_pad2d, aten._adaptive_avg_pool2d]
        buf1 = torch.ops.aten._adaptive_avg_pool2d.default(buf0, [5, 5])
        del buf0
        buf2 = buf1
        del buf1
    buf3 = empty_strided_cpu((2, ), (1, ), torch.int64)
    # Topologically Sorted Source Nodes: [], Original ATen: []
    aten.randint.low_out(-9223372036854775808, 9223372036854775807, [2], out=buf3)
    buf4 = empty_strided_cpu((1, 5), (5, 1), torch.int64)
    buf5 = empty_strided_cpu((1, ), (1, ), torch.int64)
    buf6 = empty_strided_cpu((1, ), (1, ), torch.int64)
    cpp_fused_full_randint_1(buf3, buf4, buf5, buf6)
    return (reinterpret_tensor(buf2, ((25*s0) // (2*(s0 // 2)), 1, 2*(s0 // 2)), (1, 2*((25*s0) // (2*(s0 // 2)))*(s0 // 2), (25*s0) // (2*(s0 // 2))), 0), buf4, buf5, buf6, 2*(s0 // 2), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 4
    arg1_1 = 32
    arg2_1 = 32
    arg3_1 = rand_strided((1, 4, 32, 32), (4096, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
