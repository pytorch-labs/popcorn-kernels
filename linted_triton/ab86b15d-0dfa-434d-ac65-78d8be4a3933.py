# AOT ID: ['53_inference']
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
                       int64_t* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(10L); x0+=static_cast<int64_t>(1L))
        {
            {
                {
                    auto tmp0 = in_ptr0[static_cast<int64_t>(1L)];
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
}
''')


#include "/tmp/torchinductor_sahanp/3b/c3bi5gk6mslf6u4iaqafhxm64z6u65e3eain4xlary5blqnvv6xx.h"
extern "C"  void kernel(const int64_t* in_ptr0,
                       int64_t* out_ptr0)
{
    {
        {
            {
                auto tmp0 = in_ptr0[static_cast<int64_t>(2L)];
                auto tmp1 = static_cast<int32_t>(0);
                auto tmp2 = static_cast<int64_t>(1);
                auto tmp3 = static_cast<int64_t>(10);
                auto tmp4 = randint64_cpu(tmp0, tmp1, tmp2, tmp3);
                out_ptr0[static_cast<int64_t>(0L)] = tmp4;
            }
        }
    }
}
''')


# kernel path: /tmp/torchinductor_sahanp/36/c363haaehvczooczmoinci6aneedmkn6wlyb42jiw2tx7aaolanu.py
# Topologically Sorted Source Nodes: [x_6, x_7], Original ATen: [aten.constant_pad_nd, aten.relu]
# Source node to ATen node mapping:
#   x_6 => constant_pad_nd
#   x_7 => relu_1
# Graph fragment:
#   %constant_pad_nd : [num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view, [2, 2], 3.5), kwargs = {})
#   %relu_1 : [num_users=5] = call_function[target=torch.ops.aten.relu.default](args = (%constant_pad_nd,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_constant_pad_nd_relu_2(in_out_ptr0, in_ptr0, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = (-2) + x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = 72 + 12*ks1 + 12*ks2 + 36*ks0 + 2*ks1*ks2 + 6*ks0*ks1 + 6*ks0*ks2 + ks0*ks1*ks2
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (ks2*(((-1) + ks1) * (((-1) + ks1) <= (((0) * ((0) >= ((-2) + ((3 + ks1) * ((3 + ks1) <= (((0) * ((0) >= ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1))))) + ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) * (((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) > (0))))) + (((0) * ((0) >= ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1))))) + ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) * (((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) > (0)))) * ((((0) * ((0) >= ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1))))) + ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) * (((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) > (0)))) < (3 + ks1))))) + ((-2) + ((3 + ks1) * ((3 + ks1) <= (((0) * ((0) >= ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1))))) + ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) * (((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) > (0))))) + (((0) * ((0) >= ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1))))) + ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) * (((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) > (0)))) * ((((0) * ((0) >= ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1))))) + ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) * (((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) > (0)))) < (3 + ks1)))) * (((-2) + ((3 + ks1) * ((3 + ks1) <= (((0) * ((0) >= ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1))))) + ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) * (((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) > (0))))) + (((0) * ((0) >= ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1))))) + ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) * (((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) > (0)))) * ((((0) * ((0) >= ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1))))) + ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) * (((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) > (0)))) < (3 + ks1)))) > (0))))) + (((0) * ((0) >= ((-2) + ((3 + ks1) * ((3 + ks1) <= (((0) * ((0) >= ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1))))) + ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) * (((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) > (0))))) + (((0) * ((0) >= ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1))))) + ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) * (((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) > (0)))) * ((((0) * ((0) >= ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1))))) + ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) * (((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) > (0)))) < (3 + ks1))))) + ((-2) + ((3 + ks1) * ((3 + ks1) <= (((0) * ((0) >= ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1))))) + ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) * (((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) > (0))))) + (((0) * ((0) >= ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1))))) + ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) * (((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) > (0)))) * ((((0) * ((0) >= ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1))))) + ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) * (((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) > (0)))) < (3 + ks1)))) * (((-2) + ((3 + ks1) * ((3 + ks1) <= (((0) * ((0) >= ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1))))) + ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) * (((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) > (0))))) + (((0) * ((0) >= ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1))))) + ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) * (((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) > (0)))) * ((((0) * ((0) >= ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1))))) + ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) * (((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) > (0)))) < (3 + ks1)))) > (0)))) * ((((0) * ((0) >= ((-2) + ((3 + ks1) * ((3 + ks1) <= (((0) * ((0) >= ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1))))) + ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) * (((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) > (0))))) + (((0) * ((0) >= ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1))))) + ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) * (((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) > (0)))) * ((((0) * ((0) >= ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1))))) + ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) * (((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) > (0)))) < (3 + ks1))))) + ((-2) + ((3 + ks1) * ((3 + ks1) <= (((0) * ((0) >= ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1))))) + ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) * (((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) > (0))))) + (((0) * ((0) >= ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1))))) + ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) * (((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) > (0)))) * ((((0) * ((0) >= ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1))))) + ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) * (((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) > (0)))) < (3 + ks1)))) * (((-2) + ((3 + ks1) * ((3 + ks1) <= (((0) * ((0) >= ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1))))) + ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) * (((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) > (0))))) + (((0) * ((0) >= ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1))))) + ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) * (((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) > (0)))) * ((((0) * ((0) >= ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1))))) + ((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) * (((-1) + (((((-2) + x0) // (6 + ks2)) % (6 + ks1)))) > (0)))) < (3 + ks1)))) > (0)))) < ((-1) + ks1))) + ks1*ks2*(((-1) + ks0) * (((-1) + ks0) <= (((0) * ((0) >= ((-1) + (((((-2) + x0) // (36 + 6*ks1 + 6*ks2 + ks1*ks2)) % (2 + ks0))))) + ((-1) + (((((-2) + x0) // (36 + 6*ks1 + 6*ks2 + ks1*ks2)) % (2 + ks0)))) * (((-1) + (((((-2) + x0) // (36 + 6*ks1 + 6*ks2 + ks1*ks2)) % (2 + ks0)))) > (0))))) + (((0) * ((0) >= ((-1) + (((((-2) + x0) // (36 + 6*ks1 + 6*ks2 + ks1*ks2)) % (2 + ks0))))) + ((-1) + (((((-2) + x0) // (36 + 6*ks1 + 6*ks2 + ks1*ks2)) % (2 + ks0)))) * (((-1) + (((((-2) + x0) // (36 + 6*ks1 + 6*ks2 + ks1*ks2)) % (2 + ks0)))) > (0)))) * ((((0) * ((0) >= ((-1) + (((((-2) + x0) // (36 + 6*ks1 + 6*ks2 + ks1*ks2)) % (2 + ks0))))) + ((-1) + (((((-2) + x0) // (36 + 6*ks1 + 6*ks2 + ks1*ks2)) % (2 + ks0)))) * (((-1) + (((((-2) + x0) // (36 + 6*ks1 + 6*ks2 + ks1*ks2)) % (2 + ks0)))) > (0)))) < ((-1) + ks0))) + (((-1) + ks2) * (((-1) + ks2) <= (((0) * ((0) >= ((-2) + ((3 + ks2) * ((3 + ks2) <= (((0) * ((0) >= ((-1) + ((((-2) + x0) % (6 + ks2))))) + ((-1) + ((((-2) + x0) % (6 + ks2)))) * (((-1) + ((((-2) + x0) % (6 + ks2)))) > (0))))) + (((0) * ((0) >= ((-1) + ((((-2) + x0) % (6 + ks2))))) + ((-1) + ((((-2) + x0) % (6 + ks2)))) * (((-1) + ((((-2) + x0) % (6 + ks2)))) > (0)))) * ((((0) * ((0) >= ((-1) + ((((-2) + x0) % (6 + ks2))))) + ((-1) + ((((-2) + x0) % (6 + ks2)))) * (((-1) + ((((-2) + x0) % (6 + ks2)))) > (0)))) < (3 + ks2))))) + ((-2) + ((3 + ks2) * ((3 + ks2) <= (((0) * ((0) >= ((-1) + ((((-2) + x0) % (6 + ks2))))) + ((-1) + ((((-2) + x0) % (6 + ks2)))) * (((-1) + ((((-2) + x0) % (6 + ks2)))) > (0))))) + (((0) * ((0) >= ((-1) + ((((-2) + x0) % (6 + ks2))))) + ((-1) + ((((-2) + x0) % (6 + ks2)))) * (((-1) + ((((-2) + x0) % (6 + ks2)))) > (0)))) * ((((0) * ((0) >= ((-1) + ((((-2) + x0) % (6 + ks2))))) + ((-1) + ((((-2) + x0) % (6 + ks2)))) * (((-1) + ((((-2) + x0) % (6 + ks2)))) > (0)))) < (3 + ks2)))) * (((-2) + ((3 + ks2) * ((3 + ks2) <= (((0) * ((0) >= ((-1) + ((((-2) + x0) % (6 + ks2))))) + ((-1) + ((((-2) + x0) % (6 + ks2)))) * (((-1) + ((((-2) + x0) % (6 + ks2)))) > (0))))) + (((0) * ((0) >= ((-1) + ((((-2) + x0) % (6 + ks2))))) + ((-1) + ((((-2) + x0) % (6 + ks2)))) * (((-1) + ((((-2) + x0) % (6 + ks2)))) > (0)))) * ((((0) * ((0) >= ((-1) + ((((-2) + x0) % (6 + ks2))))) + ((-1) + ((((-2) + x0) % (6 + ks2)))) * (((-1) + ((((-2) + x0) % (6 + ks2)))) > (0)))) < (3 + ks2)))) > (0))))) + (((0) * ((0) >= ((-2) + ((3 + ks2) * ((3 + ks2) <= (((0) * ((0) >= ((-1) + ((((-2) + x0) % (6 + ks2))))) + ((-1) + ((((-2) + x0) % (6 + ks2)))) * (((-1) + ((((-2) + x0) % (6 + ks2)))) > (0))))) + (((0) * ((0) >= ((-1) + ((((-2) + x0) % (6 + ks2))))) + ((-1) + ((((-2) + x0) % (6 + ks2)))) * (((-1) + ((((-2) + x0) % (6 + ks2)))) > (0)))) * ((((0) * ((0) >= ((-1) + ((((-2) + x0) % (6 + ks2))))) + ((-1) + ((((-2) + x0) % (6 + ks2)))) * (((-1) + ((((-2) + x0) % (6 + ks2)))) > (0)))) < (3 + ks2))))) + ((-2) + ((3 + ks2) * ((3 + ks2) <= (((0) * ((0) >= ((-1) + ((((-2) + x0) % (6 + ks2))))) + ((-1) + ((((-2) + x0) % (6 + ks2)))) * (((-1) + ((((-2) + x0) % (6 + ks2)))) > (0))))) + (((0) * ((0) >= ((-1) + ((((-2) + x0) % (6 + ks2))))) + ((-1) + ((((-2) + x0) % (6 + ks2)))) * (((-1) + ((((-2) + x0) % (6 + ks2)))) > (0)))) * ((((0) * ((0) >= ((-1) + ((((-2) + x0) % (6 + ks2))))) + ((-1) + ((((-2) + x0) % (6 + ks2)))) * (((-1) + ((((-2) + x0) % (6 + ks2)))) > (0)))) < (3 + ks2)))) * (((-2) + ((3 + ks2) * ((3 + ks2) <= (((0) * ((0) >= ((-1) + ((((-2) + x0) % (6 + ks2))))) + ((-1) + ((((-2) + x0) % (6 + ks2)))) * (((-1) + ((((-2) + x0) % (6 + ks2)))) > (0))))) + (((0) * ((0) >= ((-1) + ((((-2) + x0) % (6 + ks2))))) + ((-1) + ((((-2) + x0) % (6 + ks2)))) * (((-1) + ((((-2) + x0) % (6 + ks2)))) > (0)))) * ((((0) * ((0) >= ((-1) + ((((-2) + x0) % (6 + ks2))))) + ((-1) + ((((-2) + x0) % (6 + ks2)))) * (((-1) + ((((-2) + x0) % (6 + ks2)))) > (0)))) < (3 + ks2)))) > (0)))) * ((((0) * ((0) >= ((-2) + ((3 + ks2) * ((3 + ks2) <= (((0) * ((0) >= ((-1) + ((((-2) + x0) % (6 + ks2))))) + ((-1) + ((((-2) + x0) % (6 + ks2)))) * (((-1) + ((((-2) + x0) % (6 + ks2)))) > (0))))) + (((0) * ((0) >= ((-1) + ((((-2) + x0) % (6 + ks2))))) + ((-1) + ((((-2) + x0) % (6 + ks2)))) * (((-1) + ((((-2) + x0) % (6 + ks2)))) > (0)))) * ((((0) * ((0) >= ((-1) + ((((-2) + x0) % (6 + ks2))))) + ((-1) + ((((-2) + x0) % (6 + ks2)))) * (((-1) + ((((-2) + x0) % (6 + ks2)))) > (0)))) < (3 + ks2))))) + ((-2) + ((3 + ks2) * ((3 + ks2) <= (((0) * ((0) >= ((-1) + ((((-2) + x0) % (6 + ks2))))) + ((-1) + ((((-2) + x0) % (6 + ks2)))) * (((-1) + ((((-2) + x0) % (6 + ks2)))) > (0))))) + (((0) * ((0) >= ((-1) + ((((-2) + x0) % (6 + ks2))))) + ((-1) + ((((-2) + x0) % (6 + ks2)))) * (((-1) + ((((-2) + x0) % (6 + ks2)))) > (0)))) * ((((0) * ((0) >= ((-1) + ((((-2) + x0) % (6 + ks2))))) + ((-1) + ((((-2) + x0) % (6 + ks2)))) * (((-1) + ((((-2) + x0) % (6 + ks2)))) > (0)))) < (3 + ks2)))) * (((-2) + ((3 + ks2) * ((3 + ks2) <= (((0) * ((0) >= ((-1) + ((((-2) + x0) % (6 + ks2))))) + ((-1) + ((((-2) + x0) % (6 + ks2)))) * (((-1) + ((((-2) + x0) % (6 + ks2)))) > (0))))) + (((0) * ((0) >= ((-1) + ((((-2) + x0) % (6 + ks2))))) + ((-1) + ((((-2) + x0) % (6 + ks2)))) * (((-1) + ((((-2) + x0) % (6 + ks2)))) > (0)))) * ((((0) * ((0) >= ((-1) + ((((-2) + x0) % (6 + ks2))))) + ((-1) + ((((-2) + x0) % (6 + ks2)))) * (((-1) + ((((-2) + x0) % (6 + ks2)))) > (0)))) < (3 + ks2)))) > (0)))) < ((-1) + ks2)))), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full([1], 0, tl.int32)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp9 = -tmp8
    tmp10 = tmp9 - tmp9
    tmp11 = tl_math.exp(tmp10)
    tmp12 = tmp11 / tmp11
    tmp13 = tl.full(tmp12.shape, 3.5, tmp12.dtype)
    tmp14 = tl.where(tmp5, tmp12, tmp13)
    tmp15 = tl.full([1], 0, tl.int32)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tl.store(in_out_ptr0 + (x0), tmp16, xmask)




#include "/tmp/torchinductor_sahanp/3b/c3bi5gk6mslf6u4iaqafhxm64z6u65e3eain4xlary5blqnvv6xx.h"
extern "C"  void kernel(const int64_t* in_ptr0,
                       int64_t* out_ptr0)
{
    {
        {
            {
                auto tmp0 = in_ptr0[static_cast<int64_t>(0L)];
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


# kernel path: /tmp/torchinductor_sahanp/av/cavhx64d4jaybarnqzvin22kmryuq7nwmwull2k2eluopuzdv6zw.py
# Topologically Sorted Source Nodes: [randn_like], Original ATen: [aten.randn_like]
# Source node to ATen node mapping:
#   randn_like => inductor_lookup_seed_default_3, inductor_random_default
# Graph fragment:
#   %inductor_lookup_seed_default_3 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default_1, 0), kwargs = {})
#   %inductor_random_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, %sym_size_int_4], %inductor_lookup_seed_default_3, randn), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_randn_like_4(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.randn(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (x0), tmp2, xmask)




# kernel path: /tmp/torchinductor_sahanp/oi/coi2iddvjkq2wbnst6wcmzu2bvflr75sxjpdkboa4ql5llxdle7o.py
# Topologically Sorted Source Nodes: [loss_multi_margin, loss_poisson, target_poisson], Original ATen: [aten.arange, aten.ne, aten.gather, aten.rsub, aten.add, aten.clamp_min, aten.scalar_tensor, aten.where, aten.mean, aten.exp, aten.abs, aten.mul, aten.sub]
# Source node to ATen node mapping:
#   loss_multi_margin => add_40, clamp_min_5, full_default, gather, iota_5, mean, ne_4, sub_37, where
#   loss_poisson => exp_1, mean_1, mul_49, sub_49
#   target_poisson => abs_1
# Graph fragment:
#   %iota_5 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (%floordiv_5,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %ne_4 : [num_users=1] = call_function[target=torch.ops.aten.ne.Tensor](args = (%iota_5, %unsqueeze_1), kwargs = {})
#   %gather : [num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%relu_1, 1, %unsqueeze_1), kwargs = {})
#   %sub_37 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %gather), kwargs = {})
#   %add_40 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub_37, %relu_1), kwargs = {})
#   %clamp_min_5 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_40, 0), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_4, %clamp_min_5, %full_default), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%where,), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%relu_1,), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%inductor_random_default,), kwargs = {})
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%abs_1, %relu_1), kwargs = {})
#   %sub_49 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%exp_1, %mul_49), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sub_49,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_abs_add_arange_clamp_min_exp_gather_mean_mul_ne_rsub_scalar_tensor_sub_where_5(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 3
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp22 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp32 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = r0_1 + 26*x0 + x0*((2*ks1*ks2 + ks0*ks1*ks2) // 3) + 4*ks1*x0 + 4*ks2*x0 + 12*ks0*x0 + 2*ks0*ks1*x0 + 2*ks0*ks2*x0
        tmp1 = 76 + 12*ks1 + 12*ks2 + 36*ks0 + 2*ks1*ks2 + 6*ks0*ks1 + 6*ks0*ks2 + ks0*ks1*ks2
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (tl.full([XBLOCK, R0_BLOCK], 0, tl.int32)), tmp2, eviction_policy='evict_last', other=0.0)
        tmp4 = r0_1 + 26*x0 + x0*((2*ks1*ks2 + ks0*ks1*ks2) // 3) + 4*ks1*x0 + 4*ks2*x0 + 12*ks0*x0 + 2*ks0*ks1*x0 + 2*ks0*ks2*x0
        tmp5 = tmp4 != tmp3
        tmp6 = tl.broadcast_to(76 + 12*ks1 + 12*ks2 + 36*ks0 + 2*ks1*ks2 + 6*ks0*ks1 + 6*ks0*ks2 + ks0*ks1*ks2, [XBLOCK, R0_BLOCK])
        tmp7 = tmp3 + tmp6
        tmp8 = tmp3 < 0
        tmp9 = tl.where(tmp8, tmp7, tmp3)
        tl.device_assert(((0 <= tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])) & (tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK]) < 76 + 12*ks1 + 12*ks2 + 36*ks0 + 2*ks1*ks2 + 6*ks0*ks1 + 6*ks0*ks2 + ks0*ks1*ks2)) | ~(r0_mask & tmp2 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK]) < 76 + 12*ks1 + 12*ks2 + 36*ks0 + 2*ks1*ks2 + 6*ks0*ks1 + 6*ks0*ks2 + ks0*ks1*ks2")
        tmp11 = tl.load(in_ptr1 + (tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = 1.0
        tmp13 = tmp12 - tmp11
        tmp14 = tl.load(in_ptr1 + (r0_1 + 26*x0 + x0*((2*ks1*ks2 + ks0*ks1*ks2) // 3) + 4*ks1*x0 + 4*ks2*x0 + 12*ks0*x0 + 2*ks0*ks1*x0 + 2*ks0*ks2*x0), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tmp13 + tmp14
        tmp16 = 0.0
        tmp17 = triton_helpers.maximum(tmp15, tmp16)
        tmp18 = tl.where(tmp5, tmp17, tmp16)
        tmp19 = tl.full(tmp18.shape, 0, tmp18.dtype)
        tmp20 = tl.where(tmp2, tmp18, tmp19)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(r0_mask & xmask, tmp23, _tmp22)
        tmp24 = tl_math.exp(tmp14)
        tmp25 = tl.load(in_ptr2 + (r0_1 + 26*x0 + x0*((2*ks1*ks2 + ks0*ks1*ks2) // 3) + 4*ks1*x0 + 4*ks2*x0 + 12*ks0*x0 + 2*ks0*ks1*x0 + 2*ks0*ks2*x0), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp26 = tl_math.abs(tmp25)
        tmp27 = tmp26 * tmp14
        tmp28 = tmp24 - tmp27
        tmp29 = tl.full(tmp28.shape, 0, tmp28.dtype)
        tmp30 = tl.where(tmp2, tmp28, tmp29)
        tmp31 = tl.broadcast_to(tmp30, [XBLOCK, R0_BLOCK])
        tmp33 = _tmp32 + tmp31
        _tmp32 = tl.where(r0_mask & xmask, tmp33, _tmp32)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tmp32 = tl.sum(_tmp32, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp22, xmask)
    tl.store(out_ptr1 + (x0), tmp32, xmask)




# kernel path: /tmp/torchinductor_sahanp/fc/cfcquxdsmzjsrcpgebic2daoupjphqc63dgg2btk6ds4nxo2z5qt.py
# Topologically Sorted Source Nodes: [loss_multi_margin], Original ATen: [aten.arange, aten.ne, aten.gather, aten.rsub, aten.add, aten.clamp_min, aten.scalar_tensor, aten.where, aten.mean]
# Source node to ATen node mapping:
#   loss_multi_margin => add_40, clamp_min_5, full_default, gather, iota_5, mean, ne_4, sub_37, where
# Graph fragment:
#   %iota_5 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (%floordiv_5,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %ne_4 : [num_users=1] = call_function[target=torch.ops.aten.ne.Tensor](args = (%iota_5, %unsqueeze_1), kwargs = {})
#   %gather : [num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%relu_1, 1, %unsqueeze_1), kwargs = {})
#   %sub_37 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %gather), kwargs = {})
#   %add_40 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub_37, %relu_1), kwargs = {})
#   %clamp_min_5 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_40, 0), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_4, %clamp_min_5, %full_default), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%where,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_arange_clamp_min_gather_mean_ne_rsub_scalar_tensor_where_6(in_out_ptr0, in_ptr0, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 3
    R0_BLOCK: tl.constexpr = 4
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 76 + 12*ks1 + 12*ks2 + 36*ks0 + 2*ks1*ks2 + 6*ks0*ks1 + 6*ks0*ks2 + ks0*ks1*ks2
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp4 / tmp6
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp7, None)




# kernel path: /tmp/torchinductor_sahanp/vq/cvqyclaxjtabqyequ45onvcevzwlaizpi63bff7rpea7w2wk2wtt.py
# Topologically Sorted Source Nodes: [full, input_lengths], Original ATen: [aten.full, aten._to_copy]
# Source node to ATen node mapping:
#   full => full
#   input_lengths => device_put_2
# Graph fragment:
#   %full : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], %sym_sum_3), kwargs = {dtype: torch.int64, device: cpu, pin_memory: False})
#   %device_put_2 : [num_users=1] = call_function[target=torch.ops.prims.device_put.default](args = (%full, cuda:0), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_full_7(out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = 76 + 12*ks1 + 12*ks2 + 36*ks0 + 2*ks1*ks2 + 6*ks0*ks1 + 6*ks0*ks2 + ks0*ks1*ks2
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp0, None)







def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    buf0 = empty_strided_cpu((3, ), (1, ), torch.int64)
    # Topologically Sorted Source Nodes: [], Original ATen: []
    aten.randint.low_out(-9223372036854775808, 9223372036854775807, [3], out=buf0)
    buf1 = empty_strided_cpu((1, 10), (10, 1), torch.int64)
    cpp_fused_randint_0(buf0, buf1)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf2 = empty_strided_cuda((1, 10), (10, 1), torch.int64)
        buf2.copy_(buf1, False)
        del buf1
    buf3 = empty_strided_cpu((1, ), (1, ), torch.int64)
    cpp_fused_randint_1(buf0, buf3)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf4 = empty_strided_cuda((1, ), (1, ), torch.int64)
        buf4.copy_(buf3, False)
        buf5 = empty_strided_cuda((1, 76 + 12*s1 + 12*s2 + 36*s0 + 2*s1*s2 + 6*s0*s1 + 6*s0*s2 + s0*s1*s2), (76 + 12*s1 + 12*s2 + 36*s0 + 2*s1*s2 + 6*s0*s1 + 6*s0*s2 + s0*s1*s2, 1), torch.float32)
        buf8 = reinterpret_tensor(buf5, (1, 76 + 12*s1 + 12*s2 + 36*s0 + 2*s1*s2 + 6*s0*s1 + 6*s0*s2 + s0*s1*s2), (76 + 12*s1 + 12*s2 + 36*s0 + 2*s1*s2 + 6*s0*s1 + 6*s0*s2 + s0*s1*s2, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [x_6, x_7], Original ATen: [aten.constant_pad_nd, aten.relu]
        triton_poi_fused_constant_pad_nd_relu_2_xnumel = 76 + 12*s1 + 12*s2 + 36*s0 + 2*s1*s2 + 6*s0*s1 + 6*s0*s2 + s0*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_relu_2[grid(triton_poi_fused_constant_pad_nd_relu_2_xnumel)](buf8, arg3_1, 3, 64, 64, 24504, XBLOCK=128, num_warps=4, num_stages=1)
        del arg3_1
    buf6 = buf3; del buf3  # reuse
    cpp_fused_randint_3(buf0, buf6)
    del buf0
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf7 = empty_strided_cuda((1, ), (1, ), torch.int64)
        buf7.copy_(buf6, False)
        del buf6
        buf11 = empty_strided_cuda((1, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf11)
        buf12 = empty_strided_cuda((1, 76 + 12*s1 + 12*s2 + 36*s0 + 2*s1*s2 + 6*s0*s1 + 6*s0*s2 + s0*s1*s2), (76 + 12*s1 + 12*s2 + 36*s0 + 2*s1*s2 + 6*s0*s1 + 6*s0*s2 + s0*s1*s2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [randn_like], Original ATen: [aten.randn_like]
        triton_poi_fused_randn_like_4_xnumel = 76 + 12*s1 + 12*s2 + 36*s0 + 2*s1*s2 + 6*s0*s1 + 6*s0*s2 + s0*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused_randn_like_4[grid(triton_poi_fused_randn_like_4_xnumel)](buf11, buf12, 0, 24504, XBLOCK=128, num_warps=4, num_stages=1)
        del buf11
        buf9 = empty_strided_cuda((3, ), (1, ), torch.float32)
        buf13 = empty_strided_cuda((3, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [loss_multi_margin, loss_poisson, target_poisson], Original ATen: [aten.arange, aten.ne, aten.gather, aten.rsub, aten.add, aten.clamp_min, aten.scalar_tensor, aten.where, aten.mean, aten.exp, aten.abs, aten.mul, aten.sub]
        triton_red_fused_abs_add_arange_clamp_min_exp_gather_mean_mul_ne_rsub_scalar_tensor_sub_where_5_r0_numel = 26 + 4*s1 + 4*s2 + 12*s0 + 2*s0*s1 + 2*s0*s2 + ((2*s1*s2 + s0*s1*s2) // 3)
        stream0 = get_raw_stream(0)
        triton_red_fused_abs_add_arange_clamp_min_exp_gather_mean_mul_ne_rsub_scalar_tensor_sub_where_5[grid(3)](buf7, buf8, buf12, buf9, buf13, 3, 64, 64, 3, 8168, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del buf12
        buf10 = empty_strided_cuda((), (), torch.float32)
        buf16 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [loss_multi_margin], Original ATen: [aten.arange, aten.ne, aten.gather, aten.rsub, aten.add, aten.clamp_min, aten.scalar_tensor, aten.where, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_arange_clamp_min_gather_mean_ne_rsub_scalar_tensor_where_6[grid(1)](buf16, buf9, 3, 64, 64, 1, 3, XBLOCK=1, num_warps=2, num_stages=1)
        del buf9
        buf14 = empty_strided_cuda((), (), torch.float32)
        buf17 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [loss_poisson, target_poisson], Original ATen: [aten.exp, aten.abs, aten.mul, aten.sub, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_arange_clamp_min_gather_mean_ne_rsub_scalar_tensor_where_6[grid(1)](buf17, buf13, 3, 64, 64, 1, 3, XBLOCK=1, num_warps=2, num_stages=1)
        del buf13
        buf15 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [full, input_lengths], Original ATen: [aten.full, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_full_7[grid(1)](buf15, 3, 64, 64, 1, XBLOCK=1, num_warps=1, num_stages=1)
    return (buf8, buf2, buf15, buf4, buf16, buf17, )


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
