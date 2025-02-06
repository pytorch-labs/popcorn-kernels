# AOT ID: ['66_inference']
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


# kernel path: /tmp/torchinductor_sahanp/5r/c5rbjah6y5ob2jexek2eua5vfetiyuaxw6zwnumtvbmm7cku2ceb.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.bernoulli]
# Source node to ATen node mapping:
#   x => inductor_lookup_seed_default, inductor_random_default
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, %arg0_1, 1, 1], %inductor_lookup_seed_default, rand), kwargs = {})
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




# kernel path: /tmp/torchinductor_sahanp/77/c77uxzzpbovhefi7zx4nh3c5odjmb2hjx6qbnt5xkylmp3ljdilc.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.bernoulli, aten._to_copy, aten.mul, aten.add]
# Source node to ATen node mapping:
#   x => add_12, add_29, add_54, convert_element_type, lt_2, mul_12, mul_21, mul_24
# Graph fragment:
#   %lt_2 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%inductor_random_default, 0.5), kwargs = {})
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt_2, torch.float32), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%convert_element_type, 0.8864048946659319), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg3_1, %mul_21), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%convert_element_type, -1), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%add_12, 1.558387861036063), kwargs = {})
#   %add_29 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_12, 0.7791939305180315), kwargs = {})
#   %add_54 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_24, %add_29), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_add_bernoulli_mul_1(in_ptr0, in_ptr1, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
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
    tmp5 = 0.8864048946659319
    tmp6 = tmp4 * tmp5
    tmp7 = tmp0 * tmp6
    tmp8 = -1.0
    tmp9 = tmp4 + tmp8
    tmp10 = 1.558387861036063
    tmp11 = tmp9 * tmp10
    tmp12 = 0.7791939305180315
    tmp13 = tmp11 + tmp12
    tmp14 = tmp7 + tmp13
    tl.store(out_ptr0 + (x2), tmp14, xmask)




# kernel path: /tmp/torchinductor_sahanp/sl/csl3cfsmk3i347t3gkoqpyze4pcecvr72debqtbgxd6qw2qmkleu.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten._adaptive_avg_pool2d]
# Source node to ATen node mapping:
#   x_2 => _adaptive_avg_pool2d
# Graph fragment:
#   %_adaptive_avg_pool2d : [num_users=1] = call_function[target=torch.ops.aten._adaptive_avg_pool2d.default](args = (%view_1, [5, 5]), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__adaptive_avg_pool2d_2(in_ptr0, out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 32)
    x2 = ((xindex // 1024) % 3)
    x3 = xindex // 3072
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (((x0 + 32*x1 + 1024*x2 + 3072*x3) % (ks0*ks1*ks2))), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x4), tmp0, xmask)







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
        buf1 = empty_strided_cuda((1, s0, 1, 1), (s0, 1, s0, s0), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.bernoulli]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bernoulli_0[grid(s0)](buf0, buf1, 0, 3, XBLOCK=4, num_warps=1, num_stages=1)
        del buf0
        ps0 = s1*s2
        buf2 = empty_strided_cuda((1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.bernoulli, aten._to_copy, aten.mul, aten.add]
        triton_poi_fused__to_copy_add_bernoulli_mul_1_xnumel = s0*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_bernoulli_mul_1[grid(triton_poi_fused__to_copy_add_bernoulli_mul_1_xnumel)](arg3_1, buf1, buf2, 4096, 12288, XBLOCK=128, num_warps=4, num_stages=1)
        del arg3_1
        del buf1
        buf3 = empty_strided_cuda(((s0*s1*s2) // 3072, 3, 32, 32), (3072, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten._adaptive_avg_pool2d]
        triton_poi_fused__adaptive_avg_pool2d_2_xnumel = 3072*((s0*s1*s2) // 3072)
        stream0 = get_raw_stream(0)
        triton_poi_fused__adaptive_avg_pool2d_2[grid(triton_poi_fused__adaptive_avg_pool2d_2_xnumel)](buf2, buf3, 3, 64, 64, 12288, XBLOCK=256, num_warps=4, num_stages=1)
        del buf2
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten._adaptive_avg_pool2d]
        buf4 = torch.ops.aten._adaptive_avg_pool2d.default(buf3, [5, 5])
        del buf3
        buf5 = buf4
        del buf4
    return (buf5, )


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
