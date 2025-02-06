# AOT ID: ['133_inference']
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
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.bernoulli]
# Source node to ATen node mapping:
#   x_2 => inductor_lookup_seed_default, inductor_random_default_1
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, %floordiv, 1, 1], %inductor_lookup_seed_default, rand), kwargs = {})
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




# kernel path: /tmp/torchinductor_sahanp/3s/c3sqk45amy6b2mvjvltzxnp6opjo4dtrwcfyvmf6zeqnjeqne2qt.py
# Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.bernoulli]
# Source node to ATen node mapping:
#   x_5 => inductor_lookup_seed_default_1, inductor_random_default
# Graph fragment:
#   %inductor_lookup_seed_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 1), kwargs = {})
#   %inductor_random_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, %floordiv_1, 1, 1], %inductor_lookup_seed_default_1, rand), kwargs = {})
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
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (x0), tmp2, None)




# kernel path: /tmp/torchinductor_sahanp/7b/c7byv7yupsexpglu7iu67smgzoi53ekqicst3yfwlkf3u2nxf5bb.py
# Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.bernoulli, aten._to_copy, aten.div, aten.mul]
# Source node to ATen node mapping:
#   x_5 => convert_element_type_1, div_1, lt_11, mul_67
# Graph fragment:
#   %lt_11 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%inductor_random_default, 0.5), kwargs = {})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt_11, torch.float32), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%convert_element_type_1, 0.5), kwargs = {})
#   %mul_67 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_3, %div_1), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_bernoulli_div_mul_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks1)
    x2 = xindex // ks4
    x3 = xindex // ks0
    tmp0 = tl.load(in_ptr0 + (ks3*(x1 // 4) + ks2*ks3*(((x0 // 2) % 2)) + 2*ks2*ks3*(((x1 // 2) % 2)) + 4*ks2*ks3*((x0 % 2)) + 8*ks2*ks3*((x1 % 2)) + (((x0 // 4) % ks3))), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (2*((x1 % 2)) + ((x0 % 2))), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 20.0
    tmp2 = tmp0 > tmp1
    tmp3 = tl_math.exp(tmp0)
    tmp4 = libdevice.log1p(tmp3)
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp6 = libdevice.tanh(tmp5)
    tmp7 = tmp0 * tmp6
    tmp9 = 0.5
    tmp10 = tmp8 < tmp9
    tmp11 = tmp10.to(tl.float32)
    tmp12 = 2.0
    tmp13 = tmp11 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp14 > tmp1
    tmp16 = tl_math.exp(tmp14)
    tmp17 = libdevice.log1p(tmp16)
    tmp18 = tl.where(tmp15, tmp14, tmp17)
    tmp19 = libdevice.tanh(tmp18)
    tmp20 = tmp14 * tmp19
    tmp22 = tmp21 < tmp9
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp23 * tmp12
    tmp25 = tmp20 * tmp24
    tl.store(out_ptr0 + (x0 + ks3*x3*(triton_helpers.div_floor_integer(ks5,  2*(ks5 // 4)))*(triton_helpers.div_floor_integer(ks5,  8*(ks5 // 16)))), tmp25, xmask)







def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [2], out=buf0)
        buf1 = empty_strided_cuda((1, s0 // 4, 1, 1), (s0 // 4, 1, s0 // 4, s0 // 4), torch.float32)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_0_xnumel = s0 // 4
        stream0 = get_raw_stream(0)
        triton_poi_fused_bernoulli_0[grid(triton_poi_fused_bernoulli_0_xnumel)](buf0, buf1, 0, 4, XBLOCK=4, num_warps=1, num_stages=1)
        buf2 = empty_strided_cuda((1, s0 // 16, 1, 1), (s0 // 16, 1, s0 // 16, s0 // 16), torch.float32)
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_1_xnumel = s0 // 16
        stream0 = get_raw_stream(0)
        triton_poi_fused_bernoulli_1[grid(triton_poi_fused_bernoulli_1_xnumel)](buf0, buf2, 1, 1, XBLOCK=1, num_warps=1, num_stages=1)
        del buf0
        ps0 = 2*s2*(s0 // (2*(s0 // 4)))
        ps1 = 4*s1
        ps2 = 8*s1*s2*(s0 // (2*(s0 // 4)))
        buf3 = empty_strided_cuda((1, s0 // 16, 4*s1, 2*s2*(s0 // (2*(s0 // 4)))), (4*s1*s2*(s0 // 16)*(s0 // (2*(s0 // 4)))*(s0 // (8*(s0 // 16))), 4*s1*s2*(s0 // (2*(s0 // 4)))*(s0 // (8*(s0 // 16))), s2*(s0 // (2*(s0 // 4)))*(s0 // (8*(s0 // 16))), 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.bernoulli, aten._to_copy, aten.div, aten.mul]
        triton_poi_fused__to_copy_bernoulli_div_mul_2_xnumel = 8*s1*s2*(s0 // 16)*(s0 // (2*(s0 // 4)))
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_bernoulli_div_mul_2[grid(triton_poi_fused__to_copy_bernoulli_div_mul_2_xnumel)](arg3_1, buf1, buf2, buf3, 256, 256, 64, 64, 65536, 16, 65536, XBLOCK=256, num_warps=4, num_stages=1)
        del arg3_1
        del buf1
        del buf2
    return (buf3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 16
    arg1_1 = 64
    arg2_1 = 64
    arg3_1 = rand_strided((1, 16, 64, 64), (65536, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
