# AOT ID: ['105_inference']
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


# kernel path: /tmp/torchinductor_sahanp/2u/c2uio5agy6zf3faddgdmtmmahogsytx7pojobedoblrtu6wecsqf.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.bernoulli]
# Source node to ATen node mapping:
#   x_1 => inductor_lookup_seed_default, inductor_random_default_2
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default_2 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 1, 1], %inductor_lookup_seed_default, rand), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_bernoulli_0(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp2, None)




# kernel path: /tmp/torchinductor_sahanp/tf/ctfvvs4ia2uldv5lejm7tiihmdohe344fpxf6tebnnxg22vfnkew.py
# Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.bernoulli]
# Source node to ATen node mapping:
#   x_4 => inductor_lookup_seed_default_1, inductor_random_default_1
# Graph fragment:
#   %inductor_lookup_seed_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 1), kwargs = {})
#   %inductor_random_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 1, 1], %inductor_lookup_seed_default_1, rand), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_bernoulli_1(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp2, None)




# kernel path: /tmp/torchinductor_sahanp/k2/ck2ivvmtbjl4kqjpbzmvhsve7ty37nahnzreuq7xu6aimiytbumm.py
# Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.bernoulli, aten._to_copy, aten.div, aten.mul]
# Source node to ATen node mapping:
#   x_6 => convert_element_type_2, div_2, lt_2, mul_32
# Graph fragment:
#   %lt_2 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%inductor_random_default, 0.5), kwargs = {})
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt_2, torch.float32), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%convert_element_type_2, 0.5), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, %div_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_bernoulli_div_mul_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp39 = tl.load(in_ptr3 + (0))
    tmp40 = tl.broadcast_to(tmp39, [XBLOCK])
    tmp0 = (-1) + x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = 4 + ks0
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-3) + x0
    tmp7 = tl.full([1], 0, tl.int64)
    tmp8 = tmp6 >= tmp7
    tmp9 = tl.broadcast_to(ks0, [XBLOCK])
    tmp10 = tmp6 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tmp11 & tmp5
    tmp13 = tl.load(in_ptr0 + ((-3) + x0), tmp12 & xmask, other=0.0)
    tmp14 = tl.load(in_ptr1 + (tl.full([XBLOCK], 0, tl.int32)), tmp12, eviction_policy='evict_last', other=0.0)
    tmp15 = 0.5
    tmp16 = tmp14 < tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = 2.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp13 * tmp19
    tmp21 = tmp20 * tmp15
    tmp22 = 0.7071067811865476
    tmp23 = tmp20 * tmp22
    tmp24 = libdevice.erf(tmp23)
    tmp25 = 1.0
    tmp26 = tmp24 + tmp25
    tmp27 = tmp21 * tmp26
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp12, tmp27, tmp28)
    tmp30 = tl.load(in_ptr2 + (tl.full([XBLOCK], 0, tl.int32)), tmp5, eviction_policy='evict_last', other=0.0)
    tmp31 = 0.5
    tmp32 = tmp30 < tmp31
    tmp33 = tmp32.to(tl.float32)
    tmp34 = 2.0
    tmp35 = tmp33 * tmp34
    tmp36 = tmp29 * tmp35
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp5, tmp36, tmp37)
    tmp41 = 0.5
    tmp42 = tmp40 < tmp41
    tmp43 = tmp42.to(tl.float32)
    tmp44 = 2.0
    tmp45 = tmp43 * tmp44
    tmp46 = tmp38 * tmp45
    tl.store(out_ptr0 + (x0), tmp46, xmask)







def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    s0 = arg0_1
    assert_size_stride(arg1_1, (1, s0), (s0, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((3, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [3], out=buf0)
        buf1 = empty_strided_cuda((1, 1, 1), (1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.bernoulli]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bernoulli_0[grid(1)](buf0, buf1, 2, 1, XBLOCK=1, num_warps=1, num_stages=1)
        buf2 = empty_strided_cuda((1, 1, 1), (1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.bernoulli]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bernoulli_1[grid(1)](buf0, buf2, 1, 1, XBLOCK=1, num_warps=1, num_stages=1)
        buf3 = empty_strided_cuda((1, 1, 1), (1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.bernoulli]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bernoulli_0[grid(1)](buf0, buf3, 2, 1, XBLOCK=1, num_warps=1, num_stages=1)
        del buf0
        buf4 = empty_strided_cuda((1, 1, 6 + s0), (6 + s0, 6 + s0, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.bernoulli, aten._to_copy, aten.div, aten.mul]
        triton_poi_fused__to_copy_bernoulli_div_mul_2_xnumel = 6 + s0
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_bernoulli_div_mul_2[grid(triton_poi_fused__to_copy_bernoulli_div_mul_2_xnumel)](arg1_1, buf1, buf2, buf3, buf4, 10, 16, XBLOCK=16, num_warps=1, num_stages=1)
        del arg1_1
        del buf1
        del buf2
        del buf3
    return (reinterpret_tensor(buf4, (1, 6 + s0), (6 + s0, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 10
    arg1_1 = rand_strided((1, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
