# AOT ID: ['15_inference']
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


# kernel path: /tmp/torchinductor_sahanp/dr/cdriffxemrjfn3c6giz22xaqyaryykcggyljxxaif367gxeiciuc.py
# Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.abs, aten.add, aten.div, aten.avg_pool2d]
# Source node to ATen node mapping:
#   x => abs_1, add_4, div
#   x_1 => avg_pool2d
# Graph fragment:
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%arg3_1,), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%abs_1, 1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%arg3_1, %add_4), kwargs = {})
#   %avg_pool2d : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%div, [2, 2], [2, 2]), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_abs_add_avg_pool2d_div_0(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks1)
    x2 = xindex // ks2
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 2*ks4*x1 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (1 + 2*x0 + 2*ks4*x1 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (ks4 + 2*x0 + 2*ks4*x1 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (1 + ks4 + 2*x0 + 2*ks4*x1 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl_math.abs(tmp0)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 / tmp3
    tmp6 = tl_math.abs(tmp5)
    tmp7 = tmp6 + tmp2
    tmp8 = tmp5 / tmp7
    tmp9 = tmp8 + tmp4
    tmp11 = tl_math.abs(tmp10)
    tmp12 = tmp11 + tmp2
    tmp13 = tmp10 / tmp12
    tmp14 = tmp13 + tmp9
    tmp16 = tl_math.abs(tmp15)
    tmp17 = tmp16 + tmp2
    tmp18 = tmp15 / tmp17
    tmp19 = tmp18 + tmp14
    tmp20 = 0.25
    tmp21 = tmp19 * tmp20
    tl.store(out_ptr0 + (x3), tmp21, xmask)




# kernel path: /tmp/torchinductor_sahanp/s5/cs5wmqfexauwpfjgvv7tgjpbmr6qmsajyz5k6zy4s2pbz5rbssrp.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_3 => constant_pad_nd
# Graph fragment:
#   %constant_pad_nd : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view, [0, 0, 0, 0, 2, 2], 0.0), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_constant_pad_nd_1(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex // ks0
    x0 = (xindex % ks2)
    x1 = ((xindex // ks2) % ks3)
    x4 = xindex
    tmp0 = (-2) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = ks1
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (2*x0 + ((-2)*ks4*ks5) + 2*ks4*x1 + ks4*ks5*x2), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr0 + (1 + 2*x0 + ((-2)*ks4*ks5) + 2*ks4*x1 + ks4*ks5*x2), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp7 + tmp6
    tmp9 = tl.load(in_ptr0 + (ks4 + 2*x0 + ((-2)*ks4*ks5) + 2*ks4*x1 + ks4*ks5*x2), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp9 + tmp8
    tmp11 = tl.load(in_ptr0 + (1 + ks4 + 2*x0 + ((-2)*ks4*ks5) + 2*ks4*x1 + ks4*ks5*x2), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 + tmp10
    tmp13 = 0.25
    tmp14 = tmp12 * tmp13
    tmp15 = tmp14 * tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp5, tmp15, tmp16)
    tl.store(out_ptr0 + (x4), tmp17, xmask)




# kernel path: /tmp/torchinductor_sahanp/su/csu4doqbvjxgmma74msfuhijeavzvhzdqz5iqnfk7wfpnt2yhmm4.py
# Topologically Sorted Source Nodes: [x, x_1, x_2, x_3], Original ATen: [aten.abs, aten.add, aten.div, aten.avg_pool2d, aten.mul, aten.pow]
# Source node to ATen node mapping:
#   x => abs_1, add_4, div
#   x_1 => avg_pool2d
#   x_2 => avg_pool2d_1
#   x_3 => add_49, div_1, mul_50, pow_1
# Graph fragment:
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%arg3_1,), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%abs_1, 1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%arg3_1, %add_4), kwargs = {})
#   %avg_pool2d : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%div, [2, 2], [2, 2]), kwargs = {})
#   %avg_pool2d_1 : [num_users=4] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%avg_pool2d, [2, 2], [2, 2]), kwargs = {})
#   %mul_50 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, 0.0001), kwargs = {})
#   %add_49 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_50, 1.0), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_49, 0.75), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%avg_pool2d_1, %pow_1), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_abs_add_avg_pool2d_div_mul_pow_2(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks1)
    x2 = xindex // ks2
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 2*ks3*x1 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 2*ks3*x1 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (ks3 + 2*x0 + 2*ks3*x1 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (1 + ks3 + 2*x0 + 2*ks3*x1 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr1 + (ks2 + x3), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (x3 + 2*ks0*ks1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr1 + (x3 + 3*ks0*ks1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr1 + (x3 + 4*ks0*ks1), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tmp11 = tmp10 + tmp9
    tmp13 = tmp12 + tmp11
    tmp15 = tmp14 + tmp13
    tmp17 = tmp16 + tmp15
    tmp18 = 0.2
    tmp19 = tmp17 * tmp18
    tmp20 = 0.0001
    tmp21 = tmp19 * tmp20
    tmp22 = 1.0
    tmp23 = tmp21 + tmp22
    tmp24 = 0.75
    tmp25 = libdevice.pow(tmp23, tmp24)
    tmp26 = tmp8 / tmp25
    tl.store(out_ptr0 + (x3), tmp26, xmask)







def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        ps0 = s2 // 2
        ps1 = s1 // 2
        ps2 = (s1 // 2)*(s2 // 2)
        buf0 = empty_strided_cuda((1, s0, s1 // 2, s2 // 2), (s0*(s1 // 2)*(s2 // 2), (s1 // 2)*(s2 // 2), s2 // 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.abs, aten.add, aten.div, aten.avg_pool2d]
        triton_poi_fused_abs_add_avg_pool2d_div_0_xnumel = s0*(s1 // 2)*(s2 // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_abs_add_avg_pool2d_div_0[grid(triton_poi_fused_abs_add_avg_pool2d_div_0_xnumel)](arg3_1, buf0, 32, 32, 1024, 64, 64, 3072, XBLOCK=128, num_warps=4, num_stages=1)
        del arg3_1
        ps3 = (s1 // 4)*(s2 // 4)
        ps4 = s2 // 4
        ps5 = s1 // 4
        buf1 = empty_strided_cuda((1, 1, 4 + s0, s1 // 4, s2 // 4), (4*(s1 // 4)*(s2 // 4) + s0*(s1 // 4)*(s2 // 4), 4*(s1 // 4)*(s2 // 4) + s0*(s1 // 4)*(s2 // 4), (s1 // 4)*(s2 // 4), s2 // 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_1_xnumel = 4*(s1 // 4)*(s2 // 4) + s0*(s1 // 4)*(s2 // 4)
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_1[grid(triton_poi_fused_constant_pad_nd_1_xnumel)](buf0, buf1, 256, 3, 16, 16, 32, 32, 1792, XBLOCK=256, num_warps=4, num_stages=1)
        buf2 = empty_strided_cuda((1, s0, s1 // 4, s2 // 4), (s0*(s1 // 4)*(s2 // 4), (s1 // 4)*(s2 // 4), s2 // 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_1, x_2, x_3], Original ATen: [aten.abs, aten.add, aten.div, aten.avg_pool2d, aten.mul, aten.pow]
        triton_poi_fused_abs_add_avg_pool2d_div_mul_pow_2_xnumel = s0*(s1 // 4)*(s2 // 4)
        stream0 = get_raw_stream(0)
        triton_poi_fused_abs_add_avg_pool2d_div_mul_pow_2[grid(triton_poi_fused_abs_add_avg_pool2d_div_mul_pow_2_xnumel)](buf0, buf1, buf2, 16, 16, 256, 32, 32, 768, XBLOCK=128, num_warps=4, num_stages=1)
        del buf0
        del buf1
    return (reinterpret_tensor(buf2, (1, s0*(s1 // 4), s2 // 4), (s0*(s1 // 4)*(s2 // 4), s2 // 4, 1), 0), )


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
