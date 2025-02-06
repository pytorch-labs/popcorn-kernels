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


# kernel path: /tmp/torchinductor_sahanp/fy/cfync6en72arpnm5olb2vi5whmvopnetqp7nbwwzm6ujjpviqnae.py
# Topologically Sorted Source Nodes: [pow_1], Original ATen: [aten.pow]
# Source node to ATen node mapping:
#   pow_1 => pow_1
# Graph fragment:
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%unsqueeze, 2.0), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_pow_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl_math.abs(tmp0)
    tmp2 = 0.5
    tmp3 = tmp1 <= tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp3, tmp4, tmp0)
    tmp6 = tmp5 * tmp5
    tl.store(out_ptr0 + (x0), tmp6, xmask)




# kernel path: /tmp/torchinductor_sahanp/cq/ccql3pdmb27qtpfnf3asduau6p66py4ig2jjdkreqbl2ncj4cajc.py
# Topologically Sorted Source Nodes: [sign, abs_1, relu, mul, mul_1, x_2], Original ATen: [aten.sign, aten.abs, aten.relu, aten.mul, aten.pow]
# Source node to ATen node mapping:
#   abs_1 => abs_2
#   mul => mul_41
#   mul_1 => mul_47
#   relu => relu
#   sign => sign
#   x_2 => pow_2
# Graph fragment:
#   %sign : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%avg_pool3d,), kwargs = {})
#   %abs_2 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%avg_pool3d,), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%abs_2,), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign, %relu), kwargs = {})
#   %mul_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_41, 27), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_47, 0.5), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_abs_mul_pow_relu_sign_1(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp1 < tmp0
    tmp3 = tmp2.to(tl.int8)
    tmp4 = tmp0 < tmp1
    tmp5 = tmp4.to(tl.int8)
    tmp6 = tmp3 - tmp5
    tmp7 = tmp6.to(tmp0.dtype)
    tmp8 = tl_math.abs(tmp0)
    tmp9 = triton_helpers.maximum(tmp1, tmp8)
    tmp10 = tmp7 * tmp9
    tmp11 = 27.0
    tmp12 = tmp10 * tmp11
    tmp13 = libdevice.sqrt(tmp12)
    tl.store(in_out_ptr0 + (x0), tmp13, xmask)







def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 1, s0, s1, s2), (s0*s1*s2, s0*s1*s2, s1*s2, s2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pow_1], Original ATen: [aten.pow]
        triton_poi_fused_pow_0_xnumel = s0*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused_pow_0[grid(triton_poi_fused_pow_0_xnumel)](arg3_1, buf0, 262144, XBLOCK=1024, num_warps=4, num_stages=1)
        del arg3_1
        # Topologically Sorted Source Nodes: [pow_1, out], Original ATen: [aten.pow, aten.avg_pool3d]
        buf1 = torch.ops.aten.avg_pool3d.default(buf0, [3, 3, 3], [2, 2, 2], [0, 0, 0], False, True, None)
        del buf0
        buf2 = buf1
        del buf1
        buf3 = reinterpret_tensor(buf2, (1, 1, 1 + (((-3) + s0) // 2), 1 + (((-3) + s1) // 2), 1 + (((-3) + s2) // 2)), (1 + (((-3) + s0) // 2)*(((-3) + s1) // 2) + (((-3) + s0) // 2)*(((-3) + s2) // 2) + (((-3) + s1) // 2)*(((-3) + s2) // 2) + (((-3) + s0) // 2)*(((-3) + s1) // 2)*(((-3) + s2) // 2) + (((-3) + s0) // 2) + (((-3) + s1) // 2) + (((-3) + s2) // 2), 1, 1 + (((-3) + s1) // 2)*(((-3) + s2) // 2) + (((-3) + s1) // 2) + (((-3) + s2) // 2), 1 + (((-3) + s2) // 2), 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [sign, abs_1, relu, mul, mul_1, x_2], Original ATen: [aten.sign, aten.abs, aten.relu, aten.mul, aten.pow]
        triton_poi_fused_abs_mul_pow_relu_sign_1_xnumel = 1 + (((-3) + s0) // 2)*(((-3) + s1) // 2) + (((-3) + s0) // 2)*(((-3) + s2) // 2) + (((-3) + s1) // 2)*(((-3) + s2) // 2) + (((-3) + s0) // 2)*(((-3) + s1) // 2)*(((-3) + s2) // 2) + (((-3) + s0) // 2) + (((-3) + s1) // 2) + (((-3) + s2) // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_abs_mul_pow_relu_sign_1[grid(triton_poi_fused_abs_mul_pow_relu_sign_1_xnumel)](buf3, 29791, XBLOCK=256, num_warps=4, num_stages=1)
    return (reinterpret_tensor(buf3, (1, 1, 1 + (((-3) + s0) // 2), 1 + (((-3) + s1) // 2), 1 + (((-3) + s2) // 2)), (1 + (((-3) + s0) // 2)*(((-3) + s1) // 2) + (((-3) + s0) // 2)*(((-3) + s2) // 2) + (((-3) + s1) // 2)*(((-3) + s2) // 2) + (((-3) + s0) // 2)*(((-3) + s1) // 2)*(((-3) + s2) // 2) + (((-3) + s0) // 2) + (((-3) + s1) // 2) + (((-3) + s2) // 2), 1 + (((-3) + s0) // 2)*(((-3) + s1) // 2) + (((-3) + s0) // 2)*(((-3) + s2) // 2) + (((-3) + s1) // 2)*(((-3) + s2) // 2) + (((-3) + s0) // 2)*(((-3) + s1) // 2)*(((-3) + s2) // 2) + (((-3) + s0) // 2) + (((-3) + s1) // 2) + (((-3) + s2) // 2), 1 + (((-3) + s1) // 2)*(((-3) + s2) // 2) + (((-3) + s1) // 2) + (((-3) + s2) // 2), 1 + (((-3) + s2) // 2), 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 64
    arg1_1 = 64
    arg2_1 = 64
    arg3_1 = rand_strided((1, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
