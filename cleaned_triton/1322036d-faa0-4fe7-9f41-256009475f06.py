# AOT ID: ['57_inference']
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


# kernel path: /tmp/torchinductor_sahanp/gl/cglinve47z7tvhxxzya5iqhlgy7pajvsymp7xykpwnshbiai3hvh.py
# Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.rrelu_with_noise_functional, aten.pow]
# Source node to ATen node mapping:
#   x => le, mul_3, where
#   x_1 => pow_1
# Graph fragment:
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%arg2_1, 0), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg2_1, %uniform), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le, %mul_3, %arg2_1), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%where, 2.0), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_pow_rrelu_with_noise_functional_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp0 * tmp3
    tmp5 = tl.where(tmp2, tmp4, tmp0)
    tmp6 = tmp5 * tmp5
    tl.store(in_out_ptr0 + (x0), tmp6, xmask)




# kernel path: /tmp/torchinductor_sahanp/6t/c6tk44hque2ggmhk7o3btuxhwc5pa4zjlfegkageqllvaf6zz4bu.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.sign, aten.abs, aten.relu, aten.mul, aten.pow]
# Source node to ATen node mapping:
#   x_1 => abs_1, mul_31, mul_35, pow_2, relu, sign
# Graph fragment:
#   %sign : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%squeeze,), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%squeeze,), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%abs_1,), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign, %relu), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_31, 3), kwargs = {})
#   %pow_2 : [num_users=4] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_35, 0.5), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_abs_mul_pow_relu_sign_1(in_ptr0, out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + ks1*x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + ks1*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 2*x0 + ks1*x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp5 = 0.3333333333333333
    tmp6 = tmp4 * tmp5
    tmp7 = tl.full([1], 0, tl.int32)
    tmp8 = tmp7 < tmp6
    tmp9 = tmp8.to(tl.int8)
    tmp10 = tmp6 < tmp7
    tmp11 = tmp10.to(tl.int8)
    tmp12 = tmp9 - tmp11
    tmp13 = tmp12.to(tmp6.dtype)
    tmp14 = tl_math.abs(tmp6)
    tmp15 = triton_helpers.maximum(tmp7, tmp14)
    tmp16 = tmp13 * tmp15
    tmp17 = 3.0
    tmp18 = tmp16 * tmp17
    tmp19 = libdevice.sqrt(tmp18)
    tl.store(out_ptr0 + (x2), tmp19, xmask)




# kernel path: /tmp/torchinductor_sahanp/xh/cxh6eibfmv677t7za3zxmt4kexox6aokz6xi325wqdnl65a6nhff.py
# Topologically Sorted Source Nodes: [x_2, x_3], Original ATen: [aten.rrelu_with_noise_functional, aten.neg, aten._softmax]
# Source node to ATen node mapping:
#   x_2 => le_1, mul_45, where_2
#   x_3 => amax, exp, neg, sub_30, sum_1
# Graph fragment:
#   %le_1 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%pow_2, 0), kwargs = {})
#   %mul_45 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_2, %uniform_1), kwargs = {})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_1, %mul_45, %pow_2), kwargs = {})
#   %neg : [num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%where_2,), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%neg, [1], True), kwargs = {})
#   %sub_30 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%neg, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_30,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__softmax_neg_rrelu_with_noise_functional_2(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + ks0*r0_1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (x0 + ks0*r0_1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tmp0 * tmp3
        tmp5 = tl.where(tmp2, tmp4, tmp0)
        tmp6 = -tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = triton_helpers.maximum(_tmp8, tmp7)
        _tmp8 = tl.where(r0_mask & xmask, tmp9, _tmp8)
    tmp8 = triton_helpers.max2(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    _tmp20 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp10 = tl.load(in_ptr0 + (x0 + ks0*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.load(in_ptr1 + (x0 + ks0*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = 0.0
        tmp12 = tmp10 <= tmp11
        tmp14 = tmp10 * tmp13
        tmp15 = tl.where(tmp12, tmp14, tmp10)
        tmp16 = -tmp15
        tmp17 = tmp16 - tmp8
        tmp18 = tl_math.exp(tmp17)
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(r0_mask & xmask, tmp21, _tmp20)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp20, xmask)




# kernel path: /tmp/torchinductor_sahanp/lt/cltpvkmjjimtaluyuy7jksdq66plh4ytbcpbbv6p5fqimd23qudy.py
# Topologically Sorted Source Nodes: [x_2, x_3], Original ATen: [aten.rrelu_with_noise_functional, aten.neg, aten._softmax]
# Source node to ATen node mapping:
#   x_2 => le_1, mul_45, where_2
#   x_3 => div, exp, neg, sub_30
# Graph fragment:
#   %le_1 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%pow_2, 0), kwargs = {})
#   %mul_45 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_2, %uniform_1), kwargs = {})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_1, %mul_45, %pow_2), kwargs = {})
#   %neg : [num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%where_2,), kwargs = {})
#   %sub_30 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%neg, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_30,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__softmax_neg_rrelu_with_noise_functional_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    tmp0 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp0 * tmp3
    tmp5 = tl.where(tmp2, tmp4, tmp0)
    tmp6 = -tmp5
    tmp8 = tmp6 - tmp7
    tmp9 = tl_math.exp(tmp8)
    tmp11 = tmp9 / tmp10
    tl.store(out_ptr0 + (x0 + x1 + x1*(triton_helpers.div_floor_integer((-3) + ks1,  2))), tmp11, xmask)







def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    assert_size_stride(arg2_1, (1, s0, s1), (s0*s1, s1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.rrelu_with_noise_functional]
        buf0 = torch.ops.aten.uniform.default(arg2_1, 0.125, 0.3333333333333333)
        buf1 = buf0
        del buf0
        buf2 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.rrelu_with_noise_functional, aten.pow]
        triton_poi_fused_pow_rrelu_with_noise_functional_0_xnumel = s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused_pow_rrelu_with_noise_functional_0[grid(triton_poi_fused_pow_rrelu_with_noise_functional_0_xnumel)](buf2, arg2_1, 1000, XBLOCK=256, num_warps=4, num_stages=1)
        del arg2_1
        ps0 = ((-1) + s1) // 2
        buf3 = empty_strided_cuda((1, s0, ((-1) + s1) // 2), (s0*(((-1) + s1) // 2), ((-1) + s1) // 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.sign, aten.abs, aten.relu, aten.mul, aten.pow]
        triton_poi_fused_abs_mul_pow_relu_sign_1_xnumel = s0*(((-1) + s1) // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_abs_mul_pow_relu_sign_1[grid(triton_poi_fused_abs_mul_pow_relu_sign_1_xnumel)](buf2, buf3, 49, 100, 490, XBLOCK=128, num_warps=4, num_stages=1)
        del buf2
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.rrelu_with_noise_functional]
        buf4 = torch.ops.aten.uniform.default(buf3, 0.125, 0.3333333333333333)
        buf5 = buf4
        del buf4
        buf6 = empty_strided_cuda((1, 1, ((-1) + s1) // 2), (((-1) + s1) // 2, ((-1) + s1) // 2, 1), torch.float32)
        buf7 = empty_strided_cuda((1, 1, ((-1) + s1) // 2), (((-1) + s1) // 2, ((-1) + s1) // 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2, x_3], Original ATen: [aten.rrelu_with_noise_functional, aten.neg, aten._softmax]
        triton_red_fused__softmax_neg_rrelu_with_noise_functional_2_xnumel = ((-1) + s1) // 2
        stream0 = get_raw_stream(0)
        triton_red_fused__softmax_neg_rrelu_with_noise_functional_2[grid(triton_red_fused__softmax_neg_rrelu_with_noise_functional_2_xnumel)](buf3, buf5, buf6, buf7, 49, 49, 10, XBLOCK=1, R0_BLOCK=16, num_warps=2, num_stages=1)
        buf8 = empty_strided_cuda((1, s0, ((-1) + s1) // 2), (s0 + s0*(((-3) + s1) // 2), 1 + (((-3) + s1) // 2), 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2, x_3], Original ATen: [aten.rrelu_with_noise_functional, aten.neg, aten._softmax]
        triton_poi_fused__softmax_neg_rrelu_with_noise_functional_3_xnumel = s0*(((-1) + s1) // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_neg_rrelu_with_noise_functional_3[grid(triton_poi_fused__softmax_neg_rrelu_with_noise_functional_3_xnumel)](buf3, buf5, buf6, buf7, buf8, 49, 100, 490, XBLOCK=128, num_warps=4, num_stages=1)
        del buf3
        del buf5
        del buf6
        del buf7
    return (buf8, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 10
    arg1_1 = 100
    arg2_1 = rand_strided((1, 10, 100), (1000, 100, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
