# AOT ID: ['54_inference']
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
#   x_2 => inductor_lookup_seed_default, inductor_random_default
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




# kernel path: /tmp/torchinductor_sahanp/yk/cyki53lizby6kiswvsfhhodgnzumucugdy7r7z3cvi4bpbeyuw2v.py
# Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.elu, aten.avg_pool3d, aten.bernoulli, aten._to_copy, aten.div, aten.mul]
# Source node to ATen node mapping:
#   x => expm1, gt, mul, mul_1, mul_2, where
#   x_1 => avg_pool3d
#   x_2 => convert_element_type, div, lt_2, mul_26
# Graph fragment:
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%arg4_1, 0), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg4_1, 1.0), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg4_1, 1.0), kwargs = {})
#   %expm1 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%mul_1,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expm1, 1.0), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %mul, %mul_2), kwargs = {})
#   %avg_pool3d : [num_users=1] = call_function[target=torch.ops.aten.avg_pool3d.default](args = (%where, [2, 2, 2], [2, 2, 2]), kwargs = {})
#   %lt_2 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%inductor_random_default, 0.5), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt_2, torch.float32), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%convert_element_type, 0.5), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%avg_pool3d, %div), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_avg_pool3d_bernoulli_div_elu_mul_1(in_out_ptr0, in_ptr0, in_ptr1, ks0, ks1, ks2, ks3, ks4, ks5, ks6, ks7, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks1)
    x2 = ((xindex // ks2) % ks3)
    x3 = xindex // ks4
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 2*ks7*x1 + 2*ks6*ks7*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (1 + 2*x0 + 2*ks7*x1 + 2*ks6*ks7*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (ks7 + 2*x0 + 2*ks7*x1 + 2*ks6*ks7*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr0 + (1 + ks7 + 2*x0 + 2*ks7*x1 + 2*ks6*ks7*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr0 + (2*x0 + ks6*ks7 + 2*ks7*x1 + 2*ks6*ks7*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr0 + (1 + 2*x0 + ks6*ks7 + 2*ks7*x1 + 2*ks6*ks7*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr0 + (ks7 + 2*x0 + ks6*ks7 + 2*ks7*x1 + 2*ks6*ks7*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr0 + (1 + ks7 + 2*x0 + ks6*ks7 + 2*ks7*x1 + 2*ks6*ks7*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp59 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 1.0
    tmp4 = tmp0 * tmp3
    tmp5 = libdevice.expm1(tmp4)
    tmp6 = tmp5 * tmp3
    tmp7 = tl.where(tmp2, tmp4, tmp6)
    tmp9 = tmp8 > tmp1
    tmp10 = tmp8 * tmp3
    tmp11 = libdevice.expm1(tmp10)
    tmp12 = tmp11 * tmp3
    tmp13 = tl.where(tmp9, tmp10, tmp12)
    tmp14 = tmp13 + tmp7
    tmp16 = tmp15 > tmp1
    tmp17 = tmp15 * tmp3
    tmp18 = libdevice.expm1(tmp17)
    tmp19 = tmp18 * tmp3
    tmp20 = tl.where(tmp16, tmp17, tmp19)
    tmp21 = tmp20 + tmp14
    tmp23 = tmp22 > tmp1
    tmp24 = tmp22 * tmp3
    tmp25 = libdevice.expm1(tmp24)
    tmp26 = tmp25 * tmp3
    tmp27 = tl.where(tmp23, tmp24, tmp26)
    tmp28 = tmp27 + tmp21
    tmp30 = tmp29 > tmp1
    tmp31 = tmp29 * tmp3
    tmp32 = libdevice.expm1(tmp31)
    tmp33 = tmp32 * tmp3
    tmp34 = tl.where(tmp30, tmp31, tmp33)
    tmp35 = tmp34 + tmp28
    tmp37 = tmp36 > tmp1
    tmp38 = tmp36 * tmp3
    tmp39 = libdevice.expm1(tmp38)
    tmp40 = tmp39 * tmp3
    tmp41 = tl.where(tmp37, tmp38, tmp40)
    tmp42 = tmp41 + tmp35
    tmp44 = tmp43 > tmp1
    tmp45 = tmp43 * tmp3
    tmp46 = libdevice.expm1(tmp45)
    tmp47 = tmp46 * tmp3
    tmp48 = tl.where(tmp44, tmp45, tmp47)
    tmp49 = tmp48 + tmp42
    tmp51 = tmp50 > tmp1
    tmp52 = tmp50 * tmp3
    tmp53 = libdevice.expm1(tmp52)
    tmp54 = tmp53 * tmp3
    tmp55 = tl.where(tmp51, tmp52, tmp54)
    tmp56 = tmp55 + tmp49
    tmp57 = 0.125
    tmp58 = tmp56 * tmp57
    tmp60 = 0.5
    tmp61 = tmp59 < tmp60
    tmp62 = tmp61.to(tl.float32)
    tmp63 = 2.0
    tmp64 = tmp62 * tmp63
    tmp65 = tmp58 * tmp64
    tl.store(in_out_ptr0 + (x5), tmp65, xmask)




# kernel path: /tmp/torchinductor_sahanp/oy/coyf3ggisgi6pvuevv6futxfwottm6nnqvexaocwnk4phqej6dcu.py
# Topologically Sorted Source Nodes: [x_2, x_3], Original ATen: [aten.bernoulli, aten._to_copy, aten.div, aten.mul, aten.view]
# Source node to ATen node mapping:
#   x_2 => convert_element_type, div, lt_2, mul_26
#   x_3 => view
# Graph fragment:
#   %lt_2 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%inductor_random_default, 0.5), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt_2, torch.float32), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%convert_element_type, 0.5), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%avg_pool3d, %div), kwargs = {})
#   %view : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_26, [1, %floordiv, -1]), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_bernoulli_div_mul_view_2(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (ks1*ks2*((((x0 + ks1*ks2*ks5*x1) // ks3) % (ks4*ks5))) + ((x0 % ks3))), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp0, xmask)







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
        buf1 = empty_strided_cuda((1, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf1)
        buf2 = empty_strided_cuda((1, s0, 1, 1, 1), (s0, 1, s0, s0, s0), torch.float32)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.bernoulli]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bernoulli_0[grid(s0)](buf1, buf2, 0, 3, XBLOCK=4, num_warps=1, num_stages=1)
        del buf1
        ps0 = s3 // 2
        ps1 = s2 // 2
        ps2 = (s2 // 2)*(s3 // 2)
        ps3 = s1 // 2
        ps4 = (s1 // 2)*(s2 // 2)*(s3 // 2)
        buf0 = empty_strided_cuda((1, s0, s1 // 2, s2 // 2, s3 // 2), (s0*(s1 // 2)*(s2 // 2)*(s3 // 2), (s1 // 2)*(s2 // 2)*(s3 // 2), (s2 // 2)*(s3 // 2), s3 // 2, 1), torch.float32)
        buf3 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.elu, aten.avg_pool3d, aten.bernoulli, aten._to_copy, aten.div, aten.mul]
        triton_poi_fused__to_copy_avg_pool3d_bernoulli_div_elu_mul_1_xnumel = s0*(s1 // 2)*(s2 // 2)*(s3 // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool3d_bernoulli_div_elu_mul_1[grid(triton_poi_fused__to_copy_avg_pool3d_bernoulli_div_elu_mul_1_xnumel)](buf3, arg4_1, buf2, 16, 16, 256, 16, 4096, 32, 32, 32, 12288, XBLOCK=128, num_warps=4, num_stages=1)
        del arg4_1
        del buf2
        ps5 = s0*(s2 // 2)*(s3 // 2)
        buf4 = empty_strided_cuda((1, s1 // 2, s0*(s2 // 2)*(s3 // 2)), (s0*(s1 // 2)*(s2 // 2)*(s3 // 2), s0*(s2 // 2)*(s3 // 2), 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2, x_3], Original ATen: [aten.bernoulli, aten._to_copy, aten.div, aten.mul, aten.view]
        triton_poi_fused__to_copy_bernoulli_div_mul_view_2_xnumel = s0*(s1 // 2)*(s2 // 2)*(s3 // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_bernoulli_div_mul_view_2[grid(triton_poi_fused__to_copy_bernoulli_div_mul_view_2_xnumel)](buf3, buf4, 768, 16, 16, 256, 16, 3, 12288, XBLOCK=128, num_warps=4, num_stages=1)
        del buf3
    return (buf4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 3
    arg1_1 = 32
    arg2_1 = 32
    arg3_1 = 32
    arg4_1 = rand_strided((1, 3, 32, 32, 32), (98304, 32768, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
