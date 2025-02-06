# AOT ID: ['49_inference']
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


# kernel path: /tmp/torchinductor_sahanp/th/cth7bk2d6okzvn73utvci7yj5npl3ito5ft2q425hk4lhprd5qkp.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.bernoulli]
# Source node to ATen node mapping:
#   x_3 => inductor_lookup_seed_default, inductor_random_default_1
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 4, 1, 1], %inductor_lookup_seed_default, rand), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_bernoulli_0(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (x0), tmp2, xmask)




# kernel path: /tmp/torchinductor_sahanp/gv/cgvyvihy6sv26xdndzutgdsnxb3o6q5fl4u2ndlyofb7zx2nzbjm.py
# Topologically Sorted Source Nodes: [x, x_1, x_2, x_3, x_4], Original ATen: [aten.constant_pad_nd, aten.channel_shuffle, aten.tanh, aten.bernoulli, aten._to_copy, aten.mul, aten.add]
# Source node to ATen node mapping:
#   x => constant_pad_nd
#   x_1 => view
#   x_2 => tanh
#   x_3 => add_12, add_13, add_22, convert_element_type, lt, mul_14, mul_15, mul_16
#   x_4 => constant_pad_nd_1
# Graph fragment:
#   %constant_pad_nd : [num_users=3] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%arg2_1, [2, 2, 2, 2], 3.0), kwargs = {})
#   %view : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%constant_pad_nd, [1, 4, %sym_size_int, %sym_size_int_1]), kwargs = {})
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%view,), kwargs = {})
#   %lt : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%inductor_random_default_1, 0.5), kwargs = {})
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt, torch.float32), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%convert_element_type, 0.8864048946659319), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh, %mul_15), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%convert_element_type, -1), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%add_12, 1.558387861036063), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_14, 0.7791939305180315), kwargs = {})
#   %add_22 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_16, %add_13), kwargs = {})
#   %constant_pad_nd_1 : [num_users=3] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%add_22, [1, 1, 1, 1], 2.0), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_add_bernoulli_channel_shuffle_constant_pad_nd_mul_tanh_1(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // ks0) % ks1)
    x0 = (xindex % ks0)
    x2 = xindex // ks4
    x3 = xindex
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = 4 + ks2
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x0
    tmp6 = tmp5 >= tmp1
    tmp7 = 4 + ks3
    tmp8 = tmp5 < tmp7
    tmp9 = tmp2 & tmp4
    tmp10 = tmp9 & tmp6
    tmp11 = tmp10 & tmp8
    tmp12 = (-3) + x1
    tmp13 = tl.full([1], 0, tl.int64)
    tmp14 = tmp12 >= tmp13
    tmp15 = tl.broadcast_to(ks2, [XBLOCK])
    tmp16 = tmp12 < tmp15
    tmp17 = (-3) + x0
    tmp18 = tmp17 >= tmp13
    tmp19 = tl.broadcast_to(ks3, [XBLOCK])
    tmp20 = tmp17 < tmp19
    tmp21 = tmp14 & tmp16
    tmp22 = tmp21 & tmp18
    tmp23 = tmp22 & tmp20
    tmp24 = tmp23 & tmp11
    tmp25 = tl.load(in_ptr0 + ((-3) + x0 + ((-3)*ks3) + ks3*x1 + ks2*ks3*x2), tmp24 & xmask, eviction_policy='evict_last', other=3.0)
    tmp26 = libdevice.tanh(tmp25)
    tmp27 = tl.load(in_ptr1 + (x2), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = 0.5
    tmp29 = tmp27 < tmp28
    tmp30 = tmp29.to(tl.float32)
    tmp31 = 0.8864048946659319
    tmp32 = tmp30 * tmp31
    tmp33 = tmp26 * tmp32
    tmp34 = -1.0
    tmp35 = tmp30 + tmp34
    tmp36 = 1.558387861036063
    tmp37 = tmp35 * tmp36
    tmp38 = 0.7791939305180315
    tmp39 = tmp37 + tmp38
    tmp40 = tmp33 + tmp39
    tmp41 = tl.full(tmp40.shape, 2.0, tmp40.dtype)
    tmp42 = tl.where(tmp11, tmp40, tmp41)
    tl.store(out_ptr0 + (x3), tmp42, xmask)




# kernel path: /tmp/torchinductor_sahanp/fw/cfwwt7gwor2pbfwnyr3rrxx3gpflcqru5ebhetjvbycgafv2pzub.py
# Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.bernoulli]
# Source node to ATen node mapping:
#   x_7 => inductor_lookup_seed_default_1, inductor_random_default
# Graph fragment:
#   %inductor_lookup_seed_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 1), kwargs = {})
#   %inductor_random_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 4, 1, 1], %inductor_lookup_seed_default_1, rand), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_bernoulli_2(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (x0), tmp2, xmask)




# kernel path: /tmp/torchinductor_sahanp/bi/cbini4x56mkrphv2tbeulqeker32brmuxnexjxnx4koiofksxhoy.py
# Topologically Sorted Source Nodes: [x_6, x_7], Original ATen: [aten.tanh, aten.bernoulli, aten._to_copy, aten.mul, aten.add]
# Source node to ATen node mapping:
#   x_6 => tanh_1
#   x_7 => add_39, add_40, add_49, convert_element_type_1, lt_6, mul_51, mul_52, mul_53
# Graph fragment:
#   %tanh_1 : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%view_2,), kwargs = {})
#   %lt_6 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%inductor_random_default, 0.7), kwargs = {})
#   %convert_element_type_1 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt_6, torch.float32), kwargs = {})
#   %mul_52 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%convert_element_type_1, 0.8609526162463561), kwargs = {})
#   %mul_53 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh_1, %mul_52), kwargs = {})
#   %add_39 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%convert_element_type_1, -1), kwargs = {})
#   %mul_51 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%add_39, 1.513640227123543), kwargs = {})
#   %add_40 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_51, 0.4540920681370629), kwargs = {})
#   %add_49 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_53, %add_40), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_add_bernoulli_mul_tanh_3(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex % ks0)
    x3 = xindex // ks0
    x1 = xindex // ks3
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x2 + 36*(x3 // 2) + 72*((x3 % 2)) + 6*ks1*(x3 // 2) + 6*ks2*(x3 // 2) + 12*ks1*((x3 % 2)) + 12*ks2*((x3 % 2)) + ks1*ks2*(x3 // 2) + 2*ks1*ks2*((x3 % 2))), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = libdevice.tanh(tmp0)
    tmp3 = 0.7
    tmp4 = tmp2 < tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = 0.8609526162463561
    tmp7 = tmp5 * tmp6
    tmp8 = tmp1 * tmp7
    tmp9 = -1.0
    tmp10 = tmp5 + tmp9
    tmp11 = 1.513640227123543
    tmp12 = tmp10 * tmp11
    tmp13 = 0.4540920681370629
    tmp14 = tmp12 + tmp13
    tmp15 = tmp8 + tmp14
    tl.store(out_ptr0 + (x4), tmp15, xmask)







def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    s1 = arg0_1
    s2 = arg1_1
    assert_size_stride(arg2_1, (1, 4, s1, s2), (4*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [2], out=buf0)
        buf1 = empty_strided_cuda((1, 4, 1, 1), (4, 1, 4, 4), torch.float32)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.bernoulli]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bernoulli_0[grid(4)](buf0, buf1, 0, 4, XBLOCK=4, num_warps=1, num_stages=1)
        ps0 = 6 + s2
        ps1 = 6 + s1
        ps2 = 36 + 6*s1 + 6*s2 + s1*s2
        buf2 = empty_strided_cuda((1, 4, 6 + s1, 6 + s2), (144 + 24*s1 + 24*s2 + 4*s1*s2, 36 + 6*s1 + 6*s2 + s1*s2, 6 + s2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_1, x_2, x_3, x_4], Original ATen: [aten.constant_pad_nd, aten.channel_shuffle, aten.tanh, aten.bernoulli, aten._to_copy, aten.mul, aten.add]
        triton_poi_fused__to_copy_add_bernoulli_channel_shuffle_constant_pad_nd_mul_tanh_1_xnumel = 144 + 24*s1 + 24*s2 + 4*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_bernoulli_channel_shuffle_constant_pad_nd_mul_tanh_1[grid(triton_poi_fused__to_copy_add_bernoulli_channel_shuffle_constant_pad_nd_mul_tanh_1_xnumel)](arg2_1, buf1, buf2, 38, 38, 32, 32, 1444, 5776, XBLOCK=256, num_warps=4, num_stages=1)
        del arg2_1
        buf3 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.bernoulli]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bernoulli_2[grid(4)](buf0, buf3, 1, 4, XBLOCK=4, num_warps=1, num_stages=1)
        del buf0
        ps3 = 36 + 6*s1 + 6*s2 + s1*s2
        buf4 = empty_strided_cuda((1, 4, 6 + s1, 6 + s2), (144 + 24*s1 + 24*s2 + 4*s1*s2, 36 + 6*s1 + 6*s2 + s1*s2, 6 + s2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_6, x_7], Original ATen: [aten.tanh, aten.bernoulli, aten._to_copy, aten.mul, aten.add]
        triton_poi_fused__to_copy_add_bernoulli_mul_tanh_3_xnumel = 144 + 24*s1 + 24*s2 + 4*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_bernoulli_mul_tanh_3[grid(triton_poi_fused__to_copy_add_bernoulli_mul_tanh_3_xnumel)](buf2, buf3, buf4, 1444, 32, 32, 1444, 5776, XBLOCK=128, num_warps=4, num_stages=1)
        del buf2
        del buf3
    return (buf4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 32
    arg1_1 = 32
    arg2_1 = rand_strided((1, 4, 32, 32), (4096, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
