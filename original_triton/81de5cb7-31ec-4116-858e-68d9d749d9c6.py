# AOT ID: ['4_inference']
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


# kernel path: /tmp/torchinductor_sahanp/ur/curfjokdkohdpwskpg4dagtw7quy56vv5tpfgarcv3jfjuvqxxxb.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.rand]
# Source node to ATen node mapping:
#   x_1 => inductor_lookup_seed_default, inductor_random_default_3
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default_3 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 3, 3], %inductor_lookup_seed_default, rand), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_rand_0(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (x0), tmp2, xmask)




# kernel path: /tmp/torchinductor_sahanp/ys/cys2dbddaerb274k3rwwklrstq2vjs4kjstc7peq5j3zmtgve64k.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x => constant_pad_nd
# Graph fragment:
#   %constant_pad_nd : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%arg0_1, [1, 1, 1, 1, 1, 1], 0.5), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_constant_pad_nd_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 31944
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 484) % 22)
    x1 = ((xindex // 22) % 22)
    x0 = (xindex % 22)
    x3 = xindex // 10648
    x7 = (xindex % 10648)
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 20, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = (-1) + x0
    tmp9 = tmp8 >= tmp1
    tmp10 = tmp8 < tmp3
    tmp11 = tmp2 & tmp4
    tmp12 = tmp11 & tmp6
    tmp13 = tmp12 & tmp7
    tmp14 = tmp13 & tmp9
    tmp15 = tmp14 & tmp10
    tmp16 = tl.load(in_ptr0 + ((-421) + x0 + 20*x1 + 400*x2 + 8000*x3), tmp15 & xmask, other=0.5)
    tl.store(out_ptr0 + (x7 + 10656*x3), tmp16, xmask)




# kernel path: /tmp/torchinductor_sahanp/wl/cwliezc5z3tawpeh72wcydqpluumpi7cbmqgtpjxbvr6nhvqvhy4.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.bernoulli]
# Source node to ATen node mapping:
#   x_3 => inductor_lookup_seed_default_1, inductor_random_default_2
# Graph fragment:
#   %inductor_lookup_seed_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 1), kwargs = {})
#   %inductor_random_default_2 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 3, 1, 1, 1], %inductor_lookup_seed_default_1, rand), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_bernoulli_2(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (x0), tmp2, xmask)




# kernel path: /tmp/torchinductor_sahanp/he/chel47ppjtvu67wxoanzzp7w37ejhst6ogop5nltwuejmgimqp5c.py
# Topologically Sorted Source Nodes: [x_2, x_3, x_4], Original ATen: [aten.hardsigmoid, aten.bernoulli, aten._to_copy, aten.mul, aten.add, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_2 => add, clamp_max, clamp_min, div
#   x_3 => add_1, add_2, add_3, convert_element_type, lt, mul, mul_1, mul_2
#   x_4 => constant_pad_nd_1
# Graph fragment:
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 3), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add, 0), kwargs = {})
#   %clamp_max : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 6), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%clamp_max, 6), kwargs = {})
#   %lt : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%inductor_random_default_2, 0.5), kwargs = {})
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt, torch.float32), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%convert_element_type, 0.8864048946659319), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %mul_1), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%convert_element_type, -1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%add_1, 1.558387861036063), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul, 0.7791939305180315), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %add_2), kwargs = {})
#   %constant_pad_nd_1 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%add_3, [2, 2, 2, 2, 2, 2], 0.25), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_add_bernoulli_constant_pad_nd_hardsigmoid_mul_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8232
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 196) % 14)
    x1 = ((xindex // 14) % 14)
    x0 = (xindex % 14)
    x3 = xindex // 2744
    x7 = (xindex % 2744)
    tmp0 = (-2) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 10, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = (-2) + x0
    tmp9 = tmp8 >= tmp1
    tmp10 = tmp8 < tmp3
    tmp11 = tmp2 & tmp4
    tmp12 = tmp11 & tmp6
    tmp13 = tmp12 & tmp7
    tmp14 = tmp13 & tmp9
    tmp15 = tmp14 & tmp10
    tmp16 = tl.load(in_ptr0 + ((-222) + x0 + 10*x1 + 100*x2 + 1000*x3), tmp15 & xmask, other=0.0)
    tmp17 = 3.0
    tmp18 = tmp16 + tmp17
    tmp19 = 0.0
    tmp20 = triton_helpers.maximum(tmp18, tmp19)
    tmp21 = 6.0
    tmp22 = triton_helpers.minimum(tmp20, tmp21)
    tmp23 = 0.16666666666666666
    tmp24 = tmp22 * tmp23
    tmp25 = tl.load(in_ptr1 + (x3), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = 0.5
    tmp27 = tmp25 < tmp26
    tmp28 = tmp27.to(tl.float32)
    tmp29 = 0.8864048946659319
    tmp30 = tmp28 * tmp29
    tmp31 = tmp24 * tmp30
    tmp32 = -1.0
    tmp33 = tmp28 + tmp32
    tmp34 = 1.558387861036063
    tmp35 = tmp33 * tmp34
    tmp36 = 0.7791939305180315
    tmp37 = tmp35 + tmp36
    tmp38 = tmp31 + tmp37
    tmp39 = tl.full(tmp38.shape, 0.25, tmp38.dtype)
    tmp40 = tl.where(tmp15, tmp38, tmp39)
    tl.store(out_ptr0 + (x7 + 2752*x3), tmp40, xmask)




# kernel path: /tmp/torchinductor_sahanp/4c/c4cxm4dht4zlhspgefoszebmxu5g6v7xund75vk4bktvzte56ugs.py
# Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.bernoulli]
# Source node to ATen node mapping:
#   x_7 => inductor_lookup_seed_default_3, inductor_random_default
# Graph fragment:
#   %inductor_lookup_seed_default_3 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 3), kwargs = {})
#   %inductor_random_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 3, 1, 1, 1], %inductor_lookup_seed_default_3, rand), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_bernoulli_4(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (x0), tmp2, xmask)




# kernel path: /tmp/torchinductor_sahanp/tq/ctqmx5ep32mu3f7rkh5ifnmd64ia6wasduepny3wu3tzuxqtg63n.py
# Topologically Sorted Source Nodes: [x_6, x_7], Original ATen: [aten.hardsigmoid, aten.bernoulli, aten._to_copy, aten.mul, aten.add]
# Source node to ATen node mapping:
#   x_6 => add_4, clamp_max_1, clamp_min_1, div_1
#   x_7 => add_5, add_6, add_7, convert_element_type_1, lt_1, mul_3, mul_4, mul_5
# Graph fragment:
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 3), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_4, 0), kwargs = {})
#   %clamp_max_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_1, 6), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%clamp_max_1, 6), kwargs = {})
#   %lt_1 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%inductor_random_default, 0.7), kwargs = {})
#   %convert_element_type_1 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt_1, torch.float32), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%convert_element_type_1, 0.8609526162463561), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %mul_4), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%convert_element_type_1, -1), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%add_5, 1.513640227123543), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_3, 0.4540920681370629), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %add_6), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_add_bernoulli_hardsigmoid_mul_5(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 375
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 125
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp9 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = 0.16666666666666666
    tmp8 = tmp6 * tmp7
    tmp10 = 0.7
    tmp11 = tmp9 < tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp13 = 0.8609526162463561
    tmp14 = tmp12 * tmp13
    tmp15 = tmp8 * tmp14
    tmp16 = -1.0
    tmp17 = tmp12 + tmp16
    tmp18 = 1.513640227123543
    tmp19 = tmp17 * tmp18
    tmp20 = 0.4540920681370629
    tmp21 = tmp19 + tmp20
    tmp22 = tmp15 + tmp21
    tl.store(in_out_ptr0 + (x2), tmp22, xmask)







def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (1, 3, 20, 20, 20), (24000, 8000, 400, 20, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [4], out=buf0)
        buf1 = empty_strided_cuda((1, 3, 3), (9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.rand]
        stream0 = get_raw_stream(0)
        triton_poi_fused_rand_0[grid(9)](buf0, buf1, 2, 9, XBLOCK=16, num_warps=1, num_stages=1)
        buf2 = empty_strided_cuda((1, 3, 22, 22, 22), (31968, 10656, 484, 22, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_1[grid(31944)](arg0_1, buf2, 31944, XBLOCK=256, num_warps=4, num_stages=1)
        del arg0_1
        # Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.constant_pad_nd, aten.fractional_max_pool3d]
        buf3 = torch.ops.aten.fractional_max_pool3d.default(buf2, [2, 2, 2], [10, 10, 10], buf1)
        del buf2
        buf4 = buf3[0]
        del buf3
        buf6 = empty_strided_cuda((1, 3, 1, 1, 1), (3, 1, 3, 3, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.bernoulli]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bernoulli_2[grid(3)](buf0, buf6, 1, 3, XBLOCK=4, num_warps=1, num_stages=1)
        buf7 = empty_strided_cuda((1, 3, 14, 14, 14), (8256, 2752, 196, 14, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2, x_3, x_4], Original ATen: [aten.hardsigmoid, aten.bernoulli, aten._to_copy, aten.mul, aten.add, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_bernoulli_constant_pad_nd_hardsigmoid_mul_3[grid(8232)](buf4, buf6, buf7, 8232, XBLOCK=256, num_warps=4, num_stages=1)
        del buf4
        buf8 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.rand]
        stream0 = get_raw_stream(0)
        triton_poi_fused_rand_0[grid(9)](buf0, buf8, 2, 9, XBLOCK=16, num_warps=1, num_stages=1)
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.fractional_max_pool3d]
        buf9 = torch.ops.aten.fractional_max_pool3d.default(buf7, [3, 3, 3], [5, 5, 5], buf8)
        del buf7
        del buf8
        buf10 = buf9[0]
        del buf9
        buf12 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.bernoulli]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bernoulli_4[grid(3)](buf0, buf12, 3, 3, XBLOCK=4, num_warps=1, num_stages=1)
        del buf0
        buf13 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [x_6, x_7], Original ATen: [aten.hardsigmoid, aten.bernoulli, aten._to_copy, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_bernoulli_hardsigmoid_mul_5[grid(375)](buf13, buf12, 375, XBLOCK=128, num_warps=4, num_stages=1)
        del buf12
    return (buf13, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 3, 20, 20, 20), (24000, 8000, 400, 20, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
