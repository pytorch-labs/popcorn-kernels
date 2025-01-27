# AOT ID: ['3_inference']
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


# kernel path: /tmp/torchinductor_sahanp/ii/ciidg7aruzrbbya3ow4h2oio23xu3rba5knkppbwcaiymv75j3pw.py
# Topologically Sorted Source Nodes: [x, result], Original ATen: [aten.constant_pad_nd, aten.threshold]
# Source node to ATen node mapping:
#   result => full_default, le, where
#   x => constant_pad_nd
# Graph fragment:
#   %constant_pad_nd : [num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%arg0_1, [2, 2, 2, 2], 0.0), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%constant_pad_nd, 0.5), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le, %full_default, %constant_pad_nd), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_constant_pad_nd_threshold_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3888
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 36) % 36)
    x0 = (xindex % 36)
    x2 = xindex // 1296
    x4 = (xindex % 1296)
    tmp0 = (-2) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + x0
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-66) + x0 + 32*x1 + 1024*x2), tmp10 & xmask, other=0.0)
    tmp12 = 0.5
    tmp13 = tmp11 <= tmp12
    tmp14 = 0.0
    tmp15 = tl.where(tmp13, tmp14, tmp11)
    tl.store(out_ptr0 + (x4 + 1312*x2), tmp15, xmask)




# kernel path: /tmp/torchinductor_sahanp/iz/cizr2y7h2tk5v3oilthhx2gkszbdm6onmjhy27fvm4sjwvr5uka7.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.bernoulli]
# Source node to ATen node mapping:
#   x_3 => inductor_lookup_seed_default, inductor_random_default
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 3, 1], %inductor_lookup_seed_default, rand), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_bernoulli_1(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (x0), tmp2, xmask)




# kernel path: /tmp/torchinductor_sahanp/4v/c4vzqtnxc2xyoxpqd2ogp47rpeykl5k53rwlp5iz3nueib4z7ln2.py
# Topologically Sorted Source Nodes: [x_3, x_4], Original ATen: [aten.bernoulli, aten._to_copy, aten.mul, aten.add, aten.neg, aten._softmax]
# Source node to ATen node mapping:
#   x_3 => add, add_1, add_2, convert_element_type, lt, mul, mul_1, mul_2
#   x_4 => amax, exp, neg, sub, sum_1
# Graph fragment:
#   %lt : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%inductor_random_default, 0.5), kwargs = {})
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt, torch.float32), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%convert_element_type, 0.8864048946659319), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze, %mul_1), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%convert_element_type, -1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%add, 1.558387861036063), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul, 0.7791939305180315), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %add_1), kwargs = {})
#   %neg : [num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%add_2,), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%neg, [1], True), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%neg, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__softmax__to_copy_add_bernoulli_mul_neg_2(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp17 = tl.load(in_ptr0 + (10 + x0), xmask)
    tmp18 = tl.load(in_ptr1 + (1))
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK])
    tmp30 = tl.load(in_ptr0 + (20 + x0), xmask)
    tmp31 = tl.load(in_ptr1 + (2))
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK])
    tmp3 = 0.5
    tmp4 = tmp2 < tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = 0.8864048946659319
    tmp7 = tmp5 * tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = -1.0
    tmp10 = tmp5 + tmp9
    tmp11 = 1.558387861036063
    tmp12 = tmp10 * tmp11
    tmp13 = 0.7791939305180315
    tmp14 = tmp12 + tmp13
    tmp15 = tmp8 + tmp14
    tmp16 = -tmp15
    tmp20 = tmp19 < tmp3
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp21 * tmp6
    tmp23 = tmp17 * tmp22
    tmp24 = tmp21 + tmp9
    tmp25 = tmp24 * tmp11
    tmp26 = tmp25 + tmp13
    tmp27 = tmp23 + tmp26
    tmp28 = -tmp27
    tmp29 = triton_helpers.maximum(tmp16, tmp28)
    tmp33 = tmp32 < tmp3
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp34 * tmp6
    tmp36 = tmp30 * tmp35
    tmp37 = tmp34 + tmp9
    tmp38 = tmp37 * tmp11
    tmp39 = tmp38 + tmp13
    tmp40 = tmp36 + tmp39
    tmp41 = -tmp40
    tmp42 = triton_helpers.maximum(tmp29, tmp41)
    tmp43 = tmp16 - tmp42
    tmp44 = tl_math.exp(tmp43)
    tmp45 = tmp28 - tmp42
    tmp46 = tl_math.exp(tmp45)
    tmp47 = tmp44 + tmp46
    tmp48 = tmp41 - tmp42
    tmp49 = tl_math.exp(tmp48)
    tmp50 = tmp47 + tmp49
    tl.store(out_ptr0 + (x0), tmp42, xmask)
    tl.store(out_ptr1 + (x0), tmp50, xmask)




# kernel path: /tmp/torchinductor_sahanp/nw/cnwfr2lzvfoxmbo2lfsx7ovke3gfoyrcrotu5qzam4xdxj5bvbzf.py
# Topologically Sorted Source Nodes: [x_3, x_4], Original ATen: [aten.bernoulli, aten._to_copy, aten.mul, aten.add, aten.neg, aten._softmax]
# Source node to ATen node mapping:
#   x_3 => add, add_1, add_2, convert_element_type, lt, mul, mul_1, mul_2
#   x_4 => div, exp, neg, sub
# Graph fragment:
#   %lt : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%inductor_random_default, 0.5), kwargs = {})
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt, torch.float32), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%convert_element_type, 0.8864048946659319), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze, %mul_1), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%convert_element_type, -1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%add, 1.558387861036063), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul, 0.7791939305180315), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %add_1), kwargs = {})
#   %neg : [num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%add_2,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%neg, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__softmax__to_copy_add_bernoulli_mul_neg_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 30
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 10
    x0 = (xindex % 10)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
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
    tmp15 = -tmp14
    tmp17 = tmp15 - tmp16
    tmp18 = tl_math.exp(tmp17)
    tmp20 = tmp18 / tmp19
    tl.store(in_out_ptr0 + (x2), tmp20, xmask)




#include "/tmp/torchinductor_sahanp/3b/c3bi5gk6mslf6u4iaqafhxm64z6u65e3eain4xlary5blqnvv6xx.h"
extern "C"  void kernel(const int64_t* in_ptr0,
                       int64_t* out_ptr0,
                       int64_t* out_ptr1,
                       int64_t* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(5L); x0+=static_cast<int64_t>(1L))
        {
            {
                {
                    auto tmp0 = in_ptr0[static_cast<int64_t>(0L)];
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
    {
        {
            {
                auto tmp0 = static_cast<int64_t>(30);
                out_ptr1[static_cast<int64_t>(0L)] = tmp0;
            }
        }
    }
    {
        {
            {
                auto tmp0 = in_ptr0[static_cast<int64_t>(1L)];
                auto tmp1 = static_cast<int32_t>(0);
                auto tmp2 = static_cast<int64_t>(1);
                auto tmp3 = static_cast<int64_t>(6);
                auto tmp4 = randint64_cpu(tmp0, tmp1, tmp2, tmp3);
                out_ptr2[static_cast<int64_t>(0L)] = tmp4;
            }
        }
    }
}
''')





def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (1, 3, 32, 32), (3072, 1024, 32, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 3, 36, 36), (3936, 1312, 36, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, result], Original ATen: [aten.constant_pad_nd, aten.threshold]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_threshold_0[grid(3888)](arg0_1, buf0, 3888, XBLOCK=256, num_warps=4, num_stages=1)
        del arg0_1
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten._adaptive_avg_pool2d]
        buf1 = torch.ops.aten._adaptive_avg_pool2d.default(reinterpret_tensor(buf0, (1, 3, 1, 1296), (0, 1312, 0, 1), 0), [1, 10])
        del buf0
        buf2 = buf1
        del buf1
        buf3 = empty_strided_cuda((1, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf3)
        buf4 = empty_strided_cuda((1, 3, 1), (3, 1, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.bernoulli]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bernoulli_1[grid(3)](buf3, buf4, 0, 3, XBLOCK=4, num_warps=1, num_stages=1)
        del buf3
        buf5 = empty_strided_cuda((1, 1, 10), (10, 10, 1), torch.float32)
        buf6 = empty_strided_cuda((1, 1, 10), (10, 10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_3, x_4], Original ATen: [aten.bernoulli, aten._to_copy, aten.mul, aten.add, aten.neg, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax__to_copy_add_bernoulli_mul_neg_2[grid(10)](buf2, buf4, buf5, buf6, 10, XBLOCK=16, num_warps=1, num_stages=1)
        buf7 = reinterpret_tensor(buf2, (1, 3, 10), (30, 10, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [x_3, x_4], Original ATen: [aten.bernoulli, aten._to_copy, aten.mul, aten.add, aten.neg, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax__to_copy_add_bernoulli_mul_neg_3[grid(30)](buf7, buf4, buf5, buf6, 30, XBLOCK=32, num_warps=1, num_stages=1)
        del buf4
        del buf5
        del buf6
    buf8 = empty_strided_cpu((2, ), (1, ), torch.int64)
    # Topologically Sorted Source Nodes: [], Original ATen: []
    aten.randint.low_out(-9223372036854775808, 9223372036854775807, [2], out=buf8)
    buf9 = empty_strided_cpu((1, 5), (5, 1), torch.int64)
    buf10 = empty_strided_cpu((1, ), (1, ), torch.int64)
    buf11 = empty_strided_cpu((1, ), (1, ), torch.int64)
    cpp_fused_full_randint_4(buf8, buf9, buf10, buf11)
    return (reinterpret_tensor(buf7, (1, 30), (30, 1), 0), buf9, buf10, buf11, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 3, 32, 32), (3072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
