# AOT ID: ['44_forward']
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


# kernel path: /tmp/torchinductor_sahanp/wy/cwy4caf7bk3n5joanlil4bh6n4b7t5nnjwkdbir6kahcm2cpfhc3.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_2 => _low_memory_max_pool2d_with_offsets
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%unsqueeze, [1, 2], [1, 2], [0, 0], [1, 1], False), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (1 + 2*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl_math.abs(tmp0)
    tmp2 = 0.5
    tmp3 = tmp1 > tmp2
    tmp4 = tl.full([1], 0, tl.int32)
    tmp5 = tmp4 < tmp0
    tmp6 = tmp5.to(tl.int8)
    tmp7 = tmp0 < tmp4
    tmp8 = tmp7.to(tl.int8)
    tmp9 = tmp6 - tmp8
    tmp10 = tmp9.to(tmp0.dtype)
    tmp11 = tmp10 * tmp2
    tmp12 = tmp0 - tmp11
    tmp13 = 0.0
    tmp14 = tmp0 * tmp13
    tmp15 = tl.where(tmp3, tmp12, tmp14)
    tmp17 = tl_math.abs(tmp16)
    tmp18 = tmp17 > tmp2
    tmp19 = tmp4 < tmp16
    tmp20 = tmp19.to(tl.int8)
    tmp21 = tmp16 < tmp4
    tmp22 = tmp21.to(tl.int8)
    tmp23 = tmp20 - tmp22
    tmp24 = tmp23.to(tmp16.dtype)
    tmp25 = tmp24 * tmp2
    tmp26 = tmp16 - tmp25
    tmp27 = tmp16 * tmp13
    tmp28 = tl.where(tmp18, tmp26, tmp27)
    tmp29 = triton_helpers.maximum(tmp28, tmp15)
    tl.store(out_ptr0 + (x0), tmp29, xmask)




# kernel path: /tmp/torchinductor_sahanp/dv/cdvmbwuh36awblxnu6os7b23k5ginxqxrg6wq5e6relmu3ucqjn3.py
# Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.bernoulli]
# Source node to ATen node mapping:
#   x_5 => inductor_lookup_seed_default, inductor_random_default, lt
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 32, 1, 1, 1], %inductor_lookup_seed_default, rand), kwargs = {})
#   %lt : [num_users=2] = call_function[target=torch.ops.aten.lt.Scalar](args = (%inductor_random_default, 0.5), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_bernoulli_1(in_ptr0, out_ptr1, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.5
    tmp4 = tmp2 < tmp3
    tl.store(out_ptr1 + (x0), tmp4, xmask)




#include "/tmp/torchinductor_sahanp/3b/c3bi5gk6mslf6u4iaqafhxm64z6u65e3eain4xlary5blqnvv6xx.h"
extern "C"  void kernel(const int64_t* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(2080L); x0+=static_cast<int64_t>(1L))
        {
            {
                {
                    auto tmp0 = in_ptr0[static_cast<int64_t>(0L)];
                    auto tmp1 = x0;
                    auto tmp2 = c10::convert<int32_t>(tmp1);
                    auto tmp3 = static_cast<int64_t>(0);
                    auto tmp4 = static_cast<int64_t>(10);
                    auto tmp5 = randint64_cpu(tmp0, tmp2, tmp3, tmp4);
                    auto tmp6 = c10::convert<float>(tmp5);
                    out_ptr0[static_cast<int64_t>(x0)] = tmp6;
                }
            }
        }
    }
}
''')


# kernel path: /tmp/torchinductor_sahanp/57/c57qvyinktzfrnt4p4vxqav6tk36dm62espbmmwozrzc4kqbqajx.py
# Topologically Sorted Source Nodes: [x_3, loss], Original ATen: [aten.convolution, aten.mul, aten.exp, aten.sub, aten.mean]
# Source node to ATen node mapping:
#   loss => exp, mean, mul_3, sub_1
#   x_3 => convolution
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%squeeze, %primals_2, %primals_3, [2], [0], [1], True, [0], 1), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%device_put, %view_2), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%view_2,), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%exp, %mul_3), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sub_1,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_exp_mean_mul_sub_3(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 2080
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp13 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        r0_1 = r0_index // 65
        tmp0 = tl.load(in_out_ptr0 + (r0_2), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r0_2 // 65), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.int1)
        tmp9 = tl.load(in_ptr2 + (r0_2), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 2.0
        tmp6 = tmp4 * tmp5
        tmp7 = tmp2 * tmp6
        tmp8 = tl_math.exp(tmp7)
        tmp10 = tmp9 * tmp7
        tmp11 = tmp8 - tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(r0_mask, tmp14, _tmp13)
        tl.store(in_out_ptr0 + (tl.broadcast_to(r0_2, [XBLOCK, R0_BLOCK])), tmp2, r0_mask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tmp15 = 2080.0
    tmp16 = tmp13 / tmp15
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp16, None)







def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (1, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_2, (16, 32, 3), (96, 3, 1))
    assert_size_stride(primals_3, (32, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 16, 1, 32), (512, 32, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_0[grid(512)](primals_1, buf0, 512, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_1
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(reinterpret_tensor(buf0, (1, 16, 32), (0, 32, 1), 0), primals_2, stride=(2,), padding=(0,), dilation=(1,), transposed=True, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf1, (1, 32, 65), (2080, 65, 1))
        buf3 = empty_strided_cuda((1, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf3)
        buf5 = empty_strided_cuda((1, 32, 1, 1, 1), (32, 1, 1, 1, 1), torch.bool)
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.bernoulli]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bernoulli_1[grid(32)](buf3, buf5, 0, 32, XBLOCK=32, num_warps=1, num_stages=1)
        del buf3
    buf6 = empty_strided_cpu((1, ), (1, ), torch.int64)
    # Topologically Sorted Source Nodes: [], Original ATen: []
    aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf6)
    buf7 = empty_strided_cpu((1, 2080), (2080, 1), torch.float32)
    cpp_fused_randint_2(buf6, buf7)
    del buf6
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf8 = empty_strided_cuda((1, 2080), (2080, 1), torch.float32)
        buf8.copy_(buf7, False)
        del buf7
        buf2 = buf1; del buf1  # reuse
        buf9 = empty_strided_cuda((), (), torch.float32)
        buf10 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [x_3, loss], Original ATen: [aten.convolution, aten.mul, aten.exp, aten.sub, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_exp_mean_mul_sub_3[grid(1)](buf2, buf10, primals_3, buf5, buf8, 1, 2080, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del primals_3
    return (buf10, primals_2, reinterpret_tensor(buf0, (1, 16, 32), (512, 32, 1), 0), buf2, buf5, buf8, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((16, 32, 3), (96, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
