# AOT ID: ['34_inference']
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


#include "/tmp/torchinductor_sahanp/3b/c3bi5gk6mslf6u4iaqafhxm64z6u65e3eain4xlary5blqnvv6xx.h"
extern "C"  void kernel(int64_t* in_out_ptr0,
                       const int64_t ks0)
{
    {
        {
            {
                auto tmp0 = in_out_ptr0[static_cast<int64_t>(0L)];
                auto tmp1 = static_cast<int32_t>(0);
                auto tmp2 = static_cast<int64_t>(0);
                auto tmp3 = ks0;
                auto tmp4 = c10::convert<int64_t>(tmp3);
                auto tmp5 = randint64_cpu(tmp0, tmp1, tmp2, tmp4);
                in_out_ptr0[static_cast<int64_t>(0L)] = tmp5;
            }
        }
    }
}
''')


# kernel path: /tmp/torchinductor_sahanp/5x/c5xk5faqnvzrdah3stj7jtqncyw6brxtvvylxumqw77fubc7nkys.py
# Topologically Sorted Source Nodes: [x_2, x_4, loss], Original ATen: [aten.bernoulli, aten.exp, aten.mul, aten.sub, aten.mean]
# Source node to ATen node mapping:
#   loss => exp, mean, mul_31, sub_14
#   x_2 => inductor_lookup_seed_default, inductor_random_default_1
#   x_4 => inductor_lookup_seed_default_1, inductor_random_default
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 1, 1], %inductor_lookup_seed_default, rand), kwargs = {})
#   %inductor_lookup_seed_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 1), kwargs = {})
#   %inductor_random_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 1, 1], %inductor_lookup_seed_default_1, rand), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%view,), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%device_put, %view), kwargs = {})
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%exp, %mul_31), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sub_14,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_bernoulli_exp_mean_mul_sub_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, load_seed_offset, load_seed_offset1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = tl.full([1, 1], 0, tl.int32)
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = tl.load(in_ptr0 + load_seed_offset1)
    tmp4 = tl.rand(tmp3, (tmp1).to(tl.uint32))
    tmp29 = tl.load(in_ptr2 + (0))
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, R0_BLOCK])
    _tmp35 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp5 = tl.load(in_ptr1 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp6 = 0.5
        tmp7 = tmp5 <= tmp6
        tmp8 = 1.0
        tmp9 = tl.where(tmp7, tmp8, tmp5)
        tmp10 = tl_math.abs(tmp9)
        tmp11 = tmp10 <= tmp6
        tmp12 = 0.0
        tmp13 = tl.where(tmp11, tmp12, tmp9)
        tmp14 = tmp2 < tmp6
        tmp15 = tmp14.to(tl.float32)
        tmp16 = 2.0
        tmp17 = tmp15 * tmp16
        tmp18 = tmp13 * tmp17
        tmp19 = tmp18 <= tmp6
        tmp20 = tl.where(tmp19, tmp8, tmp18)
        tmp21 = tl_math.abs(tmp20)
        tmp22 = tmp21 <= tmp6
        tmp23 = tl.where(tmp22, tmp12, tmp20)
        tmp24 = tmp4 < tmp6
        tmp25 = tmp24.to(tl.float32)
        tmp26 = tmp25 * tmp16
        tmp27 = tmp23 * tmp26
        tmp28 = tl_math.exp(tmp27)
        tmp31 = tmp30.to(tl.float32)
        tmp32 = tmp31 * tmp27
        tmp33 = tmp28 - tmp32
        tmp34 = tl.broadcast_to(tmp33, [XBLOCK, R0_BLOCK])
        tmp36 = _tmp35 + tmp34
        _tmp35 = tl.where(r0_mask, tmp36, _tmp35)
    tmp35 = tl.sum(_tmp35, 1)[:, None]
    tmp37 = ks2
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tmp35 / tmp38
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp39, None)







def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    s0 = arg0_1
    assert_size_stride(arg1_1, (1, s0), (s0, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [2], out=buf0)
    buf3 = empty_strided_cpu((1, ), (1, ), torch.int64)
    # Topologically Sorted Source Nodes: [], Original ATen: []
    aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf3)
    buf4 = buf3; del buf3  # reuse
    cpp_fused_randint_0(buf4, s0)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf5 = empty_strided_cuda((1, ), (1, ), torch.int64)
        buf5.copy_(buf4, False)
        del buf4
        buf1 = empty_strided_cuda((1, 1, 1), (1, 1, 1), torch.float32)
        buf6 = reinterpret_tensor(buf1, (), (), 0); del buf1  # reuse
        buf7 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [x_2, x_4, loss], Original ATen: [aten.bernoulli, aten.exp, aten.mul, aten.sub, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_bernoulli_exp_mean_mul_sub_1[grid(1)](buf7, buf0, arg1_1, buf5, 0, 1, 10, 1, 10, XBLOCK=1, R0_BLOCK=16, num_warps=2, num_stages=1)
        del arg1_1
        del buf0
        del buf5
    return (buf7, )


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
