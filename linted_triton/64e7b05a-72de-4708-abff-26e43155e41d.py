# AOT ID: ['160_inference']
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


# kernel path: /tmp/torchinductor_sahanp/2a/c2ahxeitjzj4jtkhiq7envho5dkgsvlopenuby2apnn5ig55l4mh.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.replication_pad2d]
# Source node to ATen node mapping:
#   x => _unsafe_index, _unsafe_index_1
# Graph fragment:
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg3_1, [None, None, %clamp_max, None]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index, [None, None, None, %clamp_max_1]), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_replication_pad2d_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 68)
    x1 = ((xindex // 68) % 68)
    x2 = xindex // 4624
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (64*((63) * ((63) <= (((0) * ((0) >= ((-2) + x1)) + ((-2) + x1) * (((-2) + x1) > (0))))) + (((0) * ((0) >= ((-2) + x1)) + ((-2) + x1) * (((-2) + x1) > (0)))) * ((((0) * ((0) >= ((-2) + x1)) + ((-2) + x1) * (((-2) + x1) > (0)))) < (63))) + 4096*x2 + ((63) * ((63) <= (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0))))) + (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) * ((((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) < (63)))), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3), tmp0, xmask)




#include "/tmp/torchinductor_sahanp/3b/c3bi5gk6mslf6u4iaqafhxm64z6u65e3eain4xlary5blqnvv6xx.h"
extern "C"  void kernel(const int64_t* in_ptr0,
                       float* out_ptr0,
                       const int64_t ks0)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(25L*ks0); x0+=static_cast<int64_t>(1L))
        {
            {
                {
                    auto tmp0 = in_ptr0[static_cast<int64_t>(0L)];
                    auto tmp1 = x0;
                    auto tmp2 = c10::convert<int32_t>(tmp1);
                    auto tmp3 = static_cast<int64_t>(0);
                    auto tmp4 = static_cast<int64_t>(2);
                    auto tmp5 = randint64_cpu(tmp0, tmp2, tmp3, tmp4);
                    auto tmp6 = c10::convert<float>(tmp5);
                    out_ptr0[static_cast<int64_t>(x0)] = tmp6;
                }
            }
        }
    }
}
''')


# kernel path: /tmp/torchinductor_sahanp/ph/cphqlofxwd6fpx4zljz6javk6ezz4ippq7vglyms7gqxk7wpxmbq.py
# Topologically Sorted Source Nodes: [x_2, target_smooth_l1, loss_smooth_l1, loss_bce], Original ATen: [aten.bernoulli, aten.randn_like, aten.smooth_l1_loss, aten.binary_cross_entropy_with_logits]
# Source node to ATen node mapping:
#   loss_bce => abs_2, exp, full_default, log1p, mean_1, minimum, mul_67, neg, sub_30, sub_31, sub_32
#   loss_smooth_l1 => abs_1, div, lt_5, mean, mul_66, pow_1, sub_28, sub_29, where
#   target_smooth_l1 => inductor_lookup_seed_default_1, inductor_random_default
#   x_2 => inductor_lookup_seed_default, inductor_random_default_1
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, %arg0_1, 5, 5], %inductor_lookup_seed_default, rand), kwargs = {})
#   %inductor_lookup_seed_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 1), kwargs = {})
#   %inductor_random_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, %sym_size_int_4], %inductor_lookup_seed_default_1, randn), kwargs = {})
#   %sub_28 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_1, %inductor_random_default), kwargs = {})
#   %abs_1 : [num_users=3] = call_function[target=torch.ops.aten.abs.default](args = (%sub_28,), kwargs = {})
#   %lt_5 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%abs_1, 1.0), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%abs_1, 2), kwargs = {})
#   %mul_66 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_1, 0.5), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_66, 1.0), kwargs = {})
#   %sub_29 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%abs_1, 0.5), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%lt_5, %div, %sub_29), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%where,), kwargs = {})
#   %sub_30 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %device_put), kwargs = {})
#   %mul_67 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_30, %view_1), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %minimum : [num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%full_default, %view_1), kwargs = {})
#   %abs_2 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%view_1,), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%abs_2,), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %log1p : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp,), kwargs = {})
#   %sub_31 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%minimum, %log1p), kwargs = {})
#   %sub_32 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_67, %sub_31), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sub_32,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_bernoulli_binary_cross_entropy_with_logits_randn_like_smooth_l1_loss_2(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, load_seed_offset, load_seed_offset1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp29 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp43 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp5 = tl.load(in_ptr1 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp31 = tl.load(in_ptr2 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp0 = tl.load(in_ptr0 + load_seed_offset)
        tmp1 = r0_0
        tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
        tmp3 = tl.load(in_ptr0 + load_seed_offset1)
        tmp4 = tl.randn(tmp3, (tmp1).to(tl.uint32))
        tmp6 = 0.5
        tmp7 = tmp2 < tmp6
        tmp8 = tmp7.to(tl.float32)
        tmp9 = 0.8864048946659319
        tmp10 = tmp8 * tmp9
        tmp11 = tmp5 * tmp10
        tmp12 = -1.0
        tmp13 = tmp8 + tmp12
        tmp14 = 1.558387861036063
        tmp15 = tmp13 * tmp14
        tmp16 = 0.7791939305180315
        tmp17 = tmp15 + tmp16
        tmp18 = tmp11 + tmp17
        tmp19 = tmp18 - tmp4
        tmp20 = tl_math.abs(tmp19)
        tmp21 = 1.0
        tmp22 = tmp20 < tmp21
        tmp23 = tmp20 * tmp20
        tmp24 = tmp23 * tmp6
        tmp25 = tmp24 * tmp21
        tmp26 = tmp20 - tmp6
        tmp27 = tl.where(tmp22, tmp25, tmp26)
        tmp28 = tl.broadcast_to(tmp27, [XBLOCK, R0_BLOCK])
        tmp30 = _tmp29 + tmp28
        _tmp29 = tl.where(r0_mask, tmp30, _tmp29)
        tmp32 = tmp21 - tmp31
        tmp33 = tmp32 * tmp18
        tmp34 = 0.0
        tmp35 = triton_helpers.minimum(tmp34, tmp18)
        tmp36 = tl_math.abs(tmp18)
        tmp37 = -tmp36
        tmp38 = tl_math.exp(tmp37)
        tmp39 = libdevice.log1p(tmp38)
        tmp40 = tmp35 - tmp39
        tmp41 = tmp33 - tmp40
        tmp42 = tl.broadcast_to(tmp41, [XBLOCK, R0_BLOCK])
        tmp44 = _tmp43 + tmp42
        _tmp43 = tl.where(r0_mask, tmp44, _tmp43)
    tmp29 = tl.sum(_tmp29, 1)[:, None]
    tmp43 = tl.sum(_tmp43, 1)[:, None]
    tmp45 = 25*ks2
    tmp46 = tmp45.to(tl.float32)
    tmp47 = tmp43 / tmp46
    tmp48 = tmp29 / tmp46
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp47, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp48, None)







def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, 64, 64), (4096*s0, 4096, 64, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, s0, 68, 68), (4624*s0, 4624, 68, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.replication_pad2d]
        triton_poi_fused_replication_pad2d_0_xnumel = 4624*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_replication_pad2d_0[grid(triton_poi_fused_replication_pad2d_0_xnumel)](arg3_1, buf0, 13872, XBLOCK=128, num_warps=4, num_stages=1)
        del arg3_1
        # Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.replication_pad2d, aten._adaptive_avg_pool2d]
        buf1 = torch.ops.aten._adaptive_avg_pool2d.default(buf0, [5, 5])
        del buf0
        buf2 = buf1
        del buf1
        buf3 = empty_strided_cuda((2, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [2], out=buf3)
    buf7 = empty_strided_cpu((1, ), (1, ), torch.int64)
    # Topologically Sorted Source Nodes: [], Original ATen: []
    aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf7)
    buf8 = empty_strided_cpu((1, 25*s0), (25*s0, 1), torch.float32)
    cpp_fused_randint_1(buf7, buf8, s0)
    del buf7
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf9 = empty_strided_cuda((1, 25*s0), (25*s0, 1), torch.float32)
        buf9.copy_(buf8, False)
        del buf8
        buf6 = empty_strided_cuda((), (), torch.float32)
        buf10 = empty_strided_cuda((), (), torch.float32)
        buf12 = buf10; del buf10  # reuse
        buf11 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [x_2, target_smooth_l1, loss_smooth_l1, loss_bce], Original ATen: [aten.bernoulli, aten.randn_like, aten.smooth_l1_loss, aten.binary_cross_entropy_with_logits]
        triton_red_fused_bernoulli_binary_cross_entropy_with_logits_randn_like_smooth_l1_loss_2_r0_numel = 25*s0
        stream0 = get_raw_stream(0)
        triton_red_fused_bernoulli_binary_cross_entropy_with_logits_randn_like_smooth_l1_loss_2[grid(1)](buf12, buf11, buf3, buf2, buf9, 0, 1, 3, 1, 75, XBLOCK=1, R0_BLOCK=128, num_warps=2, num_stages=1)
        del buf2
        del buf3
        del buf9
    return (buf11, buf12, )


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
