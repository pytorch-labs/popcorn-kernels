# AOT ID: ['205_inference']
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
extern "C"  void kernel(const int64_t* in_ptr0,
                       float* out_ptr0,
                       const int64_t ks0,
                       const int64_t ks1)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(ks0*ks1); x0+=static_cast<int64_t>(1L))
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


# kernel path: /tmp/torchinductor_sahanp/pk/cpk5gi2uxnja4cxrhvv53juqi2cwb5ozagv2mzwog3mimb2en7hb.py
# Topologically Sorted Source Nodes: [x, x_1, x_2, x_3, loss], Original ATen: [aten.cat, aten.glu, aten.abs, aten.add, aten.div, aten.binary_cross_entropy_with_logits]
# Source node to ATen node mapping:
#   loss => abs_2, exp, full_default_1, log1p, mean, minimum, mul_48, neg, sub_34, sub_35, sub_36
#   x => cat
#   x_1 => glu
#   x_2 => abs_1, add_22, div
#   x_3 => glu_1
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%arg3_1, %full_default], 1), kwargs = {})
#   %glu : [num_users=2] = call_function[target=torch.ops.aten.glu.default](args = (%cat, 1), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%glu,), kwargs = {})
#   %add_22 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%abs_1, 1), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%glu, %add_22), kwargs = {})
#   %glu_1 : [num_users=5] = call_function[target=torch.ops.aten.glu.default](args = (%div, 1), kwargs = {})
#   %sub_34 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %device_put), kwargs = {})
#   %mul_48 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_34, %glu_1), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %minimum : [num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%full_default_1, %glu_1), kwargs = {})
#   %abs_2 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%glu_1,), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%abs_2,), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %log1p : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp,), kwargs = {})
#   %sub_35 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%minimum, %log1p), kwargs = {})
#   %sub_36 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_48, %sub_35), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sub_36,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_abs_add_binary_cross_entropy_with_logits_cat_div_glu_1(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp66 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp54 = tl.load(in_ptr1 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp0 = tl.full([1, 1], 0, tl.int64)
        tmp1 = tmp0 >= tmp0
        tmp2 = tl.full([1, 1], 3, tl.int64)
        tmp3 = tmp0 < tmp2
        tmp4 = tl.load(in_ptr0 + (tl.broadcast_to(r0_0 + ks0*ks1*(0), [XBLOCK, R0_BLOCK])), r0_mask & tmp3, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp0 >= tmp2
        tmp6 = tl.full([1, 1], 4, tl.int64)
        tmp7 = tmp0 < tmp6
        tmp8 = 0.0
        tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
        tmp10 = tl.where(tmp5, tmp8, tmp9)
        tmp11 = tl.where(tmp3, tmp4, tmp10)
        tmp12 = tl.full([1, 1], 2, tl.int64)
        tmp13 = tmp12 >= tmp0
        tmp14 = tmp12 < tmp2
        tmp15 = tl.load(in_ptr0 + (tl.broadcast_to(r0_0 + ks0*ks1*(2), [XBLOCK, R0_BLOCK])), r0_mask & tmp14, eviction_policy='evict_last', other=0.0)
        tmp16 = tmp12 >= tmp2
        tmp17 = tmp12 < tmp6
        tmp18 = 0.0
        tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
        tmp20 = tl.where(tmp16, tmp18, tmp19)
        tmp21 = tl.where(tmp14, tmp15, tmp20)
        tmp22 = tl.sigmoid(tmp21)
        tmp23 = tmp11 * tmp22
        tmp24 = tl_math.abs(tmp23)
        tmp25 = 1.0
        tmp26 = tmp24 + tmp25
        tmp27 = tmp23 / tmp26
        tmp28 = tl.full([1, 1], 1, tl.int64)
        tmp29 = tmp28 >= tmp0
        tmp30 = tmp28 < tmp2
        tmp31 = tl.load(in_ptr0 + (tl.broadcast_to(r0_0 + ks0*ks1*(1), [XBLOCK, R0_BLOCK])), r0_mask & tmp30, eviction_policy='evict_last', other=0.0)
        tmp32 = tmp28 >= tmp2
        tmp33 = tmp28 < tmp6
        tmp34 = 0.0
        tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
        tmp36 = tl.where(tmp32, tmp34, tmp35)
        tmp37 = tl.where(tmp30, tmp31, tmp36)
        tmp38 = tmp2 >= tmp0
        tmp39 = tmp2 < tmp2
        tmp40 = tl.load(in_ptr0 + (tl.broadcast_to(r0_0 + ks0*ks1*(3), [XBLOCK, R0_BLOCK])), r0_mask & tmp39, eviction_policy='evict_first', other=0.0)
        tmp41 = tmp2 >= tmp2
        tmp42 = tmp2 < tmp6
        tmp43 = 0.0
        tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
        tmp45 = tl.where(tmp41, tmp43, tmp44)
        tmp46 = tl.where(tmp39, tmp40, tmp45)
        tmp47 = tl.sigmoid(tmp46)
        tmp48 = tmp37 * tmp47
        tmp49 = tl_math.abs(tmp48)
        tmp50 = tmp49 + tmp25
        tmp51 = tmp48 / tmp50
        tmp52 = tl.sigmoid(tmp51)
        tmp53 = tmp27 * tmp52
        tmp55 = tmp25 - tmp54
        tmp56 = tmp55 * tmp53
        tmp57 = 0.0
        tmp58 = triton_helpers.minimum(tmp57, tmp53)
        tmp59 = tl_math.abs(tmp53)
        tmp60 = -tmp59
        tmp61 = tl_math.exp(tmp60)
        tmp62 = libdevice.log1p(tmp61)
        tmp63 = tmp58 - tmp62
        tmp64 = tmp56 - tmp63
        tmp65 = tl.broadcast_to(tmp64, [XBLOCK, R0_BLOCK])
        tmp67 = _tmp66 + tmp65
        _tmp66 = tl.where(r0_mask, tmp67, _tmp66)
        tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp53, r0_mask)
    tmp66 = tl.sum(_tmp66, 1)[:, None]
    tmp68 = ks0*ks1
    tmp69 = tmp68.to(tl.float32)
    tmp70 = tmp66 / tmp69
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp70, None)







def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, 3, s1, s2), (3*s1*s2, s1*s2, s2, 1))
    buf1 = empty_strided_cpu((1, ), (1, ), torch.int64)
    # Topologically Sorted Source Nodes: [], Original ATen: []
    aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf1)
    buf2 = empty_strided_cpu((1, 1, s1, s2), (s1*s2, s1*s2, s2, 1), torch.float32)
    cpp_fused__to_copy_randint_0(buf1, buf2, s1, s2)
    del buf1
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf3 = empty_strided_cuda((1, 1, s1, s2), (s1*s2, s1*s2, s2, 1), torch.float32)
        buf3.copy_(buf2, False)
        del buf2
        buf0 = empty_strided_cuda((1, 1, s1, s2), (s1*s2, s1*s2, s2, 1), torch.float32)
        buf4 = empty_strided_cuda((), (), torch.float32)
        buf5 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [x, x_1, x_2, x_3, loss], Original ATen: [aten.cat, aten.glu, aten.abs, aten.add, aten.div, aten.binary_cross_entropy_with_logits]
        triton_red_fused_abs_add_binary_cross_entropy_with_logits_cat_div_glu_1_r0_numel = s1*s2
        stream0 = get_raw_stream(0)
        triton_red_fused_abs_add_binary_cross_entropy_with_logits_cat_div_glu_1[grid(1)](buf5, arg3_1, buf3, buf0, 64, 64, 1, 4096, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del arg3_1
        del buf3
    return (buf0, buf5, )


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
