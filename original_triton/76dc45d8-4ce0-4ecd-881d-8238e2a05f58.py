# AOT ID: ['195_inference']
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
extern "C"  void kernel(int64_t* out_ptr0,
                       int64_t* out_ptr1,
                       const int64_t ks0,
                       const int64_t ks1)
{
    {
        {
            {
                auto tmp0 = 16L*ks0*ks1;
                auto tmp1 = c10::convert<int64_t>(tmp0);
                out_ptr0[static_cast<int64_t>(0L)] = tmp1;
            }
        }
    }
    {
        {
            {
                auto tmp0 = static_cast<int64_t>(10);
                out_ptr1[static_cast<int64_t>(0L)] = tmp0;
            }
        }
    }
}
''')


# kernel path: /tmp/torchinductor_sahanp/qc/cqcv3nfsbzczh3orr37rg34zil4thtyfsiohpkus3ky62jefb6cf.py
# Topologically Sorted Source Nodes: [x, l1_loss_value], Original ATen: [aten._unsafe_index, aten.sub]
# Source node to ATen node mapping:
#   l1_loss_value => _unsafe_index_1
#   x => _unsafe_index
# Graph fragment:
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg3_1, [None, None, %unsqueeze, %convert_element_type_3]), kwargs = {})
#   %_unsafe_index_1 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index, [None, None, %unsqueeze_1, %convert_element_type_7]), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__unsafe_index_sub_1(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // ks1) % ks2)
    x0 = (xindex % ks1)
    x2 = xindex // ks4
    x3 = xindex
    tmp0 = 2.0
    tmp1 = ks0
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 * tmp2
    tmp4 = tmp3.to(tl.float64)
    tmp5 = tl.full([1], 2.0, tl.float64)
    tmp6 = tmp5 * tmp4
    tmp7 = tmp4 / tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = x1
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp10 * tmp8
    tmp12 = tmp11.to(tl.int64)
    tmp13 = 2*ks0
    tmp14 = tmp12 + tmp13
    tmp15 = tmp12 < 0
    tmp16 = tl.where(tmp15, tmp14, tmp12)
    tmp17 = ks3
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp0 * tmp18
    tmp20 = tmp19.to(tl.float64)
    tmp21 = tmp5 * tmp20
    tmp22 = tmp20 / tmp21
    tmp23 = tmp22.to(tl.float32)
    tmp24 = x0
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp25 * tmp23
    tmp27 = tmp26.to(tl.int64)
    tmp28 = 2*ks3
    tmp29 = tmp27 + tmp28
    tmp30 = tmp27 < 0
    tmp31 = tl.where(tmp30, tmp29, tmp27)
    tmp32 = tmp1.to(tl.float64)
    tmp33 = tmp5 * tmp32
    tmp34 = tmp32 / tmp33
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp16
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tmp37 * tmp35
    tmp39 = tmp38.to(tl.int64)
    tmp40 = tmp39 + tmp1
    tmp41 = tmp39 < 0
    tmp42 = tl.where(tmp41, tmp40, tmp39)
    tmp43 = tmp17.to(tl.float64)
    tmp44 = tmp5 * tmp43
    tmp45 = tmp43 / tmp44
    tmp46 = tmp45.to(tl.float32)
    tmp47 = tmp31
    tmp48 = tmp47.to(tl.float32)
    tmp49 = tmp48 * tmp46
    tmp50 = tmp49.to(tl.int64)
    tmp51 = tmp50 + tmp17
    tmp52 = tmp50 < 0
    tmp53 = tl.where(tmp52, tmp51, tmp50)
    tmp54 = tl.load(in_ptr0 + (tmp53 + ks3*tmp42 + ks0*ks3*x2), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3), tmp54, xmask)




# kernel path: /tmp/torchinductor_sahanp/55/c55uwnhvmh3uwyx57x65gxhsk7sfoewf53h66tdn2oe5xz6mgjhw.py
# Topologically Sorted Source Nodes: [l1_loss_value], Original ATen: [aten.abs, aten.mean]
# Source node to ATen node mapping:
#   l1_loss_value => abs_1, mean
# Graph fragment:
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%_unsafe_index_1,), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%abs_1,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_abs_mean_2(in_ptr0, out_ptr0, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 6
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = r0_1 + x0*((5 + 16*ks0*ks1*ks2) // 6)
        tmp1 = 16*ks0*ks1*ks2
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (((r0_1 + x0*((5 + 16*ks0*ks1*ks2) // 6)) % (16*ks0*ks1*ks2))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl_math.abs(tmp3)
        tmp5 = tl.full(tmp4.shape, 0, tmp4.dtype)
        tmp6 = tl.where(tmp2, tmp4, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)




# kernel path: /tmp/torchinductor_sahanp/kp/ckpniguokvk7xri2v4b2uf3olbhdnjblwpplklh7bva6as55qkuk.py
# Topologically Sorted Source Nodes: [l1_loss_value], Original ATen: [aten.abs, aten.mean]
# Source node to ATen node mapping:
#   l1_loss_value => abs_1, mean
# Graph fragment:
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%_unsafe_index_1,), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%abs_1,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_abs_mean_3(in_out_ptr0, in_ptr0, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 6
    R0_BLOCK: tl.constexpr = 8
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 16*ks0*ks1*ks2
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp4 / tmp6
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp7, None)




#include "/tmp/torchinductor_sahanp/3b/c3bi5gk6mslf6u4iaqafhxm64z6u65e3eain4xlary5blqnvv6xx.h"
extern "C"  void kernel(const int64_t* in_ptr0,
                       int64_t* out_ptr0,
                       const int64_t ks0)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(10L); x0+=static_cast<int64_t>(1L))
        {
            {
                {
                    auto tmp0 = in_ptr0[static_cast<int64_t>(0L)];
                    auto tmp1 = x0;
                    auto tmp2 = c10::convert<int32_t>(tmp1);
                    auto tmp3 = static_cast<int64_t>(0);
                    auto tmp4 = ks0;
                    auto tmp5 = c10::convert<int64_t>(tmp4);
                    auto tmp6 = randint64_cpu(tmp0, tmp2, tmp3, tmp5);
                    out_ptr0[static_cast<int64_t>(x0)] = tmp6;
                }
            }
        }
    }
}
''')





def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    buf1 = empty_strided_cpu((1, ), (1, ), torch.int64)
    # Topologically Sorted Source Nodes: [], Original ATen: []
    aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf1)
    buf5 = empty_strided_cpu((1, ), (1, ), torch.int64)
    buf6 = empty_strided_cpu((1, ), (1, ), torch.int64)
    cpp_fused_full_0(buf5, buf6, s1, s2)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        ps0 = 4*s2
        ps1 = 4*s1
        ps2 = 16*s1*s2
        buf0 = empty_strided_cuda((1, s0, 4*s1, 4*s2), (16*s0*s1*s2, 16*s1*s2, 4*s2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, l1_loss_value], Original ATen: [aten._unsafe_index, aten.sub]
        triton_poi_fused__unsafe_index_sub_1_xnumel = 16*s0*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_sub_1[grid(triton_poi_fused__unsafe_index_sub_1_xnumel)](arg3_1, buf0, 32, 128, 128, 32, 16384, 49152, XBLOCK=512, num_warps=4, num_stages=1)
        del arg3_1
        buf3 = empty_strided_cuda((6, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [l1_loss_value], Original ATen: [aten.abs, aten.mean]
        triton_red_fused_abs_mean_2_r0_numel = (5 + 16*s0*s1*s2) // 6
        stream0 = get_raw_stream(0)
        triton_red_fused_abs_mean_2[grid(6)](buf0, buf3, 3, 32, 32, 6, 8192, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        buf4 = empty_strided_cuda((), (), torch.float32)
        buf7 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [l1_loss_value], Original ATen: [aten.abs, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_abs_mean_3[grid(1)](buf7, buf3, 3, 32, 32, 1, 6, XBLOCK=1, num_warps=2, num_stages=1)
        del buf3
    buf2 = empty_strided_cpu((1, 10), (10, 1), torch.int64)
    cpp_fused_randint_4(buf1, buf2, s0)
    return (reinterpret_tensor(buf0, (16*s1*s2, 1, s0), (1, 16*s0*s1*s2, 16*s1*s2), 0), buf2, buf5, buf6, buf7, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 3
    arg1_1 = 32
    arg2_1 = 32
    arg3_1 = rand_strided((1, 3, 32, 32), (3072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
