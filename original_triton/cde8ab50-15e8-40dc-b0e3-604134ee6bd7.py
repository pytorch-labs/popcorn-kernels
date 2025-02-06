# AOT ID: ['66_inference']
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


# kernel path: /tmp/torchinductor_sahanp/5q/c5qq3yqv5a3d4lyvrh32vvwwux2xmjp6fy2zku4tx6unhwzpikl4.py
# Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten._unsafe_index, aten.sigmoid]
# Source node to ATen node mapping:
#   x => _unsafe_index
#   x_1 => _unsafe_index_1
#   x_2 => sigmoid
# Graph fragment:
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg3_1, [None, None, %unsqueeze, %convert_element_type_3]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index, [None, None, %unsqueeze_1, %convert_element_type_7]), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%_unsafe_index_1,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__unsafe_index_sigmoid_0(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
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
    tmp55 = tl.sigmoid(tmp54)
    tl.store(out_ptr0 + (x3), tmp55, xmask)




# kernel path: /tmp/torchinductor_sahanp/hp/chp27flpxina5tkwoxkqpuwlmobimnruz6vx4whm4mjlt4mu37ap.py
# Topologically Sorted Source Nodes: [dist], Original ATen: [aten.sub, aten.add, aten.norm]
# Source node to ATen node mapping:
#   dist => add_74, pow_1, sub_40, sum_1
# Graph fragment:
#   %sub_40 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_2, %slice_4), kwargs = {})
#   %add_74 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%sub_40, 1e-06), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_74, 2.0), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1]), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_norm_sub_1(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 3
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = r0_1 + x0*((2 + 8*ks0*ks1*ks2) // 3)
        tmp1 = 8*ks0*ks1*ks2
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (((r0_1 + x0*((2 + 8*ks0*ks1*ks2) // 3)) % (16*ks0*ks1*ks2))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr0 + (4*ks2*((((r0_1 + x0*((2 + 8*ks0*ks1*ks2) // 3) + 8*ks0*ks1*ks2) // ks3) % (4*ks0*ks1))) + (((r0_1 + x0*((2 + 8*ks0*ks1*ks2) // 3)) % ks3))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 - tmp4
        tmp6 = 1e-06
        tmp7 = tmp5 + tmp6
        tmp8 = tmp7 * tmp7
        tmp9 = tl.full(tmp8.shape, 0, tmp8.dtype)
        tmp10 = tl.where(tmp2, tmp8, tmp9)
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp12, xmask)




# kernel path: /tmp/torchinductor_sahanp/ua/cuaxpcb4rqshdg5r4cvm7kkondg6lnd742wrtip5aie2vv3bjr3r.py
# Topologically Sorted Source Nodes: [dist, loss], Original ATen: [aten.sub, aten.add, aten.norm, aten.huber_loss]
# Source node to ATen node mapping:
#   dist => add_74, pow_1, sub_40, sum_1
#   loss => abs_1, lt_4, mean, mul_73, mul_74, mul_75, pow_2, sub_44, where
# Graph fragment:
#   %sub_40 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_2, %slice_4), kwargs = {})
#   %add_74 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%sub_40, 1e-06), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_74, 2.0), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1]), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %abs_1 : [num_users=4] = call_function[target=torch.ops.aten.abs.default](args = (%pow_2,), kwargs = {})
#   %lt_4 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%abs_1, 1.0), kwargs = {})
#   %mul_73 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%abs_1, 0.5), kwargs = {})
#   %mul_74 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_73, %abs_1), kwargs = {})
#   %sub_44 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%abs_1, 0.5), kwargs = {})
#   %mul_75 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_44, 1.0), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%lt_4, %mul_74, %mul_75), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%where,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_huber_loss_norm_sub_2(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 3
    R0_BLOCK: tl.constexpr = 4
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
    tmp5 = libdevice.sqrt(tmp4)
    tmp6 = tl_math.abs(tmp5)
    tmp7 = 1.0
    tmp8 = tmp6 < tmp7
    tmp9 = 0.5
    tmp10 = tmp6 * tmp9
    tmp11 = tmp10 * tmp6
    tmp12 = tmp6 - tmp9
    tmp13 = tmp12 * tmp7
    tmp14 = tl.where(tmp8, tmp11, tmp13)
    tmp15 = tmp14 / tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp15, None)







def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        ps0 = 4*s2
        ps1 = 4*s1
        ps2 = 16*s1*s2
        buf0 = empty_strided_cuda((1, s0, 4*s1, 4*s2), (16*s0*s1*s2, 16*s1*s2, 4*s2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten._unsafe_index, aten.sigmoid]
        triton_poi_fused__unsafe_index_sigmoid_0_xnumel = 16*s0*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_sigmoid_0[grid(triton_poi_fused__unsafe_index_sigmoid_0_xnumel)](arg3_1, buf0, 32, 128, 128, 32, 16384, 49152, XBLOCK=512, num_warps=4, num_stages=1)
        del arg3_1
        buf1 = empty_strided_cuda((1, 3), (3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [dist], Original ATen: [aten.sub, aten.add, aten.norm]
        triton_red_fused_add_norm_sub_1_r0_numel = (2 + 8*s0*s1*s2) // 3
        stream0 = get_raw_stream(0)
        triton_red_fused_add_norm_sub_1[grid(3)](buf0, buf1, 3, 32, 32, 128, 3, 8192, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del buf0
        buf2 = empty_strided_cuda((1, ), (1, ), torch.float32)
        buf3 = reinterpret_tensor(buf2, (), (), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [dist, loss], Original ATen: [aten.sub, aten.add, aten.norm, aten.huber_loss]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_huber_loss_norm_sub_2[grid(1)](buf3, buf1, 1, 3, XBLOCK=1, num_warps=2, num_stages=1)
        del buf1
    return (buf3, )


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
