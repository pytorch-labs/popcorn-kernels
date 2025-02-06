# AOT ID: ['37_forward']
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


# kernel path: /tmp/torchinductor_sahanp/5s/c5st7y7jotblweea3ck3iu5e5ddwzlx3l4joelm5ozbfditvcbxe.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   x_2 => add_17, mul_15, rsqrt, sub_23, var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_1, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_17,), kwargs = {})
#   %sub_23 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_1, %getitem_1), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_23, %rsqrt), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_0(in_ptr0, out_ptr2, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 10
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (ks2*(tl.where((-1) + ks1 + ((-1)*tl_math.abs(1 + ((-1)*ks1) + tl_math.abs((-1) + (((r0_1 // (2 + ks2)) % (2 + ks1)))))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks1) + tl_math.abs((-1) + (((r0_1 // (2 + ks2)) % (2 + ks1)))))) + 2*ks1, (-1) + ks1 + ((-1)*tl_math.abs(1 + ((-1)*ks1) + tl_math.abs((-1) + (((r0_1 // (2 + ks2)) % (2 + ks1)))))))) + ks1*ks2*(tl.where((-1) + ks0 + ((-1)*tl_math.abs(1 + ((-1)*ks0) + tl_math.abs((-1) + (r0_1 // (4 + 2*ks1 + 2*ks2 + ks1*ks2))))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks0) + tl_math.abs((-1) + (r0_1 // (4 + 2*ks1 + 2*ks2 + ks1*ks2))))) + 2*ks0, (-1) + ks0 + ((-1)*tl_math.abs(1 + ((-1)*ks0) + tl_math.abs((-1) + (r0_1 // (4 + 2*ks1 + 2*ks2 + ks1*ks2))))))) + ks0*ks1*ks2*x0 + (tl.where((-1) + ks2 + ((-1)*tl_math.abs(1 + ((-1)*ks2) + tl_math.abs((-1) + ((r0_1 % (2 + ks2)))))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks2) + tl_math.abs((-1) + ((r0_1 % (2 + ks2)))))) + 2*ks2, (-1) + ks2 + ((-1)*tl_math.abs(1 + ((-1)*ks2) + tl_math.abs((-1) + ((r0_1 % (2 + ks2))))))))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(r0_mask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(r0_mask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(r0_mask & xmask, tmp2_weight_next, tmp2_weight)
    tmp5, tmp6, tmp7 = triton_helpers.welford(tmp2_mean, tmp2_m2, tmp2_weight, 1)
    tmp2 = tmp5[:, None]
    tmp3 = tmp6[:, None]
    tmp4 = tmp7[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp8 = tl.load(in_ptr0 + (ks2*(tl.where((-1) + ks1 + ((-1)*tl_math.abs(1 + ((-1)*ks1) + tl_math.abs((-1) + (((r0_1 // (2 + ks2)) % (2 + ks1)))))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks1) + tl_math.abs((-1) + (((r0_1 // (2 + ks2)) % (2 + ks1)))))) + 2*ks1, (-1) + ks1 + ((-1)*tl_math.abs(1 + ((-1)*ks1) + tl_math.abs((-1) + (((r0_1 // (2 + ks2)) % (2 + ks1)))))))) + ks1*ks2*(tl.where((-1) + ks0 + ((-1)*tl_math.abs(1 + ((-1)*ks0) + tl_math.abs((-1) + (r0_1 // (4 + 2*ks1 + 2*ks2 + ks1*ks2))))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks0) + tl_math.abs((-1) + (r0_1 // (4 + 2*ks1 + 2*ks2 + ks1*ks2))))) + 2*ks0, (-1) + ks0 + ((-1)*tl_math.abs(1 + ((-1)*ks0) + tl_math.abs((-1) + (r0_1 // (4 + 2*ks1 + 2*ks2 + ks1*ks2))))))) + ks0*ks1*ks2*x0 + (tl.where((-1) + ks2 + ((-1)*tl_math.abs(1 + ((-1)*ks2) + tl_math.abs((-1) + ((r0_1 % (2 + ks2)))))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks2) + tl_math.abs((-1) + ((r0_1 % (2 + ks2)))))) + 2*ks2, (-1) + ks2 + ((-1)*tl_math.abs(1 + ((-1)*ks2) + tl_math.abs((-1) + ((r0_1 % (2 + ks2))))))))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp8 - tmp2
        tmp10 = 8 + 4*ks0 + 4*ks1 + 4*ks2 + 2*ks0*ks1 + 2*ks0*ks2 + 2*ks1*ks2 + ks0*ks1*ks2
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tmp3 / tmp11
        tmp13 = 1e-05
        tmp14 = tmp12 + tmp13
        tmp15 = libdevice.rsqrt(tmp14)
        tmp16 = tmp9 * tmp15
        tl.store(out_ptr2 + (r0_1 + 8*x0 + 4*ks0*x0 + 4*ks1*x0 + 4*ks2*x0 + 2*ks0*ks1*x0 + 2*ks0*ks2*x0 + 2*ks1*ks2*x0 + ks0*ks1*ks2*x0), tmp16, r0_mask & xmask)




# kernel path: /tmp/torchinductor_sahanp/3l/c3lvhxhjlplag5u4d76h7z2fldclpmxnyqcsjyinp5dqznlnhy5f.py
# Topologically Sorted Source Nodes: [x_5, x_6, x_7, x_8, x_9, x_10, x_11], Original ATen: [aten.sub, aten.add, aten.norm, aten.hardsigmoid]
# Source node to ATen node mapping:
#   x_10 => add_39
#   x_11 => add_40
#   x_5 => add_34, pow_1, pow_2, sub_30, sum_1
#   x_6 => add_35, clamp_max, clamp_min, div
#   x_7 => add_36
#   x_8 => add_37
#   x_9 => add_38
# Graph fragment:
#   %sub_30 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_2, %slice_4), kwargs = {})
#   %add_34 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%sub_30, 1e-06), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_34, 2.0), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1]), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %add_35 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_2, 3), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_35, 0), kwargs = {})
#   %clamp_max : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 6), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%clamp_max, 6), kwargs = {})
#   %add_36 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div, %select), kwargs = {})
#   %add_37 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_36, %select_1), kwargs = {})
#   %add_38 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_37, %select_2), kwargs = {})
#   %add_39 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_38, %select_3), kwargs = {})
#   %add_40 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_39, %select_4), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_hardsigmoid_norm_sub_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.load(in_ptr0 + (5))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp8 = tl.load(in_ptr0 + (1))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp10 = tl.load(in_ptr0 + (6))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp16 = tl.load(in_ptr0 + (2))
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK])
    tmp18 = tl.load(in_ptr0 + (7))
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK])
    tmp24 = tl.load(in_ptr0 + (3))
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK])
    tmp26 = tl.load(in_ptr0 + (8))
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK])
    tmp32 = tl.load(in_ptr0 + (4))
    tmp33 = tl.broadcast_to(tmp32, [XBLOCK])
    tmp34 = tl.load(in_ptr0 + (9))
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK])
    tmp49 = tl.load(in_ptr1 + (0))
    tmp50 = tl.broadcast_to(tmp49, [XBLOCK])
    tmp52 = tl.load(in_ptr2 + (0))
    tmp53 = tl.broadcast_to(tmp52, [XBLOCK])
    tmp55 = tl.load(in_ptr3 + (0))
    tmp56 = tl.broadcast_to(tmp55, [XBLOCK])
    tmp58 = tl.load(in_ptr4 + (0))
    tmp59 = tl.broadcast_to(tmp58, [XBLOCK])
    tmp61 = tl.load(in_ptr5 + (0))
    tmp62 = tl.broadcast_to(tmp61, [XBLOCK])
    tmp4 = tmp1 - tmp3
    tmp5 = 1e-06
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6 * tmp6
    tmp12 = tmp9 - tmp11
    tmp13 = tmp12 + tmp5
    tmp14 = tmp13 * tmp13
    tmp15 = tmp7 + tmp14
    tmp20 = tmp17 - tmp19
    tmp21 = tmp20 + tmp5
    tmp22 = tmp21 * tmp21
    tmp23 = tmp15 + tmp22
    tmp28 = tmp25 - tmp27
    tmp29 = tmp28 + tmp5
    tmp30 = tmp29 * tmp29
    tmp31 = tmp23 + tmp30
    tmp36 = tmp33 - tmp35
    tmp37 = tmp36 + tmp5
    tmp38 = tmp37 * tmp37
    tmp39 = tmp31 + tmp38
    tmp40 = libdevice.sqrt(tmp39)
    tmp41 = 3.0
    tmp42 = tmp40 + tmp41
    tmp43 = 0.0
    tmp44 = triton_helpers.maximum(tmp42, tmp43)
    tmp45 = 6.0
    tmp46 = triton_helpers.minimum(tmp44, tmp45)
    tmp47 = 0.16666666666666666
    tmp48 = tmp46 * tmp47
    tmp51 = tmp48 + tmp50
    tmp54 = tmp51 + tmp53
    tmp57 = tmp54 + tmp56
    tmp60 = tmp57 + tmp59
    tmp63 = tmp60 + tmp62
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp63, None)







def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9 = args
    args.clear()
    s1 = primals_1
    s2 = primals_2
    s3 = primals_3
    assert_size_stride(primals_4, (1, 10, s1, s2, s3), (10*s1*s2*s3, s1*s2*s3, s2*s3, s3, 1))
    assert_size_stride(primals_5, (10, ), (1, ))
    assert_size_stride(primals_6, (10, ), (1, ))
    assert_size_stride(primals_7, (10, ), (1, ))
    assert_size_stride(primals_8, (10, ), (1, ))
    assert_size_stride(primals_9, (10, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf3 = empty_strided_cuda((1, 10, 8 + 4*s1 + 4*s2 + 4*s3 + 2*s1*s2 + 2*s1*s3 + 2*s2*s3 + s1*s2*s3), (80 + 40*s1 + 40*s2 + 40*s3 + 20*s1*s2 + 20*s1*s3 + 20*s2*s3 + 10*s1*s2*s3, 8 + 4*s1 + 4*s2 + 4*s3 + 2*s1*s2 + 2*s1*s3 + 2*s2*s3 + s1*s2*s3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten._native_batch_norm_legit]
        triton_red_fused__native_batch_norm_legit_0_r0_numel = 8 + 4*s1 + 4*s2 + 4*s3 + 2*s1*s2 + 2*s1*s3 + 2*s2*s3 + s1*s2*s3
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_0[grid(10)](primals_4, buf3, 10, 10, 10, 10, 1728, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del primals_4
        buf4 = empty_strided_cuda((1, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [x_5, x_6, x_7, x_8, x_9, x_10, x_11], Original ATen: [aten.sub, aten.add, aten.norm, aten.hardsigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_hardsigmoid_norm_sub_1[grid(1)](buf3, primals_5, primals_6, primals_7, primals_8, primals_9, buf4, 1, XBLOCK=1, num_warps=1, num_stages=1)
        del buf3
        del primals_5
        del primals_6
        del primals_7
        del primals_8
        del primals_9
    return (buf4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = 10
    primals_2 = 10
    primals_3 = 10
    primals_4 = rand_strided((1, 10, 10, 10, 10), (10000, 1000, 100, 10, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
