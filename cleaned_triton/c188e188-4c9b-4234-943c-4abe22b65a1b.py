# AOT ID: ['101_forward']
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


# kernel path: /tmp/torchinductor_sahanp/4g/c4gs3hzjmzfkn4bqwdyxibbdmfyemrn27tq2je4vzereeoiafnfr.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.rand]
# Source node to ATen node mapping:
#   x => inductor_lookup_seed_default, inductor_random_default
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 64, 2], %inductor_lookup_seed_default, rand), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_rand_0(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (x0), tmp2, xmask)




# kernel path: /tmp/torchinductor_sahanp/wd/cwd2n5cbkrmwpctpjx7jpkooyha24ka4or4zzmsa2xq6rp54t5td.py
# Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.fractional_max_pool2d, aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   x => fractional_max_pool2d
#   x_1 => add_1, add_2, add_3, add_4, mul, mul_1, mul_2, mul_3, mul_4, mul_5, mul_6, rsqrt, sub, var_mean
# Graph fragment:
#   %fractional_max_pool2d : [num_users=1] = call_function[target=torch.ops.aten.fractional_max_pool2d.default](args = (%primals_3, [2, 2], [14, 14], %inductor_random_default), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%getitem, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%getitem, %getitem_3), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze, 0.1), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_5, 0.9), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %mul_2), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_2, 1.005128205128205), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, 0.1), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_6, 0.9), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %mul_5), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %unsqueeze_1), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %unsqueeze_3), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%getitem, %unsqueeze_6), kwargs = {})
#   %copy__1 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_5, %add_2), kwargs = {})
#   %copy__2 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_6, %add_3), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_fractional_max_pool2d_native_batch_norm_backward_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr3, out_ptr4, out_ptr5, out_ptr7, out_ptr9, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 64
    r0_numel = 196
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr0 + (1 + 2*x0), xmask, eviction_policy='evict_last')
    tmp49_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp49_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp49_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index // 14
        r0_1 = (r0_index % 14)
        r0_3 = r0_index
        tmp1 = ((-2) + ks0) / 13
        tmp2 = tmp1.to(tl.float32)
        tmp3 = r0_2
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tmp4 + tmp0
        tmp6 = tmp5 * tmp2
        tmp7 = libdevice.floor(tmp6)
        tmp8 = tmp0 * tmp2
        tmp9 = libdevice.floor(tmp8)
        tmp10 = tmp7 - tmp9
        tmp11 = tmp10.to(tl.int64)
        tmp12 = tl.full([1, 1], 13, tl.int64)
        tmp13 = tmp4 < tmp12
        tmp14 = (-2) + ks0
        tmp15 = tl.where(tmp13, tmp11, tmp14)
        tmp16 = ks0
        tmp17 = tmp15 + tmp16
        tmp18 = tmp15 < 0
        tmp19 = tl.where(tmp18, tmp17, tmp15)
        tl.device_assert(((0 <= tmp19) & (tmp19 < ks0)) | ~(r0_mask & xmask), "index out of bounds: 0 <= tmp19 < ks0")
        tmp22 = ((-2) + ks1) / 13
        tmp23 = tmp22.to(tl.float32)
        tmp24 = r0_1
        tmp25 = tmp24.to(tl.float32)
        tmp26 = tmp25 + tmp21
        tmp27 = tmp26 * tmp23
        tmp28 = libdevice.floor(tmp27)
        tmp29 = tmp21 * tmp23
        tmp30 = libdevice.floor(tmp29)
        tmp31 = tmp28 - tmp30
        tmp32 = tmp31.to(tl.int64)
        tmp33 = tmp25 < tmp12
        tmp34 = (-2) + ks1
        tmp35 = tl.where(tmp33, tmp32, tmp34)
        tmp36 = ks1
        tmp37 = tmp35 + tmp36
        tmp38 = tmp35 < 0
        tmp39 = tl.where(tmp38, tmp37, tmp35)
        tl.device_assert(((0 <= tmp39) & (tmp39 < ks1)) | ~(r0_mask & xmask), "index out of bounds: 0 <= tmp39 < ks1")
        tmp41 = tl.load(in_ptr1 + (tmp39 + ks1*tmp19 + ks0*ks1*x0), r0_mask & xmask, eviction_policy='evict_last')
        tmp42 = tl.load(in_ptr1 + (1 + tmp39 + ks1*tmp19 + ks0*ks1*x0), r0_mask & xmask, eviction_policy='evict_last')
        tmp43 = triton_helpers.maximum(tmp42, tmp41)
        tmp44 = tl.load(in_ptr1 + (ks1 + tmp39 + ks1*tmp19 + ks0*ks1*x0), r0_mask & xmask, eviction_policy='evict_last')
        tmp45 = triton_helpers.maximum(tmp44, tmp43)
        tmp46 = tl.load(in_ptr1 + (1 + ks1 + tmp39 + ks1*tmp19 + ks0*ks1*x0), r0_mask & xmask, eviction_policy='evict_last')
        tmp47 = triton_helpers.maximum(tmp46, tmp45)
        tmp48 = tl.broadcast_to(tmp47, [XBLOCK, R0_BLOCK])
        tmp49_mean_next, tmp49_m2_next, tmp49_weight_next = triton_helpers.welford_reduce(
            tmp48, tmp49_mean, tmp49_m2, tmp49_weight, roffset == 0
        )
        tmp49_mean = tl.where(r0_mask & xmask, tmp49_mean_next, tmp49_mean)
        tmp49_m2 = tl.where(r0_mask & xmask, tmp49_m2_next, tmp49_m2)
        tmp49_weight = tl.where(r0_mask & xmask, tmp49_weight_next, tmp49_weight)
        tl.store(out_ptr0 + (r0_3 + 196*x0), tmp47, r0_mask & xmask)
    tmp52, tmp53, tmp54 = triton_helpers.welford(tmp49_mean, tmp49_m2, tmp49_weight, 1)
    tmp49 = tmp52[:, None]
    tmp50 = tmp53[:, None]
    tmp51 = tmp54[:, None]
    tmp63 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp65 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        tmp55 = tl.load(out_ptr0 + (r0_3 + 196*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp56 = tmp55 - tmp49
        tmp57 = 196.0
        tmp58 = tmp50 / tmp57
        tmp59 = 1e-05
        tmp60 = tmp58 + tmp59
        tmp61 = libdevice.rsqrt(tmp60)
        tmp62 = tmp56 * tmp61
        tmp64 = tmp62 * tmp63
        tmp66 = tmp64 + tmp65
        tl.store(out_ptr3 + (r0_3 + 196*x0), tmp66, r0_mask & xmask)
        tl.store(out_ptr4 + (r0_3 + 196*x0), tmp56, r0_mask & xmask)
    tmp76 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp81 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp67 = 196.0
    tmp68 = tmp50 / tmp67
    tmp69 = 1e-05
    tmp70 = tmp68 + tmp69
    tmp71 = libdevice.rsqrt(tmp70)
    tmp72 = 1.005128205128205
    tmp73 = tmp68 * tmp72
    tmp74 = 0.1
    tmp75 = tmp73 * tmp74
    tmp77 = 0.9
    tmp78 = tmp76 * tmp77
    tmp79 = tmp75 + tmp78
    tmp80 = tmp49 * tmp74
    tmp82 = tmp81 * tmp77
    tmp83 = tmp80 + tmp82
    tl.store(out_ptr5 + (x0), tmp71, xmask)
    tl.store(out_ptr7 + (x0), tmp79, xmask)
    tl.store(out_ptr9 + (x0), tmp83, xmask)




# kernel path: /tmp/torchinductor_sahanp/ke/ckea67ksdr4pks2mcpodgvsybipmzvkeee2q7lfxwirdsmbe34x2.py
# Topologically Sorted Source Nodes: [add_], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   add_ => add
# Graph fragment:
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_4, 1), kwargs = {})
#   %copy_ : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_4, %add), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_2(in_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp1 + tmp2
    tl.store(out_ptr1 + (tl.full([XBLOCK], 0, tl.int32)), tmp3, None)







def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8 = args
    args.clear()
    s1 = primals_1
    s2 = primals_2
    assert_size_stride(primals_3, (1, 64, s1, s2), (64*s1*s2, s1*s2, s2, 1))
    assert_size_stride(primals_4, (), ())
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_7, (64, ), (1, ))
    assert_size_stride(primals_8, (64, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf0)
        buf1 = empty_strided_cuda((1, 64, 2), (128, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.rand]
        stream0 = get_raw_stream(0)
        triton_poi_fused_rand_0[grid(128)](buf0, buf1, 0, 128, XBLOCK=128, num_warps=4, num_stages=1)
        del buf0
        buf2 = empty_strided_cuda((1, 64, 14, 14), (12544, 196, 14, 1), torch.float32)
        buf7 = empty_strided_cuda((1, 64, 14, 14), (12544, 196, 14, 1), torch.float32)
        buf8 = empty_strided_cuda((1, 64, 14, 14), (12544, 196, 14, 1), torch.float32)
        buf6 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.fractional_max_pool2d, aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_fractional_max_pool2d_native_batch_norm_backward_1[grid(64)](buf1, primals_3, primals_7, primals_8, primals_6, primals_5, buf2, buf7, buf8, buf6, primals_6, primals_5, 28, 28, 64, 196, XBLOCK=1, R0_BLOCK=256, num_warps=2, num_stages=1)
        del buf1
        del buf2
        del primals_3
        del primals_5
        del primals_6
        del primals_7
        del primals_8
        # Topologically Sorted Source Nodes: [add_], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_2[grid(1)](primals_4, primals_4, 1, XBLOCK=1, num_warps=1, num_stages=1)
        del primals_4
    return (buf7, reinterpret_tensor(buf6, (64, ), (1, ), 0), buf8, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = 28
    primals_2 = 28
    primals_3 = rand_strided((1, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
