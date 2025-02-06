# AOT ID: ['52_forward']
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


# kernel path: /tmp/torchinductor_sahanp/5a/c5aqjc4trcexfb4ohskylj6gq7qrzin4ssab26kwx7oyqribz4mk.py
# Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten.hardtanh, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_1 => clamp_max, clamp_min
#   x_2 => add, add_1, mul, mul_1, rsqrt, sub, var_mean
# Graph fragment:
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%view, -1.0), kwargs = {})
#   %clamp_max : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 1.0), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clamp_max, [1]), kwargs = {correction: 0, keepdim: True})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_max, %getitem_1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %primals_2), kwargs = {})
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %primals_3), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_hardtanh_native_layer_norm_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), None)
    tmp25 = tl.load(in_ptr1 + (r0_0), None)
    tmp27 = tl.load(in_ptr2 + (r0_0), None)
    tmp1 = -1.0
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp3 = 1.0
    tmp4 = triton_helpers.minimum(tmp2, tmp3)
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp9 = tl.sum(tmp7, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp5 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.sum(tmp15, 1)[:, None]
    tmp18 = 128.0
    tmp19 = tmp17 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp4 - tmp12
    tmp24 = tmp23 * tmp22
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp22, None)
    tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp28, None)
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp12, None)




# kernel path: /tmp/torchinductor_sahanp/zh/czhfnto6vuwv6tlz23vloofa4e66do5eejkmeat3i3mqr3auehal.py
# Topologically Sorted Source Nodes: [hx], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   hx => full_default
# Graph fragment:
#   %full_default : [num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([1, 64], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)







def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25 = args
    args.clear()
    assert_size_stride(primals_1, (1, 128), (128, 1))
    assert_size_stride(primals_2, (128, ), (1, ))
    assert_size_stride(primals_3, (128, ), (1, ))
    assert_size_stride(primals_4, (192, 128), (128, 1))
    assert_size_stride(primals_5, (192, 64), (64, 1))
    assert_size_stride(primals_6, (192, ), (1, ))
    assert_size_stride(primals_7, (192, ), (1, ))
    assert_size_stride(primals_8, (192, 64), (64, 1))
    assert_size_stride(primals_9, (192, 64), (64, 1))
    assert_size_stride(primals_10, (192, ), (1, ))
    assert_size_stride(primals_11, (192, ), (1, ))
    assert_size_stride(primals_12, (192, 64), (64, 1))
    assert_size_stride(primals_13, (192, 64), (64, 1))
    assert_size_stride(primals_14, (192, ), (1, ))
    assert_size_stride(primals_15, (192, ), (1, ))
    assert_size_stride(primals_16, (192, 64), (64, 1))
    assert_size_stride(primals_17, (192, 64), (64, 1))
    assert_size_stride(primals_18, (192, ), (1, ))
    assert_size_stride(primals_19, (192, ), (1, ))
    assert_size_stride(primals_20, (192, 64), (64, 1))
    assert_size_stride(primals_21, (192, 64), (64, 1))
    assert_size_stride(primals_22, (192, ), (1, ))
    assert_size_stride(primals_23, (192, ), (1, ))
    assert_size_stride(primals_24, (10, 64), (64, 1))
    assert_size_stride(primals_25, (10, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        buf1 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        buf3 = buf1; del buf1  # reuse
        buf4 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten.hardtanh, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_hardtanh_native_layer_norm_0[grid(1)](buf3, primals_1, primals_2, primals_3, buf0, buf4, 1, 128, XBLOCK=1, num_warps=2, num_stages=1)
        del primals_2
        del primals_3
        buf5 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1[grid(64)](buf5, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf6 = empty_strided_cuda((1, 192), (192, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret], Original ATen: [aten.mm]
        extern_kernels.mm(buf4, reinterpret_tensor(primals_4, (128, 192), (1, 128), 0), out=buf6)
        buf7 = empty_strided_cuda((1, 192), (192, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret], Original ATen: [aten.mm]
        extern_kernels.mm(buf5, reinterpret_tensor(primals_5, (64, 192), (1, 64), 0), out=buf7)
        del primals_5
        # Topologically Sorted Source Nodes: [ret], Original ATen: [aten._thnn_fused_gru_cell]
        buf8 = torch.ops.aten._thnn_fused_gru_cell.default(buf6, buf7, buf5, primals_6, primals_7)
        del primals_6
        del primals_7
        buf9 = buf8[0]
        buf10 = buf8[1]
        del buf8
        buf11 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [ret_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf9, reinterpret_tensor(primals_8, (64, 192), (1, 64), 0), out=buf11)
        buf12 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [ret_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf9, reinterpret_tensor(primals_9, (64, 192), (1, 64), 0), out=buf12)
        # Topologically Sorted Source Nodes: [ret_1], Original ATen: [aten._thnn_fused_gru_cell]
        buf13 = torch.ops.aten._thnn_fused_gru_cell.default(buf11, buf12, buf9, primals_10, primals_11)
        del primals_10
        del primals_11
        buf14 = buf13[0]
        buf15 = buf13[1]
        del buf13
        buf16 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [ret_2], Original ATen: [aten.mm]
        extern_kernels.mm(buf14, reinterpret_tensor(primals_12, (64, 192), (1, 64), 0), out=buf16)
        buf17 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [ret_2], Original ATen: [aten.mm]
        extern_kernels.mm(buf14, reinterpret_tensor(primals_13, (64, 192), (1, 64), 0), out=buf17)
        # Topologically Sorted Source Nodes: [ret_2], Original ATen: [aten._thnn_fused_gru_cell]
        buf18 = torch.ops.aten._thnn_fused_gru_cell.default(buf16, buf17, buf14, primals_14, primals_15)
        del primals_14
        del primals_15
        buf19 = buf18[0]
        buf20 = buf18[1]
        del buf18
        buf21 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [ret_3], Original ATen: [aten.mm]
        extern_kernels.mm(buf19, reinterpret_tensor(primals_16, (64, 192), (1, 64), 0), out=buf21)
        buf22 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [ret_3], Original ATen: [aten.mm]
        extern_kernels.mm(buf19, reinterpret_tensor(primals_17, (64, 192), (1, 64), 0), out=buf22)
        # Topologically Sorted Source Nodes: [ret_3], Original ATen: [aten._thnn_fused_gru_cell]
        buf23 = torch.ops.aten._thnn_fused_gru_cell.default(buf21, buf22, buf19, primals_18, primals_19)
        del primals_18
        del primals_19
        buf24 = buf23[0]
        buf25 = buf23[1]
        del buf23
        buf26 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [ret_4], Original ATen: [aten.mm]
        extern_kernels.mm(buf24, reinterpret_tensor(primals_20, (64, 192), (1, 64), 0), out=buf26)
        buf27 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [ret_4], Original ATen: [aten.mm]
        extern_kernels.mm(buf24, reinterpret_tensor(primals_21, (64, 192), (1, 64), 0), out=buf27)
        # Topologically Sorted Source Nodes: [ret_4], Original ATen: [aten._thnn_fused_gru_cell]
        buf28 = torch.ops.aten._thnn_fused_gru_cell.default(buf26, buf27, buf24, primals_22, primals_23)
        del buf26
        del buf27
        del primals_22
        del primals_23
        buf29 = buf28[0]
        buf30 = buf28[1]
        del buf28
        buf31 = empty_strided_cuda((1, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_25, buf29, reinterpret_tensor(primals_24, (64, 10), (1, 64), 0), alpha=1, beta=1, out=buf31)
        del primals_25
    return (buf31, primals_1, buf0, buf3, buf4, buf5, buf9, buf10, buf14, buf15, buf19, buf20, buf24, buf25, buf29, buf30, primals_24, primals_21, primals_20, primals_17, primals_16, primals_13, primals_12, primals_9, primals_8, primals_4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((192, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((192, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((192, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((192, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((192, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((192, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((192, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((192, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((192, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((192, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((10, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
