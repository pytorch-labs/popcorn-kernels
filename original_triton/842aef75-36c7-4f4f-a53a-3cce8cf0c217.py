# AOT ID: ['18_inference']
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


# kernel path: /tmp/torchinductor_sahanp/y4/cy4nwr47xt7krmocs557ff5bnkg5lv4mcnbgt732xl4spc256f66.py
# Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten._to_copy, aten.arange, aten.clamp, aten.view, aten._unsafe_index, aten.sub, aten.mul, aten.add, aten.hardswish]
# Source node to ATen node mapping:
#   x => _unsafe_index, _unsafe_index_1, _unsafe_index_2, _unsafe_index_3, add_100, add_68, add_81, clamp_max_2, clamp_max_3, clamp_min_1, clamp_min_2, clamp_min_3, convert_element_type_1, convert_element_type_2, convert_element_type_3, iota_1, mul_42, mul_55, mul_70, sub_38, sub_41, sub_51, sub_61, sub_64, view_1
#   x_1 => add_105, clamp_max_4, clamp_min_4, div, mul_84
# Graph fragment:
#   %convert_element_type_1 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view, torch.int64), kwargs = {})
#   %iota_1 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (%mul_1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_1, torch.float32), kwargs = {})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -1.0), kwargs = {dtype: torch.float64, layout: torch.strided, device: cpu, pin_memory: False})
#   %scalar_tensor_default_5 : [num_users=2] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (%arg2_1,), kwargs = {})
#   %convert_element_type_default_3 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%scalar_tensor_default_5, torch.float64), kwargs = {})
#   %add_tensor_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%full_default_3, %convert_element_type_default_3), kwargs = {})
#   %full_default_4 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -1.0), kwargs = {dtype: torch.float64, layout: torch.strided, device: cpu, pin_memory: False})
#   %full_default_5 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 2), kwargs = {dtype: torch.int64, layout: torch.strided, device: cpu, pin_memory: False})
#   %mul_tensor_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_5, %scalar_tensor_default_5), kwargs = {})
#   %convert_element_type_default_4 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_2, torch.float64), kwargs = {})
#   %add_tensor_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%full_default_4, %convert_element_type_default_4), kwargs = {})
#   %true_divide_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.true_divide.Tensor](args = (%add_tensor_2, %add_tensor_3), kwargs = {})
#   %convert_element_type_default_5 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%true_divide_tensor_1, torch.float32), kwargs = {})
#   %mul_tensor_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_2, %convert_element_type_default_5), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_tensor_3, 0.0), kwargs = {})
#   %view_1 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clamp_min_1, [%mul_1]), kwargs = {})
#   %convert_element_type_3 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_1, torch.int64), kwargs = {})
#   %_unsafe_index_3 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg3_1, [None, None, %clamp_max, %clamp_max_1]), kwargs = {})
#   %_unsafe_index_2 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg3_1, [None, None, %clamp_max, %convert_element_type_3]), kwargs = {})
#   %sub_51 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_3, %_unsafe_index_2), kwargs = {})
#   %sub_38 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_1, %convert_element_type_3), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_38, 0.0), kwargs = {})
#   %clamp_max_2 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 1.0), kwargs = {})
#   %mul_55 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_51, %clamp_max_2), kwargs = {})
#   %add_81 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_2, %mul_55), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg3_1, [None, None, %convert_element_type_1, %clamp_max_1]), kwargs = {})
#   %_unsafe_index : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg3_1, [None, None, %convert_element_type_1, %convert_element_type_3]), kwargs = {})
#   %sub_41 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_1, %_unsafe_index), kwargs = {})
#   %mul_42 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_41, %clamp_max_2), kwargs = {})
#   %add_68 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index, %mul_42), kwargs = {})
#   %sub_64 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_81, %add_68), kwargs = {})
#   %sub_61 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %convert_element_type_1), kwargs = {})
#   %clamp_min_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_61, 0.0), kwargs = {})
#   %clamp_max_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_3, 1.0), kwargs = {})
#   %mul_70 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_64, %clamp_max_3), kwargs = {})
#   %add_100 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_68, %mul_70), kwargs = {})
#   %add_105 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_100, 3), kwargs = {})
#   %clamp_min_4 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_105, 0), kwargs = {})
#   %clamp_max_4 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_4, 6), kwargs = {})
#   %mul_84 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_100, %clamp_max_4), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_84, 6), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_hardswish_mul_sub_view_0(in_out_ptr1, in_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // ks1) % ks2)
    x0 = (xindex % ks1)
    x2 = xindex // ks4
    x4 = xindex
    tmp0 = tl.full([1], -1.0, tl.float64)
    tmp1 = ks0
    tmp2 = tmp1.to(tl.float64)
    tmp3 = tmp0 + tmp2
    tmp4 = 2.0
    tmp5 = tmp1.to(tl.float32)
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6.to(tl.float64)
    tmp8 = tmp0 + tmp7
    tmp9 = tmp3 / tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = x1
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp12 * tmp10
    tmp14 = 0.0
    tmp15 = triton_helpers.maximum(tmp13, tmp14)
    tmp16 = tmp15.to(tl.int64)
    tmp17 = ks3
    tmp18 = tmp17.to(tl.float64)
    tmp19 = tmp0 + tmp18
    tmp20 = tmp17.to(tl.float32)
    tmp21 = tmp4 * tmp20
    tmp22 = tmp21.to(tl.float64)
    tmp23 = tmp0 + tmp22
    tmp24 = tmp19 / tmp23
    tmp25 = tmp24.to(tl.float32)
    tmp26 = x0
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp27 * tmp25
    tmp29 = triton_helpers.maximum(tmp28, tmp14)
    tmp30 = tmp29.to(tl.int64)
    tmp31 = tl.load(in_ptr0 + (tmp30 + ks3*tmp16 + ks0*ks3*x2), xmask, eviction_policy='evict_last')
    tmp32 = tl.full([1], 1, tl.int64)
    tmp33 = tmp16 + tmp32
    tmp34 = (-1) + ks0
    tmp35 = triton_helpers.minimum(tmp33, tmp34)
    tmp36 = tl.load(in_ptr0 + (tmp30 + ks3*tmp35 + ks0*ks3*x2), xmask, eviction_policy='evict_last')
    tmp37 = tmp30 + tmp32
    tmp38 = (-1) + ks3
    tmp39 = triton_helpers.minimum(tmp37, tmp38)
    tmp40 = tl.load(in_ptr0 + (tmp39 + ks3*tmp35 + ks0*ks3*x2), xmask, eviction_policy='evict_last')
    tmp41 = tmp40 - tmp36
    tmp42 = tl.load(in_ptr0 + (tmp39 + ks3*tmp16 + ks0*ks3*x2), xmask, eviction_policy='evict_last')
    tmp43 = tmp42 - tmp31
    tmp44 = tmp30.to(tl.float32)
    tmp45 = tmp29 - tmp44
    tmp46 = triton_helpers.maximum(tmp45, tmp14)
    tmp47 = 1.0
    tmp48 = triton_helpers.minimum(tmp46, tmp47)
    tmp49 = tmp41 * tmp48
    tmp50 = tmp36 + tmp49
    tmp51 = tmp43 * tmp48
    tmp52 = tmp31 + tmp51
    tmp53 = tmp50 - tmp52
    tmp54 = tmp16.to(tl.float32)
    tmp55 = tmp15 - tmp54
    tmp56 = triton_helpers.maximum(tmp55, tmp14)
    tmp57 = triton_helpers.minimum(tmp56, tmp47)
    tmp58 = tmp53 * tmp57
    tmp59 = tmp52 + tmp58
    tmp60 = 3.0
    tmp61 = tmp59 + tmp60
    tmp62 = triton_helpers.maximum(tmp61, tmp14)
    tmp63 = 6.0
    tmp64 = triton_helpers.minimum(tmp62, tmp63)
    tmp65 = tmp59 * tmp64
    tmp66 = 0.16666666666666666
    tmp67 = tmp65 * tmp66
    tl.store(in_out_ptr1 + (x4), tmp67, xmask)







def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        ps0 = 2*s2
        ps1 = 2*s1
        ps2 = 4*s1*s2
        buf2 = empty_strided_cuda((1, s0, 2*s1, 2*s2), (4*s0*s1*s2, 4*s1*s2, 2*s2, 1), torch.float32)
        buf5 = buf2; del buf2  # reuse
        buf6 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten._to_copy, aten.arange, aten.clamp, aten.view, aten._unsafe_index, aten.sub, aten.mul, aten.add, aten.hardswish]
        triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_hardswish_mul_sub_view_0_xnumel = 4*s0*s1*s2
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_hardswish_mul_sub_view_0[grid(triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_hardswish_mul_sub_view_0_xnumel)](buf6, arg3_1, 32, 64, 64, 32, 4096, 12288, XBLOCK=128, num_warps=4, num_stages=1)
        del arg3_1
    return (reinterpret_tensor(buf6, (1, 2*s2, 2*s0*s1), (4*s0*s1*s2, 1, 2*s2), 0), )


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
