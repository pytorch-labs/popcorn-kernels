# AOT ID: ['185_inference']
import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import (
    grid,
)
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


# kernel path: /tmp/torchinductor_sahanp/ki/ckiemkxutabnhdezwdutb2ostamwne4qtkonqm5trapwvznmngl5.py
# Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten._unsafe_index, aten._to_copy, aten.arange, aten.clamp, aten.view, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   x => _unsafe_index
#   x_1 => _unsafe_index_1, _unsafe_index_2, _unsafe_index_3, _unsafe_index_4, add_110, add_129, add_97, clamp_max_2, clamp_max_3, clamp_min_1, clamp_min_2, clamp_min_3, convert_element_type_5, convert_element_type_6, convert_element_type_7, iota_3, mul_68, mul_81, mul_96, sub_54, sub_57, sub_67, sub_77, sub_80, view_1
# Graph fragment:
#   %scalar_tensor_default_2 : [num_users=3] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (%arg2_1,), kwargs = {})
#   %_unsafe_index : [num_users=4] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg3_1, [None, None, %unsqueeze, %convert_element_type_3]), kwargs = {})
#   %convert_element_type_5 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view, torch.int64), kwargs = {})
#   %iota_3 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (%mul_27,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_6 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_3, torch.float32), kwargs = {})
#   %full_default_6 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -1.0), kwargs = {dtype: torch.float64, layout: torch.strided, device: cpu, pin_memory: False})
#   %full_default_7 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 2), kwargs = {dtype: torch.int64, layout: torch.strided, device: cpu, pin_memory: False})
#   %mul_tensor_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_7, %scalar_tensor_default_2), kwargs = {})
#   %convert_element_type_default_7 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_7, torch.float64), kwargs = {})
#   %add_tensor_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%full_default_6, %convert_element_type_default_7), kwargs = {})
#   %full_default_8 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -1.0), kwargs = {dtype: torch.float64, layout: torch.strided, device: cpu, pin_memory: False})
#   %full_default_9 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 4), kwargs = {dtype: torch.int64, layout: torch.strided, device: cpu, pin_memory: False})
#   %mul_tensor_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_9, %scalar_tensor_default_2), kwargs = {})
#   %convert_element_type_default_8 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_8, torch.float64), kwargs = {})
#   %add_tensor_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%full_default_8, %convert_element_type_default_8), kwargs = {})
#   %true_divide_tensor_3 : [num_users=1] = call_function[target=torch.ops.aten.true_divide.Tensor](args = (%add_tensor_2, %add_tensor_3), kwargs = {})
#   %convert_element_type_default_9 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%true_divide_tensor_3, torch.float32), kwargs = {})
#   %mul_tensor_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_6, %convert_element_type_default_9), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_tensor_9, 0.0), kwargs = {})
#   %view_1 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clamp_min_1, [%mul_27]), kwargs = {})
#   %convert_element_type_7 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_1, torch.int64), kwargs = {})
#   %_unsafe_index_4 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index, [None, None, %clamp_max, %clamp_max_1]), kwargs = {})
#   %_unsafe_index_3 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index, [None, None, %clamp_max, %convert_element_type_7]), kwargs = {})
#   %sub_67 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_4, %_unsafe_index_3), kwargs = {})
#   %sub_54 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_1, %convert_element_type_7), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_54, 0.0), kwargs = {})
#   %clamp_max_2 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 1.0), kwargs = {})
#   %mul_81 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_67, %clamp_max_2), kwargs = {})
#   %add_110 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_3, %mul_81), kwargs = {})
#   %_unsafe_index_2 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index, [None, None, %convert_element_type_5, %clamp_max_1]), kwargs = {})
#   %_unsafe_index_1 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index, [None, None, %convert_element_type_5, %convert_element_type_7]), kwargs = {})
#   %sub_57 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_2, %_unsafe_index_1), kwargs = {})
#   %mul_68 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_57, %clamp_max_2), kwargs = {})
#   %add_97 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_1, %mul_68), kwargs = {})
#   %sub_80 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_110, %add_97), kwargs = {})
#   %sub_77 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %convert_element_type_5), kwargs = {})
#   %clamp_min_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_77, 0.0), kwargs = {})
#   %clamp_max_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_3, 1.0), kwargs = {})
#   %mul_96 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_80, %clamp_max_3), kwargs = {})
#   %add_129 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_97, %mul_96), kwargs = {})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_view_0(in_out_ptr1, in_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // ks1) % ks2)
    x0 = (xindex % ks1)
    x2 = xindex // ks4
    x5 = xindex
    tmp0 = 2.0
    tmp1 = ks0
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 * tmp2
    tmp4 = tmp3.to(tl.float64)
    tmp5 = tl.full([1], -1.0, tl.float64)
    tmp6 = tmp5 + tmp4
    tmp7 = 4.0
    tmp8 = tmp7 * tmp2
    tmp9 = tmp8.to(tl.float64)
    tmp10 = tmp5 + tmp9
    tmp11 = tmp6 / tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp13 = x1
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp14 * tmp12
    tmp16 = 0.0
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = tmp17.to(tl.int64)
    tmp19 = ks3
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp0 * tmp20
    tmp22 = tmp21.to(tl.float64)
    tmp23 = tmp5 + tmp22
    tmp24 = tmp7 * tmp20
    tmp25 = tmp24.to(tl.float64)
    tmp26 = tmp5 + tmp25
    tmp27 = tmp23 / tmp26
    tmp28 = tmp27.to(tl.float32)
    tmp29 = x0
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp30 * tmp28
    tmp32 = triton_helpers.maximum(tmp31, tmp16)
    tmp33 = tmp32.to(tl.int64)
    tmp34 = tl.full([1], 2.0, tl.float64)
    tmp35 = tmp1.to(tl.float64)
    tmp36 = tmp34 * tmp35
    tmp37 = tmp35 / tmp36
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tmp18
    tmp40 = tmp39.to(tl.float32)
    tmp41 = tmp40 * tmp38
    tmp42 = tmp41.to(tl.int64)
    tmp43 = tmp42 + tmp1
    tmp44 = tmp42 < 0
    tmp45 = tl.where(tmp44, tmp43, tmp42)
    tmp46 = tmp19.to(tl.float64)
    tmp47 = tmp34 * tmp46
    tmp48 = tmp46 / tmp47
    tmp49 = tmp48.to(tl.float32)
    tmp50 = tmp33
    tmp51 = tmp50.to(tl.float32)
    tmp52 = tmp51 * tmp49
    tmp53 = tmp52.to(tl.int64)
    tmp54 = tmp53 + tmp19
    tmp55 = tmp53 < 0
    tmp56 = tl.where(tmp55, tmp54, tmp53)
    tmp57 = tl.load(in_ptr0 + (tmp56 + ks3*tmp45 + ks0*ks3*x2), xmask, eviction_policy='evict_last')
    tmp58 = tl.full([1], 1, tl.int64)
    tmp59 = tmp18 + tmp58
    tmp60 = (-1) + 2*ks0
    tmp61 = triton_helpers.minimum(tmp59, tmp60)
    tmp62 = tmp61
    tmp63 = tmp62.to(tl.float32)
    tmp64 = tmp63 * tmp38
    tmp65 = tmp64.to(tl.int64)
    tmp66 = tmp65 + tmp1
    tmp67 = tmp65 < 0
    tmp68 = tl.where(tmp67, tmp66, tmp65)
    tmp69 = tl.load(in_ptr0 + (tmp56 + ks3*tmp68 + ks0*ks3*x2), xmask, eviction_policy='evict_last')
    tmp70 = tmp33 + tmp58
    tmp71 = (-1) + 2*ks3
    tmp72 = triton_helpers.minimum(tmp70, tmp71)
    tmp73 = tmp72
    tmp74 = tmp73.to(tl.float32)
    tmp75 = tmp74 * tmp49
    tmp76 = tmp75.to(tl.int64)
    tmp77 = tmp76 + tmp19
    tmp78 = tmp76 < 0
    tmp79 = tl.where(tmp78, tmp77, tmp76)
    tmp80 = tl.load(in_ptr0 + (tmp79 + ks3*tmp68 + ks0*ks3*x2), xmask, eviction_policy='evict_last')
    tmp81 = tmp80 - tmp69
    tmp82 = tl.load(in_ptr0 + (tmp79 + ks3*tmp45 + ks0*ks3*x2), xmask, eviction_policy='evict_last')
    tmp83 = tmp82 - tmp57
    tmp84 = tmp33.to(tl.float32)
    tmp85 = tmp32 - tmp84
    tmp86 = triton_helpers.maximum(tmp85, tmp16)
    tmp87 = 1.0
    tmp88 = triton_helpers.minimum(tmp86, tmp87)
    tmp89 = tmp81 * tmp88
    tmp90 = tmp69 + tmp89
    tmp91 = tmp83 * tmp88
    tmp92 = tmp57 + tmp91
    tmp93 = tmp90 - tmp92
    tmp94 = tmp18.to(tl.float32)
    tmp95 = tmp17 - tmp94
    tmp96 = triton_helpers.maximum(tmp95, tmp16)
    tmp97 = triton_helpers.minimum(tmp96, tmp87)
    tmp98 = tmp93 * tmp97
    tmp99 = tmp92 + tmp98
    tl.store(in_out_ptr1 + (x5), tmp99, xmask)


# kernel path: /tmp/torchinductor_sahanp/d6/cd6qtbhkswxka5viedqmsgowhtgrxbnvryctqiujyzdqaamfxpo2.py
# Topologically Sorted Source Nodes: [loss, target], Original ATen: [aten.exp, aten.ones_like, aten.mul, aten.sub, aten.mean]
# Source node to ATen node mapping:
#   loss => exp, mean, mul_112, sub_94
#   target => full
# Graph fragment:
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%view_2,), kwargs = {})
#   %full : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([%sym_numel_default], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_112 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full, %view_2), kwargs = {})
#   %sub_94 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%exp, %mul_112), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sub_94,), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_exp_mean_mul_ones_like_sub_1(in_ptr0, out_ptr0, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 6
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    _tmp11 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp0 = r0_1 + x0*((5 + 16*ks0*ks1*ks2) // 6)
        tmp1 = 16*ks0*ks1*ks2
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (r0_1 + x0*((5 + 16*ks0*ks1*ks2) // 6)), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl_math.exp(tmp3)
        tmp5 = 1.0
        tmp6 = tmp5 * tmp3
        tmp7 = tmp4 - tmp6
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(r0_mask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp11, xmask)


# kernel path: /tmp/torchinductor_sahanp/zx/czx52u3jbz6hbg3i5cgxo73tlcn5rmkpldz5ksp24egg234o63ej.py
# Topologically Sorted Source Nodes: [loss, target], Original ATen: [aten.exp, aten.ones_like, aten.mul, aten.sub, aten.mean]
# Source node to ATen node mapping:
#   loss => exp, mean, mul_112, sub_94
#   target => full
# Graph fragment:
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%view_2,), kwargs = {})
#   %full : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([%sym_numel_default], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_112 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full, %view_2), kwargs = {})
#   %sub_94 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%exp, %mul_112), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sub_94,), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_exp_mean_mul_ones_like_sub_2(in_out_ptr0, in_ptr0, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 6
    R0_BLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_mask = r0_index < r0_numel
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


def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        4*s2
        4*s1
        16*s1*s2
        buf2 = empty_strided_cuda((1, s0, 4*s1, 4*s2), (16*s0*s1*s2, 16*s1*s2, 4*s2, 1), torch.float32)
        buf5 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten._unsafe_index, aten._to_copy, aten.arange, aten.clamp, aten.view, aten.sub, aten.mul, aten.add]
        triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_view_0_xnumel = 16*s0*s1*s2
        get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_view_0[grid(triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_view_0_xnumel)](buf5, arg3_1, 32, 128, 128, 32, 16384, 49152, XBLOCK=256, num_warps=4, num_stages=1)
        del arg3_1
        buf6 = empty_strided_cuda((6, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [loss, target], Original ATen: [aten.exp, aten.ones_like, aten.mul, aten.sub, aten.mean]
        (5 + 16*s0*s1*s2) // 6
        get_raw_stream(0)
        triton_red_fused_exp_mean_mul_ones_like_sub_1[grid(6)](buf5, buf6, 3, 32, 32, 6, 8192, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del buf5
        buf7 = empty_strided_cuda((), (), torch.float32)
        buf8 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [loss, target], Original ATen: [aten.exp, aten.ones_like, aten.mul, aten.sub, aten.mean]
        get_raw_stream(0)
        triton_per_fused_exp_mean_mul_ones_like_sub_2[grid(1)](buf8, buf6, 3, 32, 32, 1, 6, XBLOCK=1, num_warps=2, num_stages=1)
        del buf6
    return (buf8, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 3
    arg1_1 = 32
    arg2_1 = 32
    arg3_1 = rand_strided((1, 3, 32, 32), (3072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
