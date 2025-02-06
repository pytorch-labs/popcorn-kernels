# AOT ID: ['74_inference']
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


# kernel path: /tmp/torchinductor_sahanp/gr/cgr7tc3kfrsa35ibgd3b2im4faxlfiw56kjwls5obseornvizsz2.py
# Topologically Sorted Source Nodes: [triplet_loss, mean_1], Original ATen: [aten.sub, aten.mean]
# Source node to ATen node mapping:
#   mean_1 => mean_2
#   triplet_loss => sub_63, sub_72
# Graph fragment:
#   %sub_63 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_4, %slice_8), kwargs = {})
#   %sub_72 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_4, %slice_12), kwargs = {})
#   %mean_2 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_4, [1, 2, 3]), kwargs = {})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_mean_sub_0(in_ptr0, out_ptr0, out_ptr1, out_ptr2, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    _tmp90 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_0 = (r0_index % ks0)
        r0_1 = r0_index // ks0
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (2*(((r0_0 // 2) % (ks2 // 2))) + 2*ks2*(((r0_1 // 2) % (ks1 // 2))) + ks1*ks2*((r0_0 % 2)) + 2*ks1*ks2*((r0_1 % 2))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr0 + (1 + 2*(((r0_0 // 2) % ks0)) + 2*ks2*(((r0_1 // 2) % (ks1 // 2))) + ks1*ks2*((r0_0 % 2)) + 2*ks1*ks2*((r0_1 % 2))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr0 + (ks2 + 2*(((r0_0 // 2) % ks0)) + 2*ks2*(((r0_1 // 2) % (ks1 // 2))) + ks1*ks2*((r0_0 % 2)) + 2*ks1*ks2*((r0_1 % 2))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr0 + (1 + ks2 + 2*(((r0_0 // 2) % ks0)) + 2*ks2*(((r0_1 // 2) % (ks1 // 2))) + ks1*ks2*((r0_0 % 2)) + 2*ks1*ks2*((r0_1 % 2))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp20 = tl.load(in_ptr0 + (2*(((r0_0 // 2) % ks0)) + 2*ks2*(((r0_1 // 2) % (ks1 // 2))) + ks1*ks2*((r0_0 % 2)) + 2*ks1*ks2*((r0_1 % 2))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp32 = tl.load(in_ptr0 + (2*((ks0 + r0_0) // 2) + 2*ks2*(((r0_1 // 2) % (ks1 // 2))) + ks1*ks2*(((ks0 + r0_0) % 2)) + 2*ks1*ks2*((r0_1 % 2))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp34 = tl.load(in_ptr0 + (1 + 2*((ks0 + r0_0) // 2) + 2*ks2*(((r0_1 // 2) % (ks1 // 2))) + ks1*ks2*(((ks0 + r0_0) % 2)) + 2*ks1*ks2*((r0_1 % 2))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp37 = tl.load(in_ptr0 + (ks2 + 2*((ks0 + r0_0) // 2) + 2*ks2*(((r0_1 // 2) % (ks1 // 2))) + ks1*ks2*(((ks0 + r0_0) % 2)) + 2*ks1*ks2*((r0_1 % 2))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp40 = tl.load(in_ptr0 + (1 + ks2 + 2*((ks0 + r0_0) // 2) + 2*ks2*(((r0_1 // 2) % (ks1 // 2))) + ks1*ks2*(((ks0 + r0_0) % 2)) + 2*ks1*ks2*((r0_1 % 2))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp65 = tl.load(in_ptr0 + (2*(((r0_0 // 2) % ks0)) + 2*ks2*(triton_helpers.div_floor_integer(r0_1 + (ks1 // 2),  2)) + ks1*ks2*((r0_0 % 2)) + 2*ks1*ks2*(((r0_1 + (ks1 // 2)) % 2))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp67 = tl.load(in_ptr0 + (1 + 2*(((r0_0 // 2) % ks0)) + 2*ks2*(triton_helpers.div_floor_integer(r0_1 + (ks1 // 2),  2)) + ks1*ks2*((r0_0 % 2)) + 2*ks1*ks2*(((r0_1 + (ks1 // 2)) % 2))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp70 = tl.load(in_ptr0 + (ks2 + 2*(((r0_0 // 2) % ks0)) + 2*ks2*(triton_helpers.div_floor_integer(r0_1 + (ks1 // 2),  2)) + ks1*ks2*((r0_0 % 2)) + 2*ks1*ks2*(((r0_1 + (ks1 // 2)) % 2))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp73 = tl.load(in_ptr0 + (1 + ks2 + 2*(((r0_0 // 2) % ks0)) + 2*ks2*(triton_helpers.div_floor_integer(r0_1 + (ks1 // 2),  2)) + ks1*ks2*((r0_0 % 2)) + 2*ks1*ks2*(((r0_1 + (ks1 // 2)) % 2))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0 * tmp0
        tmp3 = tmp2 * tmp2
        tmp4 = tmp3 + tmp1
        tmp6 = tmp5 * tmp5
        tmp7 = tmp6 + tmp4
        tmp9 = tmp8 * tmp8
        tmp10 = tmp9 + tmp7
        tmp11 = 0.25
        tmp12 = tmp10 * tmp11
        tmp13 = tl.full([1, 1], 0, tl.int32)
        tmp14 = tmp13 < tmp12
        tmp15 = tmp14.to(tl.int8)
        tmp16 = tmp12 < tmp13
        tmp17 = tmp16.to(tl.int8)
        tmp18 = tmp15 - tmp17
        tmp19 = tmp18.to(tmp12.dtype)
        tmp21 = tmp20 * tmp20
        tmp22 = tmp3 + tmp21
        tmp23 = tmp6 + tmp22
        tmp24 = tmp9 + tmp23
        tmp25 = tmp24 * tmp11
        tmp26 = tl_math.abs(tmp25)
        tmp27 = triton_helpers.maximum(tmp13, tmp26)
        tmp28 = tmp19 * tmp27
        tmp29 = 4.0
        tmp30 = tmp28 * tmp29
        tmp31 = libdevice.sqrt(tmp30)
        tmp33 = tmp32 * tmp32
        tmp35 = tmp34 * tmp34
        tmp36 = tmp35 + tmp33
        tmp38 = tmp37 * tmp37
        tmp39 = tmp38 + tmp36
        tmp41 = tmp40 * tmp40
        tmp42 = tmp41 + tmp39
        tmp43 = tmp42 * tmp11
        tmp44 = tmp13 < tmp43
        tmp45 = tmp44.to(tl.int8)
        tmp46 = tmp43 < tmp13
        tmp47 = tmp46.to(tl.int8)
        tmp48 = tmp45 - tmp47
        tmp49 = tmp48.to(tmp43.dtype)
        tmp50 = tl_math.abs(tmp43)
        tmp51 = triton_helpers.maximum(tmp13, tmp50)
        tmp52 = tmp49 * tmp51
        tmp53 = tmp52 * tmp29
        tmp54 = libdevice.sqrt(tmp53)
        tmp55 = tmp31 - tmp54
        tmp56 = tmp13 < tmp25
        tmp57 = tmp56.to(tl.int8)
        tmp58 = tmp25 < tmp13
        tmp59 = tmp58.to(tl.int8)
        tmp60 = tmp57 - tmp59
        tmp61 = tmp60.to(tmp25.dtype)
        tmp62 = tmp61 * tmp27
        tmp63 = tmp62 * tmp29
        tmp64 = libdevice.sqrt(tmp63)
        tmp66 = tmp65 * tmp65
        tmp68 = tmp67 * tmp67
        tmp69 = tmp68 + tmp66
        tmp71 = tmp70 * tmp70
        tmp72 = tmp71 + tmp69
        tmp74 = tmp73 * tmp73
        tmp75 = tmp74 + tmp72
        tmp76 = tmp75 * tmp11
        tmp77 = tmp13 < tmp76
        tmp78 = tmp77.to(tl.int8)
        tmp79 = tmp76 < tmp13
        tmp80 = tmp79.to(tl.int8)
        tmp81 = tmp78 - tmp80
        tmp82 = tmp81.to(tmp76.dtype)
        tmp83 = tl_math.abs(tmp76)
        tmp84 = triton_helpers.maximum(tmp13, tmp83)
        tmp85 = tmp82 * tmp84
        tmp86 = tmp85 * tmp29
        tmp87 = libdevice.sqrt(tmp86)
        tmp88 = tmp64 - tmp87
        tmp89 = tl.broadcast_to(tmp64, [XBLOCK, R0_BLOCK])
        tmp91 = _tmp90 + tmp89
        _tmp90 = tl.where(r0_mask, tmp91, _tmp90)
        tl.store(out_ptr0 + (tl.broadcast_to(r0_2, [XBLOCK, R0_BLOCK])), tmp55, r0_mask)
        tl.store(out_ptr1 + (tl.broadcast_to(r0_2, [XBLOCK, R0_BLOCK])), tmp88, r0_mask)
    tmp90 = tl.sum(_tmp90, 1)[:, None]
    tl.store(out_ptr2 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp90, None)


# kernel path: /tmp/torchinductor_sahanp/7x/c7xvv4qylix3dwsqar3td2hsqyzdfdtyshfdvcxff5abd3rcqgun.py
# Topologically Sorted Source Nodes: [triplet_loss], Original ATen: [aten.add, aten.norm]
# Source node to ATen node mapping:
#   triplet_loss => add_88, pow_3, sum_1
# Graph fragment:
#   %add_88 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%sub_63, 1e-06), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_88, 2.0), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_3, [3]), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_norm_1(in_ptr0, out_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    _tmp5 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 1e-06
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2 * tmp2
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(r0_mask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp5, xmask)


# kernel path: /tmp/torchinductor_sahanp/jz/cjzev7aryn32dheqm53g5pvtoqqnuu5olivwv5jsvlrcrnqrkxks.py
# Topologically Sorted Source Nodes: [triplet_loss, hinge_loss, mean_1, add], Original ATen: [aten.norm, aten.add, aten.sub, aten.clamp_min, aten.mean, aten.ne, aten.fill, aten.zeros_like, aten.where]
# Source node to ATen node mapping:
#   add => add_119
#   hinge_loss => add_118, clamp_min_1, full_default, full_default_1, full_default_2, full_default_3, mean_3, sub_88, where, where_1
#   mean_1 => mean_2
#   triplet_loss => add_108, clamp_min, mean, pow_4, pow_6, sub_83
# Graph fragment:
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %add_108 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%pow_4, 1.0), kwargs = {})
#   %pow_6 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_2, 0.5), kwargs = {})
#   %sub_83 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_108, %pow_6), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_83, 0), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%clamp_min,), kwargs = {})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], False), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mean_2 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%slice_4, [1, 2, 3]), kwargs = {})
#   %sub_88 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default_1, %mean_2), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_88, 0), kwargs = {})
#   %full_default : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([1], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%full_default_2, %clamp_min_1, %full_default), kwargs = {})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%full_default_3, %mean_2, %full_default), kwargs = {})
#   %add_118 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where, %where_1), kwargs = {})
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%add_118,), kwargs = {})
#   %add_119 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, %mean_3), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_clamp_min_fill_mean_ne_norm_sub_where_zeros_like_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = libdevice.sqrt(tmp0)
        tmp2 = 1.0
        tmp3 = tmp1 + tmp2
        tmp5 = libdevice.sqrt(tmp4)
        tmp6 = tmp3 - tmp5
        tmp7 = 0.0
        tmp8 = triton_helpers.maximum(tmp6, tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tmp15 = tl.load(in_ptr2 + (0))
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, 1])
    tmp12 = ks0 // 2
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp10 / tmp13
    tmp17 = ks1*(ks0 // 2)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 / tmp18
    tmp20 = 1.0
    tmp21 = tmp20 - tmp19
    tmp22 = 0.0
    tmp23 = triton_helpers.maximum(tmp21, tmp22)
    tmp24 = tl.full([1, 1], False, tl.int1)
    tmp25 = tl.where(tmp24, tmp23, tmp22)
    tmp26 = tl.full([1, 1], True, tl.int1)
    tmp27 = tl.where(tmp26, tmp19, tmp22)
    tmp28 = tmp25 + tmp27
    tmp29 = tmp28 / tmp20
    tmp30 = tmp14 + tmp29
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp30, None)


def call(args):
    _arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, 4, s1, s2), (4*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        s2 // 2
        buf0 = empty_strided_cuda((1, 1, s1 // 2, s2 // 2), ((s1 // 2)*(s2 // 2), (s1 // 2)*(s2 // 2), s2 // 2, 1), torch.float32)
        buf2 = empty_strided_cuda((1, 1, s1 // 2, s2 // 2), ((s1 // 2)*(s2 // 2), (s1 // 2)*(s2 // 2), s2 // 2, 1), torch.float32)
        buf5 = empty_strided_cuda((1, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [triplet_loss, mean_1], Original ATen: [aten.sub, aten.mean]
        (s1 // 2)*(s2 // 2)
        get_raw_stream(0)
        triton_red_fused_mean_sub_0[grid(1)](arg3_1, buf0, buf2, buf5, 32, 64, 64, 1, 1024, XBLOCK=1, R0_BLOCK=1024, num_warps=8, num_stages=1)
        del arg3_1
        buf1 = empty_strided_cuda((1, 1, s1 // 2), (s1 // 2, s1 // 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [triplet_loss], Original ATen: [aten.add, aten.norm]
        triton_red_fused_add_norm_1_xnumel = s1 // 2
        s2 // 2
        get_raw_stream(0)
        triton_red_fused_add_norm_1[grid(triton_red_fused_add_norm_1_xnumel)](buf0, buf1, 32, 32, 32, XBLOCK=1, R0_BLOCK=32, num_warps=2, num_stages=1)
        del buf0
        buf3 = empty_strided_cuda((1, 1, s1 // 2), (s1 // 2, s1 // 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [triplet_loss], Original ATen: [aten.add, aten.norm]
        triton_red_fused_add_norm_1_xnumel = s1 // 2
        s2 // 2
        get_raw_stream(0)
        triton_red_fused_add_norm_1[grid(triton_red_fused_add_norm_1_xnumel)](buf2, buf3, 32, 32, 32, XBLOCK=1, R0_BLOCK=32, num_warps=2, num_stages=1)
        del buf2
        buf4 = empty_strided_cuda((), (), torch.float32)
        buf6 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [triplet_loss, hinge_loss, mean_1, add], Original ATen: [aten.norm, aten.add, aten.sub, aten.clamp_min, aten.mean, aten.ne, aten.fill, aten.zeros_like, aten.where]
        s1 // 2
        get_raw_stream(0)
        triton_red_fused_add_clamp_min_fill_mean_ne_norm_sub_where_zeros_like_2[grid(1)](buf6, buf1, buf3, buf5, 64, 32, 1, 32, XBLOCK=1, R0_BLOCK=32, num_warps=2, num_stages=1)
        del buf1
        del buf3
        del buf5
    return (buf6, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 4
    arg1_1 = 64
    arg2_1 = 64
    arg3_1 = rand_strided((1, 4, 64, 64), (16384, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
