
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


from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_binary_cross_entropy_rand_like_sub_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr3, ks0, ks1, ks2, load_seed_offset, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 9
    R0_BLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_mask = r0_index < r0_numel
    r0_0 = (r0_index % 3)
    r0_1 = r0_index // 3
    r0_2 = r0_index
    tl.device_assert(((0 <= tl.where((triton_helpers.div_floor_integer(r0_0 + 3*r0_1,  12 + ((-6)*ks0) + ((-6)*ks1) + 3*ks0*ks1)) + ((((((r0_0 + 3*r0_1) % (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1))) // ((-2) + ks1)) % ((-2) + ks0))) < 0, ks0 + (triton_helpers.div_floor_integer(r0_0 + 3*r0_1,  12 + ((-6)*ks0) + ((-6)*ks1) + 3*ks0*ks1)) + ((((((r0_0 + 3*r0_1) % (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1))) // ((-2) + ks1)) % ((-2) + ks0))), (triton_helpers.div_floor_integer(r0_0 + 3*r0_1,  12 + ((-6)*ks0) + ((-6)*ks1) + 3*ks0*ks1)) + ((((((r0_0 + 3*r0_1) % (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1))) // ((-2) + ks1)) % ((-2) + ks0))))) & (tl.where((triton_helpers.div_floor_integer(r0_0 + 3*r0_1,  12 + ((-6)*ks0) + ((-6)*ks1) + 3*ks0*ks1)) + ((((((r0_0 + 3*r0_1) % (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1))) // ((-2) + ks1)) % ((-2) + ks0))) < 0, ks0 + (triton_helpers.div_floor_integer(r0_0 + 3*r0_1,  12 + ((-6)*ks0) + ((-6)*ks1) + 3*ks0*ks1)) + ((((((r0_0 + 3*r0_1) % (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1))) // ((-2) + ks1)) % ((-2) + ks0))), (triton_helpers.div_floor_integer(r0_0 + 3*r0_1,  12 + ((-6)*ks0) + ((-6)*ks1) + 3*ks0*ks1)) + ((((((r0_0 + 3*r0_1) % (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1))) // ((-2) + ks1)) % ((-2) + ks0)))) < ks0)) | ~(r0_mask), "index out of bounds: 0 <= tl.where((triton_helpers.div_floor_integer(r0_0 + 3*r0_1,  12 + ((-6)*ks0) + ((-6)*ks1) + 3*ks0*ks1)) + ((((((r0_0 + 3*r0_1) % (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1))) // ((-2) + ks1)) % ((-2) + ks0))) < 0, ks0 + (triton_helpers.div_floor_integer(r0_0 + 3*r0_1,  12 + ((-6)*ks0) + ((-6)*ks1) + 3*ks0*ks1)) + ((((((r0_0 + 3*r0_1) % (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1))) // ((-2) + ks1)) % ((-2) + ks0))), (triton_helpers.div_floor_integer(r0_0 + 3*r0_1,  12 + ((-6)*ks0) + ((-6)*ks1) + 3*ks0*ks1)) + ((((((r0_0 + 3*r0_1) % (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1))) // ((-2) + ks1)) % ((-2) + ks0)))) < ks0")
    tl.device_assert(((0 <= tl.where(((((r0_0 + 3*r0_1) // (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1)) % 3)) + (((((r0_0 + 3*r0_1) % (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1))) % ((-2) + ks1))) < 0, ks1 + ((((r0_0 + 3*r0_1) // (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1)) % 3)) + (((((r0_0 + 3*r0_1) % (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1))) % ((-2) + ks1))), ((((r0_0 + 3*r0_1) // (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1)) % 3)) + (((((r0_0 + 3*r0_1) % (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1))) % ((-2) + ks1))))) & (tl.where(((((r0_0 + 3*r0_1) // (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1)) % 3)) + (((((r0_0 + 3*r0_1) % (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1))) % ((-2) + ks1))) < 0, ks1 + ((((r0_0 + 3*r0_1) // (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1)) % 3)) + (((((r0_0 + 3*r0_1) % (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1))) % ((-2) + ks1))), ((((r0_0 + 3*r0_1) // (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1)) % 3)) + (((((r0_0 + 3*r0_1) % (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1))) % ((-2) + ks1)))) < ks1)) | ~(r0_mask), "index out of bounds: 0 <= tl.where(((((r0_0 + 3*r0_1) // (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1)) % 3)) + (((((r0_0 + 3*r0_1) % (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1))) % ((-2) + ks1))) < 0, ks1 + ((((r0_0 + 3*r0_1) // (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1)) % 3)) + (((((r0_0 + 3*r0_1) % (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1))) % ((-2) + ks1))), ((((r0_0 + 3*r0_1) // (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1)) % 3)) + (((((r0_0 + 3*r0_1) % (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1))) % ((-2) + ks1)))) < ks1")
    tmp2 = tl.load(in_ptr0 + (ks1*(tl.where((triton_helpers.div_floor_integer(r0_0 + 3*r0_1,  12 + ((-6)*ks0) + ((-6)*ks1) + 3*ks0*ks1)) + ((((((r0_0 + 3*r0_1) % (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1))) // ((-2) + ks1)) % ((-2) + ks0))) < 0, ks0 + (triton_helpers.div_floor_integer(r0_0 + 3*r0_1,  12 + ((-6)*ks0) + ((-6)*ks1) + 3*ks0*ks1)) + ((((((r0_0 + 3*r0_1) % (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1))) // ((-2) + ks1)) % ((-2) + ks0))), (triton_helpers.div_floor_integer(r0_0 + 3*r0_1,  12 + ((-6)*ks0) + ((-6)*ks1) + 3*ks0*ks1)) + ((((((r0_0 + 3*r0_1) % (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1))) // ((-2) + ks1)) % ((-2) + ks0))))) + ks0*ks1*(triton_helpers.div_floor_integer(r0_0 + 3*r0_1,  36 + ((-18)*ks0) + ((-18)*ks1) + 9*ks0*ks1)) + (tl.where(((((r0_0 + 3*r0_1) // (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1)) % 3)) + (((((r0_0 + 3*r0_1) % (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1))) % ((-2) + ks1))) < 0, ks1 + ((((r0_0 + 3*r0_1) // (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1)) % 3)) + (((((r0_0 + 3*r0_1) % (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1))) % ((-2) + ks1))), ((((r0_0 + 3*r0_1) // (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1)) % 3)) + (((((r0_0 + 3*r0_1) % (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1))) % ((-2) + ks1)))))), r0_mask, eviction_policy='evict_last', other=0.0)
    tl.device_assert((((((9 + r0_0 + 3*r0_1) // (12 + ((-6)*ks0) + ((-6)*ks1) + 3*ks0*ks1)) % 3)) + ((((((9 + r0_0 + 3*r0_1) % (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1))) // ((-2) + ks1)) % ((-2) + ks0))) < ks0) | ~(r0_mask), "index out of bounds: ((((9 + r0_0 + 3*r0_1) // (12 + ((-6)*ks0) + ((-6)*ks1) + 3*ks0*ks1)) % 3)) + ((((((9 + r0_0 + 3*r0_1) % (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1))) // ((-2) + ks1)) % ((-2) + ks0))) < ks0")
    tl.device_assert((((((9 + r0_0 + 3*r0_1) // (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1)) % 3)) + (((((9 + r0_0 + 3*r0_1) % (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1))) % ((-2) + ks1))) < ks1) | ~(r0_mask), "index out of bounds: ((((9 + r0_0 + 3*r0_1) // (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1)) % 3)) + (((((9 + r0_0 + 3*r0_1) % (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1))) % ((-2) + ks1))) < ks1")
    tmp16 = tl.load(in_ptr0 + (ks1*((((9 + r0_0 + 3*r0_1) // (12 + ((-6)*ks0) + ((-6)*ks1) + 3*ks0*ks1)) % 3)) + ks1*((((((9 + r0_0 + 3*r0_1) % (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1))) // ((-2) + ks1)) % ((-2) + ks0))) + ks0*ks1*(triton_helpers.div_floor_integer(9 + r0_0 + 3*r0_1,  36 + ((-18)*ks0) + ((-18)*ks1) + 9*ks0*ks1)) + ((((9 + r0_0 + 3*r0_1) // (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1)) % 3)) + (((((9 + r0_0 + 3*r0_1) % (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1))) % ((-2) + ks1)))), r0_mask, eviction_policy='evict_last', other=0.0)
    tl.device_assert((((((((18 + r0_0 + 3*r0_1) % (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1))) // ((-2) + ks1)) % ((-2) + ks0))) + (((((((18 + r0_0 + 3*r0_1) // (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1)) % (9*ks2))) // 3) % 3)) < ks0) | ~(r0_mask), "index out of bounds: ((((((18 + r0_0 + 3*r0_1) % (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1))) // ((-2) + ks1)) % ((-2) + ks0))) + (((((((18 + r0_0 + 3*r0_1) // (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1)) % (9*ks2))) // 3) % 3)) < ks0")
    tl.device_assert(((((((18 + r0_0 + 3*r0_1) % (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1))) % ((-2) + ks1))) + ((((((18 + r0_0 + 3*r0_1) // (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1)) % (9*ks2))) % 3)) < ks1) | ~(r0_mask), "index out of bounds: (((((18 + r0_0 + 3*r0_1) % (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1))) % ((-2) + ks1))) + ((((((18 + r0_0 + 3*r0_1) // (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1)) % (9*ks2))) % 3)) < ks1")
    tmp27 = tl.load(in_ptr0 + (ks1*((((((18 + r0_0 + 3*r0_1) % (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1))) // ((-2) + ks1)) % ((-2) + ks0))) + ks1*(((((((18 + r0_0 + 3*r0_1) // (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1)) % (9*ks2))) // 3) % 3)) + ks0*ks1*(((((((18 + r0_0 + 3*r0_1) // (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1)) % (9*ks2))) // 9) % ks2)) + (((((18 + r0_0 + 3*r0_1) % (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1))) % ((-2) + ks1))) + ((((((18 + r0_0 + 3*r0_1) // (4 + ((-2)*ks0) + ((-2)*ks1) + ks0*ks1)) % (9*ks2))) % 3))), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = libdevice.expm1(tmp2)
    tmp6 = tl.where(tmp4, tmp2, tmp5)
    tmp7 = 3.0
    tmp8 = tmp6 + tmp7
    tmp9 = triton_helpers.maximum(tmp8, tmp3)
    tmp10 = 6.0
    tmp11 = triton_helpers.minimum(tmp9, tmp10)
    tmp12 = 0.16666666666666666
    tmp13 = tmp11 * tmp12
    tmp17 = tmp16 > tmp3
    tmp18 = libdevice.expm1(tmp16)
    tmp19 = tl.where(tmp17, tmp16, tmp18)
    tmp20 = tmp19 + tmp7
    tmp21 = triton_helpers.maximum(tmp20, tmp3)
    tmp22 = triton_helpers.minimum(tmp21, tmp10)
    tmp23 = tmp22 * tmp12
    tmp24 = tmp13 - tmp23
    tmp28 = tmp27 > tmp3
    tmp29 = libdevice.expm1(tmp27)
    tmp30 = tl.where(tmp28, tmp27, tmp29)
    tmp31 = tmp30 + tmp7
    tmp32 = triton_helpers.maximum(tmp31, tmp3)
    tmp33 = triton_helpers.minimum(tmp32, tmp10)
    tmp34 = tmp33 * tmp12
    tmp35 = tmp13 - tmp34
    tmp36 = tl.load(in_ptr1 + load_seed_offset)
    tmp37 = r0_2
    tmp38 = tl.rand(tmp36, (tmp37).to(tl.uint32))
    tmp39 = 1.0
    tmp40 = tmp38 - tmp39
    tmp41 = -tmp13
    tmp42 = libdevice.log1p(tmp41)
    tmp43 = -100.0
    tmp44 = triton_helpers.maximum(tmp42, tmp43)
    tmp45 = tmp40 * tmp44
    tmp46 = tl_math.log(tmp13)
    tmp47 = triton_helpers.maximum(tmp46, tmp43)
    tmp48 = tmp38 * tmp47
    tmp49 = tmp45 - tmp48
    tmp50 = tl.broadcast_to(tmp49, [XBLOCK, R0_BLOCK])
    tmp52 = tl.where(r0_mask, tmp50, 0)
    tmp53 = tl.sum(tmp52, 1)[:, None]
    tl.store(out_ptr0 + (tl.broadcast_to(r0_2, [XBLOCK, R0_BLOCK])), tmp24, r0_mask)
    tl.store(out_ptr1 + (tl.broadcast_to(r0_2, [XBLOCK, R0_BLOCK])), tmp35, r0_mask)
    tl.store(out_ptr3 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp53, None)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_binary_cross_entropy_clamp_min_mean_norm_sub_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp5 = tl.load(in_ptr0 + (1))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp10 = tl.load(in_ptr0 + (2))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp18 = tl.load(in_ptr1 + (0))
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK])
    tmp22 = tl.load(in_ptr1 + (1))
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK])
    tmp27 = tl.load(in_ptr1 + (2))
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK])
    tmp36 = tl.load(in_ptr0 + (3))
    tmp37 = tl.broadcast_to(tmp36, [XBLOCK])
    tmp40 = tl.load(in_ptr0 + (4))
    tmp41 = tl.broadcast_to(tmp40, [XBLOCK])
    tmp45 = tl.load(in_ptr0 + (5))
    tmp46 = tl.broadcast_to(tmp45, [XBLOCK])
    tmp52 = tl.load(in_ptr1 + (3))
    tmp53 = tl.broadcast_to(tmp52, [XBLOCK])
    tmp56 = tl.load(in_ptr1 + (4))
    tmp57 = tl.broadcast_to(tmp56, [XBLOCK])
    tmp61 = tl.load(in_ptr1 + (5))
    tmp62 = tl.broadcast_to(tmp61, [XBLOCK])
    tmp70 = tl.load(in_ptr0 + (6))
    tmp71 = tl.broadcast_to(tmp70, [XBLOCK])
    tmp74 = tl.load(in_ptr0 + (7))
    tmp75 = tl.broadcast_to(tmp74, [XBLOCK])
    tmp79 = tl.load(in_ptr0 + (8))
    tmp80 = tl.broadcast_to(tmp79, [XBLOCK])
    tmp86 = tl.load(in_ptr1 + (6))
    tmp87 = tl.broadcast_to(tmp86, [XBLOCK])
    tmp90 = tl.load(in_ptr1 + (7))
    tmp91 = tl.broadcast_to(tmp90, [XBLOCK])
    tmp95 = tl.load(in_ptr1 + (8))
    tmp96 = tl.broadcast_to(tmp95, [XBLOCK])
    tmp106 = tl.load(in_out_ptr0 + (0))
    tmp107 = tl.broadcast_to(tmp106, [XBLOCK])
    tmp2 = 1e-06
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp3
    tmp7 = tmp6 + tmp2
    tmp8 = tmp7 * tmp7
    tmp9 = tmp4 + tmp8
    tmp12 = tmp11 + tmp2
    tmp13 = tmp12 * tmp12
    tmp14 = tmp9 + tmp13
    tmp15 = libdevice.sqrt(tmp14)
    tmp16 = 1.0
    tmp17 = tmp15 + tmp16
    tmp20 = tmp19 + tmp2
    tmp21 = tmp20 * tmp20
    tmp24 = tmp23 + tmp2
    tmp25 = tmp24 * tmp24
    tmp26 = tmp21 + tmp25
    tmp29 = tmp28 + tmp2
    tmp30 = tmp29 * tmp29
    tmp31 = tmp26 + tmp30
    tmp32 = libdevice.sqrt(tmp31)
    tmp33 = tmp17 - tmp32
    tmp34 = 0.0
    tmp35 = triton_helpers.maximum(tmp33, tmp34)
    tmp38 = tmp37 + tmp2
    tmp39 = tmp38 * tmp38
    tmp42 = tmp41 + tmp2
    tmp43 = tmp42 * tmp42
    tmp44 = tmp39 + tmp43
    tmp47 = tmp46 + tmp2
    tmp48 = tmp47 * tmp47
    tmp49 = tmp44 + tmp48
    tmp50 = libdevice.sqrt(tmp49)
    tmp51 = tmp50 + tmp16
    tmp54 = tmp53 + tmp2
    tmp55 = tmp54 * tmp54
    tmp58 = tmp57 + tmp2
    tmp59 = tmp58 * tmp58
    tmp60 = tmp55 + tmp59
    tmp63 = tmp62 + tmp2
    tmp64 = tmp63 * tmp63
    tmp65 = tmp60 + tmp64
    tmp66 = libdevice.sqrt(tmp65)
    tmp67 = tmp51 - tmp66
    tmp68 = triton_helpers.maximum(tmp67, tmp34)
    tmp69 = tmp35 + tmp68
    tmp72 = tmp71 + tmp2
    tmp73 = tmp72 * tmp72
    tmp76 = tmp75 + tmp2
    tmp77 = tmp76 * tmp76
    tmp78 = tmp73 + tmp77
    tmp81 = tmp80 + tmp2
    tmp82 = tmp81 * tmp81
    tmp83 = tmp78 + tmp82
    tmp84 = libdevice.sqrt(tmp83)
    tmp85 = tmp84 + tmp16
    tmp88 = tmp87 + tmp2
    tmp89 = tmp88 * tmp88
    tmp92 = tmp91 + tmp2
    tmp93 = tmp92 * tmp92
    tmp94 = tmp89 + tmp93
    tmp97 = tmp96 + tmp2
    tmp98 = tmp97 * tmp97
    tmp99 = tmp94 + tmp98
    tmp100 = libdevice.sqrt(tmp99)
    tmp101 = tmp85 - tmp100
    tmp102 = triton_helpers.maximum(tmp101, tmp34)
    tmp103 = tmp69 + tmp102
    tmp104 = 3.0
    tmp105 = tmp103 / tmp104
    tmp108 = 9.0
    tmp109 = tmp107 / tmp108
    tmp110 = tmp105 + tmp109
    tl.store(in_out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp110, None)


def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf2 = empty_strided_cuda((1, ), (1, ), torch.int64)

        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf2)
        buf0 = empty_strided_cuda((1, 1, 3, 3), (9, 9, 3, 1), torch.float32)
        buf1 = empty_strided_cuda((1, 1, 3, 3), (9, 9, 3, 1), torch.float32)
        buf4 = empty_strided_cuda((), (), torch.float32)

        get_raw_stream(0)
        triton_per_fused_binary_cross_entropy_rand_like_sub_0[grid(1)](arg3_1, buf2, buf0, buf1, buf4, 64, 64, 3, 0, 1, 9, XBLOCK=1, num_warps=2, num_stages=1)
        del arg3_1
        del buf2
        buf5 = buf4; del buf4

        get_raw_stream(0)
        triton_poi_fused_add_binary_cross_entropy_clamp_min_mean_norm_sub_1[grid(1)](buf5, buf0, buf1, 1, XBLOCK=1, num_warps=1, num_stages=1)
        del buf0
        del buf1
    return (buf5, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 3
    arg1_1 = 64
    arg2_1 = 64
    arg3_1 = rand_strided((1, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
