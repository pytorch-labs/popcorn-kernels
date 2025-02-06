
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
def triton_per_fused__native_batch_norm_legit__native_batch_norm_legit_functional_abs_gt_mul_sign_sub_where_0(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, out_ptr0, out_ptr1, out_ptr3, out_ptr5, out_ptr7, out_ptr8, out_ptr10, out_ptr12, out_ptr14, out_ptr15, out_ptr16, out_ptr18, out_ptr20, out_ptr22, out_ptr23, out_ptr24, out_ptr25, out_ptr27, out_ptr29, xnumel, r0_numel):
    XBLOCK: tl.constexpr = 1
    R0_BLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    tl.full([R0_BLOCK], True, tl.int1)
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 1024*x0), None)
    tmp39 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp69 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp73 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp78 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp80 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp129 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp133 = tl.load(in_ptr10 + (x0), None, eviction_policy='evict_last')
    tmp138 = tl.load(in_ptr11 + (x0), None, eviction_policy='evict_last')
    tmp140 = tl.load(in_ptr12 + (x0), None, eviction_policy='evict_last')
    tmp158 = tl.load(in_ptr13 + (x0), None, eviction_policy='evict_last')
    tmp162 = tl.load(in_ptr14 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [R0_BLOCK])
    tmp3 = tl.broadcast_to(tmp1, [R0_BLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tl.full([1], 1024, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 / tmp7
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [R0_BLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 1024.0
    tmp15 = tmp13 / tmp14
    tmp16 = 1e-05
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tmp19 = tmp0 - tmp8
    tmp20 = tmp19 * tmp18
    tmp21 = tl.broadcast_to(tmp20, [R0_BLOCK])
    tmp23 = tl.broadcast_to(tmp21, [R0_BLOCK])
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp26 = tmp25 / tmp7
    tmp27 = tmp21 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [R0_BLOCK])
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp29, 0))
    tmp32 = tmp31 / tmp14
    tmp33 = tmp32 + tmp16
    tmp34 = libdevice.rsqrt(tmp33)
    tmp35 = 1.0009775171065494
    tmp36 = tmp32 * tmp35
    tmp37 = 0.1
    tmp38 = tmp36 * tmp37
    tmp40 = 0.9
    tmp41 = tmp39 * tmp40
    tmp42 = tmp38 + tmp41
    tmp43 = tmp26 * tmp37
    tmp45 = tmp44 * tmp40
    tmp46 = tmp43 + tmp45
    tmp47 = tmp20 - tmp26
    tmp48 = tmp47 * tmp34
    tmp50 = tmp48 * tmp49
    tmp52 = tmp50 + tmp51
    tmp53 = tl.broadcast_to(tmp52, [R0_BLOCK])
    tmp55 = tl.broadcast_to(tmp53, [R0_BLOCK])
    tmp57 = triton_helpers.promote_to_tensor(tl.sum(tmp55, 0))
    tmp58 = tmp57 / tmp7
    tmp59 = tmp53 - tmp58
    tmp60 = tmp59 * tmp59
    tmp61 = tl.broadcast_to(tmp60, [R0_BLOCK])
    tmp63 = triton_helpers.promote_to_tensor(tl.sum(tmp61, 0))
    tmp64 = tmp63 / tmp14
    tmp65 = tmp64 + tmp16
    tmp66 = libdevice.rsqrt(tmp65)
    tmp67 = tmp64 * tmp35
    tmp68 = tmp67 * tmp37
    tmp70 = tmp69 * tmp40
    tmp71 = tmp68 + tmp70
    tmp72 = tmp58 * tmp37
    tmp74 = tmp73 * tmp40
    tmp75 = tmp72 + tmp74
    tmp76 = tmp52 - tmp58
    tmp77 = tmp76 * tmp66
    tmp79 = tmp77 * tmp78
    tmp81 = tmp79 + tmp80
    tmp82 = tl_math.abs(tmp81)
    tmp83 = 0.5
    tmp84 = tmp82 > tmp83
    tmp85 = tl.full([1], 0, tl.int32)
    tmp86 = tmp85 < tmp81
    tmp87 = tmp86.to(tl.int8)
    tmp88 = tmp81 < tmp85
    tmp89 = tmp88.to(tl.int8)
    tmp90 = tmp87 - tmp89
    tmp91 = tmp90.to(tmp81.dtype)
    tmp92 = tmp91 * tmp83
    tmp93 = tmp81 - tmp92
    tmp94 = 0.0
    tmp95 = tmp81 * tmp94
    tmp96 = tl.where(tmp84, tmp93, tmp95)
    tmp97 = tl.broadcast_to(tmp96, [R0_BLOCK])
    tmp99 = tl.broadcast_to(tmp97, [R0_BLOCK])
    tmp101 = triton_helpers.promote_to_tensor(tl.sum(tmp99, 0))
    tmp102 = tmp101 / tmp7
    tmp103 = tmp97 - tmp102
    tmp104 = tmp103 * tmp103
    tmp105 = tl.broadcast_to(tmp104, [R0_BLOCK])
    tmp107 = triton_helpers.promote_to_tensor(tl.sum(tmp105, 0))
    tmp108 = tmp107 / tmp14
    tmp109 = tmp108 + tmp16
    tmp110 = libdevice.rsqrt(tmp109)
    tmp111 = tmp96 - tmp102
    tmp112 = tmp111 * tmp110
    tmp113 = tl.broadcast_to(tmp112, [R0_BLOCK])
    tmp115 = tl.broadcast_to(tmp113, [R0_BLOCK])
    tmp117 = triton_helpers.promote_to_tensor(tl.sum(tmp115, 0))
    tmp118 = tmp117 / tmp7
    tmp119 = tmp113 - tmp118
    tmp120 = tmp119 * tmp119
    tmp121 = tl.broadcast_to(tmp120, [R0_BLOCK])
    tmp123 = triton_helpers.promote_to_tensor(tl.sum(tmp121, 0))
    tmp124 = tmp123 / tmp14
    tmp125 = tmp124 + tmp16
    tmp126 = libdevice.rsqrt(tmp125)
    tmp127 = tmp124 * tmp35
    tmp128 = tmp127 * tmp37
    tmp130 = tmp129 * tmp40
    tmp131 = tmp128 + tmp130
    tmp132 = tmp118 * tmp37
    tmp134 = tmp133 * tmp40
    tmp135 = tmp132 + tmp134
    tmp136 = tmp112 - tmp118
    tmp137 = tmp136 * tmp126
    tmp139 = tmp137 * tmp138
    tmp141 = tmp139 + tmp140
    tmp142 = tl.broadcast_to(tmp141, [R0_BLOCK])
    tmp144 = tl.broadcast_to(tmp142, [R0_BLOCK])
    tmp146 = triton_helpers.promote_to_tensor(tl.sum(tmp144, 0))
    tmp147 = tmp146 / tmp7
    tmp148 = tmp142 - tmp147
    tmp149 = tmp148 * tmp148
    tmp150 = tl.broadcast_to(tmp149, [R0_BLOCK])
    tmp152 = triton_helpers.promote_to_tensor(tl.sum(tmp150, 0))
    tmp153 = tmp152 / tmp14
    tmp154 = tmp153 + tmp16
    tmp155 = libdevice.rsqrt(tmp154)
    tmp156 = tmp153 * tmp35
    tmp157 = tmp156 * tmp37
    tmp159 = tmp158 * tmp40
    tmp160 = tmp157 + tmp159
    tmp161 = tmp147 * tmp37
    tmp163 = tmp162 * tmp40
    tmp164 = tmp161 + tmp163
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp18, None)
    tl.store(out_ptr3 + (x0), tmp34, None)
    tl.store(out_ptr5 + (x0), tmp42, None)
    tl.store(out_ptr7 + (x0), tmp46, None)
    tl.store(out_ptr10 + (x0), tmp66, None)
    tl.store(out_ptr12 + (x0), tmp71, None)
    tl.store(out_ptr14 + (x0), tmp75, None)
    tl.debug_barrier()
    tl.store(in_out_ptr2 + (x0), tmp110, None)
    tl.store(out_ptr18 + (x0), tmp126, None)
    tl.store(out_ptr20 + (x0), tmp131, None)
    tl.store(out_ptr22 + (x0), tmp135, None)
    tl.store(in_out_ptr1 + (r0_1 + 1024*x0), tmp141, None)
    tl.store(out_ptr25 + (x0), tmp155, None)
    tl.store(out_ptr27 + (x0), tmp160, None)
    tl.store(out_ptr29 + (x0), tmp164, None)
    tl.store(out_ptr0 + (x0), tmp8, None)
    tl.store(out_ptr1 + (x0), tmp26, None)
    tl.store(out_ptr8 + (x0), tmp58, None)
    tl.store(out_ptr15 + (x0), tmp102, None)
    tl.store(out_ptr16 + (x0), tmp118, None)
    tl.store(out_ptr23 + (x0), tmp147, None)
    tl.store(out_ptr24 + (x0), tmp152, None)


import triton

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_abs_add_div_eq_gt_masked_fill_mul_norm_randn_like_scalar_tensor_sign_sub_where_1(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, load_seed_offset, load_seed_offset1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 320
    R0_BLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_1 = r0_index
    x0 = xindex
    x3 = xindex // 32
    tmp5 = tl.load(in_out_ptr0 + (r0_1 + 32*x0), xmask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = r0_1 + 32*x0
    tmp2 = tl.randn(tmp0, (tmp1).to(tl.uint32))
    tmp3 = tl.load(in_ptr0 + load_seed_offset1)
    tmp4 = tl.randn(tmp3, (tmp1).to(tl.uint32))
    tmp7 = tmp5 - tmp6
    tmp9 = 1024.0
    tmp10 = tmp8 / tmp9
    tmp11 = 1e-05
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.rsqrt(tmp12)
    tmp14 = tmp7 * tmp13
    tmp16 = tmp14 * tmp15
    tmp18 = tmp16 + tmp17
    tmp19 = tl_math.abs(tmp18)
    tmp20 = 0.5
    tmp21 = tmp19 > tmp20
    tmp22 = tl.full([1, 1], 0, tl.int32)
    tmp23 = tmp22 < tmp18
    tmp24 = tmp23.to(tl.int8)
    tmp25 = tmp18 < tmp22
    tmp26 = tmp25.to(tl.int8)
    tmp27 = tmp24 - tmp26
    tmp28 = tmp27.to(tmp18.dtype)
    tmp29 = tmp28 * tmp20
    tmp30 = tmp18 - tmp29
    tmp31 = 0.0
    tmp32 = tmp18 * tmp31
    tmp33 = tl.where(tmp21, tmp30, tmp32)
    tmp34 = 0.1
    tmp35 = tmp4 * tmp34
    tmp36 = tmp33 + tmp35
    tmp37 = tmp33 - tmp36
    tmp38 = 1e-06
    tmp39 = tmp37 + tmp38
    tmp40 = tmp39 * tmp39
    tmp41 = tl.broadcast_to(tmp40, [XBLOCK, R0_BLOCK])
    tmp43 = tl.where(xmask, tmp41, 0)
    tmp44 = tl.sum(tmp43, 1)[:, None]
    tmp45 = 0.2
    tmp46 = tmp2 * tmp45
    tmp47 = tmp33 + tmp46
    tmp48 = tmp33 - tmp47
    tmp49 = tmp48 + tmp38
    tmp50 = tmp49 * tmp49
    tmp51 = tl.broadcast_to(tmp50, [XBLOCK, R0_BLOCK])
    tmp53 = tl.where(xmask, tmp51, 0)
    tmp54 = tl.sum(tmp53, 1)[:, None]
    tmp55 = libdevice.sqrt(tmp54)
    tmp56 = tmp55 == tmp31
    tmp57 = tmp49 / tmp55
    tmp58 = tl.where(tmp56, tmp31, tmp57)
    tmp59 = libdevice.sqrt(tmp44)
    tmp60 = tmp59 == tmp31
    tmp61 = tmp39 / tmp59
    tmp62 = tl.where(tmp60, tmp31, tmp61)
    tl.store(out_ptr0 + (r0_1 + 32*x0), tmp21, xmask)
    tl.store(in_out_ptr1 + (r0_1 + 32*x0), tmp58, xmask)
    tl.store(in_out_ptr2 + (r0_1 + 32*x0), tmp62, xmask)
    tl.store(out_ptr1 + (x0), tmp44, xmask)
    tl.store(out_ptr2 + (x0), tmp54, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_clamp_min_ge_mean_norm_sub_2(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel):
    XBLOCK: tl.constexpr = 1
    r0_numel = 320
    R0_BLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    tl.full([1], xoffset, tl.int32)
    tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_mask = r0_index < r0_numel
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, other=0.0)
    tmp4 = tl.load(in_ptr1 + (r0_0), r0_mask, other=0.0)
    tmp1 = libdevice.sqrt(tmp0)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp5 = libdevice.sqrt(tmp4)
    tmp6 = tmp3 - tmp5
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tl.broadcast_to(tmp8, [R0_BLOCK])
    tmp11 = tl.where(r0_mask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = tmp6 >= tmp7
    tmp14 = 320.0
    tmp15 = tmp12 / tmp14
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [R0_BLOCK])), tmp13, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([1], 0, tl.int32)), tmp15, None)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_3(in_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp1 + tmp2
    tl.store(out_ptr1 + (tl.full([XBLOCK], 0, tl.int32)), tmp3, None)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21 = args
    args.clear()
    assert_size_stride(primals_1, (1, 10, 32, 32), (10240, 1024, 32, 1))
    assert_size_stride(primals_2, (), ())
    assert_size_stride(primals_3, (10, ), (1, ))
    assert_size_stride(primals_4, (10, ), (1, ))
    assert_size_stride(primals_5, (10, ), (1, ))
    assert_size_stride(primals_6, (10, ), (1, ))
    assert_size_stride(primals_7, (), ())
    assert_size_stride(primals_8, (10, ), (1, ))
    assert_size_stride(primals_9, (10, ), (1, ))
    assert_size_stride(primals_10, (10, ), (1, ))
    assert_size_stride(primals_11, (10, ), (1, ))
    assert_size_stride(primals_12, (), ())
    assert_size_stride(primals_13, (10, ), (1, ))
    assert_size_stride(primals_14, (10, ), (1, ))
    assert_size_stride(primals_15, (10, ), (1, ))
    assert_size_stride(primals_16, (10, ), (1, ))
    assert_size_stride(primals_17, (), ())
    assert_size_stride(primals_18, (10, ), (1, ))
    assert_size_stride(primals_19, (10, ), (1, ))
    assert_size_stride(primals_20, (10, ), (1, ))
    assert_size_stride(primals_21, (10, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((1, 10, 1, 1), (10, 1, 10, 10), torch.float32)
        buf3 = reinterpret_tensor(buf1, (1, 10, 1, 1), (10, 1, 1, 1), 0); del buf1
        buf0 = empty_strided_cuda((1, 10, 1, 1), (10, 1, 1, 1), torch.float32)
        buf4 = empty_strided_cuda((1, 10, 1, 1), (10, 1, 1, 1), torch.float32)
        buf7 = empty_strided_cuda((1, 10, 1, 1), (10, 1, 1, 1), torch.float32)
        buf8 = empty_strided_cuda((1, 10, 32, 32), (10240, 1024, 32, 1), torch.float32)
        buf9 = empty_strided_cuda((1, 10, 1, 1, 1), (10, 1, 1, 1, 1), torch.float32)
        buf12 = empty_strided_cuda((1, 10, 1, 1, 1), (10, 1, 1, 1, 1), torch.float32)
        buf13 = reinterpret_tensor(buf8, (1, 10, 1, 32, 32), (10240, 1024, 10240, 32, 1), 0); del buf8
        buf15 = empty_strided_cuda((1, 10, 1, 1), (10, 1, 10, 10), torch.float32)
        buf17 = reinterpret_tensor(buf15, (1, 10, 1, 1), (10, 1, 1, 1), 0); del buf15
        buf14 = empty_strided_cuda((1, 10, 1, 1), (10, 1, 1, 1), torch.float32)
        buf18 = empty_strided_cuda((1, 10, 1, 1), (10, 1, 1, 1), torch.float32)
        buf21 = empty_strided_cuda((1, 10, 1, 1), (10, 1, 1, 1), torch.float32)
        buf22 = reinterpret_tensor(buf13, (1, 10, 32, 32), (10240, 1024, 32, 1), 0); del buf13
        buf23 = empty_strided_cuda((1, 10, 1, 1, 1), (10, 1, 10, 10, 10), torch.float32)
        buf24 = empty_strided_cuda((1, 10, 1, 1, 1), (10, 1, 10, 10, 10), torch.float32)
        buf26 = empty_strided_cuda((1, 10, 1, 1, 1), (10, 1, 10, 10, 10), torch.float32)

        get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit__native_batch_norm_legit_functional_abs_gt_mul_sign_sub_where_0[grid(10)](buf3, buf22, buf17, primals_1, primals_4, primals_3, primals_5, primals_6, primals_9, primals_8, primals_10, primals_11, primals_14, primals_13, primals_15, primals_16, primals_19, primals_18, buf0, buf4, buf7, primals_4, primals_3, buf9, buf12, primals_9, primals_8, buf14, buf18, buf21, primals_14, primals_13, buf23, buf24, buf26, primals_19, primals_18, 10, 1024, num_warps=8, num_stages=1)
        del primals_13
        del primals_14
        del primals_18
        del primals_19
        del primals_3
        del primals_4
        del primals_8
        del primals_9
        buf29 = empty_strided_cuda((2, ), (1, ), torch.int64)

        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [2], out=buf29)
        buf31 = empty_strided_cuda((1, 10, 32, 32), (10240, 1024, 32, 1), torch.float32)
        buf30 = empty_strided_cuda((1, 10, 32, 32), (10240, 1024, 32, 1), torch.float32)
        buf27 = reinterpret_tensor(buf22, (1, 10, 1, 32, 32), (10240, 1024, 10240, 32, 1), 0); del buf22
        buf28 = empty_strided_cuda((1, 10, 32, 32), (10240, 1024, 32, 1), torch.bool)
        buf32 = empty_strided_cuda((1, 10, 32), (320, 32, 1), torch.float32)
        buf33 = empty_strided_cuda((1, 10, 32), (320, 32, 1), torch.float32)
        buf36 = buf31; del buf31
        buf37 = buf30; del buf30

        get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_abs_add_div_eq_gt_masked_fill_mul_norm_randn_like_scalar_tensor_sign_sub_where_1[grid(320)](buf27, buf36, buf37, buf29, buf23, buf24, primals_20, primals_21, buf28, buf32, buf33, 1, 0, 320, 32, XBLOCK=1, num_warps=2, num_stages=1)
        del buf24
        del buf27
        del buf29
        del primals_21
        buf34 = empty_strided_cuda((), (), torch.float32)
        buf35 = empty_strided_cuda((1, 10, 32), (320, 32, 1), torch.bool)
        buf70 = buf34; del buf34

        get_raw_stream(0)
        triton_per_fused_add_clamp_min_ge_mean_norm_sub_2[grid(1)](buf70, buf32, buf33, buf35, 1, 320, num_warps=4, num_stages=1)
        del buf32
        del buf33

        get_raw_stream(0)
        triton_poi_fused_add_3[grid(1)](primals_2, primals_2, 1, XBLOCK=1, num_warps=1, num_stages=1)
        del primals_2

        get_raw_stream(0)
        triton_poi_fused_add_3[grid(1)](primals_7, primals_7, 1, XBLOCK=1, num_warps=1, num_stages=1)
        del primals_7

        get_raw_stream(0)
        triton_poi_fused_add_3[grid(1)](primals_12, primals_12, 1, XBLOCK=1, num_warps=1, num_stages=1)
        del primals_12

        get_raw_stream(0)
        triton_poi_fused_add_3[grid(1)](primals_17, primals_17, 1, XBLOCK=1, num_warps=1, num_stages=1)
        del primals_17
    return (buf70, primals_1, primals_5, primals_6, primals_10, primals_11, primals_15, primals_16, primals_20, buf0, buf3, buf4, buf7, buf9, buf12, buf14, buf17, buf18, buf21, reinterpret_tensor(buf26, (10, ), (1, ), 0), buf28, buf35, buf36, buf37, reinterpret_tensor(buf23, (1, 10, 1, 1, 1), (10, 1, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 10, 32, 32), (10240, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_3 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_8 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_13 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_18 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
