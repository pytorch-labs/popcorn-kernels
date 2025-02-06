
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
def triton_poi_fused_avg_pool2d_0(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (ks3*(tl.where((-1) + ks2 + ((-1)*tl_math.abs(1 + ((-1)*ks2) + tl_math.abs((-2) + 2*((x1 % (2 + (ks2 // 2)))) + ((((((x1 // (2 + (ks2 // 2))) % (4*ks1))) // 2) % 2))))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks2) + tl_math.abs((-2) + 2*((x1 % (2 + (ks2 // 2)))) + ((((((x1 // (2 + (ks2 // 2))) % (4*ks1))) // 2) % 2))))) + 2*ks2, (-1) + ks2 + ((-1)*tl_math.abs(1 + ((-1)*ks2) + tl_math.abs((-2) + 2*((x1 % (2 + (ks2 // 2)))) + ((((((x1 // (2 + (ks2 // 2))) % (4*ks1))) // 2) % 2))))))) + ks2*ks3*((((((x1 // (2 + (ks2 // 2))) % (4*ks1))) // 4) % ks1)) + (tl.where((-1) + ks3 + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs((-2) + 4*x0 + (((((x1 // (2 + (ks2 // 2))) % (4*ks1))) % 2))))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs((-2) + 4*x0 + (((((x1 // (2 + (ks2 // 2))) % (4*ks1))) % 2))))) + 2*ks3, (-1) + ks3 + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs((-2) + 4*x0 + (((((x1 // (2 + (ks2 // 2))) % (4*ks1))) % 2)))))))), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (ks3*(tl.where((-1) + ks2 + ((-1)*tl_math.abs(1 + ((-1)*ks2) + tl_math.abs((-2) + 2*((x1 % (2 + (ks2 // 2)))) + ((((((x1 // (2 + (ks2 // 2))) % (4*ks1))) // 2) % 2))))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks2) + tl_math.abs((-2) + 2*((x1 % (2 + (ks2 // 2)))) + ((((((x1 // (2 + (ks2 // 2))) % (4*ks1))) // 2) % 2))))) + 2*ks2, (-1) + ks2 + ((-1)*tl_math.abs(1 + ((-1)*ks2) + tl_math.abs((-2) + 2*((x1 % (2 + (ks2 // 2)))) + ((((((x1 // (2 + (ks2 // 2))) % (4*ks1))) // 2) % 2))))))) + ks2*ks3*((((((x1 // (2 + (ks2 // 2))) % (4*ks1))) // 4) % ks1)) + (tl.where((-1) + ks3 + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs(4*x0 + (((((x1 // (2 + (ks2 // 2))) % (4*ks1))) % 2))))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs(4*x0 + (((((x1 // (2 + (ks2 // 2))) % (4*ks1))) % 2))))) + 2*ks3, (-1) + ks3 + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs(4*x0 + (((((x1 // (2 + (ks2 // 2))) % (4*ks1))) % 2)))))))), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (ks3*(tl.where((-1) + ks2 + ((-1)*tl_math.abs(1 + ((-1)*ks2) + tl_math.abs((-2) + 2*((x1 % (2 + (ks2 // 2)))) + ((((((x1 // (2 + (ks2 // 2))) % (4*ks1))) // 2) % 2))))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks2) + tl_math.abs((-2) + 2*((x1 % (2 + (ks2 // 2)))) + ((((((x1 // (2 + (ks2 // 2))) % (4*ks1))) // 2) % 2))))) + 2*ks2, (-1) + ks2 + ((-1)*tl_math.abs(1 + ((-1)*ks2) + tl_math.abs((-2) + 2*((x1 % (2 + (ks2 // 2)))) + ((((((x1 // (2 + (ks2 // 2))) % (4*ks1))) // 2) % 2))))))) + ks2*ks3*((((((x1 // (2 + (ks2 // 2))) % (4*ks1))) // 4) % ks1)) + (tl.where((-1) + ks3 + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs(2 + 4*x0 + (((((x1 // (2 + (ks2 // 2))) % (4*ks1))) % 2))))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs(2 + 4*x0 + (((((x1 // (2 + (ks2 // 2))) % (4*ks1))) % 2))))) + 2*ks3, (-1) + ks3 + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs(2 + 4*x0 + (((((x1 // (2 + (ks2 // 2))) % (4*ks1))) % 2)))))))), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 * tmp0
    tmp3 = tmp2 * tmp2
    tmp4 = tmp3 + tmp1
    tmp6 = tmp5 * tmp5
    tmp7 = tmp6 + tmp4
    tmp8 = 0.3333333333333333
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x2), tmp9, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__softmax_neg_softplus_1(in_ptr0, out_ptr0, out_ptr1, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    _tmp24 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 2*ks0*r0_1 + ks0*r0_1*(ks1 // 2)), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.full([1, 1], 0, tl.int32)
        tmp2 = tmp1 < tmp0
        tmp3 = tmp2.to(tl.int8)
        tmp4 = tmp0 < tmp1
        tmp5 = tmp4.to(tl.int8)
        tmp6 = tmp3 - tmp5
        tmp7 = tmp6.to(tmp0.dtype)
        tmp8 = tl_math.abs(tmp0)
        tmp9 = triton_helpers.maximum(tmp1, tmp8)
        tmp10 = tmp7 * tmp9
        tmp11 = 3.0
        tmp12 = tmp10 * tmp11
        tmp13 = libdevice.sqrt(tmp12)
        tmp14 = 1.0
        tmp15 = tmp13 * tmp14
        tmp16 = 20.0
        tmp17 = tmp15 > tmp16
        tmp18 = tl_math.exp(tmp15)
        tmp19 = libdevice.log1p(tmp18)
        tmp20 = tmp19 * tmp14
        tmp21 = tl.where(tmp17, tmp13, tmp20)
        tmp22 = -tmp21
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, R0_BLOCK])
        tmp25 = triton_helpers.maximum(_tmp24, tmp23)
        _tmp24 = tl.where(r0_mask & xmask, tmp25, _tmp24)
    tmp24 = triton_helpers.max2(_tmp24, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp24, xmask)
    _tmp52 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp26 = tl.load(in_ptr0 + (x0 + 2*ks0*r0_1 + ks0*r0_1*(ks1 // 2)), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp27 = tl.full([1, 1], 0, tl.int32)
        tmp28 = tmp27 < tmp26
        tmp29 = tmp28.to(tl.int8)
        tmp30 = tmp26 < tmp27
        tmp31 = tmp30.to(tl.int8)
        tmp32 = tmp29 - tmp31
        tmp33 = tmp32.to(tmp26.dtype)
        tmp34 = tl_math.abs(tmp26)
        tmp35 = triton_helpers.maximum(tmp27, tmp34)
        tmp36 = tmp33 * tmp35
        tmp37 = 3.0
        tmp38 = tmp36 * tmp37
        tmp39 = libdevice.sqrt(tmp38)
        tmp40 = 1.0
        tmp41 = tmp39 * tmp40
        tmp42 = 20.0
        tmp43 = tmp41 > tmp42
        tmp44 = tl_math.exp(tmp41)
        tmp45 = libdevice.log1p(tmp44)
        tmp46 = tmp45 * tmp40
        tmp47 = tl.where(tmp43, tmp39, tmp46)
        tmp48 = -tmp47
        tmp49 = tmp48 - tmp24
        tmp50 = tl_math.exp(tmp49)
        tmp51 = tl.broadcast_to(tmp50, [XBLOCK, R0_BLOCK])
        tmp53 = _tmp52 + tmp51
        _tmp52 = tl.where(r0_mask & xmask, tmp53, _tmp52)
    tmp52 = tl.sum(_tmp52, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp52, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__softmax_neg_softplus_2(in_out_ptr0, in_ptr0, in_ptr1, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % ks0)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp1 < tmp0
    tmp3 = tmp2.to(tl.int8)
    tmp4 = tmp0 < tmp1
    tmp5 = tmp4.to(tl.int8)
    tmp6 = tmp3 - tmp5
    tmp7 = tmp6.to(tmp0.dtype)
    tmp8 = tl_math.abs(tmp0)
    tmp9 = triton_helpers.maximum(tmp1, tmp8)
    tmp10 = tmp7 * tmp9
    tmp11 = 3.0
    tmp12 = tmp10 * tmp11
    tmp13 = libdevice.sqrt(tmp12)
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = 20.0
    tmp17 = tmp15 > tmp16
    tmp18 = tl_math.exp(tmp15)
    tmp19 = libdevice.log1p(tmp18)
    tmp20 = tmp19 * tmp14
    tmp21 = tl.where(tmp17, tmp13, tmp20)
    tmp22 = -tmp21
    tmp24 = tmp22 - tmp23
    tmp25 = tl_math.exp(tmp24)
    tmp27 = tmp25 / tmp26
    tl.store(in_out_ptr0 + (x2), tmp27, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_unpool3d_3(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_unpool3d_4(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + ((triton_helpers.div_floor_integer(x0,  (ks0 // 4)*(triton_helpers.div_floor_integer(1 + (ks1 // 2),  4)) + (triton_helpers.div_floor_integer(1 + (ks1 // 2),  4))))*(triton_helpers.div_floor_integer(1 + (ks1 // 2),  4)) + (ks0 // 4)*(triton_helpers.div_floor_integer(x0,  (ks0 // 4)*(triton_helpers.div_floor_integer(1 + (ks1 // 2),  4)) + (triton_helpers.div_floor_integer(1 + (ks1 // 2),  4))))*(triton_helpers.div_floor_integer(1 + (ks1 // 2),  4)) + ((x0 % ((ks0 // 4)*(triton_helpers.div_floor_integer(1 + (ks1 // 2),  4)) + (triton_helpers.div_floor_integer(1 + (ks1 // 2),  4)))))), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 + tmp1
    tmp3 = 8*(triton_helpers.div_floor_integer(ks2*(triton_helpers.div_floor_integer(4 + ks0,  2 + (ks0 // 2)))*(triton_helpers.div_floor_integer(4 + ks1,  2 + (ks1 // 2))),  2)) + 8*(ks0 // 4)*(triton_helpers.div_floor_integer(ks2*(triton_helpers.div_floor_integer(4 + ks0,  2 + (ks0 // 2)))*(triton_helpers.div_floor_integer(4 + ks1,  2 + (ks1 // 2))),  2)) + 8*(triton_helpers.div_floor_integer(ks2*(triton_helpers.div_floor_integer(4 + ks0,  2 + (ks0 // 2)))*(triton_helpers.div_floor_integer(4 + ks1,  2 + (ks1 // 2))),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks1 // 2),  2)),  2)) + 8*(ks0 // 4)*(triton_helpers.div_floor_integer(ks2*(triton_helpers.div_floor_integer(4 + ks0,  2 + (ks0 // 2)))*(triton_helpers.div_floor_integer(4 + ks1,  2 + (ks1 // 2))),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks1 // 2),  2)),  2))
    tmp4 = tmp2 + tmp3
    tmp5 = tmp2 < 0
    tmp6 = tl.where(tmp5, tmp4, tmp2)
    tl.device_assert(((0 <= tmp6) & (tmp6 < 8*(triton_helpers.div_floor_integer(ks2*(triton_helpers.div_floor_integer(4 + ks0,  2 + (ks0 // 2)))*(triton_helpers.div_floor_integer(4 + ks1,  2 + (ks1 // 2))),  2)) + 8*(ks0 // 4)*(triton_helpers.div_floor_integer(ks2*(triton_helpers.div_floor_integer(4 + ks0,  2 + (ks0 // 2)))*(triton_helpers.div_floor_integer(4 + ks1,  2 + (ks1 // 2))),  2)) + 8*(triton_helpers.div_floor_integer(ks2*(triton_helpers.div_floor_integer(4 + ks0,  2 + (ks0 // 2)))*(triton_helpers.div_floor_integer(4 + ks1,  2 + (ks1 // 2))),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks1 // 2),  2)),  2)) + 8*(ks0 // 4)*(triton_helpers.div_floor_integer(ks2*(triton_helpers.div_floor_integer(4 + ks0,  2 + (ks0 // 2)))*(triton_helpers.div_floor_integer(4 + ks1,  2 + (ks1 // 2))),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks1 // 2),  2)),  2)))) | ~(xmask), "index out of bounds: 0 <= tmp6 < 8*(triton_helpers.div_floor_integer(ks2*(triton_helpers.div_floor_integer(4 + ks0,  2 + (ks0 // 2)))*(triton_helpers.div_floor_integer(4 + ks1,  2 + (ks1 // 2))),  2)) + 8*(ks0 // 4)*(triton_helpers.div_floor_integer(ks2*(triton_helpers.div_floor_integer(4 + ks0,  2 + (ks0 // 2)))*(triton_helpers.div_floor_integer(4 + ks1,  2 + (ks1 // 2))),  2)) + 8*(triton_helpers.div_floor_integer(ks2*(triton_helpers.div_floor_integer(4 + ks0,  2 + (ks0 // 2)))*(triton_helpers.div_floor_integer(4 + ks1,  2 + (ks1 // 2))),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks1 // 2),  2)),  2)) + 8*(ks0 // 4)*(triton_helpers.div_floor_integer(ks2*(triton_helpers.div_floor_integer(4 + ks0,  2 + (ks0 // 2)))*(triton_helpers.div_floor_integer(4 + ks1,  2 + (ks1 // 2))),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks1 // 2),  2)),  2))")
    tl.store(out_ptr0 + (tl.broadcast_to(2*(((tmp6 // (2 + 2*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks1 // 2),  2)),  2)))) % (2 + 2*(ks0 // 4)))) + 4*(((tmp6 // (4 + 4*(ks0 // 4) + 4*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks1 // 2),  2)),  2)) + 4*(ks0 // 4)*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks1 // 2),  2)),  2)))) % (2*(triton_helpers.div_floor_integer(ks2*(triton_helpers.div_floor_integer(4 + ks0,  2 + (ks0 // 2)))*(triton_helpers.div_floor_integer(4 + ks1,  2 + (ks1 // 2))),  2))))) + 2*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks1 // 2),  2)),  2))*(((tmp6 // (2 + 2*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks1 // 2),  2)),  2)))) % (2 + 2*(ks0 // 4)))) + 4*(ks0 // 4)*(((tmp6 // (4 + 4*(ks0 // 4) + 4*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks1 // 2),  2)),  2)) + 4*(ks0 // 4)*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks1 // 2),  2)),  2)))) % (2*(triton_helpers.div_floor_integer(ks2*(triton_helpers.div_floor_integer(4 + ks0,  2 + (ks0 // 2)))*(triton_helpers.div_floor_integer(4 + ks1,  2 + (ks1 // 2))),  2))))) + 4*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks1 // 2),  2)),  2))*(((tmp6 // (4 + 4*(ks0 // 4) + 4*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks1 // 2),  2)),  2)) + 4*(ks0 // 4)*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks1 // 2),  2)),  2)))) % (2*(triton_helpers.div_floor_integer(ks2*(triton_helpers.div_floor_integer(4 + ks0,  2 + (ks0 // 2)))*(triton_helpers.div_floor_integer(4 + ks1,  2 + (ks1 // 2))),  2))))) + 4*(ks0 // 4)*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks1 // 2),  2)),  2))*(((tmp6 // (4 + 4*(ks0 // 4) + 4*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks1 // 2),  2)),  2)) + 4*(ks0 // 4)*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks1 // 2),  2)),  2)))) % (2*(triton_helpers.div_floor_integer(ks2*(triton_helpers.div_floor_integer(4 + ks0,  2 + (ks0 // 2)))*(triton_helpers.div_floor_integer(4 + ks1,  2 + (ks1 // 2))),  2))))) + ((tmp6 % (2 + 2*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks1 // 2),  2)),  2))))), [XBLOCK])), tmp8, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_relu_5(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks1)
    x2 = xindex // ks2
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 2*((((x0 + 2*x1 + 4*x2 + 2*x1*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks5 // 2),  2)),  2)) + 4*x2*(ks4 // 4) + 4*x2*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks5 // 2),  2)),  2)) + 4*x2*(ks4 // 4)*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks5 // 2),  2)),  2))) // (2 + 2*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks5 // 2),  2)),  2)))) % (2 + 2*(ks4 // 4)))) + 4*((((x0 + 2*x1 + 4*x2 + 2*x1*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks5 // 2),  2)),  2)) + 4*x2*(ks4 // 4) + 4*x2*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks5 // 2),  2)),  2)) + 4*x2*(ks4 // 4)*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks5 // 2),  2)),  2))) // (4 + 4*(ks4 // 4) + 4*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks5 // 2),  2)),  2)) + 4*(ks4 // 4)*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks5 // 2),  2)),  2)))) % (2*(triton_helpers.div_floor_integer(ks3*(triton_helpers.div_floor_integer(4 + ks4,  2 + (ks4 // 2)))*(triton_helpers.div_floor_integer(4 + ks5,  2 + (ks5 // 2))),  2))))) + 2*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks5 // 2),  2)),  2))*((((x0 + 2*x1 + 4*x2 + 2*x1*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks5 // 2),  2)),  2)) + 4*x2*(ks4 // 4) + 4*x2*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks5 // 2),  2)),  2)) + 4*x2*(ks4 // 4)*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks5 // 2),  2)),  2))) // (2 + 2*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks5 // 2),  2)),  2)))) % (2 + 2*(ks4 // 4)))) + 4*(ks4 // 4)*((((x0 + 2*x1 + 4*x2 + 2*x1*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks5 // 2),  2)),  2)) + 4*x2*(ks4 // 4) + 4*x2*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks5 // 2),  2)),  2)) + 4*x2*(ks4 // 4)*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks5 // 2),  2)),  2))) // (4 + 4*(ks4 // 4) + 4*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks5 // 2),  2)),  2)) + 4*(ks4 // 4)*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks5 // 2),  2)),  2)))) % (2*(triton_helpers.div_floor_integer(ks3*(triton_helpers.div_floor_integer(4 + ks4,  2 + (ks4 // 2)))*(triton_helpers.div_floor_integer(4 + ks5,  2 + (ks5 // 2))),  2))))) + 4*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks5 // 2),  2)),  2))*((((x0 + 2*x1 + 4*x2 + 2*x1*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks5 // 2),  2)),  2)) + 4*x2*(ks4 // 4) + 4*x2*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks5 // 2),  2)),  2)) + 4*x2*(ks4 // 4)*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks5 // 2),  2)),  2))) // (4 + 4*(ks4 // 4) + 4*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks5 // 2),  2)),  2)) + 4*(ks4 // 4)*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks5 // 2),  2)),  2)))) % (2*(triton_helpers.div_floor_integer(ks3*(triton_helpers.div_floor_integer(4 + ks4,  2 + (ks4 // 2)))*(triton_helpers.div_floor_integer(4 + ks5,  2 + (ks5 // 2))),  2))))) + 4*(ks4 // 4)*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks5 // 2),  2)),  2))*((((x0 + 2*x1 + 4*x2 + 2*x1*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks5 // 2),  2)),  2)) + 4*x2*(ks4 // 4) + 4*x2*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks5 // 2),  2)),  2)) + 4*x2*(ks4 // 4)*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks5 // 2),  2)),  2))) // (4 + 4*(ks4 // 4) + 4*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks5 // 2),  2)),  2)) + 4*(ks4 // 4)*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks5 // 2),  2)),  2)))) % (2*(triton_helpers.div_floor_integer(ks3*(triton_helpers.div_floor_integer(4 + ks4,  2 + (ks4 // 2)))*(triton_helpers.div_floor_integer(4 + ks5,  2 + (ks5 // 2))),  2)))))), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(out_ptr0 + (x3), tmp2, xmask)


def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        (1 + (s2 // 2)) // 2
        buf0 = empty_strided_cuda((1, 2*s0*((4 + s1) // (2 + (s1 // 2)))*((4 + s2) // (2 + (s2 // 2))) + s0*(s1 // 2)*((4 + s1) // (2 + (s1 // 2)))*((4 + s2) // (2 + (s2 // 2))), 1, (1 + (s2 // 2)) // 2), (2*s0*((1 + (s2 // 2)) // 2)*((4 + s1) // (2 + (s1 // 2)))*((4 + s2) // (2 + (s2 // 2))) + s0*(s1 // 2)*((1 + (s2 // 2)) // 2)*((4 + s1) // (2 + (s1 // 2)))*((4 + s2) // (2 + (s2 // 2))), (1 + (s2 // 2)) // 2, (1 + (s2 // 2)) // 2, 1), torch.float32)

        triton_poi_fused_avg_pool2d_0_xnumel = 2*s0*((1 + (s2 // 2)) // 2)*((4 + s1) // (2 + (s1 // 2)))*((4 + s2) // (2 + (s2 // 2))) + s0*(s1 // 2)*((1 + (s2 // 2)) // 2)*((4 + s1) // (2 + (s1 // 2)))*((4 + s2) // (2 + (s2 // 2)))
        get_raw_stream(0)
        triton_poi_fused_avg_pool2d_0[grid(triton_poi_fused_avg_pool2d_0_xnumel)](arg3_1, buf0, 16, 3, 64, 64, 6528, XBLOCK=256, num_warps=4, num_stages=1)
        del arg3_1
        buf1 = empty_strided_cuda((1, 1, 2 + (s1 // 2), (1 + (s2 // 2)) // 2), (2*((1 + (s2 // 2)) // 2) + (s1 // 2)*((1 + (s2 // 2)) // 2), 2*((1 + (s2 // 2)) // 2) + (s1 // 2)*((1 + (s2 // 2)) // 2), (1 + (s2 // 2)) // 2, 1), torch.float32)
        buf2 = empty_strided_cuda((1, 1, 2 + (s1 // 2), (1 + (s2 // 2)) // 2), (2*((1 + (s2 // 2)) // 2) + (s1 // 2)*((1 + (s2 // 2)) // 2), 2*((1 + (s2 // 2)) // 2) + (s1 // 2)*((1 + (s2 // 2)) // 2), (1 + (s2 // 2)) // 2, 1), torch.float32)

        triton_red_fused__softmax_neg_softplus_1_xnumel = 2*((1 + (s2 // 2)) // 2) + (s1 // 2)*((1 + (s2 // 2)) // 2)
        s0*((4 + s1) // (2 + (s1 // 2)))*((4 + s2) // (2 + (s2 // 2)))
        get_raw_stream(0)
        triton_red_fused__softmax_neg_softplus_1[grid(triton_red_fused__softmax_neg_softplus_1_xnumel)](buf0, buf1, buf2, 16, 64, 544, 12, XBLOCK=1, R0_BLOCK=16, num_warps=2, num_stages=1)
        2*((1 + (s2 // 2)) // 2) + (s1 // 2)*((1 + (s2 // 2)) // 2)
        buf3 = reinterpret_tensor(buf0, (1, s0*((4 + s1) // (2 + (s1 // 2)))*((4 + s2) // (2 + (s2 // 2))), 2 + (s1 // 2), (1 + (s2 // 2)) // 2), (2*s0*((1 + (s2 // 2)) // 2)*((4 + s1) // (2 + (s1 // 2)))*((4 + s2) // (2 + (s2 // 2))) + s0*(s1 // 2)*((1 + (s2 // 2)) // 2)*((4 + s1) // (2 + (s1 // 2)))*((4 + s2) // (2 + (s2 // 2))), 2*((1 + (s2 // 2)) // 2) + (s1 // 2)*((1 + (s2 // 2)) // 2), (1 + (s2 // 2)) // 2, 1), 0); del buf0

        triton_poi_fused__softmax_neg_softplus_2_xnumel = 2*s0*((1 + (s2 // 2)) // 2)*((4 + s1) // (2 + (s1 // 2)))*((4 + s2) // (2 + (s2 // 2))) + s0*(s1 // 2)*((1 + (s2 // 2)) // 2)*((4 + s1) // (2 + (s1 // 2)))*((4 + s2) // (2 + (s2 // 2)))
        get_raw_stream(0)
        triton_poi_fused__softmax_neg_softplus_2[grid(triton_poi_fused__softmax_neg_softplus_2_xnumel)](buf3, buf1, buf2, 544, 6528, XBLOCK=256, num_warps=4, num_stages=1)
        del buf1
        del buf2

        buf4 = torch.ops.aten.max_pool3d_with_indices.default(reinterpret_tensor(buf3, (1, 1, s0*((4 + s1) // (2 + (s1 // 2)))*((4 + s2) // (2 + (s2 // 2))), 2 + (s1 // 2), (1 + (s2 // 2)) // 2), (0, 0, 2*((1 + (s2 // 2)) // 2) + (s1 // 2)*((1 + (s2 // 2)) // 2), (1 + (s2 // 2)) // 2, 1), 0), [2, 2, 2], [2, 2, 2])
        del buf3
        buf5 = buf4[0]
        buf6 = buf4[1]
        del buf4
        buf7 = empty_strided_cuda((1, 1, 2*((s0*((4 + s1) // (2 + (s1 // 2)))*((4 + s2) // (2 + (s2 // 2)))) // 2), 2 + 2*(s1 // 4), 2 + 2*(((-1) + (((-1) + (s2 // 2)) // 2)) // 2)), (8*((s0*((4 + s1) // (2 + (s1 // 2)))*((4 + s2) // (2 + (s2 // 2)))) // 2) + 8*(s1 // 4)*((s0*((4 + s1) // (2 + (s1 // 2)))*((4 + s2) // (2 + (s2 // 2)))) // 2) + 8*((s0*((4 + s1) // (2 + (s1 // 2)))*((4 + s2) // (2 + (s2 // 2)))) // 2)*(((-1) + (((-1) + (s2 // 2)) // 2)) // 2) + 8*(s1 // 4)*((s0*((4 + s1) // (2 + (s1 // 2)))*((4 + s2) // (2 + (s2 // 2)))) // 2)*(((-1) + (((-1) + (s2 // 2)) // 2)) // 2), 8*((s0*((4 + s1) // (2 + (s1 // 2)))*((4 + s2) // (2 + (s2 // 2)))) // 2) + 8*(s1 // 4)*((s0*((4 + s1) // (2 + (s1 // 2)))*((4 + s2) // (2 + (s2 // 2)))) // 2) + 8*((s0*((4 + s1) // (2 + (s1 // 2)))*((4 + s2) // (2 + (s2 // 2)))) // 2)*(((-1) + (((-1) + (s2 // 2)) // 2)) // 2) + 8*(s1 // 4)*((s0*((4 + s1) // (2 + (s1 // 2)))*((4 + s2) // (2 + (s2 // 2)))) // 2)*(((-1) + (((-1) + (s2 // 2)) // 2)) // 2), 4 + 4*(s1 // 4) + 4*(((-1) + (((-1) + (s2 // 2)) // 2)) // 2) + 4*(s1 // 4)*(((-1) + (((-1) + (s2 // 2)) // 2)) // 2), 2 + 2*(((-1) + (((-1) + (s2 // 2)) // 2)) // 2), 1), torch.float32)

        triton_poi_fused_max_unpool3d_3_xnumel = 8*((s0*((4 + s1) // (2 + (s1 // 2)))*((4 + s2) // (2 + (s2 // 2)))) // 2) + 8*(s1 // 4)*((s0*((4 + s1) // (2 + (s1 // 2)))*((4 + s2) // (2 + (s2 // 2)))) // 2) + 8*((s0*((4 + s1) // (2 + (s1 // 2)))*((4 + s2) // (2 + (s2 // 2)))) // 2)*(((-1) + (((-1) + (s2 // 2)) // 2)) // 2) + 8*(s1 // 4)*((s0*((4 + s1) // (2 + (s1 // 2)))*((4 + s2) // (2 + (s2 // 2)))) // 2)*(((-1) + (((-1) + (s2 // 2)) // 2)) // 2)
        get_raw_stream(0)
        triton_poi_fused_max_unpool3d_3[grid(triton_poi_fused_max_unpool3d_3_xnumel)](buf7, 6528, XBLOCK=256, num_warps=4, num_stages=1)

        triton_poi_fused_max_unpool3d_4_xnumel = ((s0*((4 + s1) // (2 + (s1 // 2)))*((4 + s2) // (2 + (s2 // 2)))) // 2)*((1 + (s2 // 2)) // 4) + (s1 // 4)*((s0*((4 + s1) // (2 + (s1 // 2)))*((4 + s2) // (2 + (s2 // 2)))) // 2)*((1 + (s2 // 2)) // 4)
        get_raw_stream(0)
        triton_poi_fused_max_unpool3d_4[grid(triton_poi_fused_max_unpool3d_4_xnumel)](buf6, buf5, buf7, 64, 64, 3, 816, XBLOCK=128, num_warps=4, num_stages=1)
        del buf5
        del buf6
        2 + 2*(((-1) + (((-1) + (s2 // 2)) // 2)) // 2)
        2 + 2*(s1 // 4)
        4 + 4*(s1 // 4) + 4*(((-1) + (((-1) + (s2 // 2)) // 2)) // 2) + 4*(s1 // 4)*(((-1) + (((-1) + (s2 // 2)) // 2)) // 2)
        buf9 = empty_strided_cuda((1, 2*((s0*((4 + s1) // (2 + (s1 // 2)))*((4 + s2) // (2 + (s2 // 2)))) // 2), 2 + 2*(s1 // 4), 2 + 2*(((-1) + (((-1) + (s2 // 2)) // 2)) // 2)), (8*((s0*((4 + s1) // (2 + (s1 // 2)))*((4 + s2) // (2 + (s2 // 2)))) // 2) + 8*(s1 // 4)*((s0*((4 + s1) // (2 + (s1 // 2)))*((4 + s2) // (2 + (s2 // 2)))) // 2) + 8*((s0*((4 + s1) // (2 + (s1 // 2)))*((4 + s2) // (2 + (s2 // 2)))) // 2)*(((-1) + (((-1) + (s2 // 2)) // 2)) // 2) + 8*(s1 // 4)*((s0*((4 + s1) // (2 + (s1 // 2)))*((4 + s2) // (2 + (s2 // 2)))) // 2)*(((-1) + (((-1) + (s2 // 2)) // 2)) // 2), 4 + 4*(s1 // 4) + 4*(((-1) + (((-1) + (s2 // 2)) // 2)) // 2) + 4*(s1 // 4)*(((-1) + (((-1) + (s2 // 2)) // 2)) // 2), 2 + 2*(((-1) + (((-1) + (s2 // 2)) // 2)) // 2), 1), torch.float32)

        triton_poi_fused_relu_5_xnumel = 8*((s0*((4 + s1) // (2 + (s1 // 2)))*((4 + s2) // (2 + (s2 // 2)))) // 2) + 8*(s1 // 4)*((s0*((4 + s1) // (2 + (s1 // 2)))*((4 + s2) // (2 + (s2 // 2)))) // 2) + 8*((s0*((4 + s1) // (2 + (s1 // 2)))*((4 + s2) // (2 + (s2 // 2)))) // 2)*(((-1) + (((-1) + (s2 // 2)) // 2)) // 2) + 8*(s1 // 4)*((s0*((4 + s1) // (2 + (s1 // 2)))*((4 + s2) // (2 + (s2 // 2)))) // 2)*(((-1) + (((-1) + (s2 // 2)) // 2)) // 2)
        get_raw_stream(0)
        triton_poi_fused_relu_5[grid(triton_poi_fused_relu_5_xnumel)](buf7, buf9, 16, 34, 544, 3, 64, 64, 6528, XBLOCK=256, num_warps=4, num_stages=1)
        del buf7
    return (buf9, )


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
