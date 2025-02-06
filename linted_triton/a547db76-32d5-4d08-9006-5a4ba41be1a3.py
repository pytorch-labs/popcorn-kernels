
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
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_dropout_0(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (x0), tmp2, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_constant_pad_nd_native_dropout_silu_1(in_out_ptr0, in_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // ks0) % ks1)
    x0 = (xindex % ks0)
    x2 = xindex // ks4
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp1 = 0.5
    tmp2 = tmp0 > tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = (-2) + x1
    tmp5 = tl.full([1], 0, tl.int64)
    tmp6 = tmp4 >= tmp5
    tmp7 = ks2
    tmp8 = tmp4 < tmp7
    tmp9 = (-2) + x0
    tmp10 = tmp9 >= tmp5
    tmp11 = ks3
    tmp12 = tmp9 < tmp11
    tmp13 = tmp6 & tmp8
    tmp14 = tmp13 & tmp10
    tmp15 = tmp14 & tmp12
    tmp16 = tl.load(in_ptr0 + ((-2) + x0 + ((-2)*ks3) + ks3*x1 + ks2*ks3*x2), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.sigmoid(tmp16)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp15, tmp18, tmp19)
    tmp21 = tmp3 * tmp20
    tmp22 = 2.0
    tmp23 = tmp21 * tmp22
    tl.store(in_out_ptr0 + (x3), tmp23, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_max_unpool2d_native_dropout_silu_2(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, ks6, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks1)
    x2 = xindex // ks2
    x7 = xindex // ks6
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 8*x1 + 16*x2 + 2*ks4*x1 + 4*ks3*x2 + 4*ks4*x2 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 8*x1 + 16*x2 + 2*ks4*x1 + 4*ks3*x2 + 4*ks4*x2 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4 + ks4 + 2*x0 + 8*x1 + 16*x2 + 2*ks4*x1 + 4*ks3*x2 + 4*ks4*x2 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (5 + ks4 + 2*x0 + 8*x1 + 16*x2 + 2*ks4*x1 + 4*ks3*x2 + 4*ks4*x2 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 > tmp0
    tmp3 = tl.full([1], 1, tl.int8)
    tmp4 = tl.full([1], 0, tl.int8)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = triton_helpers.maximum(tmp1, tmp0)
    tmp8 = tmp7 > tmp6
    tmp9 = tl.full([1], 2, tl.int8)
    tmp10 = tl.where(tmp8, tmp9, tmp5)
    tmp11 = triton_helpers.maximum(tmp7, tmp6)
    tmp13 = tmp12 > tmp11
    tmp14 = tl.full([1], 3, tl.int8)
    tmp15 = tl.where(tmp13, tmp14, tmp10)
    triton_helpers.maximum(tmp12, tmp11)
    tmp17 = tl.full([1], 2, tl.int32)
    tmp18 = tl.where((tmp15 < 0) != (tmp17 < 0), tl.where(tmp15 % tmp17 != 0, tmp15 // tmp17 - 1, tmp15 // tmp17), tmp15 // tmp17)
    tmp19 = tmp18 * tmp17
    tmp20 = tmp15 - tmp19
    tmp21 = 2*x1
    tmp22 = tmp21 + tmp18
    tmp23 = 2*x0
    tmp24 = tmp23 + tmp20
    tmp25 = ks5
    tmp26 = tmp22 * tmp25
    tmp27 = tmp26 + tmp24
    tmp28 = 16*x7 + 8*x7*(ks3 // 2) + 8*x7*(ks4 // 2) + 4*x7*(ks3 // 2)*(ks4 // 2)
    tmp29 = tmp27 + tmp28
    tl.store(out_ptr0 + (x4), tmp29, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_unpool2d_3(out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
def triton_poi_fused_max_unpool2d_4(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp6 = tl.load(in_ptr1 + (2*((x0 % ks3)) + 8*(((x0 // ks3) % ks4)) + 16*(x0 // ks5) + 2*ks2*(((x0 // ks3) % ks4)) + 4*ks1*(x0 // ks5) + 4*ks2*(x0 // ks5) + ks1*ks2*(x0 // ks5)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (1 + 2*((x0 % ks3)) + 8*(((x0 // ks3) % ks4)) + 16*(x0 // ks5) + 2*ks2*(((x0 // ks3) % ks4)) + 4*ks1*(x0 // ks5) + 4*ks2*(x0 // ks5) + ks1*ks2*(x0 // ks5)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (4 + ks2 + 2*((x0 % ks3)) + 8*(((x0 // ks3) % ks4)) + 16*(x0 // ks5) + 2*ks2*(((x0 // ks3) % ks4)) + 4*ks1*(x0 // ks5) + 4*ks2*(x0 // ks5) + ks1*ks2*(x0 // ks5)), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr1 + (5 + ks2 + 2*((x0 % ks3)) + 8*(((x0 // ks3) % ks4)) + 16*(x0 // ks5) + 2*ks2*(((x0 // ks3) % ks4)) + 4*ks1*(x0 // ks5) + 4*ks2*(x0 // ks5) + ks1*ks2*(x0 // ks5)), xmask, eviction_policy='evict_last')
    tmp1 = 16*ks0 + 8*ks0*(ks1 // 2) + 8*ks0*(ks2 // 2) + 4*ks0*(ks1 // 2)*(ks2 // 2)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 16*ks0 + 8*ks0*(ks1 // 2) + 8*ks0*(ks2 // 2) + 4*ks0*(ks1 // 2)*(ks2 // 2))) | ~(xmask), "index out of bounds: 0 <= tmp4 < 16*ks0 + 8*ks0*(ks1 // 2) + 8*ks0*(ks2 // 2) + 4*ks0*(ks1 // 2)*(ks2 // 2)")
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tl.store(out_ptr0 + (tl.broadcast_to(4*(((tmp4 // (4 + 2*(ks2 // 2))) % (4 + 2*(ks1 // 2)))) + 16*(((tmp4 // (16 + 8*(ks1 // 2) + 8*(ks2 // 2) + 4*(ks1 // 2)*(ks2 // 2))) % ks0)) + 2*(ks2 // 2)*(((tmp4 // (4 + 2*(ks2 // 2))) % (4 + 2*(ks1 // 2)))) + 8*(ks1 // 2)*(((tmp4 // (16 + 8*(ks1 // 2) + 8*(ks2 // 2) + 4*(ks1 // 2)*(ks2 // 2))) % ks0)) + 8*(ks2 // 2)*(((tmp4 // (16 + 8*(ks1 // 2) + 8*(ks2 // 2) + 4*(ks1 // 2)*(ks2 // 2))) % ks0)) + 4*(ks1 // 2)*(ks2 // 2)*(((tmp4 // (16 + 8*(ks1 // 2) + 8*(ks2 // 2) + 4*(ks1 // 2)*(ks2 // 2))) % ks0)) + ((tmp4 % (4 + 2*(ks2 // 2)))), [XBLOCK])), tmp12, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_dropout_5(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (x0), tmp2, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_constant_pad_nd_native_dropout_silu_6(in_out_ptr0, in_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // ks0) % ks1)
    x0 = (xindex % ks0)
    x2 = xindex // ks4
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp1 = 0.3
    tmp2 = tmp0 > tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = (-1) + x1
    tmp5 = tl.full([1], 0, tl.int64)
    tmp6 = tmp4 >= tmp5
    tmp7 = 4 + 2*(ks2 // 2)
    tmp8 = tmp4 < tmp7
    tmp9 = (-1) + x0
    tmp10 = tmp9 >= tmp5
    tmp11 = 4 + 2*(ks3 // 2)
    tmp12 = tmp9 < tmp11
    tmp13 = tmp6 & tmp8
    tmp14 = tmp13 & tmp10
    tmp15 = tmp14 & tmp12
    tmp16 = tl.load(in_ptr0 + (4*(((((-5) + x0 + ((-2)*(ks3 // 2)) + 4*x1 + 16*x2 + 2*x1*(ks3 // 2) + 8*x2*(ks2 // 2) + 8*x2*(ks3 // 2) + 4*x2*(ks2 // 2)*(ks3 // 2)) // (4 + 2*(ks3 // 2))) % (4 + 2*(ks2 // 2)))) + 16*(((((-5) + x0 + ((-2)*(ks3 // 2)) + 4*x1 + 16*x2 + 2*x1*(ks3 // 2) + 8*x2*(ks2 // 2) + 8*x2*(ks3 // 2) + 4*x2*(ks2 // 2)*(ks3 // 2)) // (16 + 8*(ks2 // 2) + 8*(ks3 // 2) + 4*(ks2 // 2)*(ks3 // 2))) % ks5)) + 2*(ks3 // 2)*(((((-5) + x0 + ((-2)*(ks3 // 2)) + 4*x1 + 16*x2 + 2*x1*(ks3 // 2) + 8*x2*(ks2 // 2) + 8*x2*(ks3 // 2) + 4*x2*(ks2 // 2)*(ks3 // 2)) // (4 + 2*(ks3 // 2))) % (4 + 2*(ks2 // 2)))) + 8*(ks2 // 2)*(((((-5) + x0 + ((-2)*(ks3 // 2)) + 4*x1 + 16*x2 + 2*x1*(ks3 // 2) + 8*x2*(ks2 // 2) + 8*x2*(ks3 // 2) + 4*x2*(ks2 // 2)*(ks3 // 2)) // (16 + 8*(ks2 // 2) + 8*(ks3 // 2) + 4*(ks2 // 2)*(ks3 // 2))) % ks5)) + 8*(ks3 // 2)*(((((-5) + x0 + ((-2)*(ks3 // 2)) + 4*x1 + 16*x2 + 2*x1*(ks3 // 2) + 8*x2*(ks2 // 2) + 8*x2*(ks3 // 2) + 4*x2*(ks2 // 2)*(ks3 // 2)) // (16 + 8*(ks2 // 2) + 8*(ks3 // 2) + 4*(ks2 // 2)*(ks3 // 2))) % ks5)) + 4*(ks2 // 2)*(ks3 // 2)*(((((-5) + x0 + ((-2)*(ks3 // 2)) + 4*x1 + 16*x2 + 2*x1*(ks3 // 2) + 8*x2*(ks2 // 2) + 8*x2*(ks3 // 2) + 4*x2*(ks2 // 2)*(ks3 // 2)) // (16 + 8*(ks2 // 2) + 8*(ks3 // 2) + 4*(ks2 // 2)*(ks3 // 2))) % ks5)) + ((((-5) + x0 + ((-2)*(ks3 // 2)) + 4*x1 + 16*x2 + 2*x1*(ks3 // 2) + 8*x2*(ks2 // 2) + 8*x2*(ks3 // 2) + 4*x2*(ks2 // 2)*(ks3 // 2)) % (4 + 2*(ks3 // 2))))), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.sigmoid(tmp16)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp15, tmp18, tmp19)
    tmp21 = tmp3 * tmp20
    tmp22 = 1.4285714285714286
    tmp23 = tmp21 * tmp22
    tl.store(in_out_ptr0 + (x3), tmp23, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_max_unpool2d_native_dropout_silu_7(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x3 = xindex // ks0
    x1 = ((xindex // ks0) % ks2)
    x7 = xindex // ks4
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 12*x3 + 4*x3*(ks1 // 2)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 12*x3 + 4*x3*(ks1 // 2)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (6 + 2*x0 + 2*(ks1 // 2) + 12*x3 + 4*x3*(ks1 // 2)), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (7 + 2*x0 + 2*(ks1 // 2) + 12*x3 + 4*x3*(ks1 // 2)), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 > tmp0
    tmp3 = tl.full([1], 1, tl.int8)
    tmp4 = tl.full([1], 0, tl.int8)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = triton_helpers.maximum(tmp1, tmp0)
    tmp8 = tmp7 > tmp6
    tmp9 = tl.full([1], 2, tl.int8)
    tmp10 = tl.where(tmp8, tmp9, tmp5)
    tmp11 = triton_helpers.maximum(tmp7, tmp6)
    tmp13 = tmp12 > tmp11
    tmp14 = tl.full([1], 3, tl.int8)
    tmp15 = tl.where(tmp13, tmp14, tmp10)
    triton_helpers.maximum(tmp12, tmp11)
    tmp17 = tl.full([1], 2, tl.int32)
    tmp18 = tl.where((tmp15 < 0) != (tmp17 < 0), tl.where(tmp15 % tmp17 != 0, tmp15 // tmp17 - 1, tmp15 // tmp17), tmp15 // tmp17)
    tmp19 = tmp18 * tmp17
    tmp20 = tmp15 - tmp19
    tmp21 = 2*x1
    tmp22 = tmp21 + tmp18
    tmp23 = 2*x0
    tmp24 = tmp23 + tmp20
    tmp25 = ks3
    tmp26 = tmp22 * tmp25
    tmp27 = tmp26 + tmp24
    tmp28 = 36*x7 + 12*x7*(ks1 // 2) + 12*x7*(ks5 // 2) + 4*x7*(ks1 // 2)*(ks5 // 2)
    tmp29 = tmp27 + tmp28
    tl.store(out_ptr0 + (x4), tmp29, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_unpool2d_8(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, ks6, ks7, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp6 = tl.load(in_ptr1 + (2*((x0 % ks3)) + 12*(((x0 // ks3) % ks4)) + 36*(x0 // ks5) + 4*(ks2 // 2)*(((x0 // ks3) % ks4)) + 12*(ks1 // 2)*(x0 // ks5) + 12*(ks2 // 2)*(x0 // ks5) + 4*(ks1 // 2)*(ks2 // 2)*(x0 // ks5)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (1 + 2*((x0 % ks3)) + 12*(((x0 // ks3) % ks4)) + 36*(x0 // ks5) + 4*(ks2 // 2)*(((x0 // ks3) % ks4)) + 12*(ks1 // 2)*(x0 // ks5) + 12*(ks2 // 2)*(x0 // ks5) + 4*(ks1 // 2)*(ks2 // 2)*(x0 // ks5)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (6 + 2*(ks2 // 2) + 2*((x0 % ks3)) + 12*(((x0 // ks3) % ks4)) + 36*(x0 // ks5) + 4*(ks2 // 2)*(((x0 // ks3) % ks4)) + 12*(ks1 // 2)*(x0 // ks5) + 12*(ks2 // 2)*(x0 // ks5) + 4*(ks1 // 2)*(ks2 // 2)*(x0 // ks5)), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr1 + (7 + 2*(ks2 // 2) + 2*((x0 % ks3)) + 12*(((x0 // ks3) % ks4)) + 36*(x0 // ks5) + 4*(ks2 // 2)*(((x0 // ks3) % ks4)) + 12*(ks1 // 2)*(x0 // ks5) + 12*(ks2 // 2)*(x0 // ks5) + 4*(ks1 // 2)*(ks2 // 2)*(x0 // ks5)), xmask, eviction_policy='evict_last')
    tmp1 = 36*ks0 + 12*ks0*(ks1 // 2) + 12*ks0*(ks2 // 2) + 4*ks0*(ks1 // 2)*(ks2 // 2)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 36*ks0 + 12*ks0*(ks1 // 2) + 12*ks0*(ks2 // 2) + 4*ks0*(ks1 // 2)*(ks2 // 2))) | ~(xmask), "index out of bounds: 0 <= tmp4 < 36*ks0 + 12*ks0*(ks1 // 2) + 12*ks0*(ks2 // 2) + 4*ks0*(ks1 // 2)*(ks2 // 2)")
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tl.store(out_ptr0 + (tl.broadcast_to(6*(((tmp4 // ks6) % ks7)) + 36*(((tmp4 // (36 + 12*(ks1 // 2) + 12*(ks2 // 2) + 4*(ks1 // 2)*(ks2 // 2))) % ks0)) + 2*(ks2 // 2)*(((tmp4 // ks6) % ks7)) + 12*(ks1 // 2)*(((tmp4 // (36 + 12*(ks1 // 2) + 12*(ks2 // 2) + 4*(ks1 // 2)*(ks2 // 2))) % ks0)) + 12*(ks2 // 2)*(((tmp4 // (36 + 12*(ks1 // 2) + 12*(ks2 // 2) + 4*(ks1 // 2)*(ks2 // 2))) % ks0)) + 4*(ks1 // 2)*(ks2 // 2)*(((tmp4 // (36 + 12*(ks1 // 2) + 12*(ks2 // 2) + 4*(ks1 // 2)*(ks2 // 2))) % ks0)) + ((tmp4 % ks6)), [XBLOCK])), tmp12, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_silu_9(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks1)
    x2 = xindex // ks2
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 6*((((x0 + 6*x1 + 36*x2 + 2*x1*(ks5 // 2) + 12*x2*(ks4 // 2) + 12*x2*(ks5 // 2) + 4*x2*(ks4 // 2)*(ks5 // 2)) // ks0) % ks1)) + 36*((((x0 + 6*x1 + 36*x2 + 2*x1*(ks5 // 2) + 12*x2*(ks4 // 2) + 12*x2*(ks5 // 2) + 4*x2*(ks4 // 2)*(ks5 // 2)) // (36 + 12*(ks4 // 2) + 12*(ks5 // 2) + 4*(ks4 // 2)*(ks5 // 2))) % ks3)) + 2*(ks5 // 2)*((((x0 + 6*x1 + 36*x2 + 2*x1*(ks5 // 2) + 12*x2*(ks4 // 2) + 12*x2*(ks5 // 2) + 4*x2*(ks4 // 2)*(ks5 // 2)) // ks0) % ks1)) + 12*(ks4 // 2)*((((x0 + 6*x1 + 36*x2 + 2*x1*(ks5 // 2) + 12*x2*(ks4 // 2) + 12*x2*(ks5 // 2) + 4*x2*(ks4 // 2)*(ks5 // 2)) // (36 + 12*(ks4 // 2) + 12*(ks5 // 2) + 4*(ks4 // 2)*(ks5 // 2))) % ks3)) + 12*(ks5 // 2)*((((x0 + 6*x1 + 36*x2 + 2*x1*(ks5 // 2) + 12*x2*(ks4 // 2) + 12*x2*(ks5 // 2) + 4*x2*(ks4 // 2)*(ks5 // 2)) // (36 + 12*(ks4 // 2) + 12*(ks5 // 2) + 4*(ks4 // 2)*(ks5 // 2))) % ks3)) + 4*(ks4 // 2)*(ks5 // 2)*((((x0 + 6*x1 + 36*x2 + 2*x1*(ks5 // 2) + 12*x2*(ks4 // 2) + 12*x2*(ks5 // 2) + 4*x2*(ks4 // 2)*(ks5 // 2)) // (36 + 12*(ks4 // 2) + 12*(ks5 // 2) + 4*(ks4 // 2)*(ks5 // 2))) % ks3))), xmask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
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
        buf0 = empty_strided_cuda((2, ), (1, ), torch.int64)

        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [2], out=buf0)
        buf1 = empty_strided_cuda((1, s0, 4 + s1, 4 + s2), (16*s0 + 4*s0*s1 + 4*s0*s2 + s0*s1*s2, 16 + 4*s1 + 4*s2 + s1*s2, 4 + s2, 1), torch.float32)

        triton_poi_fused_native_dropout_0_xnumel = 16*s0 + 4*s0*s1 + 4*s0*s2 + s0*s1*s2
        get_raw_stream(0)
        triton_poi_fused_native_dropout_0[grid(triton_poi_fused_native_dropout_0_xnumel)](buf0, buf1, 0, 13872, XBLOCK=256, num_warps=4, num_stages=1)
        4 + s2
        4 + s1
        16 + 4*s1 + 4*s2 + s1*s2
        buf2 = buf1; del buf1

        triton_poi_fused_constant_pad_nd_native_dropout_silu_1_xnumel = 16*s0 + 4*s0*s1 + 4*s0*s2 + s0*s1*s2
        get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_native_dropout_silu_1[grid(triton_poi_fused_constant_pad_nd_native_dropout_silu_1_xnumel)](buf2, arg3_1, 68, 68, 64, 64, 4624, 13872, XBLOCK=256, num_warps=4, num_stages=1)
        del arg3_1
        2 + (s2 // 2)
        2 + (s1 // 2)
        4 + 2*(s1 // 2) + 2*(s2 // 2) + (s1 // 2)*(s2 // 2)
        4 + 2*(s1 // 2) + 2*(s2 // 2) + (s1 // 2)*(s2 // 2)
        buf3 = empty_strided_cuda((1, s0, 2 + (s1 // 2), 2 + (s2 // 2)), (4*s0 + 2*s0*(s1 // 2) + 2*s0*(s2 // 2) + s0*(s1 // 2)*(s2 // 2), 4 + 2*(s1 // 2) + 2*(s2 // 2) + (s1 // 2)*(s2 // 2), 2 + (s2 // 2), 1), torch.int64)

        triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_max_unpool2d_native_dropout_silu_2_xnumel = 4*s0 + 2*s0*(s1 // 2) + 2*s0*(s2 // 2) + s0*(s1 // 2)*(s2 // 2)
        get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_max_unpool2d_native_dropout_silu_2[grid(triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_max_unpool2d_native_dropout_silu_2_xnumel)](buf2, buf3, 34, 34, 1156, 64, 64, 68, 1156, 3468, XBLOCK=128, num_warps=4, num_stages=1)
        buf4 = empty_strided_cuda((1, s0, 4 + 2*(s1 // 2), 4 + 2*(s2 // 2)), (16*s0 + 8*s0*(s1 // 2) + 8*s0*(s2 // 2) + 4*s0*(s1 // 2)*(s2 // 2), 16 + 8*(s1 // 2) + 8*(s2 // 2) + 4*(s1 // 2)*(s2 // 2), 4 + 2*(s2 // 2), 1), torch.float32)

        triton_poi_fused_max_unpool2d_3_xnumel = 16*s0 + 8*s0*(s1 // 2) + 8*s0*(s2 // 2) + 4*s0*(s1 // 2)*(s2 // 2)
        get_raw_stream(0)
        triton_poi_fused_max_unpool2d_3[grid(triton_poi_fused_max_unpool2d_3_xnumel)](buf4, 14700, XBLOCK=256, num_warps=4, num_stages=1)

        triton_poi_fused_max_unpool2d_4_xnumel = 4*s0 + 2*s0*(s1 // 2) + 2*s0*(s2 // 2) + s0*(s1 // 2)*(s2 // 2)
        get_raw_stream(0)
        triton_poi_fused_max_unpool2d_4[grid(triton_poi_fused_max_unpool2d_4_xnumel)](buf3, buf2, buf4, 3, 64, 64, 34, 34, 1156, 3468, XBLOCK=128, num_warps=4, num_stages=1)
        del buf2
        del buf3
        buf6 = empty_strided_cuda((1, s0, 6 + 2*(s1 // 2), 6 + 2*(s2 // 2)), (36*s0 + 12*s0*(s1 // 2) + 12*s0*(s2 // 2) + 4*s0*(s1 // 2)*(s2 // 2), 36 + 12*(s1 // 2) + 12*(s2 // 2) + 4*(s1 // 2)*(s2 // 2), 6 + 2*(s2 // 2), 1), torch.float32)

        triton_poi_fused_native_dropout_5_xnumel = 36*s0 + 12*s0*(s1 // 2) + 12*s0*(s2 // 2) + 4*s0*(s1 // 2)*(s2 // 2)
        get_raw_stream(0)
        triton_poi_fused_native_dropout_5[grid(triton_poi_fused_native_dropout_5_xnumel)](buf0, buf6, 1, 14700, XBLOCK=128, num_warps=4, num_stages=1)
        del buf0
        6 + 2*(s2 // 2)
        6 + 2*(s1 // 2)
        36 + 12*(s1 // 2) + 12*(s2 // 2) + 4*(s1 // 2)*(s2 // 2)
        buf7 = buf6; del buf6

        triton_poi_fused_constant_pad_nd_native_dropout_silu_6_xnumel = 36*s0 + 12*s0*(s1 // 2) + 12*s0*(s2 // 2) + 4*s0*(s1 // 2)*(s2 // 2)
        get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_native_dropout_silu_6[grid(triton_poi_fused_constant_pad_nd_native_dropout_silu_6_xnumel)](buf7, buf4, 70, 70, 64, 64, 4900, 3, 14700, XBLOCK=256, num_warps=4, num_stages=1)
        del buf4
        3 + (s2 // 2)
        3 + (s1 // 2)
        9 + 3*(s1 // 2) + 3*(s2 // 2) + (s1 // 2)*(s2 // 2)
        buf8 = empty_strided_cuda((1, s0, 3 + (s1 // 2), 3 + (s2 // 2)), (9*s0 + 3*s0*(s1 // 2) + 3*s0*(s2 // 2) + s0*(s1 // 2)*(s2 // 2), 9 + 3*(s1 // 2) + 3*(s2 // 2) + (s1 // 2)*(s2 // 2), 3 + (s2 // 2), 1), torch.int64)

        triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_max_unpool2d_native_dropout_silu_7_xnumel = 9*s0 + 3*s0*(s1 // 2) + 3*s0*(s2 // 2) + s0*(s1 // 2)*(s2 // 2)
        get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_max_unpool2d_native_dropout_silu_7[grid(triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_max_unpool2d_native_dropout_silu_7_xnumel)](buf7, buf8, 35, 64, 35, 70, 1225, 64, 3675, XBLOCK=128, num_warps=4, num_stages=1)
        buf9 = empty_strided_cuda((1, s0, 6 + 2*(s1 // 2), 6 + 2*(s2 // 2)), (36*s0 + 12*s0*(s1 // 2) + 12*s0*(s2 // 2) + 4*s0*(s1 // 2)*(s2 // 2), 36 + 12*(s1 // 2) + 12*(s2 // 2) + 4*(s1 // 2)*(s2 // 2), 6 + 2*(s2 // 2), 1), torch.float32)

        triton_poi_fused_max_unpool2d_3_xnumel = 36*s0 + 12*s0*(s1 // 2) + 12*s0*(s2 // 2) + 4*s0*(s1 // 2)*(s2 // 2)
        get_raw_stream(0)
        triton_poi_fused_max_unpool2d_3[grid(triton_poi_fused_max_unpool2d_3_xnumel)](buf9, 14700, XBLOCK=256, num_warps=4, num_stages=1)

        triton_poi_fused_max_unpool2d_8_xnumel = 9*s0 + 3*s0*(s1 // 2) + 3*s0*(s2 // 2) + s0*(s1 // 2)*(s2 // 2)
        get_raw_stream(0)
        triton_poi_fused_max_unpool2d_8[grid(triton_poi_fused_max_unpool2d_8_xnumel)](buf8, buf7, buf9, 3, 64, 64, 35, 35, 1225, 70, 70, 3675, XBLOCK=128, num_warps=4, num_stages=1)
        del buf8
        buf11 = buf7; del buf7

        triton_poi_fused_silu_9_xnumel = 36*s0 + 12*s0*(s1 // 2) + 12*s0*(s2 // 2) + 4*s0*(s1 // 2)*(s2 // 2)
        get_raw_stream(0)
        triton_poi_fused_silu_9[grid(triton_poi_fused_silu_9_xnumel)](buf9, buf11, 70, 70, 4900, 3, 64, 64, 14700, XBLOCK=256, num_warps=4, num_stages=1)
        del buf9
    return (buf11, )


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
