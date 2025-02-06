
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
from torch._inductor.runtime.triton_helpers import libdevice
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_copy_0(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks2)
    x2 = xindex // ks4
    x3 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = 1 + (ks1 // 2)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = x1
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp6 >= tmp7
    tmp9 = tl.broadcast_to(1 + (ks3 // 2), [XBLOCK])
    tmp10 = tmp6 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tmp11 & tmp5
    tmp13 = tl.load(in_ptr0 + ((-2) + ((-2)*ks1) + 2*x0 + 2*ks1*x1 + ks1*ks3*x2), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(in_ptr0 + ((-1) + ((-2)*ks1) + 2*x0 + 2*ks1*x1 + ks1*ks3*x2), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tmp16 = tl.load(in_ptr0 + ((-2) + ((-1)*ks1) + 2*x0 + 2*ks1*x1 + ks1*ks3*x2), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tmp18 = tl.load(in_ptr0 + ((-1) + ((-1)*ks1) + 2*x0 + 2*ks1*x1 + ks1*ks3*x2), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp12, tmp19, tmp20)
    tmp22 = tl.load(in_ptr1 + (x3), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.where(tmp11, tmp21, tmp22)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp5, tmp23, tmp24)
    tmp26 = float("nan")
    tmp27 = tl.where(tmp5, tmp25, tmp26)
    tl.store(out_ptr0 + (x3), tmp27, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // ks0) % ks1)
    x0 = (xindex % ks0)
    x4 = xindex // ks0
    x3 = xindex
    tmp39 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0
    tmp4 = tl.broadcast_to(1 + (ks2 // 2), [XBLOCK])
    tmp5 = tmp3 >= tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = x0 + ((-1)*(ks2 // 2))
    tmp8 = tl.full([1], 1, tl.int64)
    tmp9 = tmp7 < tmp8
    tmp10 = tmp9 & tmp6
    tmp11 = tl.load(in_ptr0 + (2*x4 + 2*(ks3 // 2) + x4*(ks2 // 2) + (ks2 // 2)*(ks3 // 2) + (ks2 // 2)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr0 + (x3 + ((-1)*(ks2 // 2)) + 2*(ks3 // 2) + (ks2 // 2)*(ks3 // 2)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.where(tmp9, tmp11, tmp12)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.full([1], 1, tl.int64)
    tmp17 = tmp3 < tmp16
    tmp18 = tmp17 & tmp2
    tmp19 = tl.load(in_ptr0 + (2*x4 + 2*(ks3 // 2) + x4*(ks2 // 2) + (ks2 // 2)*(ks3 // 2) + (ks2 // 2)), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.load(in_ptr0 + (x3 + 2*(ks3 // 2) + (ks2 // 2)*(ks3 // 2)), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.where(tmp17, tmp19, tmp20)
    tmp22 = tl.where(tmp5, tmp15, tmp21)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp2, tmp22, tmp23)
    tmp25 = x0
    tmp26 = 1 + (ks2 // 2)
    tmp27 = tmp25 >= tmp26
    tmp28 = x0 + ((-1)*(ks2 // 2))
    tmp29 = tl.full([1], 1, tl.int64)
    tmp30 = tmp28 < tmp29
    tmp31 = tmp30 & tmp27
    tmp32 = tl.load(in_ptr0 + (2*x4 + x4*(ks2 // 2) + (ks2 // 2)), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tl.load(in_ptr0 + (x3 + ((-1)*(ks2 // 2))), tmp27 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.where(tmp30, tmp32, tmp33)
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp27, tmp34, tmp35)
    tmp37 = tmp25 < tmp1
    tmp38 = tl.load(in_ptr0 + (2*x4 + x4*(ks2 // 2) + (ks2 // 2)), tmp37 & xmask, eviction_policy='evict_last', other=0.0)
    tmp40 = tl.where(tmp37, tmp38, tmp39)
    tmp41 = tl.where(tmp27, tmp36, tmp40)
    tmp42 = tl.where(tmp2, tmp24, tmp41)
    tl.store(out_ptr0 + (x3), tmp42, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_view_2(in_ptr0, out_ptr0, out_ptr1, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 3
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    tmp7_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp7_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp7_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        r0_2 = r0_index // ks0
        r0_1 = (r0_index % ks0)
        r0_3 = r0_index
        tmp4 = tl.load(in_ptr0 + (r0_3 + 4*x0 + 2*x0*(ks1 // 2) + 2*x0*(ks2 // 2) + x0*(ks1 // 2)*(ks2 // 2)), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = r0_2
        tmp1 = 1 + (ks1 // 2)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.load(in_ptr0 + (2 + r0_1 + 4*x0 + 2*x0*(ks1 // 2) + 2*x0*(ks2 // 2) + x0*(ks1 // 2)*(ks2 // 2) + (ks2 // 2)), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp7_mean_next, tmp7_m2_next, tmp7_weight_next = triton_helpers.welford_reduce(
            tmp6, tmp7_mean, tmp7_m2, tmp7_weight, roffset == 0
        )
        tmp7_mean = tl.where(r0_mask & xmask, tmp7_mean_next, tmp7_mean)
        tmp7_m2 = tl.where(r0_mask & xmask, tmp7_m2_next, tmp7_m2)
        tmp7_weight = tl.where(r0_mask & xmask, tmp7_weight_next, tmp7_weight)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7_mean, tmp7_m2, tmp7_weight, 1)
    tmp7 = tmp10[:, None]
    tmp8 = tmp11[:, None]
    tmp12[:, None]
    tl.store(out_ptr0 + (x0), tmp7, xmask)
    tl.store(out_ptr1 + (x0), tmp8, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_view_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // ks0) % ks1)
    x0 = (xindex % ks0)
    x2 = xindex // ks3
    x3 = xindex
    tmp4 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = 1 + (ks2 // 2)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (2 + x0 + 4*x2 + 2*x2*(ks2 // 2) + 2*x2*(ks4 // 2) + x2*(ks2 // 2)*(ks4 // 2) + (ks4 // 2)), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp7 = tmp5 - tmp6
    tmp9 = ((tl.full([], 0.0, tl.float64)) * ((tl.full([], 0.0, tl.float64)) >= (4 + 2*(ks2 // 2) + 2*(ks4 // 2) + (ks2 // 2)*(ks4 // 2))) + (4 + 2*(ks2 // 2) + 2*(ks4 // 2) + (ks2 // 2)*(ks4 // 2)) * ((4 + 2*(ks2 // 2) + 2*(ks4 // 2) + (ks2 // 2)*(ks4 // 2)) > (tl.full([], 0.0, tl.float64))))
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 / tmp10
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp7 * tmp14
    tl.store(out_ptr0 + (x3), tmp15, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_copy_4(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks2)
    x2 = xindex // ks4
    x3 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = 2 + (ks1 // 4)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = x1
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp6 >= tmp7
    tmp9 = tl.broadcast_to(2 + (ks3 // 4), [XBLOCK])
    tmp10 = tmp6 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tmp11 & tmp5
    tmp13 = tl.load(in_ptr0 + ((-6) + ((-2)*(ks1 // 2)) + 2*x0 + 4*x1 + 4*x2 + 2*x1*(ks1 // 2) + 2*x2*(ks1 // 2) + 2*x2*(ks3 // 2) + x2*(ks1 // 2)*(ks3 // 2)), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(in_ptr0 + ((-5) + ((-2)*(ks1 // 2)) + 2*x0 + 4*x1 + 4*x2 + 2*x1*(ks1 // 2) + 2*x2*(ks1 // 2) + 2*x2*(ks3 // 2) + x2*(ks1 // 2)*(ks3 // 2)), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tmp16 = tl.load(in_ptr0 + ((-4) + ((-1)*(ks1 // 2)) + 2*x0 + 4*x1 + 4*x2 + 2*x1*(ks1 // 2) + 2*x2*(ks1 // 2) + 2*x2*(ks3 // 2) + x2*(ks1 // 2)*(ks3 // 2)), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tmp18 = tl.load(in_ptr0 + ((-3) + ((-1)*(ks1 // 2)) + 2*x0 + 4*x1 + 4*x2 + 2*x1*(ks1 // 2) + 2*x2*(ks1 // 2) + 2*x2*(ks3 // 2) + x2*(ks1 // 2)*(ks3 // 2)), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp12, tmp19, tmp20)
    tmp22 = tl.load(in_ptr1 + (x3), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.where(tmp11, tmp21, tmp22)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp5, tmp23, tmp24)
    tmp26 = float("nan")
    tmp27 = tl.where(tmp5, tmp25, tmp26)
    tl.store(out_ptr0 + (x3), tmp27, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_5(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // ks0) % ks1)
    x0 = (xindex % ks0)
    x4 = xindex // ks0
    x3 = xindex
    tmp39 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0
    tmp4 = tl.broadcast_to(2 + (ks2 // 4), [XBLOCK])
    tmp5 = tmp3 >= tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = (-1) + x0 + ((-1)*(ks2 // 4))
    tmp8 = tl.full([1], 1, tl.int64)
    tmp9 = tmp7 < tmp8
    tmp10 = tmp9 & tmp6
    tmp11 = tl.load(in_ptr0 + (4 + 2*(ks2 // 4) + 3*x4 + 3*(ks3 // 4) + x4*(ks2 // 4) + (ks2 // 4)*(ks3 // 4)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr0 + (2 + x3 + 3*(ks3 // 4) + (ks2 // 4)*(ks3 // 4)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.where(tmp9, tmp11, tmp12)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.full([1], 1, tl.int64)
    tmp17 = tmp3 < tmp16
    tmp18 = tmp17 & tmp2
    tmp19 = tl.load(in_ptr0 + (4 + 2*(ks2 // 4) + 3*x4 + 3*(ks3 // 4) + x4*(ks2 // 4) + (ks2 // 4)*(ks3 // 4)), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.load(in_ptr0 + (3 + x3 + 3*(ks3 // 4) + (ks2 // 4)*(ks3 // 4) + (ks2 // 4)), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.where(tmp17, tmp19, tmp20)
    tmp22 = tl.where(tmp5, tmp15, tmp21)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp2, tmp22, tmp23)
    tmp25 = x0
    tmp26 = 2 + (ks2 // 4)
    tmp27 = tmp25 >= tmp26
    tmp28 = (-1) + x0 + ((-1)*(ks2 // 4))
    tmp29 = tl.full([1], 1, tl.int64)
    tmp30 = tmp28 < tmp29
    tmp31 = tmp30 & tmp27
    tmp32 = tl.load(in_ptr0 + (1 + 3*x4 + x4*(ks2 // 4) + (ks2 // 4)), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tl.load(in_ptr0 + ((-1) + x3 + ((-1)*(ks2 // 4))), tmp27 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.where(tmp30, tmp32, tmp33)
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp27, tmp34, tmp35)
    tmp37 = tmp25 < tmp1
    tmp38 = tl.load(in_ptr0 + (1 + 3*x4 + x4*(ks2 // 4) + (ks2 // 4)), tmp37 & xmask, eviction_policy='evict_last', other=0.0)
    tmp40 = tl.where(tmp37, tmp38, tmp39)
    tmp41 = tl.where(tmp27, tmp36, tmp40)
    tmp42 = tl.where(tmp2, tmp24, tmp41)
    tl.store(out_ptr0 + (x3), tmp42, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_view_6(in_ptr0, out_ptr0, out_ptr1, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 3
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    tmp7_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp7_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp7_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        r0_2 = r0_index // ks0
        r0_1 = (r0_index % ks0)
        r0_3 = r0_index
        tmp4 = tl.load(in_ptr0 + (r0_3 + 9*x0 + 3*x0*(ks1 // 4) + 3*x0*(ks2 // 4) + x0*(ks1 // 4)*(ks2 // 4)), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = r0_2
        tmp1 = 2 + (ks1 // 4)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.load(in_ptr0 + (3 + r0_1 + 9*x0 + 3*x0*(ks1 // 4) + 3*x0*(ks2 // 4) + x0*(ks1 // 4)*(ks2 // 4) + (ks2 // 4)), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp7_mean_next, tmp7_m2_next, tmp7_weight_next = triton_helpers.welford_reduce(
            tmp6, tmp7_mean, tmp7_m2, tmp7_weight, roffset == 0
        )
        tmp7_mean = tl.where(r0_mask & xmask, tmp7_mean_next, tmp7_mean)
        tmp7_m2 = tl.where(r0_mask & xmask, tmp7_m2_next, tmp7_m2)
        tmp7_weight = tl.where(r0_mask & xmask, tmp7_weight_next, tmp7_weight)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7_mean, tmp7_m2, tmp7_weight, 1)
    tmp7 = tmp10[:, None]
    tmp8 = tmp11[:, None]
    tmp12[:, None]
    tl.store(out_ptr0 + (x0), tmp7, xmask)
    tl.store(out_ptr1 + (x0), tmp8, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_view_7(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // ks0) % ks1)
    x0 = (xindex % ks0)
    x2 = xindex // ks3
    x3 = xindex
    tmp4 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = 2 + (ks2 // 4)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (3 + x0 + 9*x2 + 3*x2*(ks2 // 4) + 3*x2*(ks4 // 4) + x2*(ks2 // 4)*(ks4 // 4) + (ks4 // 4)), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp7 = tmp5 - tmp6
    tmp9 = ((tl.full([], 0.0, tl.float64)) * ((tl.full([], 0.0, tl.float64)) >= (9 + 3*(ks2 // 4) + 3*(ks4 // 4) + (ks2 // 4)*(ks4 // 4))) + (9 + 3*(ks2 // 4) + 3*(ks4 // 4) + (ks2 // 4)*(ks4 // 4)) * ((9 + 3*(ks2 // 4) + 3*(ks4 // 4) + (ks2 // 4)*(ks4 // 4)) > (tl.full([], 0.0, tl.float64))))
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 / tmp10
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp7 * tmp14
    tl.store(out_ptr0 + (x3), tmp15, xmask)


def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    s1 = arg0_1
    s2 = arg1_1
    assert_size_stride(arg2_1, (1, 3, s1, s2), (3*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 3, 2 + (s1 // 2), 2 + (s2 // 2)), (12 + 6*(s1 // 2) + 6*(s2 // 2) + 3*(s1 // 2)*(s2 // 2), 4 + 2*(s1 // 2) + 2*(s2 // 2) + (s1 // 2)*(s2 // 2), 2 + (s2 // 2), 1), torch.float32)
        2 + (s2 // 2)
        2 + (s1 // 2)
        4 + 2*(s1 // 2) + 2*(s2 // 2) + (s1 // 2)*(s2 // 2)
        buf1 = empty_strided_cuda((1, 3, 2 + (s1 // 2), 2 + (s2 // 2)), (12 + 6*(s1 // 2) + 6*(s2 // 2) + 3*(s1 // 2)*(s2 // 2), 4 + 2*(s1 // 2) + 2*(s2 // 2) + (s1 // 2)*(s2 // 2), 2 + (s2 // 2), 1), torch.float32)

        triton_poi_fused_copy_0_xnumel = 12 + 6*(s1 // 2) + 6*(s2 // 2) + 3*(s1 // 2)*(s2 // 2)
        get_raw_stream(0)
        triton_poi_fused_copy_0[grid(triton_poi_fused_copy_0_xnumel)](arg2_1, buf0, buf1, 34, 64, 34, 64, 1156, 3468, XBLOCK=128, num_warps=4, num_stages=1)
        del arg2_1
        buf2 = buf0; del buf0

        triton_poi_fused_1_xnumel = 12 + 6*(s1 // 2) + 6*(s2 // 2) + 3*(s1 // 2)*(s2 // 2)
        get_raw_stream(0)
        triton_poi_fused_1[grid(triton_poi_fused_1_xnumel)](buf1, buf2, 34, 34, 64, 64, 3468, XBLOCK=128, num_warps=4, num_stages=1)
        buf3 = empty_strided_cuda((1, 3, 1, 1), (3, 1, 3, 3), torch.float32)
        buf4 = empty_strided_cuda((1, 3, 1, 1), (3, 1, 3, 3), torch.float32)

        4 + 2*(s1 // 2) + 2*(s2 // 2) + (s1 // 2)*(s2 // 2)
        get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_view_2[grid(3)](buf2, buf3, buf4, 34, 64, 64, 3, 1156, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        buf6 = buf1; del buf1

        triton_poi_fused__native_batch_norm_legit_view_3_xnumel = 12 + 6*(s1 // 2) + 6*(s2 // 2) + 3*(s1 // 2)*(s2 // 2)
        get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_view_3[grid(triton_poi_fused__native_batch_norm_legit_view_3_xnumel)](buf2, buf3, buf4, buf6, 34, 34, 64, 1156, 64, 3468, XBLOCK=256, num_warps=4, num_stages=1)
        del buf2
        buf7 = empty_strided_cuda((1, 3, 3 + (s1 // 4), 3 + (s2 // 4)), (27 + 9*(s1 // 4) + 9*(s2 // 4) + 3*(s1 // 4)*(s2 // 4), 9 + 3*(s1 // 4) + 3*(s2 // 4) + (s1 // 4)*(s2 // 4), 3 + (s2 // 4), 1), torch.float32)
        3 + (s2 // 4)
        3 + (s1 // 4)
        9 + 3*(s1 // 4) + 3*(s2 // 4) + (s1 // 4)*(s2 // 4)
        buf8 = empty_strided_cuda((1, 3, 3 + (s1 // 4), 3 + (s2 // 4)), (27 + 9*(s1 // 4) + 9*(s2 // 4) + 3*(s1 // 4)*(s2 // 4), 9 + 3*(s1 // 4) + 3*(s2 // 4) + (s1 // 4)*(s2 // 4), 3 + (s2 // 4), 1), torch.float32)

        triton_poi_fused_copy_4_xnumel = 27 + 9*(s1 // 4) + 9*(s2 // 4) + 3*(s1 // 4)*(s2 // 4)
        get_raw_stream(0)
        triton_poi_fused_copy_4[grid(triton_poi_fused_copy_4_xnumel)](buf6, buf7, buf8, 19, 64, 19, 64, 361, 1083, XBLOCK=128, num_warps=4, num_stages=1)
        del buf6
        buf9 = buf7; del buf7

        triton_poi_fused_5_xnumel = 27 + 9*(s1 // 4) + 9*(s2 // 4) + 3*(s1 // 4)*(s2 // 4)
        get_raw_stream(0)
        triton_poi_fused_5[grid(triton_poi_fused_5_xnumel)](buf8, buf9, 19, 19, 64, 64, 1083, XBLOCK=128, num_warps=4, num_stages=1)
        buf10 = buf4; del buf4
        buf11 = buf3; del buf3

        9 + 3*(s1 // 4) + 3*(s2 // 4) + (s1 // 4)*(s2 // 4)
        get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_view_6[grid(3)](buf9, buf10, buf11, 19, 64, 64, 3, 361, XBLOCK=1, R0_BLOCK=512, num_warps=4, num_stages=1)
        buf13 = buf8; del buf8

        triton_poi_fused__native_batch_norm_legit_view_7_xnumel = 27 + 9*(s1 // 4) + 9*(s2 // 4) + 3*(s1 // 4)*(s2 // 4)
        get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_view_7[grid(triton_poi_fused__native_batch_norm_legit_view_7_xnumel)](buf9, buf10, buf11, buf13, 19, 19, 64, 361, 64, 1083, XBLOCK=128, num_warps=4, num_stages=1)
        del buf10
        del buf11
        del buf9
    return (buf13, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 64
    arg1_1 = 64
    arg2_1 = rand_strided((1, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
