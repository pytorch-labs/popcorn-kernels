
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
from torch._inductor.runtime.triton_helpers import math as tl_math
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_constant_pad_nd_replication_pad3d_0(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, ks6, ks7, ks8, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x6 = ((xindex // ks0) % ks1)
    x1 = ((xindex // ks3) % ks4)
    x0 = (xindex % ks3)
    x2 = ((xindex // ks7) % ks1)
    x3 = xindex // ks8
    x8 = xindex
    tmp0 = (-2) + x6
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = 4 + ks2
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = 4 + ks5
    tmp8 = tmp5 < tmp7
    tmp9 = (-2) + x0
    tmp10 = tmp9 >= tmp1
    tmp11 = 4 + ks6
    tmp12 = tmp9 < tmp11
    tmp13 = tmp2 & tmp4
    tmp14 = tmp13 & tmp6
    tmp15 = tmp14 & tmp8
    tmp16 = tmp15 & tmp10
    tmp17 = tmp16 & tmp12
    tmp18 = (-1) + ((1 + ks2) * ((1 + ks2) <= (((0) * ((0) >= ((-3) + x6)) + ((-3) + x6) * (((-3) + x6) > (0))))) + (((0) * ((0) >= ((-3) + x6)) + ((-3) + x6) * (((-3) + x6) > (0)))) * ((((0) * ((0) >= ((-3) + x6)) + ((-3) + x6) * (((-3) + x6) > (0)))) < (1 + ks2)))
    tmp19 = tl.full([1], 0, tl.int64)
    tmp20 = tmp18 >= tmp19
    tmp21 = tl.broadcast_to(ks2, [XBLOCK])
    tmp22 = tmp18 < tmp21
    tmp23 = (-1) + ((1 + ks5) * ((1 + ks5) <= (((0) * ((0) >= ((-3) + x1)) + ((-3) + x1) * (((-3) + x1) > (0))))) + (((0) * ((0) >= ((-3) + x1)) + ((-3) + x1) * (((-3) + x1) > (0)))) * ((((0) * ((0) >= ((-3) + x1)) + ((-3) + x1) * (((-3) + x1) > (0)))) < (1 + ks5)))
    tmp24 = tmp23 >= tmp19
    tmp25 = tl.broadcast_to(ks5, [XBLOCK])
    tmp26 = tmp23 < tmp25
    tmp27 = (-1) + ((1 + ks6) * ((1 + ks6) <= (((0) * ((0) >= ((-3) + x0)) + ((-3) + x0) * (((-3) + x0) > (0))))) + (((0) * ((0) >= ((-3) + x0)) + ((-3) + x0) * (((-3) + x0) > (0)))) * ((((0) * ((0) >= ((-3) + x0)) + ((-3) + x0) * (((-3) + x0) > (0)))) < (1 + ks6)))
    tmp28 = tmp27 >= tmp19
    tmp29 = tl.broadcast_to(ks6, [XBLOCK])
    tmp30 = tmp27 < tmp29
    tmp31 = tmp20 & tmp22
    tmp32 = tmp31 & tmp24
    tmp33 = tmp32 & tmp26
    tmp34 = tmp33 & tmp28
    tmp35 = tmp34 & tmp30
    tmp36 = tmp35 & tmp17
    tmp37 = tl.load(in_ptr0 + ((-1) + ((-1)*ks6) + ks6*((1 + ks5) * ((1 + ks5) <= (((0) * ((0) >= ((-3) + x1)) + ((-3) + x1) * (((-3) + x1) > (0))))) + (((0) * ((0) >= ((-3) + x1)) + ((-3) + x1) * (((-3) + x1) > (0)))) * ((((0) * ((0) >= ((-3) + x1)) + ((-3) + x1) * (((-3) + x1) > (0)))) < (1 + ks5))) + ((-1)*ks5*ks6) + ks5*ks6*((1 + ks2) * ((1 + ks2) <= (((0) * ((0) >= ((-3) + x2)) + ((-3) + x2) * (((-3) + x2) > (0))))) + (((0) * ((0) >= ((-3) + x2)) + ((-3) + x2) * (((-3) + x2) > (0)))) * ((((0) * ((0) >= ((-3) + x2)) + ((-3) + x2) * (((-3) + x2) > (0)))) < (1 + ks2))) + ks2*ks5*ks6*x3 + ((1 + ks6) * ((1 + ks6) <= (((0) * ((0) >= ((-3) + x0)) + ((-3) + x0) * (((-3) + x0) > (0))))) + (((0) * ((0) >= ((-3) + x0)) + ((-3) + x0) * (((-3) + x0) > (0)))) * ((((0) * ((0) >= ((-3) + x0)) + ((-3) + x0) * (((-3) + x0) > (0)))) < (1 + ks6)))), tmp36 & xmask, eviction_policy='evict_last', other=0.5)
    tmp38 = tl.full(tmp37.shape, 0.25, tmp37.dtype)
    tmp39 = tl.where(tmp17, tmp37, tmp38)
    tl.store(out_ptr0 + (x8), tmp39, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_constant_pad_nd_replication_pad3d_1(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, ks6, ks7, ks8, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x6 = ((xindex // ks0) % ks1)
    x1 = ((xindex // ks3) % ks4)
    x0 = (xindex % ks3)
    x2 = ((xindex // ks7) % ks1)
    x3 = xindex // ks8
    x8 = xindex
    tmp0 = (-3) + x6
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = 12 + ks2
    tmp4 = tmp0 < tmp3
    tmp5 = (-3) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = 12 + ks5
    tmp8 = tmp5 < tmp7
    tmp9 = (-3) + x0
    tmp10 = tmp9 >= tmp1
    tmp11 = 12 + ks6
    tmp12 = tmp9 < tmp11
    tmp13 = tmp2 & tmp4
    tmp14 = tmp13 & tmp6
    tmp15 = tmp14 & tmp8
    tmp16 = tmp15 & tmp10
    tmp17 = tmp16 & tmp12
    tmp18 = tl.load(in_ptr0 + (8*((7 + ks5) * ((7 + ks5) <= (((0) * ((0) >= ((-5) + x1)) + ((-5) + x1) * (((-5) + x1) > (0))))) + (((0) * ((0) >= ((-5) + x1)) + ((-5) + x1) * (((-5) + x1) > (0)))) * ((((0) * ((0) >= ((-5) + x1)) + ((-5) + x1) * (((-5) + x1) > (0)))) < (7 + ks5))) + 64*((7 + ks2) * ((7 + ks2) <= (((0) * ((0) >= ((-5) + x2)) + ((-5) + x2) * (((-5) + x2) > (0))))) + (((0) * ((0) >= ((-5) + x2)) + ((-5) + x2) * (((-5) + x2) > (0)))) * ((((0) * ((0) >= ((-5) + x2)) + ((-5) + x2) * (((-5) + x2) > (0)))) < (7 + ks2))) + 512*x3 + ks6*((7 + ks5) * ((7 + ks5) <= (((0) * ((0) >= ((-5) + x1)) + ((-5) + x1) * (((-5) + x1) > (0))))) + (((0) * ((0) >= ((-5) + x1)) + ((-5) + x1) * (((-5) + x1) > (0)))) * ((((0) * ((0) >= ((-5) + x1)) + ((-5) + x1) * (((-5) + x1) > (0)))) < (7 + ks5))) + 8*ks5*((7 + ks2) * ((7 + ks2) <= (((0) * ((0) >= ((-5) + x2)) + ((-5) + x2) * (((-5) + x2) > (0))))) + (((0) * ((0) >= ((-5) + x2)) + ((-5) + x2) * (((-5) + x2) > (0)))) * ((((0) * ((0) >= ((-5) + x2)) + ((-5) + x2) * (((-5) + x2) > (0)))) < (7 + ks2))) + 8*ks6*((7 + ks2) * ((7 + ks2) <= (((0) * ((0) >= ((-5) + x2)) + ((-5) + x2) * (((-5) + x2) > (0))))) + (((0) * ((0) >= ((-5) + x2)) + ((-5) + x2) * (((-5) + x2) > (0)))) * ((((0) * ((0) >= ((-5) + x2)) + ((-5) + x2) * (((-5) + x2) > (0)))) < (7 + ks2))) + 64*ks2*x3 + 64*ks5*x3 + 64*ks6*x3 + ks5*ks6*((7 + ks2) * ((7 + ks2) <= (((0) * ((0) >= ((-5) + x2)) + ((-5) + x2) * (((-5) + x2) > (0))))) + (((0) * ((0) >= ((-5) + x2)) + ((-5) + x2) * (((-5) + x2) > (0)))) * ((((0) * ((0) >= ((-5) + x2)) + ((-5) + x2) * (((-5) + x2) > (0)))) < (7 + ks2))) + 8*ks2*ks5*x3 + 8*ks2*ks6*x3 + 8*ks5*ks6*x3 + ks2*ks5*ks6*x3 + ((7 + ks6) * ((7 + ks6) <= (((0) * ((0) >= ((-5) + x0)) + ((-5) + x0) * (((-5) + x0) > (0))))) + (((0) * ((0) >= ((-5) + x0)) + ((-5) + x0) * (((-5) + x0) > (0)))) * ((((0) * ((0) >= ((-5) + x0)) + ((-5) + x0) * (((-5) + x0) > (0)))) < (7 + ks6)))), tmp17 & xmask, eviction_policy='evict_last', other=0.75)
    tl.store(out_ptr0 + (x8), tmp18, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_exp_mean_mul_ones_like_sub_2(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, ks6, ks7, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 46
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
        tmp0 = r0_1 + x0*((45 + 5832*ks0 + 324*ks0*ks1 + 324*ks0*ks2 + 324*ks0*ks3 + 18*ks0*ks1*ks2 + 18*ks0*ks1*ks3 + 18*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 46)
        tmp1 = 5832*ks0 + 324*ks0*ks1 + 324*ks0*ks2 + 324*ks0*ks3 + 18*ks0*ks1*ks2 + 18*ks0*ks1*ks3 + 18*ks0*ks2*ks3 + ks0*ks1*ks2*ks3
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (18*((((r0_1 + x0*((45 + 5832*ks0 + 324*ks0*ks1 + 324*ks0*ks2 + 324*ks0*ks3 + 18*ks0*ks1*ks2 + 18*ks0*ks1*ks3 + 18*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 46)) // ks6) % ks7)) + 324*((((r0_1 + x0*((45 + 5832*ks0 + 324*ks0*ks1 + 324*ks0*ks2 + 324*ks0*ks3 + 18*ks0*ks1*ks2 + 18*ks0*ks1*ks3 + 18*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 46)) // ks4) % ks5)) + 5832*((((r0_1 + x0*((45 + 5832*ks0 + 324*ks0*ks1 + 324*ks0*ks2 + 324*ks0*ks3 + 18*ks0*ks1*ks2 + 18*ks0*ks1*ks3 + 18*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 46)) // (5832 + 324*ks1 + 324*ks2 + 324*ks3 + 18*ks1*ks2 + 18*ks1*ks3 + 18*ks2*ks3 + ks1*ks2*ks3)) % ks0)) + ks3*((((r0_1 + x0*((45 + 5832*ks0 + 324*ks0*ks1 + 324*ks0*ks2 + 324*ks0*ks3 + 18*ks0*ks1*ks2 + 18*ks0*ks1*ks3 + 18*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 46)) // ks6) % ks7)) + 18*ks2*((((r0_1 + x0*((45 + 5832*ks0 + 324*ks0*ks1 + 324*ks0*ks2 + 324*ks0*ks3 + 18*ks0*ks1*ks2 + 18*ks0*ks1*ks3 + 18*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 46)) // ks4) % ks5)) + 18*ks3*((((r0_1 + x0*((45 + 5832*ks0 + 324*ks0*ks1 + 324*ks0*ks2 + 324*ks0*ks3 + 18*ks0*ks1*ks2 + 18*ks0*ks1*ks3 + 18*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 46)) // ks4) % ks5)) + 324*ks1*((((r0_1 + x0*((45 + 5832*ks0 + 324*ks0*ks1 + 324*ks0*ks2 + 324*ks0*ks3 + 18*ks0*ks1*ks2 + 18*ks0*ks1*ks3 + 18*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 46)) // (5832 + 324*ks1 + 324*ks2 + 324*ks3 + 18*ks1*ks2 + 18*ks1*ks3 + 18*ks2*ks3 + ks1*ks2*ks3)) % ks0)) + 324*ks2*((((r0_1 + x0*((45 + 5832*ks0 + 324*ks0*ks1 + 324*ks0*ks2 + 324*ks0*ks3 + 18*ks0*ks1*ks2 + 18*ks0*ks1*ks3 + 18*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 46)) // (5832 + 324*ks1 + 324*ks2 + 324*ks3 + 18*ks1*ks2 + 18*ks1*ks3 + 18*ks2*ks3 + ks1*ks2*ks3)) % ks0)) + 324*ks3*((((r0_1 + x0*((45 + 5832*ks0 + 324*ks0*ks1 + 324*ks0*ks2 + 324*ks0*ks3 + 18*ks0*ks1*ks2 + 18*ks0*ks1*ks3 + 18*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 46)) // (5832 + 324*ks1 + 324*ks2 + 324*ks3 + 18*ks1*ks2 + 18*ks1*ks3 + 18*ks2*ks3 + ks1*ks2*ks3)) % ks0)) + ks2*ks3*((((r0_1 + x0*((45 + 5832*ks0 + 324*ks0*ks1 + 324*ks0*ks2 + 324*ks0*ks3 + 18*ks0*ks1*ks2 + 18*ks0*ks1*ks3 + 18*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 46)) // ks4) % ks5)) + 18*ks1*ks2*((((r0_1 + x0*((45 + 5832*ks0 + 324*ks0*ks1 + 324*ks0*ks2 + 324*ks0*ks3 + 18*ks0*ks1*ks2 + 18*ks0*ks1*ks3 + 18*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 46)) // (5832 + 324*ks1 + 324*ks2 + 324*ks3 + 18*ks1*ks2 + 18*ks1*ks3 + 18*ks2*ks3 + ks1*ks2*ks3)) % ks0)) + 18*ks1*ks3*((((r0_1 + x0*((45 + 5832*ks0 + 324*ks0*ks1 + 324*ks0*ks2 + 324*ks0*ks3 + 18*ks0*ks1*ks2 + 18*ks0*ks1*ks3 + 18*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 46)) // (5832 + 324*ks1 + 324*ks2 + 324*ks3 + 18*ks1*ks2 + 18*ks1*ks3 + 18*ks2*ks3 + ks1*ks2*ks3)) % ks0)) + 18*ks2*ks3*((((r0_1 + x0*((45 + 5832*ks0 + 324*ks0*ks1 + 324*ks0*ks2 + 324*ks0*ks3 + 18*ks0*ks1*ks2 + 18*ks0*ks1*ks3 + 18*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 46)) // (5832 + 324*ks1 + 324*ks2 + 324*ks3 + 18*ks1*ks2 + 18*ks1*ks3 + 18*ks2*ks3 + ks1*ks2*ks3)) % ks0)) + ks1*ks2*ks3*((((r0_1 + x0*((45 + 5832*ks0 + 324*ks0*ks1 + 324*ks0*ks2 + 324*ks0*ks3 + 18*ks0*ks1*ks2 + 18*ks0*ks1*ks3 + 18*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 46)) // (5832 + 324*ks1 + 324*ks2 + 324*ks3 + 18*ks1*ks2 + 18*ks1*ks3 + 18*ks2*ks3 + ks1*ks2*ks3)) % ks0)) + (((r0_1 + x0*((45 + 5832*ks0 + 324*ks0*ks1 + 324*ks0*ks2 + 324*ks0*ks3 + 18*ks0*ks1*ks2 + 18*ks0*ks1*ks3 + 18*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 46)) % ks6))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_exp_mean_mul_ones_like_sub_3(in_out_ptr0, in_ptr0, ks0, ks1, ks2, ks3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 46
    R0_BLOCK: tl.constexpr = 64
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
    tmp5 = 5832*ks0 + 324*ks0*ks1 + 324*ks0*ks2 + 324*ks0*ks3 + 18*ks0*ks1*ks2 + 18*ks0*ks1*ks3 + 18*ks0*ks2*ks3 + ks0*ks1*ks2*ks3
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp4 / tmp6
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp7, None)


def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    s3 = arg3_1
    assert_size_stride(arg4_1, (1, s0, s1, s2, s3), (s0*s1*s2*s3, s1*s2*s3, s2*s3, s3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        64 + 8*s2 + 8*s3 + s2*s3
        8 + s1
        8 + s3
        8 + s2
        64 + 8*s2 + 8*s3 + s2*s3
        512 + 64*s1 + 64*s2 + 64*s3 + 8*s1*s2 + 8*s1*s3 + 8*s2*s3 + s1*s2*s3
        buf0 = empty_strided_cuda((1, s0, 8 + s1, 8 + s2, 8 + s3), (512*s0 + 64*s0*s1 + 64*s0*s2 + 64*s0*s3 + 8*s0*s1*s2 + 8*s0*s1*s3 + 8*s0*s2*s3 + s0*s1*s2*s3, 512 + 64*s1 + 64*s2 + 64*s3 + 8*s1*s2 + 8*s1*s3 + 8*s2*s3 + s1*s2*s3, 64 + 8*s2 + 8*s3 + s2*s3, 8 + s3, 1), torch.float32)

        triton_poi_fused_constant_pad_nd_replication_pad3d_0_xnumel = 512*s0 + 64*s0*s1 + 64*s0*s2 + 64*s0*s3 + 8*s0*s1*s2 + 8*s0*s1*s3 + 8*s0*s2*s3 + s0*s1*s2*s3
        get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_replication_pad3d_0[grid(triton_poi_fused_constant_pad_nd_replication_pad3d_0_xnumel)](arg4_1, buf0, 1600, 40, 32, 40, 40, 32, 32, 1600, 64000, 192000, XBLOCK=512, num_warps=8, num_stages=1)
        del arg4_1
        324 + 18*s2 + 18*s3 + s2*s3
        18 + s1
        18 + s3
        18 + s2
        324 + 18*s2 + 18*s3 + s2*s3
        5832 + 324*s1 + 324*s2 + 324*s3 + 18*s1*s2 + 18*s1*s3 + 18*s2*s3 + s1*s2*s3
        buf1 = empty_strided_cuda((1, s0, 18 + s1, 18 + s2, 18 + s3), (5832*s0 + 324*s0*s1 + 324*s0*s2 + 324*s0*s3 + 18*s0*s1*s2 + 18*s0*s1*s3 + 18*s0*s2*s3 + s0*s1*s2*s3, 5832 + 324*s1 + 324*s2 + 324*s3 + 18*s1*s2 + 18*s1*s3 + 18*s2*s3 + s1*s2*s3, 324 + 18*s2 + 18*s3 + s2*s3, 18 + s3, 1), torch.float32)

        triton_poi_fused_constant_pad_nd_replication_pad3d_1_xnumel = 5832*s0 + 324*s0*s1 + 324*s0*s2 + 324*s0*s3 + 18*s0*s1*s2 + 18*s0*s1*s3 + 18*s0*s2*s3 + s0*s1*s2*s3
        get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_replication_pad3d_1[grid(triton_poi_fused_constant_pad_nd_replication_pad3d_1_xnumel)](buf0, buf1, 2500, 50, 32, 50, 50, 32, 32, 2500, 125000, 375000, XBLOCK=512, num_warps=8, num_stages=1)
        del buf0
        buf2 = empty_strided_cuda((46, ), (1, ), torch.float32)

        (45 + 5832*s0 + 324*s0*s1 + 324*s0*s2 + 324*s0*s3 + 18*s0*s1*s2 + 18*s0*s1*s3 + 18*s0*s2*s3 + s0*s1*s2*s3) // 46
        get_raw_stream(0)
        triton_red_fused_exp_mean_mul_ones_like_sub_2[grid(46)](buf1, buf2, 3, 32, 32, 32, 2500, 50, 50, 50, 46, 8153, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del buf1
        buf3 = empty_strided_cuda((), (), torch.float32)
        buf4 = buf3; del buf3

        get_raw_stream(0)
        triton_per_fused_exp_mean_mul_ones_like_sub_3[grid(1)](buf4, buf2, 3, 32, 32, 32, 1, 46, XBLOCK=1, num_warps=2, num_stages=1)
        del buf2
    return (buf4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 3
    arg1_1 = 32
    arg2_1 = 32
    arg3_1 = 32
    arg4_1 = rand_strided((1, 3, 32, 32, 32), (98304, 32768, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
