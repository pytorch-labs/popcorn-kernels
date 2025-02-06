
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
def triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_replication_pad2d_sub_view_0(in_out_ptr1, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 72) % 72)
    x0 = (xindex % 72)
    x2 = xindex // 5184
    x4 = xindex
    tmp0 = 4.0
    tmp1 = 32.0
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float64)
    tmp4 = tl.full([1], -1.0, tl.float64)
    tmp5 = tmp4 + tmp3
    tmp6 = 2.0
    tmp7 = tmp6 * tmp1
    tmp8 = 8.0
    tmp9 = tmp8 + tmp7
    tmp10 = tmp9.to(tl.float64)
    tmp11 = tmp4 + tmp10
    tmp12 = tmp5 / tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp14 = x1
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp15 * tmp13
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = tmp18.to(tl.int32)
    tmp20 = tl.full([1], 1, tl.int64)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full([1], 35, tl.int64)
    tmp23 = triton_helpers.minimum(tmp21, tmp22)
    tmp24 = x0
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp25 * tmp13
    tmp27 = triton_helpers.maximum(tmp26, tmp17)
    tmp28 = tmp27.to(tl.int32)
    tmp29 = tl.load(in_ptr0 + (32*((31) * ((31) <= (((0) * ((0) >= ((-2) + tmp23)) + ((-2) + tmp23) * (((-2) + tmp23) > (0))))) + (((0) * ((0) >= ((-2) + tmp23)) + ((-2) + tmp23) * (((-2) + tmp23) > (0)))) * ((((0) * ((0) >= ((-2) + tmp23)) + ((-2) + tmp23) * (((-2) + tmp23) > (0)))) < (31))) + 1024*x2 + ((31) * ((31) <= (((0) * ((0) >= ((-2) + tmp28)) + ((-2) + tmp28) * (((-2) + tmp28) > (0))))) + (((0) * ((0) >= ((-2) + tmp28)) + ((-2) + tmp28) * (((-2) + tmp28) > (0)))) * ((((0) * ((0) >= ((-2) + tmp28)) + ((-2) + tmp28) * (((-2) + tmp28) > (0)))) < (31)))), xmask, eviction_policy='evict_last')
    tmp30 = tmp28 + tmp20
    tmp31 = triton_helpers.minimum(tmp30, tmp22)
    tmp32 = tl.load(in_ptr0 + (32*((31) * ((31) <= (((0) * ((0) >= ((-2) + tmp23)) + ((-2) + tmp23) * (((-2) + tmp23) > (0))))) + (((0) * ((0) >= ((-2) + tmp23)) + ((-2) + tmp23) * (((-2) + tmp23) > (0)))) * ((((0) * ((0) >= ((-2) + tmp23)) + ((-2) + tmp23) * (((-2) + tmp23) > (0)))) < (31))) + 1024*x2 + ((31) * ((31) <= (((0) * ((0) >= ((-2) + tmp31)) + ((-2) + tmp31) * (((-2) + tmp31) > (0))))) + (((0) * ((0) >= ((-2) + tmp31)) + ((-2) + tmp31) * (((-2) + tmp31) > (0)))) * ((((0) * ((0) >= ((-2) + tmp31)) + ((-2) + tmp31) * (((-2) + tmp31) > (0)))) < (31)))), xmask, eviction_policy='evict_last')
    tmp33 = tmp32 - tmp29
    tmp34 = tmp28.to(tl.float32)
    tmp35 = tmp27 - tmp34
    tmp36 = triton_helpers.maximum(tmp35, tmp17)
    tmp37 = 1.0
    tmp38 = triton_helpers.minimum(tmp36, tmp37)
    tmp39 = tmp33 * tmp38
    tmp40 = tmp29 + tmp39
    tmp41 = tl.load(in_ptr0 + (32*((31) * ((31) <= (((0) * ((0) >= ((-2) + tmp19)) + ((-2) + tmp19) * (((-2) + tmp19) > (0))))) + (((0) * ((0) >= ((-2) + tmp19)) + ((-2) + tmp19) * (((-2) + tmp19) > (0)))) * ((((0) * ((0) >= ((-2) + tmp19)) + ((-2) + tmp19) * (((-2) + tmp19) > (0)))) < (31))) + 1024*x2 + ((31) * ((31) <= (((0) * ((0) >= ((-2) + tmp28)) + ((-2) + tmp28) * (((-2) + tmp28) > (0))))) + (((0) * ((0) >= ((-2) + tmp28)) + ((-2) + tmp28) * (((-2) + tmp28) > (0)))) * ((((0) * ((0) >= ((-2) + tmp28)) + ((-2) + tmp28) * (((-2) + tmp28) > (0)))) < (31)))), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr0 + (32*((31) * ((31) <= (((0) * ((0) >= ((-2) + tmp19)) + ((-2) + tmp19) * (((-2) + tmp19) > (0))))) + (((0) * ((0) >= ((-2) + tmp19)) + ((-2) + tmp19) * (((-2) + tmp19) > (0)))) * ((((0) * ((0) >= ((-2) + tmp19)) + ((-2) + tmp19) * (((-2) + tmp19) > (0)))) < (31))) + 1024*x2 + ((31) * ((31) <= (((0) * ((0) >= ((-2) + tmp31)) + ((-2) + tmp31) * (((-2) + tmp31) > (0))))) + (((0) * ((0) >= ((-2) + tmp31)) + ((-2) + tmp31) * (((-2) + tmp31) > (0)))) * ((((0) * ((0) >= ((-2) + tmp31)) + ((-2) + tmp31) * (((-2) + tmp31) > (0)))) < (31)))), xmask, eviction_policy='evict_last')
    tmp43 = tmp42 - tmp41
    tmp44 = tmp43 * tmp38
    tmp45 = tmp41 + tmp44
    tmp46 = tmp40 - tmp45
    tmp47 = tmp19.to(tl.float32)
    tmp48 = tmp18 - tmp47
    tmp49 = triton_helpers.maximum(tmp48, tmp17)
    tmp50 = triton_helpers.minimum(tmp49, tmp37)
    tmp51 = tmp46 * tmp50
    tmp52 = tmp45 + tmp51
    tl.store(in_out_ptr1 + (x4), tmp52, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_binary_cross_entropy_celu_constant_pad_nd_sub_1(in_out_ptr0, in_ptr0, out_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    R0_BLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_mask = r0_index < r0_numel
    r0_1 = ((r0_index // 7) % 7)
    r0_0 = (r0_index % 7)
    r0_2 = r0_index // 49
    r0_3 = r0_index
    tmp0 = (-1) + r0_1
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 5, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + r0_0
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (tl.broadcast_to((-6) + r0_0 + 5*r0_1 + 25*r0_2, [XBLOCK, R0_BLOCK])), r0_mask & tmp10, other=0.5)
    tmp12 = 0.0
    tmp13 = tmp11 > tmp12
    tmp14 = libdevice.expm1(tmp11)
    tmp15 = tl.where(tmp13, tmp11, tmp14)
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = -tmp16
    tmp18 = libdevice.log1p(tmp17)
    tmp19 = -100.0
    tmp20 = triton_helpers.maximum(tmp18, tmp19)
    tmp21 = tmp12 * tmp20
    tmp22 = tl_math.log(tmp16)
    tmp23 = triton_helpers.maximum(tmp22, tmp19)
    tmp24 = tmp21 - tmp23
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
    tmp27 = tl.where(r0_mask, tmp25, 0)
    tmp28 = tl.sum(tmp27, 1)[:, None]
    tmp29 = 49*ks0
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp28 / tmp30
    tl.store(out_ptr0 + (tl.broadcast_to(r0_3, [XBLOCK, R0_BLOCK])), tmp16, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp31, None)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_clamp_min_mean_norm_ones_like_sub_2(in_out_ptr1, in_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    _tmp63 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (7*r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr0 + (1 + 7*r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.load(in_ptr0 + (2 + 7*r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.load(in_ptr0 + (3 + 7*r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp21 = tl.load(in_ptr0 + (4 + 7*r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp26 = tl.load(in_ptr0 + (5 + 7*r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp31 = tl.load(in_ptr0 + (6 + 7*r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = 1.0
        tmp2 = tmp0 - tmp1
        tmp3 = 1e-06
        tmp4 = tmp2 + tmp3
        tmp5 = tmp4 * tmp4
        tmp7 = tmp6 - tmp1
        tmp8 = tmp7 + tmp3
        tmp9 = tmp8 * tmp8
        tmp10 = tmp5 + tmp9
        tmp12 = tmp11 - tmp1
        tmp13 = tmp12 + tmp3
        tmp14 = tmp13 * tmp13
        tmp15 = tmp10 + tmp14
        tmp17 = tmp16 - tmp1
        tmp18 = tmp17 + tmp3
        tmp19 = tmp18 * tmp18
        tmp20 = tmp15 + tmp19
        tmp22 = tmp21 - tmp1
        tmp23 = tmp22 + tmp3
        tmp24 = tmp23 * tmp23
        tmp25 = tmp20 + tmp24
        tmp27 = tmp26 - tmp1
        tmp28 = tmp27 + tmp3
        tmp29 = tmp28 * tmp28
        tmp30 = tmp25 + tmp29
        tmp32 = tmp31 - tmp1
        tmp33 = tmp32 + tmp3
        tmp34 = tmp33 * tmp33
        tmp35 = tmp30 + tmp34
        tmp36 = libdevice.sqrt(tmp35)
        tmp37 = tmp36 + tmp1
        tmp38 = tmp0 + tmp3
        tmp39 = tmp38 * tmp38
        tmp40 = tmp6 + tmp3
        tmp41 = tmp40 * tmp40
        tmp42 = tmp39 + tmp41
        tmp43 = tmp11 + tmp3
        tmp44 = tmp43 * tmp43
        tmp45 = tmp42 + tmp44
        tmp46 = tmp16 + tmp3
        tmp47 = tmp46 * tmp46
        tmp48 = tmp45 + tmp47
        tmp49 = tmp21 + tmp3
        tmp50 = tmp49 * tmp49
        tmp51 = tmp48 + tmp50
        tmp52 = tmp26 + tmp3
        tmp53 = tmp52 * tmp52
        tmp54 = tmp51 + tmp53
        tmp55 = tmp31 + tmp3
        tmp56 = tmp55 * tmp55
        tmp57 = tmp54 + tmp56
        tmp58 = libdevice.sqrt(tmp57)
        tmp59 = tmp37 - tmp58
        tmp60 = 0.0
        tmp61 = triton_helpers.maximum(tmp59, tmp60)
        tmp62 = tl.broadcast_to(tmp61, [XBLOCK, R0_BLOCK])
        tmp64 = _tmp63 + tmp62
        _tmp63 = tl.where(r0_mask, tmp64, _tmp63)
    tmp63 = tl.sum(_tmp63, 1)[:, None]
    tmp65 = 7*ks0
    tmp66 = tmp65.to(tl.float32)
    tmp67 = tmp63 / tmp66
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp67, None)


def call(args):
    arg0_1, _arg1_1, _arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    assert_size_stride(arg3_1, (1, s0, 32, 32), (1024*s0, 1024, 32, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf3 = empty_strided_cuda((1, s0, 72, 72), (5184*s0, 5184, 72, 1), torch.float32)
        buf5 = buf3; del buf3
        buf7 = buf5; del buf5

        triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_replication_pad2d_sub_view_0_xnumel = 5184*s0
        get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_replication_pad2d_sub_view_0[grid(triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_replication_pad2d_sub_view_0_xnumel)](buf7, arg3_1, 15552, XBLOCK=256, num_warps=4, num_stages=1)
        del arg3_1

        buf8 = torch.ops.aten._adaptive_avg_pool2d.default(buf7, [5, 5])
        del buf7
        buf9 = buf8
        del buf8
        buf10 = empty_strided_cuda((1, s0, 7, 7), (49*s0, 49, 7, 1), torch.float32)
        buf11 = empty_strided_cuda((), (), torch.float32)
        buf15 = buf11; del buf11

        49*s0
        get_raw_stream(0)
        triton_per_fused_binary_cross_entropy_celu_constant_pad_nd_sub_1[grid(1)](buf15, buf9, buf10, 3, 1, 147, XBLOCK=1, num_warps=2, num_stages=1)
        del buf9
        buf14 = empty_strided_cuda((), (), torch.float32)
        buf16 = buf14; del buf14

        7*s0
        get_raw_stream(0)
        triton_red_fused_add_clamp_min_mean_norm_ones_like_sub_2[grid(1)](buf16, buf10, 3, 1, 21, XBLOCK=1, R0_BLOCK=32, num_warps=2, num_stages=1)
    return (buf10, buf15, buf16, )


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
