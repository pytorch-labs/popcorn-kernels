
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
def triton_poi_fused_max_unpool2d_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
def triton_poi_fused_abs_le_max_pool2d_with_indices_max_unpool2d_scalar_tensor_where_1(in_ptr0, out_ptr1, ks0, ks1, ks2, ks3, ks4, ks5, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks1)
    x2 = xindex // ks2
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 2*ks4*x1 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (1 + 2*x0 + 2*ks4*x1 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (ks4 + 2*x0 + 2*ks4*x1 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (1 + ks4 + 2*x0 + 2*ks4*x1 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr0 + (2*((x3 % ks0)) + 2*ks4*(((x3 // ks0) % ks1)) + ks3*ks4*(x3 // ks2)), xmask, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr0 + (1 + 2*((x3 % ks0)) + 2*ks4*(((x3 // ks0) % ks1)) + ks3*ks4*(x3 // ks2)), xmask, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr0 + (ks4 + 2*((x3 % ks0)) + 2*ks4*(((x3 // ks0) % ks1)) + ks3*ks4*(x3 // ks2)), xmask, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr0 + (1 + ks4 + 2*((x3 % ks0)) + 2*ks4*(((x3 // ks0) % ks1)) + ks3*ks4*(x3 // ks2)), xmask, eviction_policy='evict_last')
    tmp1 = tl_math.abs(tmp0)
    tmp2 = 0.5
    tmp3 = tmp1 <= tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp3, tmp4, tmp0)
    tmp7 = tl_math.abs(tmp6)
    tmp8 = tmp7 <= tmp2
    tmp9 = tl.where(tmp8, tmp4, tmp6)
    tmp10 = tmp9 > tmp5
    tmp11 = tl.full([1], 1, tl.int8)
    tmp12 = tl.full([1], 0, tl.int8)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = triton_helpers.maximum(tmp9, tmp5)
    tmp16 = tl_math.abs(tmp15)
    tmp17 = tmp16 <= tmp2
    tmp18 = tl.where(tmp17, tmp4, tmp15)
    tmp19 = tmp18 > tmp14
    tmp20 = tl.full([1], 2, tl.int8)
    tmp21 = tl.where(tmp19, tmp20, tmp13)
    tmp22 = triton_helpers.maximum(tmp18, tmp14)
    tmp24 = tl_math.abs(tmp23)
    tmp25 = tmp24 <= tmp2
    tmp26 = tl.where(tmp25, tmp4, tmp23)
    tmp27 = tmp26 > tmp22
    tmp28 = tl.full([1], 3, tl.int8)
    tmp29 = tl.where(tmp27, tmp28, tmp21)
    triton_helpers.maximum(tmp26, tmp22)
    tmp31 = tl.full([1], 2, tl.int32)
    tmp32 = tl.where((tmp29 < 0) != (tmp31 < 0), tl.where(tmp29 % tmp31 != 0, tmp29 // tmp31 - 1, tmp29 // tmp31), tmp29 // tmp31)
    tmp33 = tmp32 * tmp31
    tmp34 = tmp29 - tmp33
    tmp35 = 2*(((x3 // ks0) % ks1))
    tmp36 = tmp35 + tmp32
    tmp37 = 2*((x3 % ks0))
    tmp38 = tmp37 + tmp34
    tmp39 = ks4
    tmp40 = tmp36 * tmp39
    tmp41 = tmp40 + tmp38
    tmp42 = ks3*ks4*(x3 // ks2)
    tmp43 = tmp41 + tmp42
    tmp44 = ks3*ks4*ks5
    tmp45 = tmp43 + tmp44
    tmp46 = tmp43 < 0
    tmp47 = tl.where(tmp46, tmp45, tmp43)
    tl.device_assert(((0 <= tmp47) & (tmp47 < ks3*ks4*ks5)) | ~(xmask), "index out of bounds: 0 <= tmp47 < ks3*ks4*ks5")
    tmp50 = tl_math.abs(tmp49)
    tmp51 = tmp50 <= tmp2
    tmp52 = tl.where(tmp51, tmp4, tmp49)
    tmp54 = tl_math.abs(tmp53)
    tmp55 = tmp54 <= tmp2
    tmp56 = tl.where(tmp55, tmp4, tmp53)
    tmp57 = triton_helpers.maximum(tmp56, tmp52)
    tmp59 = tl_math.abs(tmp58)
    tmp60 = tmp59 <= tmp2
    tmp61 = tl.where(tmp60, tmp4, tmp58)
    tmp62 = triton_helpers.maximum(tmp61, tmp57)
    tmp64 = tl_math.abs(tmp63)
    tmp65 = tmp64 <= tmp2
    tmp66 = tl.where(tmp65, tmp4, tmp63)
    tmp67 = triton_helpers.maximum(tmp66, tmp62)
    tl.store(out_ptr1 + (tl.broadcast_to((tmp47 % (ks3*ks4*ks5)), [XBLOCK])), tmp67, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__log_softmax_2(in_ptr0, out_ptr0, out_ptr1, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + ks0*x1 + ks0*ks2*((((x0 + ks0*x1 + ks0*ks2*r0_2) // (ks0*ks2)) % ks1))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = triton_helpers.maximum(_tmp2, tmp1)
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = triton_helpers.max2(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_2 = r0_index
        tmp4 = tl.load(in_ptr0 + (x0 + ks0*x1 + ks0*ks2*((((x0 + ks0*x1 + ks0*ks2*r0_2) // (ks0*ks2)) % ks1))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp4 - tmp2
        tmp6 = tl_math.exp(tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp8, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_clamp_min_fill_mean_ne_nll_loss2d_forward_randint_sub_where_zeros_like_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, load_seed_offset, ks1, ks2, ks3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    _tmp22 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp26 = tl.full([XBLOCK, R0_BLOCK], 0, tl.int64)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_2 = r0_index
        r0_0 = (r0_index % ks2)
        r0_1 = r0_index // ks2
        tmp13 = tl.load(in_ptr2 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp0 = tl.load(in_ptr0 + load_seed_offset)
        tmp1 = r0_2
        tmp2 = tl.full([1, 1], 0, tl.int64)
        tmp3 = ks1
        tmp4 = triton_helpers.randint64(tmp0, (tmp1).to(tl.uint32), tmp2, tmp3)
        tmp5 = tl.full([1, 1], -100, tl.int64)
        tmp6 = tmp4 != tmp5
        tmp7 = tl.where(tmp6, tmp4, tmp2)
        tmp8 = tmp7 + tmp3
        tmp9 = tmp7 < 0
        tmp10 = tl.where(tmp9, tmp8, tmp7)
        tl.device_assert(((0 <= tmp10) & (tmp10 < ks1)) | ~(r0_mask), "index out of bounds: 0 <= tmp10 < ks1")
        tmp12 = tl.load(in_ptr1 + (r0_0 + ks2*r0_1 + ks2*ks3*((((r0_0 + ks2*r0_1 + ks2*ks3*tmp10) // (ks2*ks3)) % ks1))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp14 = tmp12 - tmp13
        tmp16 = tl_math.log(tmp15)
        tmp17 = tmp14 - tmp16
        tmp18 = -tmp17
        tmp19 = 0.0
        tmp20 = tl.where(tmp6, tmp18, tmp19)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(r0_mask, tmp23, _tmp22)
        tmp24 = tmp6.to(tl.int64)
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(r0_mask, tmp27, _tmp26)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tmp28 = tmp26.to(tl.float32)
    tmp29 = tmp22 / tmp28
    tmp30 = 1.0
    tmp31 = tmp30 - tmp29
    tmp32 = 0.0
    tmp33 = triton_helpers.maximum(tmp31, tmp32)
    tmp34 = tl.full([1, 1], False, tl.int1)
    tmp35 = tl.where(tmp34, tmp33, tmp32)
    tmp36 = tl.full([1, 1], True, tl.int1)
    tmp37 = tl.where(tmp36, tmp29, tmp32)
    tmp38 = tmp35 + tmp37
    tmp39 = tmp38 / tmp30
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp39, None)


def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf2 = empty_strided_cuda((1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1), torch.float32)

        triton_poi_fused_max_unpool2d_0_xnumel = s0*s1*s2
        get_raw_stream(0)
        triton_poi_fused_max_unpool2d_0[grid(triton_poi_fused_max_unpool2d_0_xnumel)](buf2, 12288, XBLOCK=256, num_warps=4, num_stages=1)
        s2 // 2
        s1 // 2
        (s1 // 2)*(s2 // 2)

        triton_poi_fused_abs_le_max_pool2d_with_indices_max_unpool2d_scalar_tensor_where_1_xnumel = s0*(s1 // 2)*(s2 // 2)
        get_raw_stream(0)
        triton_poi_fused_abs_le_max_pool2d_with_indices_max_unpool2d_scalar_tensor_where_1[grid(triton_poi_fused_abs_le_max_pool2d_with_indices_max_unpool2d_scalar_tensor_where_1_xnumel)](arg3_1, buf2, 32, 32, 1024, 64, 64, 3, 3072, XBLOCK=256, num_warps=4, num_stages=1)
        del arg3_1
        buf1 = empty_strided_cuda((1, ), (1, ), torch.int64)

        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf1)
        buf4 = empty_strided_cuda((1, 1, s1, s2), (s1*s2, s1*s2, s2, 1), torch.float32)
        buf5 = empty_strided_cuda((1, 1, s1, s2), (s1*s2, s1*s2, s2, 1), torch.float32)

        triton_red_fused__log_softmax_2_xnumel = s1*s2
        get_raw_stream(0)
        triton_red_fused__log_softmax_2[grid(triton_red_fused__log_softmax_2_xnumel)](buf2, buf4, buf5, 64, 3, 64, 4096, 3, XBLOCK=128, R0_BLOCK=4, num_warps=4, num_stages=1)
        buf6 = empty_strided_cuda((), (), torch.float32)
        buf8 = buf6; del buf6

        s1*s2
        get_raw_stream(0)
        triton_red_fused_add_clamp_min_fill_mean_ne_nll_loss2d_forward_randint_sub_where_zeros_like_3[grid(1)](buf8, buf1, buf2, buf4, buf5, 0, 3, 64, 64, 1, 4096, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del buf1
        del buf2
        del buf4
        del buf5
    return (buf8, )


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
