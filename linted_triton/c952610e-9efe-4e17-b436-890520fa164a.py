
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
def triton_red_fused__log_softmax_0(in_ptr0, out_ptr2, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    _tmp28 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + ks0*ks1*ks2*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = -1.0
        tmp2 = triton_helpers.maximum(tmp0, tmp1)
        tmp3 = 1.0
        tmp4 = triton_helpers.minimum(tmp2, tmp3)
        tmp5 = triton_helpers.maximum(tmp4, tmp1)
        tmp6 = triton_helpers.minimum(tmp5, tmp3)
        tmp7 = tmp6 * tmp3
        tmp8 = 20.0
        tmp9 = tmp7 > tmp8
        tmp10 = tl_math.exp(tmp7)
        tmp11 = libdevice.log1p(tmp10)
        tmp12 = tmp11 * tmp3
        tmp13 = tl.where(tmp9, tmp6, tmp12)
        tmp14 = tmp13 * tmp3
        tmp15 = tmp14 > tmp8
        tmp16 = tl_math.exp(tmp14)
        tmp17 = libdevice.log1p(tmp16)
        tmp18 = tmp17 * tmp3
        tmp19 = tl.where(tmp15, tmp13, tmp18)
        tmp20 = 0.0
        tmp21 = triton_helpers.minimum(tmp20, tmp19)
        tmp22 = tl_math.abs(tmp19)
        tmp23 = -tmp22
        tmp24 = tl_math.exp(tmp23)
        tmp25 = libdevice.log1p(tmp24)
        tmp26 = tmp21 - tmp25
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, R0_BLOCK])
        tmp29 = triton_helpers.maximum(_tmp28, tmp27)
        _tmp28 = tl.where(r0_mask & xmask, tmp29, _tmp28)
    tmp28 = triton_helpers.max2(_tmp28, 1)[:, None]
    _tmp60 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp30 = tl.load(in_ptr0 + (r0_1 + ks0*ks1*ks2*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp31 = -1.0
        tmp32 = triton_helpers.maximum(tmp30, tmp31)
        tmp33 = 1.0
        tmp34 = triton_helpers.minimum(tmp32, tmp33)
        tmp35 = triton_helpers.maximum(tmp34, tmp31)
        tmp36 = triton_helpers.minimum(tmp35, tmp33)
        tmp37 = tmp36 * tmp33
        tmp38 = 20.0
        tmp39 = tmp37 > tmp38
        tmp40 = tl_math.exp(tmp37)
        tmp41 = libdevice.log1p(tmp40)
        tmp42 = tmp41 * tmp33
        tmp43 = tl.where(tmp39, tmp36, tmp42)
        tmp44 = tmp43 * tmp33
        tmp45 = tmp44 > tmp38
        tmp46 = tl_math.exp(tmp44)
        tmp47 = libdevice.log1p(tmp46)
        tmp48 = tmp47 * tmp33
        tmp49 = tl.where(tmp45, tmp43, tmp48)
        tmp50 = 0.0
        tmp51 = triton_helpers.minimum(tmp50, tmp49)
        tmp52 = tl_math.abs(tmp49)
        tmp53 = -tmp52
        tmp54 = tl_math.exp(tmp53)
        tmp55 = libdevice.log1p(tmp54)
        tmp56 = tmp51 - tmp55
        tmp57 = tmp56 - tmp28
        tmp58 = tl_math.exp(tmp57)
        tmp59 = tl.broadcast_to(tmp58, [XBLOCK, R0_BLOCK])
        tmp61 = _tmp60 + tmp59
        _tmp60 = tl.where(r0_mask & xmask, tmp61, _tmp60)
    tmp60 = tl.sum(_tmp60, 1)[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp62 = tl.load(in_ptr0 + (r0_1 + ks0*ks1*ks2*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp63 = -1.0
        tmp64 = triton_helpers.maximum(tmp62, tmp63)
        tmp65 = 1.0
        tmp66 = triton_helpers.minimum(tmp64, tmp65)
        tmp67 = triton_helpers.maximum(tmp66, tmp63)
        tmp68 = triton_helpers.minimum(tmp67, tmp65)
        tmp69 = tmp68 * tmp65
        tmp70 = 20.0
        tmp71 = tmp69 > tmp70
        tmp72 = tl_math.exp(tmp69)
        tmp73 = libdevice.log1p(tmp72)
        tmp74 = tmp73 * tmp65
        tmp75 = tl.where(tmp71, tmp68, tmp74)
        tmp76 = tmp75 * tmp65
        tmp77 = tmp76 > tmp70
        tmp78 = tl_math.exp(tmp76)
        tmp79 = libdevice.log1p(tmp78)
        tmp80 = tmp79 * tmp65
        tmp81 = tl.where(tmp77, tmp75, tmp80)
        tmp82 = 0.0
        tmp83 = triton_helpers.minimum(tmp82, tmp81)
        tmp84 = tl_math.abs(tmp81)
        tmp85 = -tmp84
        tmp86 = tl_math.exp(tmp85)
        tmp87 = libdevice.log1p(tmp86)
        tmp88 = tmp83 - tmp87
        tmp89 = tmp88 - tmp28
        tmp90 = tl_math.log(tmp60)
        tmp91 = tmp89 - tmp90
        tl.store(out_ptr2 + (r0_1 + ks0*ks1*ks2*x0), tmp91, r0_mask & xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_flip_norm_sub_1(in_ptr0, out_ptr0, out_ptr1, ks0, ks1, ks2, ks3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp14 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + ks0*ks1*ks2*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr0 + (r0_1 + ks0*ks1*ks2*x0 + ks0*ks1*ks2*(ks3 // 2)), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr0 + (r0_1 + ((-1)*ks0*ks1*ks2) + ks0*ks1*ks2*ks3 + ((-1)*ks0*ks1*ks2*x0)), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 - tmp1
        tmp3 = 1e-06
        tmp4 = tmp2 + tmp3
        tmp5 = tmp4 * tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask & xmask, tmp8, _tmp7)
        tmp10 = tmp0 - tmp9
        tmp11 = tmp10 + tmp3
        tmp12 = tmp11 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(r0_mask & xmask, tmp15, _tmp14)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp7, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_clamp_min_mean_norm_sub_2(in_out_ptr0, in_ptr0, in_ptr1, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    tmp12 = ks0 // 2
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp10 / tmp13
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp14, None)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_mse_loss_3(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp0 = r0_1 + x0*(triton_helpers.div_floor_integer(1 + ks1*ks2*ks3*(ks0 // 2),  2))
        tmp1 = ks1*ks2*ks3*(ks0 // 2)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (((r0_1 + x0*(triton_helpers.div_floor_integer(1 + ks1*ks2*ks3*(ks0 // 2),  2))) % (ks1*ks2*ks3*(ks0 // 2)))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr0 + (ks1*ks2*ks3*(ks0 // 2) + (((r0_1 + x0*(triton_helpers.div_floor_integer(1 + ks1*ks2*ks3*(ks0 // 2),  2))) % (ks1*ks2*ks3*(ks0 // 2))))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 - tmp4
        tmp6 = tmp5 * tmp5
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp10, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_mse_loss_4(in_out_ptr0, in_ptr0, ks0, ks1, ks2, ks3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    R0_BLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None]
    tmp4 = ks1*ks2*ks3*(ks0 // 2)
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp6, None)


def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    s3 = arg3_1
    assert_size_stride(arg4_1, (s0, s1, s2, s3), (s1*s2*s3, s2*s3, s3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf2 = empty_strided_cuda((s0, s1*s2*s3), (s1*s2*s3, 1), torch.float32)

        s1*s2*s3
        get_raw_stream(0)
        triton_red_fused__log_softmax_0[grid(s0)](arg4_1, buf2, 3, 32, 32, 10, 3072, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del arg4_1
        buf3 = empty_strided_cuda((s0 // 2, ), (1, ), torch.float32)
        buf4 = empty_strided_cuda((s0 // 2, ), (1, ), torch.float32)

        triton_red_fused_add_flip_norm_sub_1_xnumel = s0 // 2
        s1*s2*s3
        get_raw_stream(0)
        triton_red_fused_add_flip_norm_sub_1[grid(triton_red_fused_add_flip_norm_sub_1_xnumel)](buf2, buf3, buf4, 3, 32, 32, 10, 5, 3072, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        buf5 = empty_strided_cuda((), (), torch.float32)
        buf8 = buf5; del buf5

        s0 // 2
        get_raw_stream(0)
        triton_red_fused_add_clamp_min_mean_norm_sub_2[grid(1)](buf8, buf3, buf4, 10, 1, 5, XBLOCK=1, R0_BLOCK=8, num_warps=2, num_stages=1)
        del buf3
        del buf4
        buf6 = empty_strided_cuda((2, ), (1, ), torch.float32)

        (1 + s1*s2*s3*(s0 // 2)) // 2
        get_raw_stream(0)
        triton_red_fused_mse_loss_3[grid(2)](buf2, buf6, 10, 3, 32, 32, 2, 7680, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        buf7 = empty_strided_cuda((), (), torch.float32)
        buf9 = buf7; del buf7

        get_raw_stream(0)
        triton_per_fused_mse_loss_4[grid(1)](buf9, buf6, 10, 3, 32, 32, 1, 2, XBLOCK=1, num_warps=2, num_stages=1)
        del buf6
    return (buf2, buf8, buf9, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 10
    arg1_1 = 3
    arg2_1 = 32
    arg3_1 = 32
    arg4_1 = rand_strided((10, 3, 32, 32), (3072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
