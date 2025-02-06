
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
def triton_poi_fused_bernoulli_0(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
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
def triton_poi_fused__to_copy_abs_add_avg_pool3d_bernoulli_mish_mul_pow_relu_sign_1(in_out_ptr0, in_ptr0, in_ptr1, ks0, ks1, ks2, ks3, ks4, ks5, ks6, ks7, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks1)
    x2 = ((xindex // ks2) % ks3)
    x3 = xindex // ks4
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 2*ks7*x1 + 2*ks6*ks7*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 2*x0 + 2*ks7*x1 + 2*ks6*ks7*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (ks7 + 2*x0 + 2*ks7*x1 + 2*ks6*ks7*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (1 + ks7 + 2*x0 + 2*ks7*x1 + 2*ks6*ks7*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (2*x0 + ks6*ks7 + 2*ks7*x1 + 2*ks6*ks7*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (1 + 2*x0 + ks6*ks7 + 2*ks7*x1 + 2*ks6*ks7*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (ks7 + 2*x0 + ks6*ks7 + 2*ks7*x1 + 2*ks6*ks7*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr0 + (1 + ks7 + 2*x0 + ks6*ks7 + 2*ks7*x1 + 2*ks6*ks7*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 * tmp0
    tmp3 = tmp2 * tmp2
    tmp4 = tmp3 + tmp1
    tmp6 = tmp5 * tmp5
    tmp7 = tmp6 + tmp4
    tmp9 = tmp8 * tmp8
    tmp10 = tmp9 + tmp7
    tmp12 = tmp11 * tmp11
    tmp13 = tmp12 + tmp10
    tmp15 = tmp14 * tmp14
    tmp16 = tmp15 + tmp13
    tmp18 = tmp17 * tmp17
    tmp19 = tmp18 + tmp16
    tmp21 = tmp20 * tmp20
    tmp22 = tmp21 + tmp19
    tmp23 = 0.125
    tmp24 = tmp22 * tmp23
    tmp25 = tl.full([1], 0, tl.int32)
    tmp26 = tmp25 < tmp24
    tmp27 = tmp26.to(tl.int8)
    tmp28 = tmp24 < tmp25
    tmp29 = tmp28.to(tl.int8)
    tmp30 = tmp27 - tmp29
    tmp31 = tmp30.to(tmp24.dtype)
    tmp32 = tl_math.abs(tmp24)
    tmp33 = triton_helpers.maximum(tmp25, tmp32)
    tmp34 = tmp31 * tmp33
    tmp35 = 8.0
    tmp36 = tmp34 * tmp35
    tmp37 = libdevice.sqrt(tmp36)
    tmp38 = 20.0
    tmp39 = tmp37 > tmp38
    tmp40 = tl_math.exp(tmp37)
    tmp41 = libdevice.log1p(tmp40)
    tmp42 = tl.where(tmp39, tmp37, tmp41)
    tmp43 = libdevice.tanh(tmp42)
    tmp44 = tmp37 * tmp43
    tmp46 = 0.5
    tmp47 = tmp45 < tmp46
    tmp48 = tmp47.to(tl.float32)
    tmp49 = 0.8864048946659319
    tmp50 = tmp48 * tmp49
    tmp51 = tmp44 * tmp50
    tmp52 = -1.0
    tmp53 = tmp48 + tmp52
    tmp54 = 1.558387861036063
    tmp55 = tmp53 * tmp54
    tmp56 = 0.7791939305180315
    tmp57 = tmp55 + tmp56
    tmp58 = tmp51 + tmp57
    tmp59 = tmp58 * tmp58
    tl.store(in_out_ptr0 + (x5), tmp59, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_abs_add_avg_pool3d_bernoulli_mish_mul_pow_relu_sign_2(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, ks6, ks7, ks8, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks1)
    x2 = ((xindex // ks2) % ks3)
    x3 = xindex // ks4
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 2*ks5*x1 + 2*ks5*ks6*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 2*ks5*x1 + 2*ks5*ks6*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (ks5 + 2*x0 + 2*ks5*x1 + 2*ks5*ks6*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (1 + ks5 + 2*x0 + 2*ks5*x1 + 2*ks5*ks6*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (ks8 + 2*x0 + 2*ks5*x1 + 2*ks5*ks6*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (1 + ks8 + 2*x0 + 2*ks5*x1 + 2*ks5*ks6*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (ks5 + ks8 + 2*x0 + 2*ks5*x1 + 2*ks5*ks6*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (1 + ks5 + ks8 + 2*x0 + 2*ks5*x1 + 2*ks5*ks6*x2 + ks5*ks6*ks7*x3), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp15 = 0.125
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (x4), tmp16, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__to_copy_add_arange_clamp_min_div_gather_mean_ne_randint_rsub_scalar_tensor_soft_margin_loss_where_3(in_out_ptr0, in_ptr0, in_ptr1, load_seed_offset, ks1, ks2, ks3, ks4, ks5, ks6, ks7, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    _tmp57 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp65 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_0 = r0_index
        tmp32 = tl.load(in_ptr1 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp0 = tl.load(in_ptr0 + load_seed_offset)
        tmp1 = tl.full([1, 1], 0, tl.int32)
        tmp2 = tl.full([1, 1], 0, tl.int64)
        tmp3 = tl.full([1, 1], 10, tl.int64)
        tmp4 = triton_helpers.randint64(tmp0, (tmp1).to(tl.uint32), tmp2, tmp3)
        tmp5 = ks1*ks2*ks3*ks4
        tmp6 = tmp4 + tmp5
        tmp7 = tmp4 < 0
        tmp8 = tl.where(tmp7, tmp6, tmp4)
        tl.device_assert((0 <= tmp8) & (tmp8 < ks4*(ks5 // 4)*(ks6 // 4)*(ks7 // 4)), "index out of bounds: 0 <= tmp8 < ks4*(ks5 // 4)*(ks6 // 4)*(ks7 // 4)")
        tmp10 = tl.load(in_ptr1 + ((tmp8 % (ks1*ks2*ks3*ks4))), None, eviction_policy='evict_last')
        tmp11 = tmp1 < tmp10
        tmp12 = tmp11.to(tl.int8)
        tmp13 = tmp10 < tmp1
        tmp14 = tmp13.to(tl.int8)
        tmp15 = tmp12 - tmp14
        tmp16 = tmp15.to(tmp10.dtype)
        tmp17 = tl_math.abs(tmp10)
        tmp18 = triton_helpers.maximum(tmp1, tmp17)
        tmp19 = tmp16 * tmp18
        tmp20 = 8.0
        tmp21 = tmp19 * tmp20
        tmp22 = libdevice.sqrt(tmp21)
        tmp23 = 20.0
        tmp24 = tmp22 > tmp23
        tmp25 = tl_math.exp(tmp22)
        tmp26 = libdevice.log1p(tmp25)
        tmp27 = tl.where(tmp24, tmp22, tmp26)
        tmp28 = libdevice.tanh(tmp27)
        tmp29 = tmp22 * tmp28
        tmp30 = 1.0
        tmp31 = tmp30 - tmp29
        tmp33 = tmp1 < tmp32
        tmp34 = tmp33.to(tl.int8)
        tmp35 = tmp32 < tmp1
        tmp36 = tmp35.to(tl.int8)
        tmp37 = tmp34 - tmp36
        tmp38 = tmp37.to(tmp32.dtype)
        tmp39 = tl_math.abs(tmp32)
        tmp40 = triton_helpers.maximum(tmp1, tmp39)
        tmp41 = tmp38 * tmp40
        tmp42 = tmp41 * tmp20
        tmp43 = libdevice.sqrt(tmp42)
        tmp44 = tmp43 > tmp23
        tmp45 = tl_math.exp(tmp43)
        tmp46 = libdevice.log1p(tmp45)
        tmp47 = tl.where(tmp44, tmp43, tmp46)
        tmp48 = libdevice.tanh(tmp47)
        tmp49 = tmp43 * tmp48
        tmp50 = tmp31 + tmp49
        tmp51 = r0_0
        tmp52 = tmp51 != tmp4
        tmp53 = 0.0
        tmp54 = triton_helpers.maximum(tmp50, tmp53)
        tmp55 = tl.where(tmp52, tmp54, tmp53)
        tmp56 = tl.broadcast_to(tmp55, [XBLOCK, R0_BLOCK])
        tmp58 = _tmp57 + tmp56
        _tmp57 = tl.where(r0_mask, tmp58, _tmp57)
        tmp59 = -tmp49
        tmp60 = tmp4.to(tl.float32)
        tmp61 = tmp59 * tmp60
        tmp62 = tl_math.exp(tmp61)
        tmp63 = libdevice.log1p(tmp62)
        tmp64 = tl.broadcast_to(tmp63, [XBLOCK, R0_BLOCK])
        tmp66 = _tmp65 + tmp64
        _tmp65 = tl.where(r0_mask, tmp66, _tmp65)
    tmp57 = tl.sum(_tmp57, 1)[:, None]
    tmp65 = tl.sum(_tmp65, 1)[:, None]
    tmp67 = ks1*ks2*ks3*ks4
    tmp68 = tmp67.to(tl.float32)
    tmp69 = tmp57 / tmp68
    tmp70 = tmp65 / tmp68
    tmp71 = tmp69 + tmp70
    tmp72 = 0.5
    tmp73 = tmp71 * tmp72
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp73, None)


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
        buf1 = empty_strided_cuda((2, ), (1, ), torch.int64)

        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [2], out=buf1)
        buf2 = empty_strided_cuda((1, s0, 1, 1, 1), (s0, 1, s0, s0, s0), torch.float32)

        get_raw_stream(0)
        triton_poi_fused_bernoulli_0[grid(s0)](buf1, buf2, 0, 3, XBLOCK=4, num_warps=1, num_stages=1)
        s3 // 2
        s2 // 2
        (s2 // 2)*(s3 // 2)
        s1 // 2
        (s1 // 2)*(s2 // 2)*(s3 // 2)
        buf0 = empty_strided_cuda((1, s0, s1 // 2, s2 // 2, s3 // 2), (s0*(s1 // 2)*(s2 // 2)*(s3 // 2), (s1 // 2)*(s2 // 2)*(s3 // 2), (s2 // 2)*(s3 // 2), s3 // 2, 1), torch.float32)
        buf3 = buf0; del buf0

        triton_poi_fused__to_copy_abs_add_avg_pool3d_bernoulli_mish_mul_pow_relu_sign_1_xnumel = s0*(s1 // 2)*(s2 // 2)*(s3 // 2)
        get_raw_stream(0)
        triton_poi_fused__to_copy_abs_add_avg_pool3d_bernoulli_mish_mul_pow_relu_sign_1[grid(triton_poi_fused__to_copy_abs_add_avg_pool3d_bernoulli_mish_mul_pow_relu_sign_1_xnumel)](buf3, arg4_1, buf2, 16, 16, 256, 16, 4096, 32, 32, 32, 12288, XBLOCK=128, num_warps=4, num_stages=1)
        del arg4_1
        del buf2
        s3 // 4
        s2 // 4
        (s2 // 4)*(s3 // 4)
        s1 // 4
        (s1 // 4)*(s2 // 4)*(s3 // 4)
        buf4 = empty_strided_cuda((1, s0, s1 // 4, s2 // 4, s3 // 4), (s0*(s1 // 4)*(s2 // 4)*(s3 // 4), (s1 // 4)*(s2 // 4)*(s3 // 4), (s2 // 4)*(s3 // 4), s3 // 4, 1), torch.float32)

        triton_poi_fused__to_copy_abs_add_avg_pool3d_bernoulli_mish_mul_pow_relu_sign_2_xnumel = s0*(s1 // 4)*(s2 // 4)*(s3 // 4)
        get_raw_stream(0)
        triton_poi_fused__to_copy_abs_add_avg_pool3d_bernoulli_mish_mul_pow_relu_sign_2[grid(triton_poi_fused__to_copy_abs_add_avg_pool3d_bernoulli_mish_mul_pow_relu_sign_2_xnumel)](buf3, buf4, 8, 8, 64, 8, 512, 16, 16, 16, 256, 1536, XBLOCK=128, num_warps=4, num_stages=1)
        del buf3
        buf6 = empty_strided_cuda((), (), torch.float32)
        buf8 = buf6; del buf6

        s0*(s1 // 4)*(s2 // 4)*(s3 // 4)
        get_raw_stream(0)
        triton_red_fused__to_copy_add_arange_clamp_min_div_gather_mean_ne_randint_rsub_scalar_tensor_soft_margin_loss_where_3[grid(1)](buf8, buf1, buf4, 1, 8, 8, 8, 3, 32, 32, 32, 1, 1536, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del buf1
        del buf4
    return (buf8, )


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
