
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
def triton_red_fused_add_norm_randn_like_sub_0(in_ptr0, in_ptr1, out_ptr2, out_ptr3, load_seed_offset, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 5
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    _tmp75 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp80 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp3 = tl.load(in_ptr1 + (2*r0_1 + ks1*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp22 = tl.load(in_ptr1 + (1 + 2*r0_1 + ks1*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp38 = tl.load(in_ptr1 + (2*r0_1 + 5*ks1 + ks1*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp53 = tl.load(in_ptr1 + (1 + 2*r0_1 + 5*ks1 + ks1*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = tl.load(in_ptr0 + load_seed_offset)
        tmp1 = r0_1 + x0*(ks1 // 2)
        tmp2 = tl.randn(tmp0, (tmp1).to(tl.uint32))
        tmp4 = tl_math.abs(tmp3)
        tmp5 = 0.5
        tmp6 = tmp4 > tmp5
        tmp7 = tl.full([1, 1], 0, tl.int32)
        tmp8 = tmp7 < tmp3
        tmp9 = tmp8.to(tl.int8)
        tmp10 = tmp3 < tmp7
        tmp11 = tmp10.to(tl.int8)
        tmp12 = tmp9 - tmp11
        tmp13 = tmp12.to(tmp3.dtype)
        tmp14 = tmp13 * tmp5
        tmp15 = tmp3 - tmp14
        tmp16 = 0.0
        tmp17 = tmp3 * tmp16
        tmp18 = tl.where(tmp6, tmp15, tmp17)
        tmp19 = triton_helpers.maximum(tmp18, tmp16)
        tmp20 = 6.0
        tmp21 = triton_helpers.minimum(tmp19, tmp20)
        tmp23 = tl_math.abs(tmp22)
        tmp24 = tmp23 > tmp5
        tmp25 = tmp7 < tmp22
        tmp26 = tmp25.to(tl.int8)
        tmp27 = tmp22 < tmp7
        tmp28 = tmp27.to(tl.int8)
        tmp29 = tmp26 - tmp28
        tmp30 = tmp29.to(tmp22.dtype)
        tmp31 = tmp30 * tmp5
        tmp32 = tmp22 - tmp31
        tmp33 = tmp22 * tmp16
        tmp34 = tl.where(tmp24, tmp32, tmp33)
        tmp35 = triton_helpers.maximum(tmp34, tmp16)
        tmp36 = triton_helpers.minimum(tmp35, tmp20)
        tmp37 = triton_helpers.maximum(tmp36, tmp21)
        tmp39 = tl_math.abs(tmp38)
        tmp40 = tmp39 > tmp5
        tmp41 = tmp7 < tmp38
        tmp42 = tmp41.to(tl.int8)
        tmp43 = tmp38 < tmp7
        tmp44 = tmp43.to(tl.int8)
        tmp45 = tmp42 - tmp44
        tmp46 = tmp45.to(tmp38.dtype)
        tmp47 = tmp46 * tmp5
        tmp48 = tmp38 - tmp47
        tmp49 = tmp38 * tmp16
        tmp50 = tl.where(tmp40, tmp48, tmp49)
        tmp51 = triton_helpers.maximum(tmp50, tmp16)
        tmp52 = triton_helpers.minimum(tmp51, tmp20)
        tmp54 = tl_math.abs(tmp53)
        tmp55 = tmp54 > tmp5
        tmp56 = tmp7 < tmp53
        tmp57 = tmp56.to(tl.int8)
        tmp58 = tmp53 < tmp7
        tmp59 = tmp58.to(tl.int8)
        tmp60 = tmp57 - tmp59
        tmp61 = tmp60.to(tmp53.dtype)
        tmp62 = tmp61 * tmp5
        tmp63 = tmp53 - tmp62
        tmp64 = tmp53 * tmp16
        tmp65 = tl.where(tmp55, tmp63, tmp64)
        tmp66 = triton_helpers.maximum(tmp65, tmp16)
        tmp67 = triton_helpers.minimum(tmp66, tmp20)
        tmp68 = triton_helpers.maximum(tmp67, tmp52)
        tmp69 = tmp37 - tmp68
        tmp70 = tmp37 - tmp2
        tmp71 = 1e-06
        tmp72 = tmp70 + tmp71
        tmp73 = tmp72 * tmp72
        tmp74 = tl.broadcast_to(tmp73, [XBLOCK, R0_BLOCK])
        tmp76 = _tmp75 + tmp74
        _tmp75 = tl.where(r0_mask & xmask, tmp76, _tmp75)
        tmp77 = tmp69 + tmp71
        tmp78 = tmp77 * tmp77
        tmp79 = tl.broadcast_to(tmp78, [XBLOCK, R0_BLOCK])
        tmp81 = _tmp80 + tmp79
        _tmp80 = tl.where(r0_mask & xmask, tmp81, _tmp80)
    tmp75 = tl.sum(_tmp75, 1)[:, None]
    tmp80 = tl.sum(_tmp80, 1)[:, None]
    tl.store(out_ptr2 + (x0), tmp75, xmask)
    tl.store(out_ptr3 + (x0), tmp80, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_clamp_min_mean_norm_sub_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp5 = tl.load(in_ptr1 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp11 = tl.load(in_ptr0 + (1))
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK])
    tmp15 = tl.load(in_ptr1 + (1))
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp21 = tl.load(in_ptr0 + (2))
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK])
    tmp25 = tl.load(in_ptr1 + (2))
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK])
    tmp31 = tl.load(in_ptr0 + (3))
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK])
    tmp35 = tl.load(in_ptr1 + (3))
    tmp36 = tl.broadcast_to(tmp35, [XBLOCK])
    tmp41 = tl.load(in_ptr0 + (4))
    tmp42 = tl.broadcast_to(tmp41, [XBLOCK])
    tmp45 = tl.load(in_ptr1 + (4))
    tmp46 = tl.broadcast_to(tmp45, [XBLOCK])
    tmp2 = libdevice.sqrt(tmp1)
    tmp3 = 1.0
    tmp4 = tmp2 + tmp3
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tmp4 - tmp7
    tmp9 = 0.0
    tmp10 = triton_helpers.maximum(tmp8, tmp9)
    tmp13 = libdevice.sqrt(tmp12)
    tmp14 = tmp13 + tmp3
    tmp17 = libdevice.sqrt(tmp16)
    tmp18 = tmp14 - tmp17
    tmp19 = triton_helpers.maximum(tmp18, tmp9)
    tmp20 = tmp10 + tmp19
    tmp23 = libdevice.sqrt(tmp22)
    tmp24 = tmp23 + tmp3
    tmp27 = libdevice.sqrt(tmp26)
    tmp28 = tmp24 - tmp27
    tmp29 = triton_helpers.maximum(tmp28, tmp9)
    tmp30 = tmp20 + tmp29
    tmp33 = libdevice.sqrt(tmp32)
    tmp34 = tmp33 + tmp3
    tmp37 = libdevice.sqrt(tmp36)
    tmp38 = tmp34 - tmp37
    tmp39 = triton_helpers.maximum(tmp38, tmp9)
    tmp40 = tmp30 + tmp39
    tmp43 = libdevice.sqrt(tmp42)
    tmp44 = tmp43 + tmp3
    tmp47 = libdevice.sqrt(tmp46)
    tmp48 = tmp44 - tmp47
    tmp49 = triton_helpers.maximum(tmp48, tmp9)
    tmp50 = tmp40 + tmp49
    tmp51 = 5.0
    tmp52 = tmp50 / tmp51
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp52, None)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    s1 = arg0_1
    assert_size_stride(arg1_1, (1, 10, s1), (10*s1, s1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf2 = empty_strided_cuda((1, ), (1, ), torch.int64)

        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf2)
        buf4 = empty_strided_cuda((1, 5), (5, 1), torch.float32)
        buf1 = empty_strided_cuda((1, 5), (5, 1), torch.float32)

        s1 // 2
        get_raw_stream(0)
        triton_red_fused_add_norm_randn_like_sub_0[grid(5)](buf2, arg1_1, buf4, buf1, 0, 20, 5, 10, XBLOCK=1, R0_BLOCK=16, num_warps=2, num_stages=1)
        del arg1_1
        del buf2
        buf5 = empty_strided_cuda((), (), torch.float32)

        get_raw_stream(0)
        triton_poi_fused_add_clamp_min_mean_norm_sub_1[grid(1)](buf1, buf4, buf5, 1, XBLOCK=1, num_warps=1, num_stages=1)
        del buf1
        del buf4
    return (buf5, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 20
    arg1_1 = rand_strided((1, 10, 20), (200, 20, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
