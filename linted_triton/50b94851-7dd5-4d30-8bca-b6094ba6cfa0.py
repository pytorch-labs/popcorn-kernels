
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
def triton_red_fused__log_softmax__softmax_div_mul_randn_like_sub_sum_xlogy_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr3, out_ptr4, ks0, ks1, load_seed_offset, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    _tmp28 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_0 = r0_index
        tmp13 = tl.load(in_ptr1 + (r0_0 // (16 + 4*ks0 + 4*ks1 + ks0*ks1)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp0 = (-2) + (((r0_0 // (4 + ks1)) % (4 + ks0)))
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = ks0
        tmp4 = tmp0 < tmp3
        tmp5 = (-2) + ((r0_0 % (4 + ks1)))
        tmp6 = tmp5 >= tmp1
        tmp7 = ks1
        tmp8 = tmp5 < tmp7
        tmp9 = tmp2 & tmp4
        tmp10 = tmp9 & tmp6
        tmp11 = tmp10 & tmp8
        tmp12 = tl.load(in_ptr0 + (tl.broadcast_to((-2) + ((-2)*ks1) + ks1*(((r0_0 // (4 + ks1)) % (4 + ks0))) + ks0*ks1*(r0_0 // (16 + 4*ks0 + 4*ks1 + ks0*ks1)) + ((r0_0 % (4 + ks1))), [XBLOCK, R0_BLOCK])), r0_mask & tmp11, eviction_policy='evict_last', other=0.0)
        tmp14 = 0.5
        tmp15 = tmp13 < tmp14
        tmp16 = tmp15.to(tl.float32)
        tmp17 = 0.8864048946659319
        tmp18 = tmp16 * tmp17
        tmp19 = tmp12 * tmp18
        tmp20 = -1.0
        tmp21 = tmp16 + tmp20
        tmp22 = 1.558387861036063
        tmp23 = tmp21 * tmp22
        tmp24 = 0.7791939305180315
        tmp25 = tmp23 + tmp24
        tmp26 = tmp19 + tmp25
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, R0_BLOCK])
        tmp29 = triton_helpers.maximum(_tmp28, tmp27)
        _tmp28 = tl.where(r0_mask, tmp29, _tmp28)
    tmp28 = triton_helpers.max2(_tmp28, 1)[:, None]
    _tmp60 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_0 = r0_index
        tmp43 = tl.load(in_ptr1 + (r0_0 // (16 + 4*ks0 + 4*ks1 + ks0*ks1)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp30 = (-2) + (((r0_0 // (4 + ks1)) % (4 + ks0)))
        tmp31 = tl.full([1, 1], 0, tl.int64)
        tmp32 = tmp30 >= tmp31
        tmp33 = ks0
        tmp34 = tmp30 < tmp33
        tmp35 = (-2) + ((r0_0 % (4 + ks1)))
        tmp36 = tmp35 >= tmp31
        tmp37 = ks1
        tmp38 = tmp35 < tmp37
        tmp39 = tmp32 & tmp34
        tmp40 = tmp39 & tmp36
        tmp41 = tmp40 & tmp38
        tmp42 = tl.load(in_ptr0 + (tl.broadcast_to((-2) + ((-2)*ks1) + ks1*(((r0_0 // (4 + ks1)) % (4 + ks0))) + ks0*ks1*(r0_0 // (16 + 4*ks0 + 4*ks1 + ks0*ks1)) + ((r0_0 % (4 + ks1))), [XBLOCK, R0_BLOCK])), r0_mask & tmp41, eviction_policy='evict_last', other=0.0)
        tmp44 = 0.5
        tmp45 = tmp43 < tmp44
        tmp46 = tmp45.to(tl.float32)
        tmp47 = 0.8864048946659319
        tmp48 = tmp46 * tmp47
        tmp49 = tmp42 * tmp48
        tmp50 = -1.0
        tmp51 = tmp46 + tmp50
        tmp52 = 1.558387861036063
        tmp53 = tmp51 * tmp52
        tmp54 = 0.7791939305180315
        tmp55 = tmp53 + tmp54
        tmp56 = tmp49 + tmp55
        tmp57 = tmp56 - tmp28
        tmp58 = tl_math.exp(tmp57)
        tmp59 = tl.broadcast_to(tmp58, [XBLOCK, R0_BLOCK])
        tmp61 = _tmp60 + tmp59
        _tmp60 = tl.where(r0_mask, tmp61, _tmp60)
    tmp60 = tl.sum(_tmp60, 1)[:, None]
    _tmp96 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_0 = r0_index
        tmp75 = tl.load(in_ptr1 + (r0_0 // (16 + 4*ks0 + 4*ks1 + ks0*ks1)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp62 = (-2) + (((r0_0 // (4 + ks1)) % (4 + ks0)))
        tmp63 = tl.full([1, 1], 0, tl.int64)
        tmp64 = tmp62 >= tmp63
        tmp65 = ks0
        tmp66 = tmp62 < tmp65
        tmp67 = (-2) + ((r0_0 % (4 + ks1)))
        tmp68 = tmp67 >= tmp63
        tmp69 = ks1
        tmp70 = tmp67 < tmp69
        tmp71 = tmp64 & tmp66
        tmp72 = tmp71 & tmp68
        tmp73 = tmp72 & tmp70
        tmp74 = tl.load(in_ptr0 + (tl.broadcast_to((-2) + ((-2)*ks1) + ks1*(((r0_0 // (4 + ks1)) % (4 + ks0))) + ks0*ks1*(r0_0 // (16 + 4*ks0 + 4*ks1 + ks0*ks1)) + ((r0_0 % (4 + ks1))), [XBLOCK, R0_BLOCK])), r0_mask & tmp73, eviction_policy='evict_last', other=0.0)
        tmp76 = 0.5
        tmp77 = tmp75 < tmp76
        tmp78 = tmp77.to(tl.float32)
        tmp79 = 0.8864048946659319
        tmp80 = tmp78 * tmp79
        tmp81 = tmp74 * tmp80
        tmp82 = -1.0
        tmp83 = tmp78 + tmp82
        tmp84 = 1.558387861036063
        tmp85 = tmp83 * tmp84
        tmp86 = 0.7791939305180315
        tmp87 = tmp85 + tmp86
        tmp88 = tmp81 + tmp87
        tmp89 = tmp88 - tmp28
        tmp90 = tl_math.log(tmp60)
        tmp91 = tmp89 - tmp90
        tmp92 = tl.load(in_ptr2 + load_seed_offset)
        tmp93 = r0_0
        tmp94 = tl.randn(tmp92, (tmp93).to(tl.uint32))
        tmp95 = tl.broadcast_to(tmp94, [XBLOCK, R0_BLOCK])
        tmp97 = triton_helpers.maximum(_tmp96, tmp95)
        _tmp96 = tl.where(r0_mask, tmp97, _tmp96)
        tl.store(out_ptr3 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp91, r0_mask)
        tl.store(out_ptr4 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp94, r0_mask)
    tmp96 = triton_helpers.max2(_tmp96, 1)[:, None]
    _tmp102 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_0 = r0_index
        tmp98 = tl.load(out_ptr4 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp99 = tmp98 - tmp96
        tmp100 = tl_math.exp(tmp99)
        tmp101 = tl.broadcast_to(tmp100, [XBLOCK, R0_BLOCK])
        tmp103 = _tmp102 + tmp101
        _tmp102 = tl.where(r0_mask, tmp103, _tmp102)
    tmp102 = tl.sum(_tmp102, 1)[:, None]
    _tmp120 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_0 = r0_index
        tmp104 = tl.load(out_ptr4 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp116 = tl.load(out_ptr3 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp105 = tmp104 - tmp96
        tmp106 = tl_math.exp(tmp105)
        tmp107 = tmp106 / tmp102
        tmp108 = libdevice.isnan(tmp107).to(tl.int1)
        tmp109 = 0.0
        tmp110 = tmp107 == tmp109
        tmp111 = tl_math.log(tmp107)
        tmp112 = tmp107 * tmp111
        tmp113 = tl.where(tmp110, tmp109, tmp112)
        tmp114 = float("nan")
        tmp115 = tl.where(tmp108, tmp114, tmp113)
        tmp117 = tmp107 * tmp116
        tmp118 = tmp115 - tmp117
        tmp119 = tl.broadcast_to(tmp118, [XBLOCK, R0_BLOCK])
        tmp121 = _tmp120 + tmp119
        _tmp120 = tl.where(r0_mask, tmp121, _tmp120)
    tmp120 = tl.sum(_tmp120, 1)[:, None]
    tmp122 = 1.0
    tmp123 = tmp120 * tmp122
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp123, None)


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
        buf1 = empty_strided_cuda((1, s0, 1, 1), (s0, 1, s0, s0), torch.float32)

        get_raw_stream(0)
        triton_poi_fused_bernoulli_0[grid(s0)](buf0, buf1, 0, 3, XBLOCK=4, num_warps=1, num_stages=1)
        buf8 = empty_strided_cuda((1, 16*s0 + 4*s0*s1 + 4*s0*s2 + s0*s1*s2), (16*s0 + 4*s0*s1 + 4*s0*s2 + s0*s1*s2, 1), torch.float32)
        buf2 = empty_strided_cuda((1, 16*s0 + 4*s0*s1 + 4*s0*s2 + s0*s1*s2), (16*s0 + 4*s0*s1 + 4*s0*s2 + s0*s1*s2, 1), torch.float32)
        buf3 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        buf9 = reinterpret_tensor(buf3, (), (), 0); del buf3
        buf10 = buf9; del buf9

        16*s0 + 4*s0*s1 + 4*s0*s2 + s0*s1*s2
        get_raw_stream(0)
        triton_red_fused__log_softmax__softmax_div_mul_randn_like_sub_sum_xlogy_1[grid(1)](buf10, arg3_1, buf1, buf0, buf8, buf2, 32, 32, 1, 1, 3888, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del arg3_1
        del buf0
        del buf1
        del buf2
        del buf8
    return (buf10, )


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
