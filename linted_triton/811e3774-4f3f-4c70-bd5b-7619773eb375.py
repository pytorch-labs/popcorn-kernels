
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
def triton_red_fused__softmax_div_log_mul_ones_like_sub_sum_xlogy_0(in_out_ptr0, in_ptr0, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    _tmp14 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = -1.0
        tmp2 = triton_helpers.maximum(tmp0, tmp1)
        tmp3 = 1.0
        tmp4 = triton_helpers.minimum(tmp2, tmp3)
        tmp5 = -0.5
        tmp6 = triton_helpers.maximum(tmp4, tmp5)
        tmp7 = 0.5
        tmp8 = triton_helpers.minimum(tmp6, tmp7)
        tmp9 = -0.2
        tmp10 = triton_helpers.maximum(tmp8, tmp9)
        tmp11 = 0.2
        tmp12 = triton_helpers.minimum(tmp10, tmp11)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
        tmp15 = triton_helpers.maximum(_tmp14, tmp13)
        _tmp14 = tl.where(r0_mask, tmp15, _tmp14)
    tmp14 = triton_helpers.max2(_tmp14, 1)[:, None]
    _tmp32 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_0 = r0_index
        tmp16 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp17 = -1.0
        tmp18 = triton_helpers.maximum(tmp16, tmp17)
        tmp19 = 1.0
        tmp20 = triton_helpers.minimum(tmp18, tmp19)
        tmp21 = -0.5
        tmp22 = triton_helpers.maximum(tmp20, tmp21)
        tmp23 = 0.5
        tmp24 = triton_helpers.minimum(tmp22, tmp23)
        tmp25 = -0.2
        tmp26 = triton_helpers.maximum(tmp24, tmp25)
        tmp27 = 0.2
        tmp28 = triton_helpers.minimum(tmp26, tmp27)
        tmp29 = tmp28 - tmp14
        tmp30 = tl_math.exp(tmp29)
        tmp31 = tl.broadcast_to(tmp30, [XBLOCK, R0_BLOCK])
        tmp33 = _tmp32 + tmp31
        _tmp32 = tl.where(r0_mask, tmp33, _tmp32)
    tmp32 = tl.sum(_tmp32, 1)[:, None]
    _tmp65 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_0 = r0_index
        tmp46 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp34 = 1.0
        tmp35 = ks0*ks1*ks2
        tmp36 = tmp35.to(tl.float32)
        tmp37 = tmp34 / tmp36
        tmp38 = libdevice.isnan(tmp37).to(tl.int1)
        tmp39 = 0.0
        tmp40 = tmp37 == tmp39
        tmp41 = tl_math.log(tmp37)
        tmp42 = tmp37 * tmp41
        tmp43 = tl.where(tmp40, tmp39, tmp42)
        tmp44 = float("nan")
        tmp45 = tl.where(tmp38, tmp44, tmp43)
        tmp47 = -1.0
        tmp48 = triton_helpers.maximum(tmp46, tmp47)
        tmp49 = triton_helpers.minimum(tmp48, tmp34)
        tmp50 = -0.5
        tmp51 = triton_helpers.maximum(tmp49, tmp50)
        tmp52 = 0.5
        tmp53 = triton_helpers.minimum(tmp51, tmp52)
        tmp54 = -0.2
        tmp55 = triton_helpers.maximum(tmp53, tmp54)
        tmp56 = 0.2
        tmp57 = triton_helpers.minimum(tmp55, tmp56)
        tmp58 = tmp57 - tmp14
        tmp59 = tl_math.exp(tmp58)
        tmp60 = tmp59 / tmp32
        tmp61 = tl_math.log(tmp60)
        tmp62 = tmp37 * tmp61
        tmp63 = tmp45 - tmp62
        tmp64 = tl.broadcast_to(tmp63, [XBLOCK, R0_BLOCK])
        tmp66 = _tmp65 + tmp64
        _tmp65 = tl.where(r0_mask, tmp66, _tmp65)
    tmp65 = tl.sum(_tmp65, 1)[:, None]
    tmp67 = 1.0
    tmp68 = tmp65 * tmp67
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp68, None)


def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        buf2 = reinterpret_tensor(buf0, (), (), 0); del buf0
        buf3 = buf2; del buf2

        s0*s1*s2
        get_raw_stream(0)
        triton_red_fused__softmax_div_log_mul_ones_like_sub_sum_xlogy_0[grid(1)](buf3, arg3_1, 3, 32, 32, 1, 3072, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del arg3_1
    return (buf3, )


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
