
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
def triton_red_fused_add_clamp_min_constant_pad_nd_fill_hardswish_mean_ne_ones_like_sub_where_zeros_like_0(in_out_ptr0, in_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    _tmp44 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_0 = r0_index
        tmp0 = 1.0
        tmp1 = tmp0 != tmp0
        tmp2 = (-3) + r0_0
        tmp3 = tl.full([1, 1], 0, tl.int64)
        tmp4 = tmp2 >= tmp3
        tmp5 = 4 + ks0
        tmp6 = tmp2 < tmp5
        tmp7 = tmp4 & tmp6
        tmp8 = tl.broadcast_to((-5) + r0_0, [XBLOCK, R0_BLOCK])
        tmp9 = tl.full([1, 1], 0, tl.int64)
        tmp10 = tmp8 >= tmp9
        tmp11 = tl.broadcast_to(ks0, [XBLOCK, R0_BLOCK])
        tmp12 = tmp8 < tmp11
        tmp13 = tmp10 & tmp12
        tmp14 = tmp13 & tmp7
        tmp15 = tl.load(in_ptr0 + (tl.broadcast_to((-5) + r0_0, [XBLOCK, R0_BLOCK])), r0_mask & tmp14, eviction_policy='evict_first', other=0.0)
        tmp16 = 3.0
        tmp17 = tmp15 + tmp16
        tmp18 = 0.0
        tmp19 = triton_helpers.maximum(tmp17, tmp18)
        tmp20 = 6.0
        tmp21 = triton_helpers.minimum(tmp19, tmp20)
        tmp22 = tmp15 * tmp21
        tmp23 = 0.16666666666666666
        tmp24 = tmp22 * tmp23
        tmp25 = tl.full(tmp24.shape, 1.0, tmp24.dtype)
        tmp26 = tl.where(tmp7, tmp24, tmp25)
        tmp27 = 3.0
        tmp28 = tmp26 + tmp27
        tmp29 = 0.0
        tmp30 = triton_helpers.maximum(tmp28, tmp29)
        tmp31 = 6.0
        tmp32 = triton_helpers.minimum(tmp30, tmp31)
        tmp33 = tmp26 * tmp32
        tmp34 = 0.16666666666666666
        tmp35 = tmp33 * tmp34
        tmp36 = tmp0 - tmp35
        tmp37 = triton_helpers.maximum(tmp36, tmp29)
        tmp38 = tl.where(tmp1, tmp37, tmp29)
        tmp39 = -1.0
        tmp40 = tmp0 != tmp39
        tmp41 = tl.where(tmp40, tmp35, tmp29)
        tmp42 = tmp38 + tmp41
        tmp43 = tl.broadcast_to(tmp42, [XBLOCK, R0_BLOCK])
        tmp45 = _tmp44 + tmp43
        _tmp44 = tl.where(r0_mask, tmp45, _tmp44)
    tmp44 = tl.sum(_tmp44, 1)[:, None]
    tmp46 = 10 + ks0
    tmp47 = tmp46.to(tl.float32)
    tmp48 = tmp44 / tmp47
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp48, None)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    s0 = arg0_1
    assert_size_stride(arg1_1, (1, s0), (s0, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((), (), torch.float32)
        buf1 = buf0; del buf0

        10 + s0
        get_raw_stream(0)
        triton_red_fused_add_clamp_min_constant_pad_nd_fill_hardswish_mean_ne_ones_like_sub_where_zeros_like_0[grid(1)](buf1, arg1_1, 10, 1, 20, XBLOCK=1, R0_BLOCK=32, num_warps=2, num_stages=1)
        del arg1_1
    return (buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 10
    arg1_1 = rand_strided((1, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
