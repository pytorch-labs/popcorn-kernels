
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
def triton_red_fused__log_softmax_0(in_ptr0, out_ptr2, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    _tmp5 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + ((ks1 // 2)*((((((r0_0 // (2*(ks1 // 2))) % (2*(ks0 // 2)))) // 2) % (ks0 // 2))) + (ks0 // 2)*(ks1 // 2)*((((r0_0 % (2*(ks1 // 2)))) % 2)) + 2*(ks0 // 2)*(ks1 // 2)*(((((r0_0 // (2*(ks1 // 2))) % (2*(ks0 // 2)))) % 2)) + 4*(ks0 // 2)*(ks1 // 2)*(triton_helpers.div_floor_integer(r0_0,  4*(ks0 // 2)*(ks1 // 2))) + (((((r0_0 % (2*(ks1 // 2)))) // 2) % (ks1 // 2)))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr0 + ((ks1 // 2)*((((((r0_0 // (2*(ks1 // 2))) % (2*(ks0 // 2)))) // 2) % (ks0 // 2))) + 8*(ks0 // 2)*(ks1 // 2) + (ks0 // 2)*(ks1 // 2)*((((r0_0 % (2*(ks1 // 2)))) % 2)) + 2*(ks0 // 2)*(ks1 // 2)*(((((r0_0 // (2*(ks1 // 2))) % (2*(ks0 // 2)))) % 2)) + 4*(ks0 // 2)*(ks1 // 2)*(triton_helpers.div_floor_integer(r0_0,  4*(ks0 // 2)*(ks1 // 2))) + (((((r0_0 % (2*(ks1 // 2)))) // 2) % (ks1 // 2)))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp6 = triton_helpers.maximum(_tmp5, tmp4)
        _tmp5 = tl.where(r0_mask, tmp6, _tmp5)
    tmp5 = triton_helpers.max2(_tmp5, 1)[:, None]
    _tmp14 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_0 = r0_index
        tmp7 = tl.load(in_ptr0 + ((ks1 // 2)*((((((r0_0 // (2*(ks1 // 2))) % (2*(ks0 // 2)))) // 2) % (ks0 // 2))) + (ks0 // 2)*(ks1 // 2)*((((r0_0 % (2*(ks1 // 2)))) % 2)) + 2*(ks0 // 2)*(ks1 // 2)*(((((r0_0 // (2*(ks1 // 2))) % (2*(ks0 // 2)))) % 2)) + 4*(ks0 // 2)*(ks1 // 2)*(triton_helpers.div_floor_integer(r0_0,  4*(ks0 // 2)*(ks1 // 2))) + (((((r0_0 % (2*(ks1 // 2)))) // 2) % (ks1 // 2)))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr0 + ((ks1 // 2)*((((((r0_0 // (2*(ks1 // 2))) % (2*(ks0 // 2)))) // 2) % (ks0 // 2))) + 8*(ks0 // 2)*(ks1 // 2) + (ks0 // 2)*(ks1 // 2)*((((r0_0 % (2*(ks1 // 2)))) % 2)) + 2*(ks0 // 2)*(ks1 // 2)*(((((r0_0 // (2*(ks1 // 2))) % (2*(ks0 // 2)))) % 2)) + 4*(ks0 // 2)*(ks1 // 2)*(triton_helpers.div_floor_integer(r0_0,  4*(ks0 // 2)*(ks1 // 2))) + (((((r0_0 % (2*(ks1 // 2)))) // 2) % (ks1 // 2)))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.sigmoid(tmp8)
        tmp10 = tmp7 * tmp9
        tmp11 = tmp10 - tmp5
        tmp12 = tl_math.exp(tmp11)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(r0_mask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_0 = r0_index
        tmp16 = tl.load(in_ptr0 + ((ks1 // 2)*((((((r0_0 // (2*(ks1 // 2))) % (2*(ks0 // 2)))) // 2) % (ks0 // 2))) + (ks0 // 2)*(ks1 // 2)*((((r0_0 % (2*(ks1 // 2)))) % 2)) + 2*(ks0 // 2)*(ks1 // 2)*(((((r0_0 // (2*(ks1 // 2))) % (2*(ks0 // 2)))) % 2)) + 4*(ks0 // 2)*(ks1 // 2)*(triton_helpers.div_floor_integer(r0_0,  4*(ks0 // 2)*(ks1 // 2))) + (((((r0_0 % (2*(ks1 // 2)))) // 2) % (ks1 // 2)))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr0 + ((ks1 // 2)*((((((r0_0 // (2*(ks1 // 2))) % (2*(ks0 // 2)))) // 2) % (ks0 // 2))) + 8*(ks0 // 2)*(ks1 // 2) + (ks0 // 2)*(ks1 // 2)*((((r0_0 % (2*(ks1 // 2)))) % 2)) + 2*(ks0 // 2)*(ks1 // 2)*(((((r0_0 // (2*(ks1 // 2))) % (2*(ks0 // 2)))) % 2)) + 4*(ks0 // 2)*(ks1 // 2)*(triton_helpers.div_floor_integer(r0_0,  4*(ks0 // 2)*(ks1 // 2))) + (((((r0_0 % (2*(ks1 // 2)))) // 2) % (ks1 // 2)))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp18 = tl.sigmoid(tmp17)
        tmp19 = tmp16 * tmp18
        tmp20 = tmp19 - tmp5
        tmp21 = tl_math.log(tmp14)
        tmp22 = tmp20 - tmp21
        tl.store(out_ptr2 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp22, r0_mask)


def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, 1, s0, s1, s2), (s0*s1*s2, s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)

        buf0 = torch.ops.aten.max_pool3d_with_indices.default(arg3_1, [2, 2, 2], [2, 2, 2])
        del arg3_1
        buf1 = buf0[0]
        del buf0
        buf5 = empty_strided_cuda((1, 8*(s1 // 2)*(s2 // 2)), (2*(s0 // (4*(s0 // 8)))*(s0 // (8*(s0 // 16)))*(s1 // 2)*(s2 // 2), 1), torch.float32)

        8*(s1 // 2)*(s2 // 2)
        get_raw_stream(0)
        triton_red_fused__log_softmax_0[grid(1)](buf1, buf5, 32, 32, 1, 2048, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del buf1
    return (buf5, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 32
    arg1_1 = 32
    arg2_1 = 32
    arg3_1 = rand_strided((1, 1, 32, 32, 32), (32768, 32768, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
