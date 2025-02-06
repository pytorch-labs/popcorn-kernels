
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
def triton_poi_fused_max_pool2d_with_indices_replication_pad2d_0(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks1)
    x2 = xindex // ks2
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (ks4*(((-1) + ks3) * (((-1) + ks3) <= (((0) * ((0) >= ((-2) + 2*x1)) + ((-2) + 2*x1) * (((-2) + 2*x1) > (0))))) + (((0) * ((0) >= ((-2) + 2*x1)) + ((-2) + 2*x1) * (((-2) + 2*x1) > (0)))) * ((((0) * ((0) >= ((-2) + 2*x1)) + ((-2) + 2*x1) * (((-2) + 2*x1) > (0)))) < ((-1) + ks3))) + ks3*ks4*x2 + (((-1) + ks4) * (((-1) + ks4) <= (((0) * ((0) >= ((-2) + 2*x0)) + ((-2) + 2*x0) * (((-2) + 2*x0) > (0))))) + (((0) * ((0) >= ((-2) + 2*x0)) + ((-2) + 2*x0) * (((-2) + 2*x0) > (0)))) * ((((0) * ((0) >= ((-2) + 2*x0)) + ((-2) + 2*x0) * (((-2) + 2*x0) > (0)))) < ((-1) + ks4)))), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (ks4*(((-1) + ks3) * (((-1) + ks3) <= (((0) * ((0) >= ((-2) + 2*x1)) + ((-2) + 2*x1) * (((-2) + 2*x1) > (0))))) + (((0) * ((0) >= ((-2) + 2*x1)) + ((-2) + 2*x1) * (((-2) + 2*x1) > (0)))) * ((((0) * ((0) >= ((-2) + 2*x1)) + ((-2) + 2*x1) * (((-2) + 2*x1) > (0)))) < ((-1) + ks3))) + ks3*ks4*x2 + (((-1) + ks4) * (((-1) + ks4) <= (((0) * ((0) >= ((-1) + 2*x0)) + ((-1) + 2*x0) * (((-1) + 2*x0) > (0))))) + (((0) * ((0) >= ((-1) + 2*x0)) + ((-1) + 2*x0) * (((-1) + 2*x0) > (0)))) * ((((0) * ((0) >= ((-1) + 2*x0)) + ((-1) + 2*x0) * (((-1) + 2*x0) > (0)))) < ((-1) + ks4)))), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (ks4*(((-1) + ks3) * (((-1) + ks3) <= (((0) * ((0) >= ((-1) + 2*x1)) + ((-1) + 2*x1) * (((-1) + 2*x1) > (0))))) + (((0) * ((0) >= ((-1) + 2*x1)) + ((-1) + 2*x1) * (((-1) + 2*x1) > (0)))) * ((((0) * ((0) >= ((-1) + 2*x1)) + ((-1) + 2*x1) * (((-1) + 2*x1) > (0)))) < ((-1) + ks3))) + ks3*ks4*x2 + (((-1) + ks4) * (((-1) + ks4) <= (((0) * ((0) >= ((-2) + 2*x0)) + ((-2) + 2*x0) * (((-2) + 2*x0) > (0))))) + (((0) * ((0) >= ((-2) + 2*x0)) + ((-2) + 2*x0) * (((-2) + 2*x0) > (0)))) * ((((0) * ((0) >= ((-2) + 2*x0)) + ((-2) + 2*x0) * (((-2) + 2*x0) > (0)))) < ((-1) + ks4)))), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (ks4*(((-1) + ks3) * (((-1) + ks3) <= (((0) * ((0) >= ((-1) + 2*x1)) + ((-1) + 2*x1) * (((-1) + 2*x1) > (0))))) + (((0) * ((0) >= ((-1) + 2*x1)) + ((-1) + 2*x1) * (((-1) + 2*x1) > (0)))) * ((((0) * ((0) >= ((-1) + 2*x1)) + ((-1) + 2*x1) * (((-1) + 2*x1) > (0)))) < ((-1) + ks3))) + ks3*ks4*x2 + (((-1) + ks4) * (((-1) + ks4) <= (((0) * ((0) >= ((-1) + 2*x0)) + ((-1) + 2*x0) * (((-1) + 2*x0) > (0))))) + (((0) * ((0) >= ((-1) + 2*x0)) + ((-1) + 2*x0) * (((-1) + 2*x0) > (0)))) * ((((0) * ((0) >= ((-1) + 2*x0)) + ((-1) + 2*x0) * (((-1) + 2*x0) > (0)))) < ((-1) + ks4)))), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (x3), tmp6, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_binary_cross_entropy_with_logits_zeros_like_1(in_out_ptr0, in_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    _tmp16 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = -tmp0
        tmp2 = tmp1 - tmp1
        tmp3 = tl_math.exp(tmp2)
        tmp4 = tmp3 / tmp3
        tmp5 = 1.0
        tmp6 = tmp5 * tmp4
        tmp7 = 0.0
        tmp8 = triton_helpers.minimum(tmp7, tmp4)
        tmp9 = tl_math.abs(tmp4)
        tmp10 = -tmp9
        tmp11 = tl_math.exp(tmp10)
        tmp12 = libdevice.log1p(tmp11)
        tmp13 = tmp8 - tmp12
        tmp14 = tmp6 - tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(r0_mask, tmp17, _tmp16)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tmp18 = 25*ks0
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp16 / tmp19
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp20, None)


def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        2 + (s2 // 2)
        2 + (s1 // 2)
        4 + 2*(s1 // 2) + 2*(s2 // 2) + (s1 // 2)*(s2 // 2)
        buf0 = empty_strided_cuda((1, s0, 2 + (s1 // 2), 2 + (s2 // 2)), (4*s0 + 2*s0*(s1 // 2) + 2*s0*(s2 // 2) + s0*(s1 // 2)*(s2 // 2), 4 + 2*(s1 // 2) + 2*(s2 // 2) + (s1 // 2)*(s2 // 2), 2 + (s2 // 2), 1), torch.float32)

        triton_poi_fused_max_pool2d_with_indices_replication_pad2d_0_xnumel = 4*s0 + 2*s0*(s1 // 2) + 2*s0*(s2 // 2) + s0*(s1 // 2)*(s2 // 2)
        get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_replication_pad2d_0[grid(triton_poi_fused_max_pool2d_with_indices_replication_pad2d_0_xnumel)](arg3_1, buf0, 34, 34, 1156, 64, 64, 3468, XBLOCK=256, num_warps=4, num_stages=1)
        del arg3_1

        buf1 = torch.ops.aten.adaptive_max_pool2d.default(buf0, [5, 5])
        del buf0
        buf2 = buf1[0]
        del buf1
        buf4 = empty_strided_cuda((), (), torch.float32)
        buf5 = buf4; del buf4

        25*s0
        get_raw_stream(0)
        triton_red_fused_binary_cross_entropy_with_logits_zeros_like_1[grid(1)](buf5, buf2, 3, 1, 75, XBLOCK=1, R0_BLOCK=128, num_warps=2, num_stages=1)
        del buf2
    return (buf5, )


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
