
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
def triton_red_fused__softmax_hardswish_0(in_ptr0, out_ptr0, out_ptr1, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    _tmp11 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + ks0*ks1*r0_1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 3.0
        tmp2 = tmp0 + tmp1
        tmp3 = 0.0
        tmp4 = triton_helpers.maximum(tmp2, tmp3)
        tmp5 = 6.0
        tmp6 = triton_helpers.minimum(tmp4, tmp5)
        tmp7 = tmp0 * tmp6
        tmp8 = 1.0
        tmp9 = tmp7 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp12 = triton_helpers.maximum(_tmp11, tmp10)
        _tmp11 = tl.where(r0_mask & xmask, tmp12, _tmp11)
    tmp11 = triton_helpers.max2(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp11, xmask)
    _tmp28 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp13 = tl.load(in_ptr0 + (x0 + ks0*ks1*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp14 = 3.0
        tmp15 = tmp13 + tmp14
        tmp16 = 0.0
        tmp17 = triton_helpers.maximum(tmp15, tmp16)
        tmp18 = 6.0
        tmp19 = triton_helpers.minimum(tmp17, tmp18)
        tmp20 = tmp13 * tmp19
        tmp21 = 1.0
        tmp22 = tmp20 * tmp21
        tmp23 = tmp22 - tmp11
        tmp24 = 0.16666666666666666
        tmp25 = tmp23 * tmp24
        tmp26 = tl_math.exp(tmp25)
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, R0_BLOCK])
        tmp29 = _tmp28 + tmp27
        _tmp28 = tl.where(r0_mask & xmask, tmp29, _tmp28)
    tmp28 = tl.sum(_tmp28, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp28, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__softmax_hardswish_log_sigmoid_forward_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % ks0)
    tmp0 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 - tmp10
    tmp12 = 0.16666666666666666
    tmp13 = tmp11 * tmp12
    tmp14 = tl_math.exp(tmp13)
    tmp16 = tmp14 / tmp15
    tmp17 = triton_helpers.minimum(tmp3, tmp16)
    tmp18 = tl_math.abs(tmp16)
    tmp19 = -tmp18
    tmp20 = tl_math.exp(tmp19)
    tmp21 = libdevice.log1p(tmp20)
    tmp22 = tmp17 - tmp21
    tl.store(out_ptr0 + (x2), tmp22, xmask)


def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 1, s1, s2), (s1*s2, s1*s2, s2, 1), torch.float32)
        buf1 = empty_strided_cuda((1, 1, s1, s2), (s1*s2, s1*s2, s2, 1), torch.float32)

        triton_red_fused__softmax_hardswish_0_xnumel = s1*s2
        get_raw_stream(0)
        triton_red_fused__softmax_hardswish_0[grid(triton_red_fused__softmax_hardswish_0_xnumel)](arg3_1, buf0, buf1, 32, 32, 1024, 3, XBLOCK=64, R0_BLOCK=4, num_warps=2, num_stages=1)
        s1*s2
        buf2 = empty_strided_cuda((1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1), torch.float32)

        triton_poi_fused__softmax_hardswish_log_sigmoid_forward_1_xnumel = s0*s1*s2
        get_raw_stream(0)
        triton_poi_fused__softmax_hardswish_log_sigmoid_forward_1[grid(triton_poi_fused__softmax_hardswish_log_sigmoid_forward_1_xnumel)](arg3_1, buf0, buf1, buf2, 1024, 3072, XBLOCK=128, num_warps=4, num_stages=1)
        del arg3_1
        del buf0
        del buf1
    return (buf2, )


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
