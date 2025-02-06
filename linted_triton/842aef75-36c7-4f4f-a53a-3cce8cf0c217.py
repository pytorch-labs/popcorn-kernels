
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
def triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_hardswish_mul_sub_view_0(in_out_ptr1, in_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // ks1) % ks2)
    x0 = (xindex % ks1)
    x2 = xindex // ks4
    x4 = xindex
    tmp0 = tl.full([1], -1.0, tl.float64)
    tmp1 = ks0
    tmp2 = tmp1.to(tl.float64)
    tmp3 = tmp0 + tmp2
    tmp4 = 2.0
    tmp5 = tmp1.to(tl.float32)
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6.to(tl.float64)
    tmp8 = tmp0 + tmp7
    tmp9 = tmp3 / tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = x1
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp12 * tmp10
    tmp14 = 0.0
    tmp15 = triton_helpers.maximum(tmp13, tmp14)
    tmp16 = tmp15.to(tl.int64)
    tmp17 = ks3
    tmp18 = tmp17.to(tl.float64)
    tmp19 = tmp0 + tmp18
    tmp20 = tmp17.to(tl.float32)
    tmp21 = tmp4 * tmp20
    tmp22 = tmp21.to(tl.float64)
    tmp23 = tmp0 + tmp22
    tmp24 = tmp19 / tmp23
    tmp25 = tmp24.to(tl.float32)
    tmp26 = x0
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp27 * tmp25
    tmp29 = triton_helpers.maximum(tmp28, tmp14)
    tmp30 = tmp29.to(tl.int64)
    tmp31 = tl.load(in_ptr0 + (tmp30 + ks3*tmp16 + ks0*ks3*x2), xmask, eviction_policy='evict_last')
    tmp32 = tl.full([1], 1, tl.int64)
    tmp33 = tmp16 + tmp32
    tmp34 = (-1) + ks0
    tmp35 = triton_helpers.minimum(tmp33, tmp34)
    tmp36 = tl.load(in_ptr0 + (tmp30 + ks3*tmp35 + ks0*ks3*x2), xmask, eviction_policy='evict_last')
    tmp37 = tmp30 + tmp32
    tmp38 = (-1) + ks3
    tmp39 = triton_helpers.minimum(tmp37, tmp38)
    tmp40 = tl.load(in_ptr0 + (tmp39 + ks3*tmp35 + ks0*ks3*x2), xmask, eviction_policy='evict_last')
    tmp41 = tmp40 - tmp36
    tmp42 = tl.load(in_ptr0 + (tmp39 + ks3*tmp16 + ks0*ks3*x2), xmask, eviction_policy='evict_last')
    tmp43 = tmp42 - tmp31
    tmp44 = tmp30.to(tl.float32)
    tmp45 = tmp29 - tmp44
    tmp46 = triton_helpers.maximum(tmp45, tmp14)
    tmp47 = 1.0
    tmp48 = triton_helpers.minimum(tmp46, tmp47)
    tmp49 = tmp41 * tmp48
    tmp50 = tmp36 + tmp49
    tmp51 = tmp43 * tmp48
    tmp52 = tmp31 + tmp51
    tmp53 = tmp50 - tmp52
    tmp54 = tmp16.to(tl.float32)
    tmp55 = tmp15 - tmp54
    tmp56 = triton_helpers.maximum(tmp55, tmp14)
    tmp57 = triton_helpers.minimum(tmp56, tmp47)
    tmp58 = tmp53 * tmp57
    tmp59 = tmp52 + tmp58
    tmp60 = 3.0
    tmp61 = tmp59 + tmp60
    tmp62 = triton_helpers.maximum(tmp61, tmp14)
    tmp63 = 6.0
    tmp64 = triton_helpers.minimum(tmp62, tmp63)
    tmp65 = tmp59 * tmp64
    tmp66 = 0.16666666666666666
    tmp67 = tmp65 * tmp66
    tl.store(in_out_ptr1 + (x4), tmp67, xmask)


def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        2*s2
        2*s1
        4*s1*s2
        buf2 = empty_strided_cuda((1, s0, 2*s1, 2*s2), (4*s0*s1*s2, 4*s1*s2, 2*s2, 1), torch.float32)
        buf5 = buf2; del buf2
        buf6 = buf5; del buf5

        triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_hardswish_mul_sub_view_0_xnumel = 4*s0*s1*s2
        get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_hardswish_mul_sub_view_0[grid(triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_hardswish_mul_sub_view_0_xnumel)](buf6, arg3_1, 32, 64, 64, 32, 4096, 12288, XBLOCK=128, num_warps=4, num_stages=1)
        del arg3_1
    return (reinterpret_tensor(buf6, (1, 2*s2, 2*s0*s1), (4*s0*s1*s2, 1, 2*s2), 0), )


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
