
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
from torch._inductor.runtime.triton_helpers import libdevice
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__unsafe_index_gelu_tanh_0(in_out_ptr0, in_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // ks1) % ks2)
    x0 = (xindex % ks1)
    x2 = xindex // ks4
    x3 = xindex
    tmp0 = 4.0
    tmp1 = ks0
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 * tmp2
    tmp4 = tmp3.to(tl.float64)
    tmp5 = tl.full([1], 2.0, tl.float64)
    tmp6 = tmp5 * tmp4
    tmp7 = tmp4 / tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = x1
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp10 * tmp8
    tmp12 = tmp11.to(tl.int64)
    tmp13 = 4*ks0
    tmp14 = tmp12 + tmp13
    tmp15 = tmp12 < 0
    tmp16 = tl.where(tmp15, tmp14, tmp12)
    tmp17 = ks3
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp0 * tmp18
    tmp20 = tmp19.to(tl.float64)
    tmp21 = tmp5 * tmp20
    tmp22 = tmp20 / tmp21
    tmp23 = tmp22.to(tl.float32)
    tmp24 = x0
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp25 * tmp23
    tmp27 = tmp26.to(tl.int64)
    tmp28 = 4*ks3
    tmp29 = tmp27 + tmp28
    tmp30 = tmp27 < 0
    tmp31 = tl.where(tmp30, tmp29, tmp27)
    tmp32 = 2.0
    tmp33 = tmp32 * tmp2
    tmp34 = tmp33.to(tl.float64)
    tmp35 = tmp5 * tmp34
    tmp36 = tmp34 / tmp35
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tmp16
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tmp39 * tmp37
    tmp41 = tmp40.to(tl.int64)
    tmp42 = 2*ks0
    tmp43 = tmp41 + tmp42
    tmp44 = tmp41 < 0
    tmp45 = tl.where(tmp44, tmp43, tmp41)
    tmp46 = tmp32 * tmp18
    tmp47 = tmp46.to(tl.float64)
    tmp48 = tmp5 * tmp47
    tmp49 = tmp47 / tmp48
    tmp50 = tmp49.to(tl.float32)
    tmp51 = tmp31
    tmp52 = tmp51.to(tl.float32)
    tmp53 = tmp52 * tmp50
    tmp54 = tmp53.to(tl.int64)
    tmp55 = 2*ks3
    tmp56 = tmp54 + tmp55
    tmp57 = tmp54 < 0
    tmp58 = tl.where(tmp57, tmp56, tmp54)
    tmp59 = tmp1.to(tl.float64)
    tmp60 = tmp5 * tmp59
    tmp61 = tmp59 / tmp60
    tmp62 = tmp61.to(tl.float32)
    tmp63 = tmp45
    tmp64 = tmp63.to(tl.float32)
    tmp65 = tmp64 * tmp62
    tmp66 = tmp65.to(tl.int64)
    tmp67 = tmp66 + tmp1
    tmp68 = tmp66 < 0
    tmp69 = tl.where(tmp68, tmp67, tmp66)
    tmp70 = tmp17.to(tl.float64)
    tmp71 = tmp5 * tmp70
    tmp72 = tmp70 / tmp71
    tmp73 = tmp72.to(tl.float32)
    tmp74 = tmp58
    tmp75 = tmp74.to(tl.float32)
    tmp76 = tmp75 * tmp73
    tmp77 = tmp76.to(tl.int64)
    tmp78 = tmp77 + tmp17
    tmp79 = tmp77 < 0
    tmp80 = tl.where(tmp79, tmp78, tmp77)
    tmp81 = tl.load(in_ptr0 + (tmp80 + ks3*tmp69 + ks0*ks3*x2), xmask, eviction_policy='evict_last')
    tmp82 = 0.5
    tmp83 = tmp81 * tmp82
    tmp84 = 0.7071067811865476
    tmp85 = tmp81 * tmp84
    tmp86 = libdevice.erf(tmp85)
    tmp87 = 1.0
    tmp88 = tmp86 + tmp87
    tmp89 = tmp83 * tmp88
    tmp90 = libdevice.tanh(tmp89)
    tmp91 = tmp90 * tmp82
    tmp92 = tmp90 * tmp84
    tmp93 = libdevice.erf(tmp92)
    tmp94 = tmp93 + tmp87
    tmp95 = tmp91 * tmp94
    tmp96 = libdevice.tanh(tmp95)
    tl.store(in_out_ptr0 + (x3), tmp96, xmask)


def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        8*s2
        8*s1
        64*s1*s2
        buf0 = empty_strided_cuda((1, s0, 8*s1, 8*s2), (64*s0*s1*s2, 64*s1*s2, 8*s2, 1), torch.float32)
        buf1 = buf0; del buf0

        triton_poi_fused__unsafe_index_gelu_tanh_0_xnumel = 64*s0*s1*s2
        get_raw_stream(0)
        triton_poi_fused__unsafe_index_gelu_tanh_0[grid(triton_poi_fused__unsafe_index_gelu_tanh_0_xnumel)](buf1, arg3_1, 32, 256, 256, 32, 65536, 196608, XBLOCK=512, num_warps=8, num_stages=1)
        del arg3_1
    return (buf1, )


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
