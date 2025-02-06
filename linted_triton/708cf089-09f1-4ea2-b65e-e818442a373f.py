
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
def triton_red_fused__native_batch_norm_legit__native_batch_norm_legit_functional_mean_mish_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr3, out_ptr4, out_ptr6, out_ptr8, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 64
    r0_numel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(r0_mask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(r0_mask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(r0_mask & xmask, tmp2_weight_next, tmp2_weight)
    tmp5, tmp6, tmp7 = triton_helpers.welford(tmp2_mean, tmp2_m2, tmp2_weight, 1)
    tmp5[:, None]
    tmp3 = tmp6[:, None]
    tmp7[:, None]
    tmp8 = 4096.0
    tmp9 = tmp3 / tmp8
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp12, xmask)
    tmp15_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        r0_1 = r0_index
        tmp13 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_reduce(
            tmp14, tmp15_mean, tmp15_m2, tmp15_weight, roffset == 0
        )
        tmp15_mean = tl.where(r0_mask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(r0_mask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(r0_mask & xmask, tmp15_weight_next, tmp15_weight)
    tmp18, tmp19, tmp20 = triton_helpers.welford(tmp15_mean, tmp15_m2, tmp15_weight, 1)
    tmp15 = tmp18[:, None]
    tmp19[:, None]
    tmp20[:, None]
    tl.store(out_ptr0 + (x0), tmp15, xmask)
    tmp32_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp32_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp32_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        r0_1 = r0_index
        tmp21 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp22 = tmp21 - tmp15
        tmp23 = tmp22 * tmp12
        tmp24 = 20.0
        tmp25 = tmp23 > tmp24
        tmp26 = tl_math.exp(tmp23)
        tmp27 = libdevice.log1p(tmp26)
        tmp28 = tl.where(tmp25, tmp23, tmp27)
        tmp29 = libdevice.tanh(tmp28)
        tmp30 = tmp23 * tmp29
        tmp31 = tl.broadcast_to(tmp30, [XBLOCK, R0_BLOCK])
        tmp32_mean_next, tmp32_m2_next, tmp32_weight_next = triton_helpers.welford_reduce(
            tmp31, tmp32_mean, tmp32_m2, tmp32_weight, roffset == 0
        )
        tmp32_mean = tl.where(r0_mask & xmask, tmp32_mean_next, tmp32_mean)
        tmp32_m2 = tl.where(r0_mask & xmask, tmp32_m2_next, tmp32_m2)
        tmp32_weight = tl.where(r0_mask & xmask, tmp32_weight_next, tmp32_weight)
    tmp35, tmp36, tmp37 = triton_helpers.welford(tmp32_mean, tmp32_m2, tmp32_weight, 1)
    tmp32 = tmp35[:, None]
    tmp33 = tmp36[:, None]
    tmp37[:, None]
    tl.store(out_ptr1 + (x0), tmp32, xmask)
    tmp55 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        r0_1 = r0_index
        tmp38 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp39 = tmp38 - tmp15
        tmp40 = tmp39 * tmp12
        tmp41 = 20.0
        tmp42 = tmp40 > tmp41
        tmp43 = tl_math.exp(tmp40)
        tmp44 = libdevice.log1p(tmp43)
        tmp45 = tl.where(tmp42, tmp40, tmp44)
        tmp46 = libdevice.tanh(tmp45)
        tmp47 = tmp40 * tmp46
        tmp48 = tmp47 - tmp32
        tmp49 = 4096.0
        tmp50 = tmp33 / tmp49
        tmp51 = 1e-05
        tmp52 = tmp50 + tmp51
        tmp53 = libdevice.rsqrt(tmp52)
        tmp54 = tmp48 * tmp53
        tmp56 = tmp54 * tmp55
        tmp58 = tmp56 + tmp57
        tl.store(out_ptr3 + (r0_1 + 4096*x0), tmp58, r0_mask & xmask)
    tmp68 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp75 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp59 = 4096.0
    tmp60 = tmp33 / tmp59
    tmp61 = 1e-05
    tmp62 = tmp60 + tmp61
    tmp63 = libdevice.rsqrt(tmp62)
    tmp64 = 1.0002442002442002
    tmp65 = tmp60 * tmp64
    tmp66 = 0.1
    tmp67 = tmp65 * tmp66
    tmp69 = 0.9
    tmp70 = tmp68 * tmp69
    tmp71 = tmp67 + tmp70
    tmp72 = 1.0
    tmp73 = tmp71 / tmp72
    tmp74 = tmp32 * tmp66
    tmp76 = tmp75 * tmp69
    tmp77 = tmp74 + tmp76
    tmp78 = tmp77 / tmp72
    tl.store(out_ptr4 + (x0), tmp63, xmask)
    tl.store(out_ptr6 + (x0), tmp73, xmask)
    tl.store(out_ptr8 + (x0), tmp78, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (1, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        buf3 = reinterpret_tensor(buf1, (1, 64, 1, 1), (64, 1, 1, 1), 0); del buf1
        buf0 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 1, 1), torch.float32)
        buf4 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        buf8 = empty_strided_cuda((1, 64, 64, 64), (262144, 4096, 64, 1), torch.float32)
        buf7 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32)

        get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit__native_batch_norm_legit_functional_mean_mish_0[grid(64)](buf3, primals_1, primals_4, primals_5, primals_3, primals_2, buf0, buf4, buf8, buf7, primals_3, primals_2, 64, 4096, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del primals_2
        del primals_3
        del primals_4
        del primals_5
    return (buf8, primals_1, buf0, buf3, reinterpret_tensor(buf7, (64, ), (1, ), 0), reinterpret_tensor(buf4, (1, 64, 1, 1), (64, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([primals_1, primals_2, primals_3, primals_4, primals_5])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
