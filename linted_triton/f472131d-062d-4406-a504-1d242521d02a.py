
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
def triton_red_fused__native_batch_norm_legit_0(in_ptr0, out_ptr2, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 10
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
        tmp0 = tl.load(in_ptr0 + (ks2*(tl.where((-1) + ks1 + ((-1)*tl_math.abs(1 + ((-1)*ks1) + tl_math.abs((-1) + (((r0_1 // (2 + ks2)) % (2 + ks1)))))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks1) + tl_math.abs((-1) + (((r0_1 // (2 + ks2)) % (2 + ks1)))))) + 2*ks1, (-1) + ks1 + ((-1)*tl_math.abs(1 + ((-1)*ks1) + tl_math.abs((-1) + (((r0_1 // (2 + ks2)) % (2 + ks1)))))))) + ks1*ks2*(tl.where((-1) + ks0 + ((-1)*tl_math.abs(1 + ((-1)*ks0) + tl_math.abs((-1) + (r0_1 // (4 + 2*ks1 + 2*ks2 + ks1*ks2))))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks0) + tl_math.abs((-1) + (r0_1 // (4 + 2*ks1 + 2*ks2 + ks1*ks2))))) + 2*ks0, (-1) + ks0 + ((-1)*tl_math.abs(1 + ((-1)*ks0) + tl_math.abs((-1) + (r0_1 // (4 + 2*ks1 + 2*ks2 + ks1*ks2))))))) + ks0*ks1*ks2*x0 + (tl.where((-1) + ks2 + ((-1)*tl_math.abs(1 + ((-1)*ks2) + tl_math.abs((-1) + ((r0_1 % (2 + ks2)))))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks2) + tl_math.abs((-1) + ((r0_1 % (2 + ks2)))))) + 2*ks2, (-1) + ks2 + ((-1)*tl_math.abs(1 + ((-1)*ks2) + tl_math.abs((-1) + ((r0_1 % (2 + ks2))))))))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(r0_mask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(r0_mask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(r0_mask & xmask, tmp2_weight_next, tmp2_weight)
    tmp5, tmp6, tmp7 = triton_helpers.welford(tmp2_mean, tmp2_m2, tmp2_weight, 1)
    tmp2 = tmp5[:, None]
    tmp3 = tmp6[:, None]
    tmp7[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        r0_1 = r0_index
        tmp8 = tl.load(in_ptr0 + (ks2*(tl.where((-1) + ks1 + ((-1)*tl_math.abs(1 + ((-1)*ks1) + tl_math.abs((-1) + (((r0_1 // (2 + ks2)) % (2 + ks1)))))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks1) + tl_math.abs((-1) + (((r0_1 // (2 + ks2)) % (2 + ks1)))))) + 2*ks1, (-1) + ks1 + ((-1)*tl_math.abs(1 + ((-1)*ks1) + tl_math.abs((-1) + (((r0_1 // (2 + ks2)) % (2 + ks1)))))))) + ks1*ks2*(tl.where((-1) + ks0 + ((-1)*tl_math.abs(1 + ((-1)*ks0) + tl_math.abs((-1) + (r0_1 // (4 + 2*ks1 + 2*ks2 + ks1*ks2))))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks0) + tl_math.abs((-1) + (r0_1 // (4 + 2*ks1 + 2*ks2 + ks1*ks2))))) + 2*ks0, (-1) + ks0 + ((-1)*tl_math.abs(1 + ((-1)*ks0) + tl_math.abs((-1) + (r0_1 // (4 + 2*ks1 + 2*ks2 + ks1*ks2))))))) + ks0*ks1*ks2*x0 + (tl.where((-1) + ks2 + ((-1)*tl_math.abs(1 + ((-1)*ks2) + tl_math.abs((-1) + ((r0_1 % (2 + ks2)))))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks2) + tl_math.abs((-1) + ((r0_1 % (2 + ks2)))))) + 2*ks2, (-1) + ks2 + ((-1)*tl_math.abs(1 + ((-1)*ks2) + tl_math.abs((-1) + ((r0_1 % (2 + ks2))))))))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp8 - tmp2
        tmp10 = 8 + 4*ks0 + 4*ks1 + 4*ks2 + 2*ks0*ks1 + 2*ks0*ks2 + 2*ks1*ks2 + ks0*ks1*ks2
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tmp3 / tmp11
        tmp13 = 1e-05
        tmp14 = tmp12 + tmp13
        tmp15 = libdevice.rsqrt(tmp14)
        tmp16 = tmp9 * tmp15
        tl.store(out_ptr2 + (r0_1 + 8*x0 + 4*ks0*x0 + 4*ks1*x0 + 4*ks2*x0 + 2*ks0*ks1*x0 + 2*ks0*ks2*x0 + 2*ks1*ks2*x0 + ks0*ks1*ks2*x0), tmp16, r0_mask & xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_hardsigmoid_norm_sub_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.load(in_ptr0 + (5))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp8 = tl.load(in_ptr0 + (1))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp10 = tl.load(in_ptr0 + (6))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp16 = tl.load(in_ptr0 + (2))
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK])
    tmp18 = tl.load(in_ptr0 + (7))
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK])
    tmp24 = tl.load(in_ptr0 + (3))
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK])
    tmp26 = tl.load(in_ptr0 + (8))
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK])
    tmp32 = tl.load(in_ptr0 + (4))
    tmp33 = tl.broadcast_to(tmp32, [XBLOCK])
    tmp34 = tl.load(in_ptr0 + (9))
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK])
    tmp49 = tl.load(in_ptr1 + (0))
    tmp50 = tl.broadcast_to(tmp49, [XBLOCK])
    tmp52 = tl.load(in_ptr2 + (0))
    tmp53 = tl.broadcast_to(tmp52, [XBLOCK])
    tmp55 = tl.load(in_ptr3 + (0))
    tmp56 = tl.broadcast_to(tmp55, [XBLOCK])
    tmp58 = tl.load(in_ptr4 + (0))
    tmp59 = tl.broadcast_to(tmp58, [XBLOCK])
    tmp61 = tl.load(in_ptr5 + (0))
    tmp62 = tl.broadcast_to(tmp61, [XBLOCK])
    tmp4 = tmp1 - tmp3
    tmp5 = 1e-06
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6 * tmp6
    tmp12 = tmp9 - tmp11
    tmp13 = tmp12 + tmp5
    tmp14 = tmp13 * tmp13
    tmp15 = tmp7 + tmp14
    tmp20 = tmp17 - tmp19
    tmp21 = tmp20 + tmp5
    tmp22 = tmp21 * tmp21
    tmp23 = tmp15 + tmp22
    tmp28 = tmp25 - tmp27
    tmp29 = tmp28 + tmp5
    tmp30 = tmp29 * tmp29
    tmp31 = tmp23 + tmp30
    tmp36 = tmp33 - tmp35
    tmp37 = tmp36 + tmp5
    tmp38 = tmp37 * tmp37
    tmp39 = tmp31 + tmp38
    tmp40 = libdevice.sqrt(tmp39)
    tmp41 = 3.0
    tmp42 = tmp40 + tmp41
    tmp43 = 0.0
    tmp44 = triton_helpers.maximum(tmp42, tmp43)
    tmp45 = 6.0
    tmp46 = triton_helpers.minimum(tmp44, tmp45)
    tmp47 = 0.16666666666666666
    tmp48 = tmp46 * tmp47
    tmp51 = tmp48 + tmp50
    tmp54 = tmp51 + tmp53
    tmp57 = tmp54 + tmp56
    tmp60 = tmp57 + tmp59
    tmp63 = tmp60 + tmp62
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp63, None)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9 = args
    args.clear()
    s1 = primals_1
    s2 = primals_2
    s3 = primals_3
    assert_size_stride(primals_4, (1, 10, s1, s2, s3), (10*s1*s2*s3, s1*s2*s3, s2*s3, s3, 1))
    assert_size_stride(primals_5, (10, ), (1, ))
    assert_size_stride(primals_6, (10, ), (1, ))
    assert_size_stride(primals_7, (10, ), (1, ))
    assert_size_stride(primals_8, (10, ), (1, ))
    assert_size_stride(primals_9, (10, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf3 = empty_strided_cuda((1, 10, 8 + 4*s1 + 4*s2 + 4*s3 + 2*s1*s2 + 2*s1*s3 + 2*s2*s3 + s1*s2*s3), (80 + 40*s1 + 40*s2 + 40*s3 + 20*s1*s2 + 20*s1*s3 + 20*s2*s3 + 10*s1*s2*s3, 8 + 4*s1 + 4*s2 + 4*s3 + 2*s1*s2 + 2*s1*s3 + 2*s2*s3 + s1*s2*s3, 1), torch.float32)

        8 + 4*s1 + 4*s2 + 4*s3 + 2*s1*s2 + 2*s1*s3 + 2*s2*s3 + s1*s2*s3
        get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_0[grid(10)](primals_4, buf3, 10, 10, 10, 10, 1728, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del primals_4
        buf4 = empty_strided_cuda((1, ), (1, ), torch.float32)

        get_raw_stream(0)
        triton_poi_fused_add_hardsigmoid_norm_sub_1[grid(1)](buf3, primals_5, primals_6, primals_7, primals_8, primals_9, buf4, 1, XBLOCK=1, num_warps=1, num_stages=1)
        del buf3
        del primals_5
        del primals_6
        del primals_7
        del primals_8
        del primals_9
    return (buf4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = 10
    primals_2 = 10
    primals_3 = 10
    primals_4 = rand_strided((1, 10, 10, 10, 10), (10000, 1000, 100, 10, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
