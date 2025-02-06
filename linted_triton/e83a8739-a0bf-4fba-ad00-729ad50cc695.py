
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
def triton_per_fused__native_batch_norm_legit_functional_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, out_ptr3, out_ptr5, out_ptr7, xnumel, r0_numel):
    XBLOCK: tl.constexpr = 1
    r0_numel = 784
    R0_BLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_mask = r0_index < r0_numel
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 784*x0), r0_mask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [R0_BLOCK])
    tl.where(r0_mask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [R0_BLOCK])
    tmp6 = tl.where(r0_mask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 784, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [R0_BLOCK])
    tmp15 = tl.where(r0_mask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = tmp0 - tmp10
    tmp18 = 784.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = 1.0012770891189575
    tmp29 = tmp19 * tmp28
    tmp30 = 0.1
    tmp31 = tmp29 * tmp30
    tmp33 = 0.9
    tmp34 = tmp32 * tmp33
    tmp35 = tmp31 + tmp34
    tmp36 = tmp10 * tmp30
    tmp38 = tmp37 * tmp33
    tmp39 = tmp36 + tmp38
    tl.store(out_ptr2 + (r0_1 + 784*x0), tmp27, r0_mask)
    tl.store(out_ptr3 + (x0), tmp22, None)
    tl.store(out_ptr5 + (x0), tmp35, None)
    tl.store(out_ptr7 + (x0), tmp39, None)
    tl.store(out_ptr0 + (x0), tmp10, None)


import triton

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_rand_1(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 20
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
def triton_poi_fused_fractional_max_pool2d_2(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex // 196
    x1 = ((xindex // 14) % 14)
    x0 = (xindex % 14)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x2), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr0 + (1 + 2*x2), xmask, eviction_policy='evict_last')
    tmp1 = x1
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp2 + tmp0
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp6 = libdevice.floor(tmp5)
    tmp7 = tmp0 * tmp4
    tmp8 = libdevice.floor(tmp7)
    tmp9 = tmp6 - tmp8
    tmp10 = tmp9.to(tl.int64)
    tmp11 = tl.full([1], 13, tl.int64)
    tmp12 = tmp2 < tmp11
    tmp13 = tl.full([1], 26, tl.int64)
    tmp14 = tl.where(tmp12, tmp10, tmp13)
    tmp15 = tl.full([XBLOCK], 28, tl.int32)
    tmp16 = tmp14 + tmp15
    tmp17 = tmp14 < 0
    tmp18 = tl.where(tmp17, tmp16, tmp14)
    tl.device_assert(((0 <= tmp18) & (tmp18 < 28)) | ~(xmask), "index out of bounds: 0 <= tmp18 < 28")
    tmp21 = x0
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp22 + tmp20
    tmp24 = tmp23 * tmp4
    tmp25 = libdevice.floor(tmp24)
    tmp26 = tmp20 * tmp4
    tmp27 = libdevice.floor(tmp26)
    tmp28 = tmp25 - tmp27
    tmp29 = tmp28.to(tl.int64)
    tmp30 = tmp22 < tmp11
    tmp31 = tl.where(tmp30, tmp29, tmp13)
    tmp32 = tmp31 + tmp15
    tmp33 = tmp31 < 0
    tmp34 = tl.where(tmp33, tmp32, tmp31)
    tl.device_assert(((0 <= tmp34) & (tmp34 < 28)) | ~(xmask), "index out of bounds: 0 <= tmp34 < 28")
    tmp36 = tl.load(in_ptr1 + (tmp34 + 28*tmp18 + 784*x2), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr1 + (1 + tmp34 + 28*tmp18 + 784*x2), xmask, eviction_policy='evict_last')
    tmp38 = triton_helpers.maximum(tmp37, tmp36)
    tmp39 = tl.load(in_ptr1 + (28 + tmp34 + 28*tmp18 + 784*x2), xmask, eviction_policy='evict_last')
    tmp40 = triton_helpers.maximum(tmp39, tmp38)
    tmp41 = tl.load(in_ptr1 + (29 + tmp34 + 28*tmp18 + 784*x2), xmask, eviction_policy='evict_last')
    tmp42 = triton_helpers.maximum(tmp41, tmp40)
    tmp43 = tmp37 > tmp36
    tmp44 = libdevice.isnan(tmp37).to(tl.int1)
    tmp45 = tmp43 | tmp44
    tmp46 = 1 + tmp34 + 28*tmp18
    tmp47 = tmp46.to(tl.int32)
    tmp48 = tmp34 + 28*tmp18
    tmp49 = tmp48.to(tl.int32)
    tmp50 = tl.where(tmp45, tmp47, tmp49)
    tmp51 = tmp39 > tmp38
    tmp52 = libdevice.isnan(tmp39).to(tl.int1)
    tmp53 = tmp51 | tmp52
    tmp54 = 28 + tmp34 + 28*tmp18
    tmp55 = tmp54.to(tl.int32)
    tmp56 = tl.where(tmp53, tmp55, tmp50)
    tmp57 = tmp41 > tmp40
    tmp58 = libdevice.isnan(tmp41).to(tl.int1)
    tmp59 = tmp57 | tmp58
    tmp60 = 29 + tmp34 + 28*tmp18
    tmp61 = tmp60.to(tl.int32)
    tmp62 = tl.where(tmp59, tmp61, tmp56)
    tl.store(out_ptr0 + (x4), tmp42, xmask)
    tl.store(out_ptr1 + (x4), tmp62, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__softmax_3(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 196
    r0_numel = 10
    R0_BLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_mask = r0_index < r0_numel
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 196*r0_1), r0_mask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tmp5 = tmp0 - tmp4
    tmp6 = tl_math.exp(tmp5)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(r0_mask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr1 + (x0), tmp10, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit__softmax_4(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 10
    r0_numel = 196
    R0_BLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_mask = r0_index < r0_numel
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 196*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 - tmp1
    tmp3 = tl_math.exp(tmp2)
    tmp5 = tmp3 / tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tl.where(r0_mask & xmask, tmp6, 0)
    tmp9 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp11 = tl.where(r0_mask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = tl.full([XBLOCK, 1], 196, tl.int32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 / tmp14
    tmp16 = tmp6 - tmp15
    tmp17 = tmp16 * tmp16
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
    tmp20 = tl.where(r0_mask & xmask, tmp18, 0)
    tmp21 = tl.sum(tmp20, 1)[:, None]
    tmp22 = 196.0
    tmp23 = tmp21 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tl.store(in_out_ptr0 + (r0_1 + 196*x0), tmp5, r0_mask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp26, xmask)
    tl.store(out_ptr0 + (x0), tmp15, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_nll_loss2d_forward_randint_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr2, load_seed_offset, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 196
    R0_BLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_mask = r0_index < r0_numel
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = r0_0
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tl.full([1, 1], 10, tl.int64)
    tmp4 = triton_helpers.randint64(tmp0, (tmp1).to(tl.uint32), tmp2, tmp3)
    tmp5 = tl.full([1, 1], -100, tl.int64)
    tmp6 = tmp4 != tmp5
    tmp7 = tmp6.to(tl.int64)
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(r0_mask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.where(tmp6, tmp4, tmp2)
    tmp13 = tl.full([XBLOCK, R0_BLOCK], 10, tl.int32)
    tmp14 = tmp12 + tmp13
    tmp15 = tmp12 < 0
    tmp16 = tl.where(tmp15, tmp14, tmp12)
    tl.device_assert(((0 <= tmp16) & (tmp16 < 10)) | ~(r0_mask), "index out of bounds: 0 <= tmp16 < 10")
    tmp18 = tl.load(in_ptr1 + (r0_0 + 196*tmp16), r0_mask, other=0.0)
    tmp19 = tl.load(in_ptr2 + (tmp16), r0_mask, eviction_policy='evict_last')
    tmp20 = tmp18 - tmp19
    tmp21 = tl.load(in_ptr3 + (tmp16), r0_mask, eviction_policy='evict_last')
    tmp22 = tmp20 * tmp21
    tmp23 = 1e-10
    tmp24 = tmp22 + tmp23
    tmp25 = tl_math.log(tmp24)
    tmp26 = -tmp25
    tmp27 = 0.0
    tmp28 = tl.where(tmp6, tmp26, tmp27)
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, R0_BLOCK])
    tmp31 = tl.where(r0_mask, tmp29, 0)
    tmp32 = tl.sum(tmp31, 1)[:, None]
    tmp33 = tmp11.to(tl.float32)
    tmp34 = tmp32 / tmp33
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp4, r0_mask)
    tl.store(out_ptr2 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp33, None)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp34, None)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_6(in_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp1 + tmp2
    tl.store(out_ptr1 + (tl.full([XBLOCK], 0, tl.int32)), tmp3, None)


def call(args):
    _primals_1, _primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8 = args
    args.clear()
    assert_size_stride(primals_3, (1, 10, 28, 28), (7840, 784, 28, 1))
    assert_size_stride(primals_4, (), ())
    assert_size_stride(primals_5, (10, ), (1, ))
    assert_size_stride(primals_6, (10, ), (1, ))
    assert_size_stride(primals_7, (10, ), (1, ))
    assert_size_stride(primals_8, (10, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 10, 1, 1), (10, 1, 10, 10), torch.float32)
        buf4 = empty_strided_cuda((1, 10, 28, 28), (7840, 784, 28, 1), torch.float32)
        buf3 = empty_strided_cuda((1, 10, 1, 1), (10, 1, 10, 10), torch.float32)

        get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_0[grid(10)](primals_3, primals_7, primals_8, primals_6, primals_5, buf0, buf4, buf3, primals_6, primals_5, 10, 784, num_warps=8, num_stages=1)
        del primals_5
        del primals_6
        del primals_7
        del primals_8
        buf5 = empty_strided_cuda((2, ), (1, ), torch.int64)

        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [2], out=buf5)
        buf6 = empty_strided_cuda((1, 10, 2), (20, 2, 1), torch.float32)

        get_raw_stream(0)
        triton_poi_fused_rand_1[grid(20)](buf5, buf6, 0, 20, XBLOCK=32, num_warps=1, num_stages=1)
        buf7 = empty_strided_cuda((1, 10, 14, 14), (1984, 196, 14, 1), torch.float32)
        buf8 = empty_strided_cuda((1, 10, 14, 14), (1968, 196, 14, 1), torch.int64)

        get_raw_stream(0)
        triton_poi_fused_fractional_max_pool2d_2[grid(1960)](buf6, buf4, buf7, buf8, 1960, XBLOCK=256, num_warps=4, num_stages=1)
        del buf6
        buf9 = empty_strided_cuda((1, 1, 14, 14), (196, 196, 14, 1), torch.float32)
        buf10 = empty_strided_cuda((1, 1, 14, 14), (196, 196, 14, 1), torch.float32)

        get_raw_stream(0)
        triton_per_fused__softmax_3[grid(196)](buf7, buf9, buf10, 196, 10, XBLOCK=1, num_warps=2, num_stages=1)
        buf11 = buf7; del buf7
        buf12 = empty_strided_cuda((1, 10, 1), (10, 1, 1), torch.float32)
        buf13 = empty_strided_cuda((1, 10, 1), (10, 1, 10), torch.float32)
        buf15 = reinterpret_tensor(buf13, (1, 10, 1), (10, 1, 1), 0); del buf13

        get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit__softmax_4[grid(10)](buf11, buf15, buf9, buf10, buf12, 10, 196, XBLOCK=1, num_warps=2, num_stages=1)
        del buf10
        del buf9
        buf16 = empty_strided_cuda((1, 14, 14), (196, 14, 1), torch.int64)
        buf19 = empty_strided_cuda((), (), torch.float32)
        buf18 = empty_strided_cuda((), (), torch.float32)
        buf28 = buf19; del buf19

        get_raw_stream(0)
        triton_per_fused_nll_loss2d_forward_randint_5[grid(1)](buf28, buf5, buf11, buf12, buf15, buf16, buf18, 1, 1, 196, XBLOCK=1, num_warps=2, num_stages=1)
        del buf5

        get_raw_stream(0)
        triton_poi_fused_add_6[grid(1)](primals_4, primals_4, 1, XBLOCK=1, num_warps=1, num_stages=1)
        del primals_4
    return (buf28, primals_3, reinterpret_tensor(buf3, (10, ), (1, ), 0), buf4, buf8, buf11, buf12, buf15, buf16, buf18, reinterpret_tensor(buf0, (1, 10, 1, 1), (10, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = 28
    primals_2 = 28
    primals_3 = rand_strided((1, 10, 28, 28), (7840, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_5 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
