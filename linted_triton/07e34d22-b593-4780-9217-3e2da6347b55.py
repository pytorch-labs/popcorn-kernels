
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
def triton_poi_fused_avg_pool2d_reflection_pad2d_relu_0(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks1)
    x2 = xindex // ks2
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (ks4*(tl.where((-1) + ks3 + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs((-2) + 2*x1))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs((-2) + 2*x1))) + 2*ks3, (-1) + ks3 + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs((-2) + 2*x1))))) + ks3*ks4*x2 + (tl.where((-1) + ks4 + ((-1)*tl_math.abs(1 + ((-1)*ks4) + tl_math.abs((-2) + 2*x0))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks4) + tl_math.abs((-2) + 2*x0))) + 2*ks4, (-1) + ks4 + ((-1)*tl_math.abs(1 + ((-1)*ks4) + tl_math.abs((-2) + 2*x0)))))), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (ks4*(tl.where((-1) + ks3 + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs((-2) + 2*x1))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs((-2) + 2*x1))) + 2*ks3, (-1) + ks3 + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs((-2) + 2*x1))))) + ks3*ks4*x2 + (tl.where((-1) + ks4 + ((-1)*tl_math.abs(1 + ((-1)*ks4) + tl_math.abs((-1) + 2*x0))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks4) + tl_math.abs((-1) + 2*x0))) + 2*ks4, (-1) + ks4 + ((-1)*tl_math.abs(1 + ((-1)*ks4) + tl_math.abs((-1) + 2*x0)))))), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (ks4*(tl.where((-1) + ks3 + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs((-1) + 2*x1))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs((-1) + 2*x1))) + 2*ks3, (-1) + ks3 + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs((-1) + 2*x1))))) + ks3*ks4*x2 + (tl.where((-1) + ks4 + ((-1)*tl_math.abs(1 + ((-1)*ks4) + tl_math.abs((-2) + 2*x0))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks4) + tl_math.abs((-2) + 2*x0))) + 2*ks4, (-1) + ks4 + ((-1)*tl_math.abs(1 + ((-1)*ks4) + tl_math.abs((-2) + 2*x0)))))), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (ks4*(tl.where((-1) + ks3 + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs((-1) + 2*x1))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs((-1) + 2*x1))) + 2*ks3, (-1) + ks3 + ((-1)*tl_math.abs(1 + ((-1)*ks3) + tl_math.abs((-1) + 2*x1))))) + ks3*ks4*x2 + (tl.where((-1) + ks4 + ((-1)*tl_math.abs(1 + ((-1)*ks4) + tl_math.abs((-1) + 2*x0))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks4) + tl_math.abs((-1) + 2*x0))) + 2*ks4, (-1) + ks4 + ((-1)*tl_math.abs(1 + ((-1)*ks4) + tl_math.abs((-1) + 2*x0)))))), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp1, tmp3)
    tmp5 = tmp4 + tmp2
    tmp7 = triton_helpers.maximum(tmp1, tmp6)
    tmp8 = tmp7 + tmp5
    tmp10 = triton_helpers.maximum(tmp1, tmp9)
    tmp11 = tmp10 + tmp8
    tmp12 = 0.25
    tmp13 = tmp11 * tmp12
    tl.store(out_ptr0 + (x3), tmp13, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_pow_1(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = libdevice.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tmp9 = tmp8 * tmp8
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_avg_pool2d_relu_2(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks1)
    x2 = xindex // ks2
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x2 + 2*x0 + 2*x1 + x2*(triton_helpers.div_floor_integer((-1) + (ks3 // 2),  2)) + x2*(triton_helpers.div_floor_integer((-1) + (ks4 // 2),  2)) + 2*x1*(triton_helpers.div_floor_integer((-1) + (ks4 // 2),  2)) + x2*(triton_helpers.div_floor_integer((-1) + (ks3 // 2),  2))*(triton_helpers.div_floor_integer((-1) + (ks4 // 2),  2))), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (1 + x2 + 2*x0 + 2*x1 + x2*(triton_helpers.div_floor_integer((-1) + (ks3 // 2),  2)) + x2*(triton_helpers.div_floor_integer((-1) + (ks4 // 2),  2)) + 2*x1*(triton_helpers.div_floor_integer((-1) + (ks4 // 2),  2)) + x2*(triton_helpers.div_floor_integer((-1) + (ks3 // 2),  2))*(triton_helpers.div_floor_integer((-1) + (ks4 // 2),  2))), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr0 + (1 + x2 + 2*x0 + 2*x1 + x2*(triton_helpers.div_floor_integer((-1) + (ks3 // 2),  2)) + x2*(triton_helpers.div_floor_integer((-1) + (ks4 // 2),  2)) + 2*x1*(triton_helpers.div_floor_integer((-1) + (ks4 // 2),  2)) + x2*(triton_helpers.div_floor_integer((-1) + (ks3 // 2),  2))*(triton_helpers.div_floor_integer((-1) + (ks4 // 2),  2)) + (triton_helpers.div_floor_integer((-1) + (ks4 // 2),  2))), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr0 + (2 + x2 + 2*x0 + 2*x1 + x2*(triton_helpers.div_floor_integer((-1) + (ks3 // 2),  2)) + x2*(triton_helpers.div_floor_integer((-1) + (ks4 // 2),  2)) + 2*x1*(triton_helpers.div_floor_integer((-1) + (ks4 // 2),  2)) + x2*(triton_helpers.div_floor_integer((-1) + (ks3 // 2),  2))*(triton_helpers.div_floor_integer((-1) + (ks4 // 2),  2)) + (triton_helpers.div_floor_integer((-1) + (ks4 // 2),  2))), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp1 < tmp0
    tmp3 = tmp2.to(tl.int8)
    tmp4 = tmp0 < tmp1
    tmp5 = tmp4.to(tl.int8)
    tmp6 = tmp3 - tmp5
    tmp7 = tmp6.to(tmp0.dtype)
    tmp8 = tl_math.abs(tmp0)
    tmp9 = triton_helpers.maximum(tmp1, tmp8)
    tmp10 = tmp7 * tmp9
    tmp11 = 27.0
    tmp12 = tmp10 * tmp11
    tmp13 = libdevice.sqrt(tmp12)
    tmp14 = triton_helpers.maximum(tmp1, tmp13)
    tmp16 = tmp1 < tmp15
    tmp17 = tmp16.to(tl.int8)
    tmp18 = tmp15 < tmp1
    tmp19 = tmp18.to(tl.int8)
    tmp20 = tmp17 - tmp19
    tmp21 = tmp20.to(tmp15.dtype)
    tmp22 = tl_math.abs(tmp15)
    tmp23 = triton_helpers.maximum(tmp1, tmp22)
    tmp24 = tmp21 * tmp23
    tmp25 = tmp24 * tmp11
    tmp26 = libdevice.sqrt(tmp25)
    tmp27 = triton_helpers.maximum(tmp1, tmp26)
    tmp28 = tmp27 + tmp14
    tmp30 = tmp1 < tmp29
    tmp31 = tmp30.to(tl.int8)
    tmp32 = tmp29 < tmp1
    tmp33 = tmp32.to(tl.int8)
    tmp34 = tmp31 - tmp33
    tmp35 = tmp34.to(tmp29.dtype)
    tmp36 = tl_math.abs(tmp29)
    tmp37 = triton_helpers.maximum(tmp1, tmp36)
    tmp38 = tmp35 * tmp37
    tmp39 = tmp38 * tmp11
    tmp40 = libdevice.sqrt(tmp39)
    tmp41 = triton_helpers.maximum(tmp1, tmp40)
    tmp42 = tmp41 + tmp28
    tmp44 = tmp1 < tmp43
    tmp45 = tmp44.to(tl.int8)
    tmp46 = tmp43 < tmp1
    tmp47 = tmp46.to(tl.int8)
    tmp48 = tmp45 - tmp47
    tmp49 = tmp48.to(tmp43.dtype)
    tmp50 = tl_math.abs(tmp43)
    tmp51 = triton_helpers.maximum(tmp1, tmp50)
    tmp52 = tmp49 * tmp51
    tmp53 = tmp52 * tmp11
    tmp54 = libdevice.sqrt(tmp53)
    tmp55 = triton_helpers.maximum(tmp1, tmp54)
    tmp56 = tmp55 + tmp42
    tmp57 = 0.25
    tmp58 = tmp56 * tmp57
    tl.store(out_ptr0 + (x3), tmp58, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_binary_cross_entropy_gelu_zeros_like_3(in_out_ptr0, in_ptr0, ks0, ks1, ks2, ks3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    _tmp21 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_0 = (r0_index % ks0)
        r0_1 = r0_index // ks0
        tmp0 = tl.load(in_ptr0 + (r0_0 + ks1*r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.5
        tmp2 = tmp0 * tmp1
        tmp3 = 0.7071067811865476
        tmp4 = tmp0 * tmp3
        tmp5 = libdevice.erf(tmp4)
        tmp6 = 1.0
        tmp7 = tmp5 + tmp6
        tmp8 = tmp2 * tmp7
        tmp9 = -tmp8
        tmp10 = libdevice.log1p(tmp9)
        tmp11 = -100.0
        tmp12 = triton_helpers.maximum(tmp10, tmp11)
        tmp13 = -1.0
        tmp14 = tmp13 * tmp12
        tmp15 = tl_math.log(tmp8)
        tmp16 = triton_helpers.maximum(tmp15, tmp11)
        tmp17 = 0.0
        tmp18 = tmp17 * tmp16
        tmp19 = tmp14 - tmp18
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(r0_mask, tmp22, _tmp21)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tmp23 = 1 + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks2 // 2),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks3 // 2),  2)),  2)) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks2 // 2),  2)),  2)) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (ks3 // 2),  2)),  2))
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp21 / tmp24
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp25, None)


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

        triton_poi_fused_avg_pool2d_reflection_pad2d_relu_0_xnumel = 4*s0 + 2*s0*(s1 // 2) + 2*s0*(s2 // 2) + s0*(s1 // 2)*(s2 // 2)
        get_raw_stream(0)
        triton_poi_fused_avg_pool2d_reflection_pad2d_relu_0[grid(triton_poi_fused_avg_pool2d_reflection_pad2d_relu_0_xnumel)](arg3_1, buf0, 34, 34, 1156, 64, 64, 3468, XBLOCK=128, num_warps=4, num_stages=1)
        del arg3_1
        buf1 = reinterpret_tensor(buf0, (1, 1, s0, 2 + (s1 // 2), 2 + (s2 // 2)), (4*s0 + 2*s0*(s1 // 2) + 2*s0*(s2 // 2) + s0*(s1 // 2)*(s2 // 2), 4*s0 + 2*s0*(s1 // 2) + 2*s0*(s2 // 2) + s0*(s1 // 2)*(s2 // 2), 4 + 2*(s1 // 2) + 2*(s2 // 2) + (s1 // 2)*(s2 // 2), 2 + (s2 // 2), 1), 0); del buf0

        triton_poi_fused_pow_1_xnumel = 4*s0 + 2*s0*(s1 // 2) + 2*s0*(s2 // 2) + s0*(s1 // 2)*(s2 // 2)
        get_raw_stream(0)
        triton_poi_fused_pow_1[grid(triton_poi_fused_pow_1_xnumel)](buf1, 3468, XBLOCK=128, num_warps=4, num_stages=1)

        buf2 = torch.ops.aten.avg_pool3d.default(buf1, [3, 3, 3], [2, 2, 2], [0, 0, 0], False, True, None)
        del buf1
        buf3 = buf2
        del buf2
        (1 + (((-1) + (s2 // 2)) // 2)) // 2
        (1 + (((-1) + (s1 // 2)) // 2)) // 2
        ((1 + (((-1) + (s1 // 2)) // 2)) // 2)*((1 + (((-1) + (s2 // 2)) // 2)) // 2)
        buf4 = empty_strided_cuda((1, 1 + (((-3) + s0) // 2), (1 + (((-1) + (s1 // 2)) // 2)) // 2, (1 + (((-1) + (s2 // 2)) // 2)) // 2), (((1 + (((-1) + (s1 // 2)) // 2)) // 2)*((1 + (((-1) + (s2 // 2)) // 2)) // 2) + ((1 + (((-1) + (s1 // 2)) // 2)) // 2)*((1 + (((-1) + (s2 // 2)) // 2)) // 2)*(((-3) + s0) // 2), ((1 + (((-1) + (s1 // 2)) // 2)) // 2)*((1 + (((-1) + (s2 // 2)) // 2)) // 2), (1 + (((-1) + (s2 // 2)) // 2)) // 2, 1), torch.float32)

        triton_poi_fused_avg_pool2d_relu_2_xnumel = ((1 + (((-1) + (s1 // 2)) // 2)) // 2)*((1 + (((-1) + (s2 // 2)) // 2)) // 2) + ((1 + (((-1) + (s1 // 2)) // 2)) // 2)*((1 + (((-1) + (s2 // 2)) // 2)) // 2)*(((-3) + s0) // 2)
        get_raw_stream(0)
        triton_poi_fused_avg_pool2d_relu_2[grid(triton_poi_fused_avg_pool2d_relu_2_xnumel)](buf3, buf4, 8, 8, 64, 64, 64, 64, XBLOCK=64, num_warps=1, num_stages=1)
        del buf3
        1 + (((-1) + (((-1) + (s2 // 2)) // 2)) // 2)
        buf5 = empty_strided_cuda((), (), torch.float32)
        buf6 = buf5; del buf5

        1 + (((-1) + (((-1) + (s1 // 2)) // 2)) // 2)*(((-1) + (((-1) + (s2 // 2)) // 2)) // 2) + (((-1) + (((-1) + (s1 // 2)) // 2)) // 2) + (((-1) + (((-1) + (s2 // 2)) // 2)) // 2)
        get_raw_stream(0)
        triton_red_fused_binary_cross_entropy_gelu_zeros_like_3[grid(1)](buf6, buf4, 8, 8, 64, 64, 1, 64, XBLOCK=1, R0_BLOCK=64, num_warps=2, num_stages=1)
        del buf4
    return (buf6, )


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
