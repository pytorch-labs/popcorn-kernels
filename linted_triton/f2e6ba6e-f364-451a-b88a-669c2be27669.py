# AOT ID: ['31_inference']
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


# kernel path: /tmp/torchinductor_sahanp/ry/cryyxblkvqzpcricnsstxokawtwtkqjdioazvdyziwh72s5is76g.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x => _low_memory_max_pool2d_with_offsets
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%unsqueeze, [1, 2], [1, 2], [0, 0], [1, 1], False), kwargs = {})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(out_ptr0 + (x0), tmp2, xmask)


# kernel path: /tmp/torchinductor_sahanp/u6/cu6nnazcj35j4s6ky3kheno3qscszvghjmh7hjxhe72sjtkt6tzx.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_2 => constant_pad_nd
# Graph fragment:
#   %constant_pad_nd : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%unsqueeze_2, [0, 0, 2, 2], 0.0), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_constant_pad_nd_1(in_ptr0, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 16
    x2 = xindex
    tmp0 = (-2) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = ks0
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + ((-64) + 2*x2), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr0 + ((-63) + 2*x2), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp9 = tmp8 * tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp5, tmp9, tmp10)
    tl.store(out_ptr0 + (x2), tmp11, xmask)


# kernel path: /tmp/torchinductor_sahanp/jl/cjlwcur74h7gszc7hw4x42yv3tvkhqvtwoxjp6kechmq4nt5mi4l.py
# Topologically Sorted Source Nodes: [x_2, x_3, loss], Original ATen: [aten.mul, aten.add, aten.pow, aten.div, aten.softplus, aten.huber_loss]
# Source node to ATen node mapping:
#   loss => abs_1, lt_1, mean, mul_72, mul_73, mul_74, sub_22, where, where_1
#   x_2 => add_54, div, mul_52, pow_1
#   x_3 => div_1, exp, gt, log1p, mul_65
# Graph fragment:
#   %mul_52 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_4, 0.0001), kwargs = {})
#   %add_54 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_52, 1.0), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_54, 0.75), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%squeeze_2, %pow_1), kwargs = {})
#   %mul_65 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, 1.0), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%mul_65, 20.0), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_65,), kwargs = {})
#   %log1p : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp,), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%log1p, 1.0), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %div, %div_1), kwargs = {})
#   %abs_1 : [num_users=4] = call_function[target=torch.ops.aten.abs.default](args = (%where,), kwargs = {})
#   %lt_1 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%abs_1, 1.0), kwargs = {})
#   %mul_72 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%abs_1, 0.5), kwargs = {})
#   %mul_73 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_72, %abs_1), kwargs = {})
#   %sub_22 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%abs_1, 0.5), kwargs = {})
#   %mul_74 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_22, 1.0), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%lt_1, %mul_73, %mul_74), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%where_1,), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_div_huber_loss_mul_pow_softplus_2(in_out_ptr0, in_ptr0, in_ptr1, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    _tmp37 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (2*r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr0 + (1 + 2*r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (16 + r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr1 + (32 + r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr1 + (48 + r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.load(in_ptr1 + (64 + r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp2 = triton_helpers.maximum(tmp1, tmp0)
        tmp5 = tmp4 + tmp3
        tmp7 = tmp6 + tmp5
        tmp9 = tmp8 + tmp7
        tmp11 = tmp10 + tmp9
        tmp12 = 0.2
        tmp13 = tmp11 * tmp12
        tmp14 = 0.0001
        tmp15 = tmp13 * tmp14
        tmp16 = 1.0
        tmp17 = tmp15 + tmp16
        tmp18 = 0.75
        tmp19 = libdevice.pow(tmp17, tmp18)
        tmp20 = tmp2 / tmp19
        tmp21 = tmp20 * tmp16
        tmp22 = 20.0
        tmp23 = tmp21 > tmp22
        tmp24 = tl_math.exp(tmp21)
        tmp25 = libdevice.log1p(tmp24)
        tmp26 = tmp25 * tmp16
        tmp27 = tl.where(tmp23, tmp20, tmp26)
        tmp28 = tl_math.abs(tmp27)
        tmp29 = tmp28 < tmp16
        tmp30 = 0.5
        tmp31 = tmp28 * tmp30
        tmp32 = tmp31 * tmp28
        tmp33 = tmp28 - tmp30
        tmp34 = tmp33 * tmp16
        tmp35 = tl.where(tmp29, tmp32, tmp34)
        tmp36 = tl.broadcast_to(tmp35, [XBLOCK, R0_BLOCK])
        tmp38 = _tmp37 + tmp36
        _tmp37 = tl.where(r0_mask, tmp38, _tmp37)
    tmp37 = tl.sum(_tmp37, 1)[:, None]
    tmp39 = 16*ks0
    tmp40 = tmp39.to(tl.float32)
    tmp41 = tmp37 / tmp40
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp41, None)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    s0 = arg0_1
    assert_size_stride(arg1_1, (1, s0, 64), (64*s0, 64, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, s0, 1, 32), (32*s0, 32, 32*s0, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_0_xnumel = 32*s0
        get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_0[grid(triton_poi_fused_max_pool2d_with_indices_0_xnumel)](arg1_1, buf0, 320, XBLOCK=256, num_warps=4, num_stages=1)
        del arg1_1
        buf1 = empty_strided_cuda((1, 1, 4 + s0, 16), (64 + 16*s0, 64 + 16*s0, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_1_xnumel = 64 + 16*s0
        get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_1[grid(triton_poi_fused_constant_pad_nd_1_xnumel)](buf0, buf1, 10, 224, XBLOCK=256, num_warps=4, num_stages=1)
        buf3 = empty_strided_cuda((), (), torch.float32)
        buf4 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [x_2, x_3, loss], Original ATen: [aten.mul, aten.add, aten.pow, aten.div, aten.softplus, aten.huber_loss]
        16*s0
        get_raw_stream(0)
        triton_red_fused_add_div_huber_loss_mul_pow_softplus_2[grid(1)](buf4, buf0, buf1, 10, 1, 160, XBLOCK=1, R0_BLOCK=256, num_warps=2, num_stages=1)
        del buf0
        del buf1
    return (buf4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 10
    arg1_1 = rand_strided((1, 10, 64), (640, 64, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
