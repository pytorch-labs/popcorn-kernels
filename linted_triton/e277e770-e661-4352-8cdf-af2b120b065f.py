# AOT ID: ['186_inference']
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


# kernel path: /tmp/torchinductor_sahanp/75/c75uhi3u7elcuuabyap7qu227jqcphe7ehzzfm5id6zpag4y7od5.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_1 => constant_pad_nd
# Graph fragment:
#   %constant_pad_nd : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%unsqueeze_1, [0, 0, 2, 2], 0.0), kwargs = {})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_constant_pad_nd_0(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // ks0) % ks1)
    x0 = (xindex % ks0)
    x2 = xindex // ks3
    x3 = xindex
    tmp0 = (-2) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = ks2
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (((-2)*ks4) + 2*x0 + ks4*x1 + ks2*ks4*x2), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr0 + (1 + ((-2)*ks4) + 2*x0 + ks4*x1 + ks2*ks4*x2), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp7 + tmp6
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = tmp10 * tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp5, tmp11, tmp12)
    tl.store(out_ptr0 + (x3), tmp13, xmask)


# kernel path: /tmp/torchinductor_sahanp/wu/cwukxm2odxubyz7ecofd6oogqkdporv35ip5kyzpptsxqbmab2mp.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.mul, aten.add, aten.pow, aten.div]
# Source node to ATen node mapping:
#   x_1 => add_41, div, mul_34, pow_1
# Graph fragment:
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_1, 0.0001), kwargs = {})
#   %add_41 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_34, 1.0), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_41, 0.75), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%squeeze, %pow_1), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_div_mul_pow_1(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x3 = xindex // ks0
    x2 = xindex // ks2
    x4 = (xindex % ks2)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + ks1*x3), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + ks1*x3), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x4 + 4*ks0*x2 + ks0*ks3*x2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (ks0 + x4 + 4*ks0*x2 + ks0*ks3*x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x4 + 2*ks0 + 4*ks0*x2 + ks0*ks3*x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr1 + (x4 + 3*ks0 + 4*ks0*x2 + ks0*ks3*x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (x4 + 4*ks0 + 4*ks0*x2 + ks0*ks3*x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp7 = tmp6 + tmp5
    tmp9 = tmp8 + tmp7
    tmp11 = tmp10 + tmp9
    tmp13 = tmp12 + tmp11
    tmp14 = 0.2
    tmp15 = tmp13 * tmp14
    tmp16 = 0.0001
    tmp17 = tmp15 * tmp16
    tmp18 = 1.0
    tmp19 = tmp17 + tmp18
    tmp20 = 0.75
    tmp21 = libdevice.pow(tmp19, tmp20)
    tmp22 = tmp4 / tmp21
    tl.store(out_ptr0 + (x5), tmp22, xmask)


# kernel path: /tmp/torchinductor_sahanp/ex/cexnz42zi4jczi3sbjkpvgp6ip4cwab33tm2ux55qog7anckqbpz.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten.sub, aten.add, aten.norm]
# Source node to ATen node mapping:
#   loss => add_74, add_83, pow_2, pow_4, sub_47, sub_53, sum_1, sum_2
# Graph fragment:
#   %sub_47 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %view), kwargs = {})
#   %add_74 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%sub_47, 1e-06), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_74, 2.0), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_2, [1]), kwargs = {})
#   %sub_53 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %view), kwargs = {})
#   %add_83 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%sub_53, 1e-06), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_83, 2.0), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_4, [1]), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_norm_sub_2(in_ptr0, out_ptr0, out_ptr1, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (2*((r0_1 % (ks2 // 4))) + ks0*(triton_helpers.div_floor_integer(r0_1,  ks2 // 4)) + ks0*ks1*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr0 + (1 + 2*((r0_1 % (ks2 // 4))) + ks0*(triton_helpers.div_floor_integer(r0_1,  ks2 // 4)) + ks0*ks1*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp1 + tmp0
        tmp3 = 0.5
        tmp4 = tmp2 * tmp3
        tmp5 = tmp4 - tmp4
        tmp6 = 1e-06
        tmp7 = tmp5 + tmp6
        tmp8 = tmp7 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp10, xmask)


# kernel path: /tmp/torchinductor_sahanp/gq/cgqvdkizzsebdty6l53evx73hcpprubodtyttqcia3zh43eqhn5h.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten.norm, aten.add, aten.sub, aten.clamp_min, aten.mean]
# Source node to ATen node mapping:
#   loss => add_89, clamp_min, mean, pow_3, pow_5, sub_60
# Graph fragment:
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %add_89 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%pow_3, 1.0), kwargs = {})
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_2, 0.5), kwargs = {})
#   %sub_60 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_89, %pow_5), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_60, 0), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%clamp_min,), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_clamp_min_mean_norm_sub_3(in_out_ptr0, in_ptr0, in_ptr1, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = libdevice.sqrt(tmp0)
        tmp2 = 1.0
        tmp3 = tmp1 + tmp2
        tmp5 = libdevice.sqrt(tmp4)
        tmp6 = tmp3 - tmp5
        tmp7 = 0.0
        tmp8 = triton_helpers.maximum(tmp6, tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tmp12 = ks0
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp10 / tmp13
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp14, None)


def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (s0, s1, s2), (s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        s2 // 2
        4 + s1
        4*(s2 // 2) + s1*(s2 // 2)
        buf0 = empty_strided_cuda((s0, 1, 4 + s1, s2 // 2), (4*(s2 // 2) + s1*(s2 // 2), 4*s0*(s2 // 2) + s0*s1*(s2 // 2), s2 // 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_0_xnumel = 4*s0*(s2 // 2) + s0*s1*(s2 // 2)
        get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_0[grid(triton_poi_fused_constant_pad_nd_0_xnumel)](arg3_1, buf0, 32, 7, 3, 224, 64, 2240, XBLOCK=128, num_warps=4, num_stages=1)
        s1*(s2 // 2)
        buf1 = empty_strided_cuda((s0, s1, s2 // 2), (s1*(s2 // 2), s2 // 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.mul, aten.add, aten.pow, aten.div]
        triton_poi_fused_add_div_mul_pow_1_xnumel = s0*s1*(s2 // 2)
        get_raw_stream(0)
        triton_poi_fused_add_div_mul_pow_1[grid(triton_poi_fused_add_div_mul_pow_1_xnumel)](arg3_1, buf0, buf1, 32, 64, 96, 3, 960, XBLOCK=256, num_warps=4, num_stages=1)
        del arg3_1
        del buf0
        buf2 = empty_strided_cuda((s0, ), (1, ), torch.float32)
        buf3 = empty_strided_cuda((s0, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [loss], Original ATen: [aten.sub, aten.add, aten.norm]
        s1*(s2 // 4)
        get_raw_stream(0)
        triton_red_fused_add_norm_sub_2[grid(s0)](buf1, buf2, buf3, 32, 3, 64, 10, 48, XBLOCK=16, R0_BLOCK=8, num_warps=2, num_stages=1)
        del buf1
        buf4 = empty_strided_cuda((), (), torch.float32)
        buf5 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [loss], Original ATen: [aten.norm, aten.add, aten.sub, aten.clamp_min, aten.mean]
        get_raw_stream(0)
        triton_red_fused_add_clamp_min_mean_norm_sub_3[grid(1)](buf5, buf2, buf3, 10, 1, 10, XBLOCK=1, R0_BLOCK=16, num_warps=2, num_stages=1)
        del buf2
        del buf3
    return (buf5, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 10
    arg1_1 = 3
    arg2_1 = 64
    arg3_1 = rand_strided((10, 3, 64), (192, 64, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
