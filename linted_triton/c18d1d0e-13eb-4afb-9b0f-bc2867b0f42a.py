# AOT ID: ['102_inference']
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


# kernel path: /tmp/torchinductor_sahanp/pt/cptwwyqidth7dqasoci257uqbiu2wzesaumggpjnart3tm6ib62i.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   x_2 => pow_3, sum_2
# Graph fragment:
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%slice_5, 2), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_3, [1], True), kwargs = {})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_linalg_vector_norm_0(in_ptr0, out_ptr0, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (ks1*r0_1 + ks1*(ks0 // 2) + (tl.where((-1) + ks1 + ((-1)*tl_math.abs(1 + ((-1)*ks1) + tl_math.abs((-2) + (tl.where(3 + ks1 + ((-1)*tl_math.abs(3 + ks1 + ((-1)*tl_math.abs((-3) + x0)))) < 0, 7 + ((-1)*tl_math.abs(3 + ks1 + ((-1)*tl_math.abs((-3) + x0)))) + 2*ks1, 3 + ks1 + ((-1)*tl_math.abs(3 + ks1 + ((-1)*tl_math.abs((-3) + x0))))))))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks1) + tl_math.abs((-2) + (tl.where(3 + ks1 + ((-1)*tl_math.abs(3 + ks1 + ((-1)*tl_math.abs((-3) + x0)))) < 0, 7 + ((-1)*tl_math.abs(3 + ks1 + ((-1)*tl_math.abs((-3) + x0)))) + 2*ks1, 3 + ks1 + ((-1)*tl_math.abs(3 + ks1 + ((-1)*tl_math.abs((-3) + x0))))))))) + 2*ks1, (-1) + ks1 + ((-1)*tl_math.abs(1 + ((-1)*ks1) + tl_math.abs((-2) + (tl.where(3 + ks1 + ((-1)*tl_math.abs(3 + ks1 + ((-1)*tl_math.abs((-3) + x0)))) < 0, 7 + ((-1)*tl_math.abs(3 + ks1 + ((-1)*tl_math.abs((-3) + x0)))) + 2*ks1, 3 + ks1 + ((-1)*tl_math.abs(3 + ks1 + ((-1)*tl_math.abs((-3) + x0)))))))))))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0 * tmp0
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(r0_mask & xmask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp3, xmask)


# kernel path: /tmp/torchinductor_sahanp/jw/cjwvrvpxjttsc6nff3g7o6bx33g5ssdypidu2acywvgrc6khidha.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.linalg_vector_norm, aten.clamp_min, aten.div, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   x_2 => clamp_min, clamp_min_1, div, div_1, mul_56, pow_1, pow_2, pow_4, sum_1, sum_3
# Graph fragment:
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%slice_2, 2), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1], True), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_2, 1e-08), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%slice_2, %clamp_min), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_2, 0.5), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_4, 1e-08), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%slice_5, %clamp_min_1), kwargs = {})
#   %mul_56 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %div), kwargs = {})
#   %sum_3 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_56, [1]), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_clamp_min_div_linalg_vector_norm_mul_sum_1(in_out_ptr0, in_ptr0, in_ptr1, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (ks0*r0_1 + (tl.where((-1) + ks0 + ((-1)*tl_math.abs(1 + ((-1)*ks0) + tl_math.abs((-2) + (tl.where(3 + ks0 + ((-1)*tl_math.abs(3 + ks0 + ((-1)*tl_math.abs((-3) + x0)))) < 0, 7 + ((-1)*tl_math.abs(3 + ks0 + ((-1)*tl_math.abs((-3) + x0)))) + 2*ks0, 3 + ks0 + ((-1)*tl_math.abs(3 + ks0 + ((-1)*tl_math.abs((-3) + x0))))))))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks0) + tl_math.abs((-2) + (tl.where(3 + ks0 + ((-1)*tl_math.abs(3 + ks0 + ((-1)*tl_math.abs((-3) + x0)))) < 0, 7 + ((-1)*tl_math.abs(3 + ks0 + ((-1)*tl_math.abs((-3) + x0)))) + 2*ks0, 3 + ks0 + ((-1)*tl_math.abs(3 + ks0 + ((-1)*tl_math.abs((-3) + x0))))))))) + 2*ks0, (-1) + ks0 + ((-1)*tl_math.abs(1 + ((-1)*ks0) + tl_math.abs((-2) + (tl.where(3 + ks0 + ((-1)*tl_math.abs(3 + ks0 + ((-1)*tl_math.abs((-3) + x0)))) < 0, 7 + ((-1)*tl_math.abs(3 + ks0 + ((-1)*tl_math.abs((-3) + x0)))) + 2*ks0, 3 + ks0 + ((-1)*tl_math.abs(3 + ks0 + ((-1)*tl_math.abs((-3) + x0)))))))))))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0 * tmp0
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(r0_mask & xmask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tmp11 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    _tmp17 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp5 = tl.load(in_ptr0 + (ks0*r0_1 + (tl.where((-1) + ks0 + ((-1)*tl_math.abs(1 + ((-1)*ks0) + tl_math.abs((-2) + (tl.where(3 + ks0 + ((-1)*tl_math.abs(3 + ks0 + ((-1)*tl_math.abs((-3) + x0)))) < 0, 7 + ((-1)*tl_math.abs(3 + ks0 + ((-1)*tl_math.abs((-3) + x0)))) + 2*ks0, 3 + ks0 + ((-1)*tl_math.abs(3 + ks0 + ((-1)*tl_math.abs((-3) + x0))))))))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks0) + tl_math.abs((-2) + (tl.where(3 + ks0 + ((-1)*tl_math.abs(3 + ks0 + ((-1)*tl_math.abs((-3) + x0)))) < 0, 7 + ((-1)*tl_math.abs(3 + ks0 + ((-1)*tl_math.abs((-3) + x0)))) + 2*ks0, 3 + ks0 + ((-1)*tl_math.abs(3 + ks0 + ((-1)*tl_math.abs((-3) + x0))))))))) + 2*ks0, (-1) + ks0 + ((-1)*tl_math.abs(1 + ((-1)*ks0) + tl_math.abs((-2) + (tl.where(3 + ks0 + ((-1)*tl_math.abs(3 + ks0 + ((-1)*tl_math.abs((-3) + x0)))) < 0, 7 + ((-1)*tl_math.abs(3 + ks0 + ((-1)*tl_math.abs((-3) + x0)))) + 2*ks0, 3 + ks0 + ((-1)*tl_math.abs(3 + ks0 + ((-1)*tl_math.abs((-3) + x0)))))))))))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.load(in_ptr0 + (ks0*r0_1 + ks0*(ks1 // 2) + (tl.where((-1) + ks0 + ((-1)*tl_math.abs(1 + ((-1)*ks0) + tl_math.abs((-2) + (tl.where(3 + ks0 + ((-1)*tl_math.abs(3 + ks0 + ((-1)*tl_math.abs((-3) + x0)))) < 0, 7 + ((-1)*tl_math.abs(3 + ks0 + ((-1)*tl_math.abs((-3) + x0)))) + 2*ks0, 3 + ks0 + ((-1)*tl_math.abs(3 + ks0 + ((-1)*tl_math.abs((-3) + x0))))))))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks0) + tl_math.abs((-2) + (tl.where(3 + ks0 + ((-1)*tl_math.abs(3 + ks0 + ((-1)*tl_math.abs((-3) + x0)))) < 0, 7 + ((-1)*tl_math.abs(3 + ks0 + ((-1)*tl_math.abs((-3) + x0)))) + 2*ks0, 3 + ks0 + ((-1)*tl_math.abs(3 + ks0 + ((-1)*tl_math.abs((-3) + x0))))))))) + 2*ks0, (-1) + ks0 + ((-1)*tl_math.abs(1 + ((-1)*ks0) + tl_math.abs((-2) + (tl.where(3 + ks0 + ((-1)*tl_math.abs(3 + ks0 + ((-1)*tl_math.abs((-3) + x0)))) < 0, 7 + ((-1)*tl_math.abs(3 + ks0 + ((-1)*tl_math.abs((-3) + x0)))) + 2*ks0, 3 + ks0 + ((-1)*tl_math.abs(3 + ks0 + ((-1)*tl_math.abs((-3) + x0)))))))))))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = libdevice.sqrt(tmp3)
        tmp7 = 1e-08
        tmp8 = triton_helpers.maximum(tmp6, tmp7)
        tmp9 = tmp5 / tmp8
        tmp12 = libdevice.sqrt(tmp11)
        tmp13 = triton_helpers.maximum(tmp12, tmp7)
        tmp14 = tmp10 / tmp13
        tmp15 = tmp9 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(r0_mask & xmask, tmp18, _tmp17)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(in_out_ptr0 + (x0), tmp17, xmask)


# kernel path: /tmp/torchinductor_sahanp/3k/c3kx6q5nkvt4ta5oz3sqdzj6wacfbypcryn6jejxhnmzrgfld462.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten.exp, aten.mul, aten.sub, aten.mean]
# Source node to ATen node mapping:
#   loss => exp, mean, mul_62, sub_45
# Graph fragment:
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sum_3,), kwargs = {})
#   %mul_62 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_3, %sum_3), kwargs = {})
#   %sub_45 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%exp, %mul_62), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sub_45,), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_exp_mean_mul_sub_2(in_out_ptr0, in_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    _tmp5 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl_math.exp(tmp0)
        tmp2 = tmp0 * tmp0
        tmp3 = tmp1 - tmp2
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(r0_mask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tmp7 = 10 + ks0
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp5 / tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp9, None)


def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    assert_size_stride(arg2_1, (1, s0, s1), (s0*s1, s1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((1, 1, 10 + s1), (10 + s1, 10 + s1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_0_xnumel = 10 + s1
        s0 + ((-1)*(s0 // 2))
        get_raw_stream(0)
        triton_red_fused_linalg_vector_norm_0[grid(triton_red_fused_linalg_vector_norm_0_xnumel)](arg2_1, buf1, 10, 20, 30, 5, XBLOCK=1, R0_BLOCK=8, num_warps=2, num_stages=1)
        buf0 = empty_strided_cuda((1, 1, 10 + s1), (10 + s1, 10 + s1, 1), torch.float32)
        buf2 = reinterpret_tensor(buf0, (1, 10 + s1), (10 + s1, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.linalg_vector_norm, aten.clamp_min, aten.div, aten.mul, aten.sum]
        triton_red_fused_clamp_min_div_linalg_vector_norm_mul_sum_1_xnumel = 10 + s1
        s0 // 2
        get_raw_stream(0)
        triton_red_fused_clamp_min_div_linalg_vector_norm_mul_sum_1[grid(triton_red_fused_clamp_min_div_linalg_vector_norm_mul_sum_1_xnumel)](buf2, arg2_1, buf1, 20, 10, 30, 5, XBLOCK=1, R0_BLOCK=8, num_warps=2, num_stages=1)
        del arg2_1
        del buf1
        buf3 = empty_strided_cuda((), (), torch.float32)
        buf4 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [loss], Original ATen: [aten.exp, aten.mul, aten.sub, aten.mean]
        10 + s1
        get_raw_stream(0)
        triton_red_fused_exp_mean_mul_sub_2[grid(1)](buf4, buf2, 20, 1, 30, XBLOCK=1, R0_BLOCK=32, num_warps=2, num_stages=1)
        del buf2
    return (buf4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 10
    arg1_1 = 20
    arg2_1 = rand_strided((1, 10, 20), (200, 20, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
