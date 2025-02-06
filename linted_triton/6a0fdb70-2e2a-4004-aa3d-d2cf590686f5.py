# AOT ID: ['168_inference']
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


# kernel path: /tmp/torchinductor_sahanp/6n/c6nulscxl4jmdru7l4ovqq2stmp776x75m5tcm6gnkrnx7wtxi36.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.sign, aten.abs, aten.relu, aten.mul]
# Source node to ATen node mapping:
#   x_1 => abs_3, mul_26, mul_30, relu, sign
# Graph fragment:
#   %sign : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%squeeze,), kwargs = {})
#   %abs_3 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%squeeze,), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%abs_3,), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign, %relu), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_26, 3), kwargs = {})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_abs_mul_relu_sign_0(in_ptr0, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (tl.where((-1) + ks0 + ((-1)*tl_math.abs(1 + ((-1)*ks0) + tl_math.abs((-2) + 2*x0))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks0) + tl_math.abs((-2) + 2*x0))) + 2*ks0, (-1) + ks0 + ((-1)*tl_math.abs(1 + ((-1)*ks0) + tl_math.abs((-2) + 2*x0))))), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (tl.where((-1) + ks0 + ((-1)*tl_math.abs(1 + ((-1)*ks0) + tl_math.abs((-1) + 2*x0))) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks0) + tl_math.abs((-1) + 2*x0))) + 2*ks0, (-1) + ks0 + ((-1)*tl_math.abs(1 + ((-1)*ks0) + tl_math.abs((-1) + 2*x0))))), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (tl.where((-1) + ks0 + ((-1)*tl_math.abs(1 + ((-1)*ks0) + 2*x0)) < 0, (-1) + ((-1)*tl_math.abs(1 + ((-1)*ks0) + 2*x0)) + 2*ks0, (-1) + ks0 + ((-1)*tl_math.abs(1 + ((-1)*ks0) + 2*x0)))), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 * tmp0
    tmp3 = tmp2 * tmp2
    tmp4 = tmp3 + tmp1
    tmp6 = tmp5 * tmp5
    tmp7 = tmp6 + tmp4
    tmp8 = 0.3333333333333333
    tmp9 = tmp7 * tmp8
    tmp10 = tl.full([1], 0, tl.int32)
    tmp11 = tmp10 < tmp9
    tmp12 = tmp11.to(tl.int8)
    tmp13 = tmp9 < tmp10
    tmp14 = tmp13.to(tl.int8)
    tmp15 = tmp12 - tmp14
    tmp16 = tmp15.to(tmp9.dtype)
    tmp17 = tl_math.abs(tmp9)
    tmp18 = triton_helpers.maximum(tmp10, tmp17)
    tmp19 = tmp16 * tmp18
    tmp20 = 3.0
    tmp21 = tmp19 * tmp20
    tl.store(out_ptr0 + (x0), tmp21, xmask)


# kernel path: /tmp/torchinductor_sahanp/nc/cncmjlnbzmucbim5atrfxblzz4bugydkdaftxviaqqxn257vaxxl.py
# Topologically Sorted Source Nodes: [x_3, loss, randint_like], Original ATen: [aten.sign, aten.abs, aten.relu, aten.mul, aten.pow, aten.exp, aten.randint_like, aten.sub, aten.mean]
# Source node to ATen node mapping:
#   loss => exp, mean, mul_77, sub_37
#   randint_like => convert_element_type_default, inductor_lookup_seed_default, inductor_randint_default
#   x_3 => abs_6, mul_63, mul_67, pow_4, relu_1, sign_1
# Graph fragment:
#   %sign_1 : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%squeeze_1,), kwargs = {})
#   %abs_6 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%squeeze_1,), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%abs_6,), kwargs = {})
#   %mul_63 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign_1, %relu_1), kwargs = {})
#   %mul_67 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_63, 3), kwargs = {})
#   %pow_4 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_67, 0.5), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%pow_4,), kwargs = {})
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_randint_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_randint.default](args = (0, 10, [1, 1, %sym_size_int_2], %inductor_lookup_seed_default), kwargs = {})
#   %convert_element_type_default : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%inductor_randint_default, torch.float32), kwargs = {})
#   %mul_77 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default, %pow_4), kwargs = {})
#   %sub_37 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%exp, %mul_77), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sub_37,), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_abs_exp_mean_mul_pow_randint_like_relu_sign_sub_1(in_out_ptr1, in_ptr0, in_ptr1, ks0, load_seed_offset, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    _tmp36 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (tl.where(((-1)*tl_math.abs(((-1)*((1 + ks0) // 2)) + tl_math.abs((-1) + 2*r0_0))) + ((1 + ks0) // 2) < 0, ((-1)*tl_math.abs(((-1)*((1 + ks0) // 2)) + tl_math.abs((-1) + 2*r0_0))) + ((1 + ks0) // 2) + ((3 + ks0) // 2), ((-1)*tl_math.abs(((-1)*((1 + ks0) // 2)) + tl_math.abs((-1) + 2*r0_0))) + ((1 + ks0) // 2))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr0 + (tl.where(((-1)*tl_math.abs(((-1)*((1 + ks0) // 2)) + 2*r0_0)) + ((1 + ks0) // 2) < 0, ((-1)*tl_math.abs(((-1)*((1 + ks0) // 2)) + 2*r0_0)) + ((1 + ks0) // 2) + ((3 + ks0) // 2), ((-1)*tl_math.abs(((-1)*((1 + ks0) // 2)) + 2*r0_0)) + ((1 + ks0) // 2))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr0 + (tl.where(((-1)*tl_math.abs(1 + ((-1)*((1 + ks0) // 2)) + 2*r0_0)) + ((1 + ks0) // 2) < 0, ((-1)*tl_math.abs(1 + ((-1)*((1 + ks0) // 2)) + 2*r0_0)) + ((1 + ks0) // 2) + ((3 + ks0) // 2), ((-1)*tl_math.abs(1 + ((-1)*((1 + ks0) // 2)) + 2*r0_0)) + ((1 + ks0) // 2))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = libdevice.sqrt(tmp0)
        tmp2 = tmp1 * tmp1
        tmp4 = libdevice.sqrt(tmp3)
        tmp5 = tmp4 * tmp4
        tmp6 = tmp5 + tmp2
        tmp8 = libdevice.sqrt(tmp7)
        tmp9 = tmp8 * tmp8
        tmp10 = tmp9 + tmp6
        tmp11 = 0.3333333333333333
        tmp12 = tmp10 * tmp11
        tmp13 = tl_math.abs(tmp12)
        tmp14 = tl.full([1, 1], 0, tl.int32)
        tmp15 = triton_helpers.maximum(tmp14, tmp13)
        tmp16 = tmp14 < tmp12
        tmp17 = tmp16.to(tl.int8)
        tmp18 = tmp12 < tmp14
        tmp19 = tmp18.to(tl.int8)
        tmp20 = tmp17 - tmp19
        tmp21 = tmp20.to(tmp12.dtype)
        tmp22 = tmp21 * tmp15
        tmp23 = 3.0
        tmp24 = tmp22 * tmp23
        tmp25 = libdevice.sqrt(tmp24)
        tmp26 = tl_math.exp(tmp25)
        tmp27 = tl.load(in_ptr1 + load_seed_offset)
        tmp28 = r0_0
        tmp29 = tl.full([1, 1], 0, tl.int64)
        tmp30 = tl.full([1, 1], 10, tl.int64)
        tmp31 = triton_helpers.randint64(tmp27, (tmp28).to(tl.uint32), tmp29, tmp30)
        tmp32 = tmp31.to(tl.float32)
        tmp33 = tmp32 * tmp25
        tmp34 = tmp26 - tmp33
        tmp35 = tl.broadcast_to(tmp34, [XBLOCK, R0_BLOCK])
        tmp37 = _tmp36 + tmp35
        _tmp36 = tl.where(r0_mask, tmp37, _tmp36)
    tmp36 = tl.sum(_tmp36, 1)[:, None]
    tmp38 = 1 + ((1 + ks0) // 4)
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tmp36 / tmp39
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp40, None)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    s0 = arg0_1
    assert_size_stride(arg1_1, (1, 1, s0), (s0, s0, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 1, (3 + s0) // 2), ((3 + s0) // 2, (3 + s0) // 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.sign, aten.abs, aten.relu, aten.mul]
        triton_poi_fused_abs_mul_relu_sign_0_xnumel = (3 + s0) // 2
        get_raw_stream(0)
        triton_poi_fused_abs_mul_relu_sign_0[grid(triton_poi_fused_abs_mul_relu_sign_0_xnumel)](arg1_1, buf0, 64, 33, XBLOCK=64, num_warps=1, num_stages=1)
        del arg1_1
        buf3 = empty_strided_cuda((1, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf3)
        buf4 = empty_strided_cuda((), (), torch.float32)
        buf5 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [x_3, loss, randint_like], Original ATen: [aten.sign, aten.abs, aten.relu, aten.mul, aten.pow, aten.exp, aten.randint_like, aten.sub, aten.mean]
        1 + ((1 + s0) // 4)
        get_raw_stream(0)
        triton_red_fused_abs_exp_mean_mul_pow_randint_like_relu_sign_sub_1[grid(1)](buf5, buf0, buf3, 64, 0, 1, 17, XBLOCK=1, R0_BLOCK=32, num_warps=2, num_stages=1)
        del buf0
        del buf3
    return (buf5, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 64
    arg1_1 = rand_strided((1, 1, 64), (64, 64, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
