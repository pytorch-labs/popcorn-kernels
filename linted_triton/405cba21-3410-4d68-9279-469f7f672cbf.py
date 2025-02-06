# AOT ID: ['59_forward']
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


# kernel path: /tmp/torchinductor_sahanp/pj/cpjumco5b7yd62ayhv6lqxtskjir3c5hrzexysavabz53ea22gh5.py
# Topologically Sorted Source Nodes: [x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, input_1, input_2, x_8], Original ATen: [aten.abs, aten.gt, aten.sign, aten.mul, aten.sub, aten.where, aten.celu, aten.add, aten.view]
# Source node to ATen node mapping:
#   input_1 => abs_4, gt_5, mul_6, mul_7, sign_3, sub_3, where_5
#   input_2 => expm1_2, gt_6, where_6
#   x => abs_1, gt, mul, mul_1, sign, sub, where
#   x_1 => expm1, gt_1, where_1
#   x_2 => abs_2, gt_2, mul_2, mul_3, sign_1, sub_1, where_2
#   x_3 => expm1_1, gt_3, where_3
#   x_4 => abs_3, gt_4, mul_4, mul_5, sign_2, sub_2, where_4
#   x_5 => add
#   x_6 => add_1
#   x_7 => add_2
#   x_8 => view_3
# Graph fragment:
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%primals_1,), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%abs_1, 0.5), kwargs = {})
#   %sign : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%primals_1,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign, 0.5), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_1, %mul), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_1, 0), kwargs = {})
#   %where : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt, %sub, %mul_1), kwargs = {})
#   %expm1 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%where,), kwargs = {})
#   %gt_1 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%where, 0), kwargs = {})
#   %where_1 : [num_users=4] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %where, %expm1), kwargs = {})
#   %abs_2 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%where_1,), kwargs = {})
#   %gt_2 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%abs_2, 0.5), kwargs = {})
#   %sign_1 : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%where_1,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign_1, 0.5), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_1, %mul_2), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_1, 0), kwargs = {})
#   %where_2 : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %sub_1, %mul_3), kwargs = {})
#   %expm1_1 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%where_2,), kwargs = {})
#   %gt_3 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%where_2, 0), kwargs = {})
#   %where_3 : [num_users=4] = call_function[target=torch.ops.aten.where.self](args = (%gt_3, %where_2, %expm1_1), kwargs = {})
#   %abs_3 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%where_3,), kwargs = {})
#   %gt_4 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%abs_3, 0.5), kwargs = {})
#   %sign_2 : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%where_3,), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign_2, 0.5), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_3, %mul_4), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_3, 0), kwargs = {})
#   %where_4 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_4, %sub_2, %mul_5), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_4, %view), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %view_1), kwargs = {})
#   %add_2 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %view_2), kwargs = {})
#   %abs_4 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%add_2,), kwargs = {})
#   %gt_5 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%abs_4, 0.5), kwargs = {})
#   %sign_3 : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%add_2,), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign_3, 0.5), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_2, %mul_6), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, 0), kwargs = {})
#   %where_5 : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt_5, %sub_3, %mul_7), kwargs = {})
#   %expm1_2 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%where_5,), kwargs = {})
#   %gt_6 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%where_5, 0), kwargs = {})
#   %where_6 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_6, %where_5, %expm1_2), kwargs = {})
#   %view_3 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%where_6, [1, -1]), kwargs = {})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_abs_add_celu_gt_mul_sign_sub_view_where_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp46 = tl.load(in_ptr1 + (x0), xmask)
    tmp48 = tl.load(in_ptr2 + (x0), xmask)
    tmp50 = tl.load(in_ptr3 + (x0), xmask)
    tmp1 = tl_math.abs(tmp0)
    tmp2 = 0.5
    tmp3 = tmp1 > tmp2
    tmp4 = tl.full([1], 0, tl.int32)
    tmp5 = tmp4 < tmp0
    tmp6 = tmp5.to(tl.int8)
    tmp7 = tmp0 < tmp4
    tmp8 = tmp7.to(tl.int8)
    tmp9 = tmp6 - tmp8
    tmp10 = tmp9.to(tmp0.dtype)
    tmp11 = tmp10 * tmp2
    tmp12 = tmp0 - tmp11
    tmp13 = 0.0
    tmp14 = tmp0 * tmp13
    tmp15 = tl.where(tmp3, tmp12, tmp14)
    tmp16 = tmp15 > tmp13
    tmp17 = libdevice.expm1(tmp15)
    tmp18 = tl.where(tmp16, tmp15, tmp17)
    tmp19 = tl_math.abs(tmp18)
    tmp20 = tmp19 > tmp2
    tmp21 = tmp4 < tmp18
    tmp22 = tmp21.to(tl.int8)
    tmp23 = tmp18 < tmp4
    tmp24 = tmp23.to(tl.int8)
    tmp25 = tmp22 - tmp24
    tmp26 = tmp25.to(tmp18.dtype)
    tmp27 = tmp26 * tmp2
    tmp28 = tmp18 - tmp27
    tmp29 = tmp18 * tmp13
    tmp30 = tl.where(tmp20, tmp28, tmp29)
    tmp31 = tmp30 > tmp13
    tmp32 = libdevice.expm1(tmp30)
    tmp33 = tl.where(tmp31, tmp30, tmp32)
    tmp34 = tl_math.abs(tmp33)
    tmp35 = tmp34 > tmp2
    tmp36 = tmp4 < tmp33
    tmp37 = tmp36.to(tl.int8)
    tmp38 = tmp33 < tmp4
    tmp39 = tmp38.to(tl.int8)
    tmp40 = tmp37 - tmp39
    tmp41 = tmp40.to(tmp33.dtype)
    tmp42 = tmp41 * tmp2
    tmp43 = tmp33 - tmp42
    tmp44 = tmp33 * tmp13
    tmp45 = tl.where(tmp35, tmp43, tmp44)
    tmp47 = tmp45 + tmp46
    tmp49 = tmp47 + tmp48
    tmp51 = tmp49 + tmp50
    tmp52 = tl_math.abs(tmp51)
    tmp53 = tmp52 > tmp2
    tmp54 = tmp4 < tmp51
    tmp55 = tmp54.to(tl.int8)
    tmp56 = tmp51 < tmp4
    tmp57 = tmp56.to(tl.int8)
    tmp58 = tmp55 - tmp57
    tmp59 = tmp58.to(tmp51.dtype)
    tmp60 = tmp59 * tmp2
    tmp61 = tmp51 - tmp60
    tmp62 = tmp51 * tmp13
    tmp63 = tl.where(tmp53, tmp61, tmp62)
    tmp64 = tmp63 > tmp13
    tmp65 = libdevice.expm1(tmp63)
    tmp66 = tl.where(tmp64, tmp63, tmp65)
    tl.store(in_out_ptr0 + (x0), tmp51, xmask)
    tl.store(out_ptr0 + (x0), tmp66, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (1, 10), (10, 1))
    assert_size_stride(primals_2, (10, ), (1, ))
    assert_size_stride(primals_3, (10, ), (1, ))
    assert_size_stride(primals_4, (10, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 10), (10, 1), torch.float32)
        buf1 = buf0; del buf0  # reuse
        buf2 = empty_strided_cuda((1, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, input_1, input_2, x_8], Original ATen: [aten.abs, aten.gt, aten.sign, aten.mul, aten.sub, aten.where, aten.celu, aten.add, aten.view]
        get_raw_stream(0)
        triton_poi_fused_abs_add_celu_gt_mul_sign_sub_view_where_0[grid(10)](buf1, primals_1, primals_2, primals_3, primals_4, buf2, 10, XBLOCK=16, num_warps=1, num_stages=1)
        del primals_1
        del primals_2
        del primals_3
        del primals_4
    return (buf2, buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([primals_1, primals_2, primals_3, primals_4])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
