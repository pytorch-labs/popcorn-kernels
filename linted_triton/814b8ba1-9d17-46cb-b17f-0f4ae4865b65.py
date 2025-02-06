# AOT ID: ['2_inference']
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


# kernel path: /tmp/torchinductor_sahanp/7r/c7raslsvzwpsnnqr77ir7ky2occfpmh6szdui6qy2ov7nwamer44.py
# Topologically Sorted Source Nodes: [x, x_1, x_3], Original ATen: [aten.reflection_pad1d, aten.mish, aten._softmax]
# Source node to ATen node mapping:
#   x => _unsafe_index
#   x_1 => exp, gt, log1p, mul, tanh, where
#   x_3 => amax, exp_1, sub_2, sum_1
# Graph fragment:
#   %_unsafe_index : [num_users=4] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg0_1, [None, %sub_1]), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%_unsafe_index, 20), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%_unsafe_index,), kwargs = {})
#   %log1p : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp,), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %_unsafe_index, %log1p), kwargs = {})
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%where,), kwargs = {})
#   %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index, %tanh), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul, [1], True), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul, %amax), kwargs = {})
#   %exp_1 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_2,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_1, [1], True), kwargs = {})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__softmax_mish_reflection_pad1d_0(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 14
    R0_BLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_mask = r0_index < r0_numel
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (9 + ((-1)*tl_math.abs((-9) + tl_math.abs((-2) + r0_0)))), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = 20.0
    tmp2 = tmp0 > tmp1
    tmp3 = tl_math.exp(tmp0)
    tmp4 = libdevice.log1p(tmp3)
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp6 = libdevice.tanh(tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(r0_mask, tmp8, float("-inf"))
    tmp11 = triton_helpers.max2(tmp10, 1)[:, None]
    tmp12 = tmp7 - tmp11
    tmp13 = tl_math.exp(tmp12)
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.where(r0_mask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp11, None)
    tl.store(out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp17, None)


# kernel path: /tmp/torchinductor_sahanp/7u/c7uux3bd7bglppawm2ogkob7aufjfawlkdzh3rcg6mf4bnlor6xt.py
# Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.replication_pad3d]
# Source node to ATen node mapping:
#   x_5 => _unsafe_index_1, _unsafe_index_2, _unsafe_index_3
# Graph fragment:
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%view, [None, %clamp_max, None, None]), kwargs = {})
#   %_unsafe_index_2 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_1, [None, None, %clamp_max_1, None]), kwargs = {})
#   %_unsafe_index_3 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_2, [None, None, None, %clamp_max_2]), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_replication_pad3d_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (9 + ((-1)*tl_math.abs((-9) + tl_math.abs((-2) + ((13) * ((13) <= (((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0))))) + (((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) * ((((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) < (13))))))), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (0))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp12 = tl.load(in_ptr2 + (0))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK])
    tmp1 = 20.0
    tmp2 = tmp0 > tmp1
    tmp3 = tl_math.exp(tmp0)
    tmp4 = libdevice.log1p(tmp3)
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp6 = libdevice.tanh(tmp5)
    tmp7 = tmp0 * tmp6
    tmp10 = tmp7 - tmp9
    tmp11 = tl_math.exp(tmp10)
    tmp14 = tmp11 / tmp13
    tl.store(out_ptr0 + (x2), tmp14, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (1, 10), (10, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        buf1 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_1, x_3], Original ATen: [aten.reflection_pad1d, aten.mish, aten._softmax]
        get_raw_stream(0)
        triton_per_fused__softmax_mish_reflection_pad1d_0[grid(1)](arg0_1, buf0, buf1, 1, 14, XBLOCK=1, num_warps=2, num_stages=1)
        buf2 = empty_strided_cuda((1, 3, 3, 16), (144, 48, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.replication_pad3d]
        get_raw_stream(0)
        triton_poi_fused_replication_pad3d_1[grid(144)](arg0_1, buf0, buf1, buf2, 144, XBLOCK=128, num_warps=4, num_stages=1)
        del arg0_1
        del buf0
        del buf1
        # Topologically Sorted Source Nodes: [x_5, x_6], Original ATen: [aten.replication_pad3d, aten.adaptive_max_pool3d]
        buf3 = torch.ops.aten.adaptive_max_pool3d.default(buf2, [8, 8, 8])
        del buf2
        buf4 = buf3[0]
        del buf3
    return (reinterpret_tensor(buf4, (1, 64, 8), (512, 8, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
