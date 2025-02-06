# AOT ID: ['42_inference']
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


# kernel path: /tmp/torchinductor_sahanp/23/c23xbjxwws65e2bjnk36symml4wm2mms64abtjx2omi6detmvjoy.py
# Topologically Sorted Source Nodes: [x_3, x_4, x_6], Original ATen: [aten.mish, aten._unsafe_index, aten.mean]
# Source node to ATen node mapping:
#   x_3 => exp, gt, log1p, mul_24, tanh, where
#   x_4 => _unsafe_index
#   x_6 => mean
# Graph fragment:
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%view_1, 20), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%view_1,), kwargs = {})
#   %log1p : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp,), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %view_1, %log1p), kwargs = {})
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%where,), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, %tanh), kwargs = {})
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%mul_24, [None, None, %unsqueeze_1, %convert_element_type_3]), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%unsqueeze_2, [-1, -2, -3], True), kwargs = {})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__unsafe_index_mean_mish_0(in_out_ptr0, in_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    _tmp29 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_2 = r0_index // 64
        r0_1 = (r0_index % 64)
        tmp0 = tl.full([1, 1], 2.0, tl.float64)
        tmp1 = ks0
        tmp2 = tmp1.to(tl.float64)
        tmp3 = tmp0 * tmp2
        tmp4 = tmp2 / tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp6 = r0_2
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp7 * tmp5
        tmp9 = tmp8.to(tl.int64)
        tmp10 = tmp9 + tmp1
        tmp11 = tmp9 < 0
        tmp12 = tl.where(tmp11, tmp10, tmp9)
        tmp13 = r0_1
        tmp14 = tmp13.to(tl.float32)
        tmp15 = 0.5
        tmp16 = tmp14 * tmp15
        tmp17 = tmp16.to(tl.int64)
        tmp18 = tl.load(in_ptr0 + (2*tmp17 + 64*tmp12 + 64*ks0*x0), r0_mask & xmask, eviction_policy='evict_last')
        tmp19 = tl.load(in_ptr0 + (1 + 2*tmp17 + 64*tmp12 + 64*ks0*x0), r0_mask & xmask, eviction_policy='evict_last')
        tmp20 = triton_helpers.maximum(tmp19, tmp18)
        tmp21 = 20.0
        tmp22 = tmp20 > tmp21
        tmp23 = tl_math.exp(tmp20)
        tmp24 = libdevice.log1p(tmp23)
        tmp25 = tl.where(tmp22, tmp20, tmp24)
        tmp26 = libdevice.tanh(tmp25)
        tmp27 = tmp20 * tmp26
        tmp28 = tl.broadcast_to(tmp27, [XBLOCK, R0_BLOCK])
        tmp30 = _tmp29 + tmp28
        _tmp29 = tl.where(r0_mask & xmask, tmp30, _tmp29)
    tmp29 = tl.sum(_tmp29, 1)[:, None]
    tmp31 = 128*ks0
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp29 / tmp32
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp33, xmask)


def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    assert_size_stride(arg2_1, (1, s0, s1, 64), (64*s0*s1, 64*s1, 64, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((1, s0, 1, 1, 1), (s0, 1, s0, s0, s0), torch.float32)
        buf2 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [x_3, x_4, x_6], Original ATen: [aten.mish, aten._unsafe_index, aten.mean]
        128*s1
        get_raw_stream(0)
        triton_red_fused__unsafe_index_mean_mish_0[grid(s0)](buf2, arg2_1, 64, 3, 8192, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del arg2_1
    return (reinterpret_tensor(buf2, (1, s0, 1, 1), (s0, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 3
    arg1_1 = 64
    arg2_1 = rand_strided((1, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
