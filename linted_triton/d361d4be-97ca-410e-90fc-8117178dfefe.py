# AOT ID: ['157_inference']
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


# kernel path: /tmp/torchinductor_sahanp/fl/cfl4obgan2tqkvmmymklj3gteoiqfbij35z5ngww7yzvj6kb6i7h.py
# Topologically Sorted Source Nodes: [target, loss, x_1], Original ATen: [aten.zeros_like, aten.binary_cross_entropy, aten.sigmoid]
# Source node to ATen node mapping:
#   loss => full_default, full_default_1, log, log1p, maximum, maximum_1, mean, mul_56, mul_57, neg, sub_33, sub_34
#   target => full
#   x_1 => sigmoid
# Graph fragment:
#   %full : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([1, %sym_size_int_15], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sub_33 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full, 1), kwargs = {})
#   %sigmoid : [num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view,), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%sigmoid,), kwargs = {})
#   %log1p : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%neg,), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -100), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %maximum : [num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%log1p, %full_default), kwargs = {})
#   %mul_56 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_33, %maximum), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sigmoid,), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -100), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %maximum_1 : [num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%log, %full_default_1), kwargs = {})
#   %mul_57 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full, %maximum_1), kwargs = {})
#   %sub_34 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_56, %mul_57), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sub_34,), kwargs = {})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_binary_cross_entropy_sigmoid_zeros_like_0(in_out_ptr0, in_ptr0, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    _tmp14 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.sigmoid(tmp0)
        tmp2 = -tmp1
        tmp3 = libdevice.log1p(tmp2)
        tmp4 = -100.0
        tmp5 = triton_helpers.maximum(tmp3, tmp4)
        tmp6 = -1.0
        tmp7 = tmp6 * tmp5
        tmp8 = tl_math.log(tmp1)
        tmp9 = triton_helpers.maximum(tmp8, tmp4)
        tmp10 = 0.0
        tmp11 = tmp10 * tmp9
        tmp12 = tmp7 - tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(r0_mask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tmp16 = (ks0 // 32)*(ks1 // 32)*(ks2 // 32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp14 / tmp17
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp18, None)


def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, 1, s0, s1, s2), (s0*s1*s2, s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.max_pool3d_with_indices]
        buf0 = torch.ops.aten.max_pool3d_with_indices.default(arg3_1, [2, 2, 2], [2, 2, 2])
        del arg3_1
        buf1 = buf0[0]
        del buf0
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.max_pool3d_with_indices]
        buf3 = torch.ops.aten.max_pool3d_with_indices.default(buf1, [2, 2, 2], [2, 2, 2])
        del buf1
        buf4 = buf3[0]
        del buf3
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.max_pool3d_with_indices]
        buf6 = torch.ops.aten.max_pool3d_with_indices.default(buf4, [2, 2, 2], [2, 2, 2])
        del buf4
        buf7 = buf6[0]
        del buf6
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.max_pool3d_with_indices]
        buf9 = torch.ops.aten.max_pool3d_with_indices.default(buf7, [2, 2, 2], [2, 2, 2])
        del buf7
        buf10 = buf9[0]
        del buf9
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.max_pool3d_with_indices]
        buf12 = torch.ops.aten.max_pool3d_with_indices.default(buf10, [2, 2, 2], [2, 2, 2])
        del buf10
        buf13 = buf12[0]
        del buf12
        buf15 = empty_strided_cuda((), (), torch.float32)
        buf16 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [target, loss, x_1], Original ATen: [aten.zeros_like, aten.binary_cross_entropy, aten.sigmoid]
        (s0 // 32)*(s1 // 32)*(s2 // 32)
        get_raw_stream(0)
        triton_red_fused_binary_cross_entropy_sigmoid_zeros_like_0[grid(1)](buf16, buf13, 64, 64, 64, 1, 8, XBLOCK=1, R0_BLOCK=8, num_warps=2, num_stages=1)
        del buf13
    return (buf16, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 64
    arg1_1 = 64
    arg2_1 = 64
    arg3_1 = rand_strided((1, 1, 64, 64, 64), (262144, 262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
