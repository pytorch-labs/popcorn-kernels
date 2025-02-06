# AOT ID: ['154_inference']
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


# kernel path: /tmp/torchinductor_sahanp/b2/cb2qgvxdegnfkx36bw6sbnkv444osh6jtxj4y4wkrru23jn6ohos.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.adaptive_max_pool2d]
# Source node to ATen node mapping:
#   x => _low_memory_max_pool2d_with_offsets
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%unsqueeze, [1, 10], [1, 10], [0, 0], [1, 1], False), kwargs = {})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_adaptive_max_pool2d_0(in_ptr0, out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (10*x0 + ks1*x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 10*x0 + ks1*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 10*x0 + ks1*x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 10*x0 + ks1*x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4 + 10*x0 + ks1*x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (5 + 10*x0 + ks1*x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (6 + 10*x0 + ks1*x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (7 + 10*x0 + ks1*x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (8 + 10*x0 + ks1*x1), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (9 + 10*x0 + ks1*x1), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tmp18 = triton_helpers.maximum(tmp17, tmp16)
    tl.store(out_ptr0 + (x2), tmp18, xmask)


# kernel path: /tmp/torchinductor_sahanp/c4/cc4yvm3x4rkhqib7v7vkhulhkpt6rhhzzzhqgorae4igoqipiltt.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   x_1 => amax, exp, sub_10, sum_1
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%squeeze_1, [1], True), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%squeeze_1, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_10,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__softmax_1(in_ptr0, out_ptr0, out_ptr1, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + ks0*r0_1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = triton_helpers.maximum(_tmp2, tmp1)
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = triton_helpers.max2(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp4 = tl.load(in_ptr0 + (x0 + ks0*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp4 - tmp2
        tmp6 = tl_math.exp(tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp8, xmask)


# kernel path: /tmp/torchinductor_sahanp/na/cnaophugc7lclfcsbjtnqykcamlc66e645ipqpwussf3kio4avba.py
# Topologically Sorted Source Nodes: [ones_like, target, loss, x_1, log], Original ATen: [aten.ones_like, aten.div, aten.xlogy, aten._softmax, aten.log, aten.mul, aten.sub, aten.sum]
# Source node to ATen node mapping:
#   log => log
#   loss => div_2, eq_22, full_default, full_default_1, isnan, log_1, mul_31, mul_35, sub_23, sum_2, where, where_1
#   ones_like => full
#   target => div_1
#   x_1 => div, exp, sub_10
# Graph fragment:
#   %full : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, %arg0_1, %sym_size_int_2], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div_1 : [num_users=5] = call_function[target=torch.ops.aten.div.Tensor](args = (%full, %arg0_1), kwargs = {})
#   %isnan : [num_users=1] = call_function[target=torch.ops.aten.isnan.default](args = (%div_1,), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], nan), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %eq_22 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%div_1, 0), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %log_1 : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%div_1,), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %log_1), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_22, %full_default, %mul_35), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%isnan, %full_default_1, %where), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%squeeze_1, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_10,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%div,), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %log), kwargs = {})
#   %sub_23 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_1, %mul_31), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%sub_23,), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_2, 1), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__softmax_div_log_mul_ones_like_sub_sum_xlogy_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    _tmp22 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_2 = r0_index
        r0_0 = (r0_index % ks1)
        tmp12 = tl.load(in_ptr0 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr1 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.load(in_ptr2 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp0 = 1.0
        tmp1 = ks0
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tmp0 / tmp2
        tmp4 = libdevice.isnan(tmp3).to(tl.int1)
        tmp5 = 0.0
        tmp6 = tmp3 == tmp5
        tmp7 = tl_math.log(tmp3)
        tmp8 = tmp3 * tmp7
        tmp9 = tl.where(tmp6, tmp5, tmp8)
        tmp10 = float("nan")
        tmp11 = tl.where(tmp4, tmp10, tmp9)
        tmp14 = tmp12 - tmp13
        tmp15 = tl_math.exp(tmp14)
        tmp17 = tmp15 / tmp16
        tmp18 = tl_math.log(tmp17)
        tmp19 = tmp3 * tmp18
        tmp20 = tmp11 - tmp19
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(r0_mask, tmp23, _tmp22)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tmp24 = 1.0
    tmp25 = tmp22 * tmp24
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp25, None)


def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    assert_size_stride(arg2_1, (1, s0, s1), (s0*s1, s1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        s1 // 10
        buf0 = empty_strided_cuda((1, s0, 1, s1 // 10), (s0*(s1 // 10), s1 // 10, s1 // 10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.adaptive_max_pool2d]
        triton_poi_fused_adaptive_max_pool2d_0_xnumel = s0*(s1 // 10)
        get_raw_stream(0)
        triton_poi_fused_adaptive_max_pool2d_0[grid(triton_poi_fused_adaptive_max_pool2d_0_xnumel)](arg2_1, buf0, 10, 100, 200, XBLOCK=128, num_warps=4, num_stages=1)
        del arg2_1
        buf1 = empty_strided_cuda((1, 1, s1 // 10), (s1 // 10, s1 // 10, 1), torch.float32)
        buf2 = empty_strided_cuda((1, 1, s1 // 10), (s1 // 10, s1 // 10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._softmax]
        triton_red_fused__softmax_1_xnumel = s1 // 10
        get_raw_stream(0)
        triton_red_fused__softmax_1[grid(triton_red_fused__softmax_1_xnumel)](buf0, buf1, buf2, 10, 10, 20, XBLOCK=1, R0_BLOCK=32, num_warps=2, num_stages=1)
        buf3 = empty_strided_cuda((), (), torch.float32)
        buf4 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [ones_like, target, loss, x_1, log], Original ATen: [aten.ones_like, aten.div, aten.xlogy, aten._softmax, aten.log, aten.mul, aten.sub, aten.sum]
        s0*(s1 // 10)
        get_raw_stream(0)
        triton_red_fused__softmax_div_log_mul_ones_like_sub_sum_xlogy_2[grid(1)](buf4, buf0, buf1, buf2, 20, 10, 1, 200, XBLOCK=1, R0_BLOCK=256, num_warps=2, num_stages=1)
        del buf0
        del buf1
        del buf2
    return (buf4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 20
    arg1_1 = 100
    arg2_1 = rand_strided((1, 20, 100), (2000, 100, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
