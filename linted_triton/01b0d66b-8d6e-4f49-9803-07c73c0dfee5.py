# AOT ID: ['7_inference']
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


# kernel path: /tmp/torchinductor_sahanp/3c/c3cjjfvzaxbib5lnkva2syg25hlkfyzaz56sk255pzmhytek65sh.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten._log_softmax]
# Source node to ATen node mapping:
#   loss => amax
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view, [1], True), kwargs = {})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__log_softmax_0(in_ptr0, out_ptr0, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    _tmp41 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp0 = r0_1 + x0*((1 + ks0*ks1*ks2) // 2)
        tmp1 = ks0*ks1*ks2
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (((r0_1 + x0*((1 + ks0*ks1*ks2) // 2)) % (ks0*ks1*ks2))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 > tmp4
        tmp6 = 0.01
        tmp7 = tmp3 * tmp6
        tmp8 = tl.where(tmp5, tmp3, tmp7)
        tmp9 = tl_math.abs(tmp8)
        tmp10 = 0.5
        tmp11 = tmp9 > tmp10
        tmp12 = tl.full([1, 1], 0, tl.int32)
        tmp13 = tmp12 < tmp8
        tmp14 = tmp13.to(tl.int8)
        tmp15 = tmp8 < tmp12
        tmp16 = tmp15.to(tl.int8)
        tmp17 = tmp14 - tmp16
        tmp18 = tmp17.to(tmp8.dtype)
        tmp19 = tmp18 * tmp10
        tmp20 = tmp8 - tmp19
        tmp21 = tmp8 * tmp4
        tmp22 = tl.where(tmp11, tmp20, tmp21)
        tmp23 = tmp22 > tmp4
        tmp24 = tmp22 * tmp6
        tmp25 = tl.where(tmp23, tmp22, tmp24)
        tmp26 = tl_math.abs(tmp25)
        tmp27 = tmp26 > tmp10
        tmp28 = tmp12 < tmp25
        tmp29 = tmp28.to(tl.int8)
        tmp30 = tmp25 < tmp12
        tmp31 = tmp30.to(tl.int8)
        tmp32 = tmp29 - tmp31
        tmp33 = tmp32.to(tmp25.dtype)
        tmp34 = tmp33 * tmp10
        tmp35 = tmp25 - tmp34
        tmp36 = tmp25 * tmp4
        tmp37 = tl.where(tmp27, tmp35, tmp36)
        tmp38 = tl.full(tmp37.shape, float("-inf"), tmp37.dtype)
        tmp39 = tl.where(tmp2, tmp37, tmp38)
        tmp40 = tl.broadcast_to(tmp39, [XBLOCK, R0_BLOCK])
        tmp42 = triton_helpers.maximum(_tmp41, tmp40)
        _tmp41 = tl.where(r0_mask & xmask, tmp42, _tmp41)
    tmp41 = triton_helpers.max2(_tmp41, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp41, xmask)


# kernel path: /tmp/torchinductor_sahanp/pk/cpkbcdxfsknsp3rqg7idd7tjvbyn7xjtf75kf7geio63eeogca6l.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten._log_softmax]
# Source node to ATen node mapping:
#   loss => amax
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view, [1], True), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__log_softmax_1(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    R0_BLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = triton_helpers.max2(tmp1, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp3, None)


# kernel path: /tmp/torchinductor_sahanp/ae/caemggk5zkiv3rwxevghnb4mxf34t6ia2ije62rv2e573l5flg64.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten._log_softmax]
# Source node to ATen node mapping:
#   loss => exp, sub_51, sum_1
# Graph fragment:
#   %sub_51 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %amax), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_51,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__log_softmax_2(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    _tmp44 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp0 = r0_1 + x0*((1 + ks0*ks1*ks2) // 2)
        tmp1 = ks0*ks1*ks2
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (((r0_1 + x0*((1 + ks0*ks1*ks2) // 2)) % (ks0*ks1*ks2))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 > tmp4
        tmp6 = 0.01
        tmp7 = tmp3 * tmp6
        tmp8 = tl.where(tmp5, tmp3, tmp7)
        tmp9 = tl_math.abs(tmp8)
        tmp10 = 0.5
        tmp11 = tmp9 > tmp10
        tmp12 = tl.full([1, 1], 0, tl.int32)
        tmp13 = tmp12 < tmp8
        tmp14 = tmp13.to(tl.int8)
        tmp15 = tmp8 < tmp12
        tmp16 = tmp15.to(tl.int8)
        tmp17 = tmp14 - tmp16
        tmp18 = tmp17.to(tmp8.dtype)
        tmp19 = tmp18 * tmp10
        tmp20 = tmp8 - tmp19
        tmp21 = tmp8 * tmp4
        tmp22 = tl.where(tmp11, tmp20, tmp21)
        tmp23 = tmp22 > tmp4
        tmp24 = tmp22 * tmp6
        tmp25 = tl.where(tmp23, tmp22, tmp24)
        tmp26 = tl_math.abs(tmp25)
        tmp27 = tmp26 > tmp10
        tmp28 = tmp12 < tmp25
        tmp29 = tmp28.to(tl.int8)
        tmp30 = tmp25 < tmp12
        tmp31 = tmp30.to(tl.int8)
        tmp32 = tmp29 - tmp31
        tmp33 = tmp32.to(tmp25.dtype)
        tmp34 = tmp33 * tmp10
        tmp35 = tmp25 - tmp34
        tmp36 = tmp25 * tmp4
        tmp37 = tl.where(tmp27, tmp35, tmp36)
        tmp38 = tl.load(in_ptr1 + (tl.full([XBLOCK, R0_BLOCK], 0, tl.int32)), tmp2, eviction_policy='evict_last', other=0.0)
        tmp39 = tmp37 - tmp38
        tmp40 = tl_math.exp(tmp39)
        tmp41 = tl.full(tmp40.shape, 0, tmp40.dtype)
        tmp42 = tl.where(tmp2, tmp40, tmp41)
        tmp43 = tl.broadcast_to(tmp42, [XBLOCK, R0_BLOCK])
        tmp45 = _tmp44 + tmp43
        _tmp44 = tl.where(r0_mask & xmask, tmp45, _tmp44)
    tmp44 = tl.sum(_tmp44, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp44, xmask)


# kernel path: /tmp/torchinductor_sahanp/lq/clq2b2yiu4u3dsiqxemo7xsmupkelalelqsnxdzynqquombiisxg.py
# Topologically Sorted Source Nodes: [target, loss], Original ATen: [aten.randint, aten.nll_loss_forward, aten._log_softmax]
# Source node to ATen node mapping:
#   loss => convert_element_type, div, exp, full_default_1, ne_1, ne_2, neg, sub_51, sum_1, sum_2, sum_3, where_5
#   target => inductor_lookup_seed_default, inductor_randint_default
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_randint_default : [num_users=4] = call_function[target=torch.ops.prims.inductor_randint.default](args = (0, 10, [1], %inductor_lookup_seed_default), kwargs = {})
#   %ne_1 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%inductor_randint_default, -100), kwargs = {})
#   %sub_51 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %amax), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_51,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_5 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_1, %neg, %full_default_1), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_5,), kwargs = {})
#   %ne_2 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%inductor_randint_default, -100), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne_2,), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_2, torch.float32), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_3, %convert_element_type), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__log_softmax_nll_loss_forward_randint_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, load_seed_offset, ks1, ks2, ks3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    R0_BLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), None)
    tmp51 = tl.load(in_out_ptr0 + (0))
    tmp52 = tl.broadcast_to(tmp51, [XBLOCK, 1])
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None]
    tmp4 = tl.load(in_ptr1 + load_seed_offset)
    tmp5 = tl.full([1, 1], 0, tl.int32)
    tmp6 = tl.full([1, 1], 0, tl.int64)
    tmp7 = tl.full([1, 1], 10, tl.int64)
    tmp8 = triton_helpers.randint64(tmp4, (tmp5).to(tl.uint32), tmp6, tmp7)
    tmp9 = tl.full([1, 1], -100, tl.int64)
    tmp10 = tmp8 != tmp9
    tmp11 = tl.where(tmp10, tmp8, tmp6)
    tmp12 = ks1*ks2*ks3
    tmp13 = tmp11 + tmp12
    tmp14 = tmp11 < 0
    tmp15 = tl.where(tmp14, tmp13, tmp11)
    tl.device_assert((0 <= tmp15) & (tmp15 < ks1*ks2*ks3), "index out of bounds: 0 <= tmp15 < ks1*ks2*ks3")
    tmp17 = tl.load(in_ptr2 + ((tmp15 % (ks1*ks2*ks3))), None, eviction_policy='evict_last')
    tmp18 = 0.0
    tmp19 = tmp17 > tmp18
    tmp20 = 0.01
    tmp21 = tmp17 * tmp20
    tmp22 = tl.where(tmp19, tmp17, tmp21)
    tmp23 = tl_math.abs(tmp22)
    tmp24 = 0.5
    tmp25 = tmp23 > tmp24
    tmp26 = tmp5 < tmp22
    tmp27 = tmp26.to(tl.int8)
    tmp28 = tmp22 < tmp5
    tmp29 = tmp28.to(tl.int8)
    tmp30 = tmp27 - tmp29
    tmp31 = tmp30.to(tmp22.dtype)
    tmp32 = tmp31 * tmp24
    tmp33 = tmp22 - tmp32
    tmp34 = tmp22 * tmp18
    tmp35 = tl.where(tmp25, tmp33, tmp34)
    tmp36 = tmp35 > tmp18
    tmp37 = tmp35 * tmp20
    tmp38 = tl.where(tmp36, tmp35, tmp37)
    tmp39 = tl_math.abs(tmp38)
    tmp40 = tmp39 > tmp24
    tmp41 = tmp5 < tmp38
    tmp42 = tmp41.to(tl.int8)
    tmp43 = tmp38 < tmp5
    tmp44 = tmp43.to(tl.int8)
    tmp45 = tmp42 - tmp44
    tmp46 = tmp45.to(tmp38.dtype)
    tmp47 = tmp46 * tmp24
    tmp48 = tmp38 - tmp47
    tmp49 = tmp38 * tmp18
    tmp50 = tl.where(tmp40, tmp48, tmp49)
    tmp53 = tmp50 - tmp52
    tmp54 = tl_math.log(tmp3)
    tmp55 = tmp53 - tmp54
    tmp56 = -tmp55
    tmp57 = tl.where(tmp10, tmp56, tmp18)
    tmp58 = tmp10.to(tl.int32)
    tmp59 = tmp58.to(tl.float32)
    tmp60 = tmp57 / tmp59
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp60, None)


def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf0)
        buf1 = empty_strided_cuda((1, 1, 2), (2, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [loss], Original ATen: [aten._log_softmax]
        (1 + s0*s1*s2) // 2
        get_raw_stream(0)
        triton_red_fused__log_softmax_0[grid(2)](arg3_1, buf1, 3, 64, 64, 2, 6144, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        buf2 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [loss], Original ATen: [aten._log_softmax]
        get_raw_stream(0)
        triton_per_fused__log_softmax_1[grid(1)](buf1, buf2, 1, 2, XBLOCK=1, num_warps=2, num_stages=1)
        buf3 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [loss], Original ATen: [aten._log_softmax]
        (1 + s0*s1*s2) // 2
        get_raw_stream(0)
        triton_red_fused__log_softmax_2[grid(2)](arg3_1, buf2, buf3, 3, 64, 64, 2, 6144, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        buf5 = reinterpret_tensor(buf2, (), (), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [target, loss], Original ATen: [aten.randint, aten.nll_loss_forward, aten._log_softmax]
        get_raw_stream(0)
        triton_per_fused__log_softmax_nll_loss_forward_randint_3[grid(1)](buf5, buf3, buf0, arg3_1, 0, 3, 64, 64, 1, 2, XBLOCK=1, num_warps=2, num_stages=1)
        del arg3_1
        del buf0
        del buf3
    return (buf5, )


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
