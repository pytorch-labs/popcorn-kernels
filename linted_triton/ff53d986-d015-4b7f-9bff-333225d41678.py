# AOT ID: ['59_inference']
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


# kernel path: /tmp/torchinductor_sahanp/2u/c2uio5agy6zf3faddgdmtmmahogsytx7pojobedoblrtu6wecsqf.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.bernoulli]
# Source node to ATen node mapping:
#   x_3 => inductor_lookup_seed_default, inductor_random_default_1
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 1, 1], %inductor_lookup_seed_default, rand), kwargs = {})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_bernoulli_0(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp2, None)


# kernel path: /tmp/torchinductor_sahanp/tf/ctfvvs4ia2uldv5lejm7tiihmdohe344fpxf6tebnnxg22vfnkew.py
# Topologically Sorted Source Nodes: [x_8], Original ATen: [aten.bernoulli]
# Source node to ATen node mapping:
#   x_8 => inductor_lookup_seed_default_1, inductor_random_default
# Graph fragment:
#   %inductor_lookup_seed_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 1), kwargs = {})
#   %inductor_random_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 1, 1], %inductor_lookup_seed_default_1, rand), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_bernoulli_1(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp2, None)


# kernel path: /tmp/torchinductor_sahanp/bg/cbguopnzzncjj6kozmjtfezyvrxccpjszhs5jd2v3kuthotvp6yb.py
# Topologically Sorted Source Nodes: [x_9, x_8], Original ATen: [aten.log_sigmoid_forward, aten.bernoulli, aten._to_copy, aten.div, aten.mul]
# Source node to ATen node mapping:
#   x_8 => convert_element_type_1, div_3, lt_1, mul_38
#   x_9 => abs_2, exp_1, full_default_1, log1p_1, minimum_1, neg_1, sub_24
# Graph fragment:
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %lt_1 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%inductor_random_default, 0.5), kwargs = {})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt_1, torch.float32), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%convert_element_type_1, 0.5), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, %div_3), kwargs = {})
#   %minimum_1 : [num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%full_default_1, %squeeze_1), kwargs = {})
#   %abs_2 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%squeeze_1,), kwargs = {})
#   %neg_1 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%abs_2,), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_1,), kwargs = {})
#   %log1p_1 : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_1,), kwargs = {})
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%minimum_1, %log1p_1), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_bernoulli_div_log_sigmoid_forward_mul_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp12 = tl.load(in_ptr1 + (0))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK])
    tmp33 = tl.load(in_ptr2 + (0))
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK])
    tmp1 = libdevice.tanh(tmp0)
    tmp2 = tmp0 - tmp1
    tmp3 = 3.0
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = tmp2 * tmp8
    tmp10 = 0.16666666666666666
    tmp11 = tmp9 * tmp10
    tmp14 = 0.5
    tmp15 = tmp13 < tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = 2.0
    tmp18 = tmp16 * tmp17
    tmp19 = tmp11 * tmp18
    tmp20 = triton_helpers.minimum(tmp5, tmp19)
    tmp21 = tl_math.abs(tmp19)
    tmp22 = -tmp21
    tmp23 = tl_math.exp(tmp22)
    tmp24 = libdevice.log1p(tmp23)
    tmp25 = tmp20 - tmp24
    tmp26 = libdevice.tanh(tmp25)
    tmp27 = tmp25 - tmp26
    tmp28 = tmp27 + tmp3
    tmp29 = triton_helpers.maximum(tmp28, tmp5)
    tmp30 = triton_helpers.minimum(tmp29, tmp7)
    tmp31 = tmp27 * tmp30
    tmp32 = tmp31 * tmp10
    tmp35 = tmp34 < tmp14
    tmp36 = tmp35.to(tl.float32)
    tmp37 = tmp36 * tmp17
    tmp38 = tmp32 * tmp37
    tmp39 = triton_helpers.minimum(tmp5, tmp38)
    tmp40 = tl_math.abs(tmp38)
    tmp41 = -tmp40
    tmp42 = tl_math.exp(tmp41)
    tmp43 = libdevice.log1p(tmp42)
    tmp44 = tmp39 - tmp43
    tl.store(in_out_ptr0 + (x0), tmp44, xmask)


def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [2], out=buf0)
        buf1 = empty_strided_cuda((1, 1, 1), (1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.bernoulli]
        get_raw_stream(0)
        triton_poi_fused_bernoulli_0[grid(1)](buf0, buf1, 0, 1, XBLOCK=1, num_warps=1, num_stages=1)
        buf2 = empty_strided_cuda((1, 1, 1), (1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten.bernoulli]
        get_raw_stream(0)
        triton_poi_fused_bernoulli_1[grid(1)](buf0, buf2, 1, 1, XBLOCK=1, num_warps=1, num_stages=1)
        del buf0
        buf3 = empty_strided_cuda((1, 1, s0*s1*s2), (s0*s1*s2, s0*s1*s2, 1), torch.float32)
        buf4 = reinterpret_tensor(buf3, (1, s0*s1*s2), (s0*s1*s2, 1), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [x_9, x_8], Original ATen: [aten.log_sigmoid_forward, aten.bernoulli, aten._to_copy, aten.div, aten.mul]
        triton_poi_fused__to_copy_bernoulli_div_log_sigmoid_forward_mul_2_xnumel = s0*s1*s2
        get_raw_stream(0)
        triton_poi_fused__to_copy_bernoulli_div_log_sigmoid_forward_mul_2[grid(triton_poi_fused__to_copy_bernoulli_div_log_sigmoid_forward_mul_2_xnumel)](buf4, arg3_1, buf1, buf2, 12288, XBLOCK=128, num_warps=4, num_stages=1)
        del arg3_1
        del buf1
        del buf2
    return (buf4, )


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
