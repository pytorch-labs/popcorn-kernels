# AOT ID: ['186_forward']
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


# kernel path: /tmp/torchinductor_sahanp/zf/czfwmztsbhovcz6x2n3zqxi4r3snqtybbss76kbh2lwkf44yvoin.py
# Topologically Sorted Source Nodes: [x, x_2, x_3], Original ATen: [aten.bernoulli, aten._to_copy, aten.add, aten.mul, aten.view, aten._native_batch_norm_legit, aten.div]
# Source node to ATen node mapping:
#   x => add, add_1, add_10, convert_element_type, inductor_lookup_seed_default, inductor_random_default_2, lt, mul_2, mul_3, mul_4
#   x_2 => add_23, mul_32, rsqrt, sub_10, var_mean, view_2, view_3, view_4
#   x_3 => convert_element_type_1, div, inductor_lookup_seed_default_1, inductor_random_default_1, lt_3, mul_43
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default_2 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 10, 1, 1], %inductor_lookup_seed_default, rand), kwargs = {})
#   %lt : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%inductor_random_default_2, 0.5), kwargs = {})
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt, torch.float32), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%convert_element_type, -1), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%add, 1.558387861036063), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_2, 0.7791939305180315), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%convert_element_type, 0.8864048946659319), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_3, %mul_3), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %add_1), kwargs = {})
#   %view_2 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_10, [-1, 10, %primals_1, %primals_2]), kwargs = {})
#   %view_3 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_2, [1, 10, %primals_1, %primals_2]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_3, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_23 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_23,), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_3, %getitem_1), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %rsqrt), kwargs = {})
#   %view_4 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_32, [1, 10, %primals_1, %primals_2]), kwargs = {})
#   %inductor_lookup_seed_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 1), kwargs = {})
#   %inductor_random_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 10, 1, 1], %inductor_lookup_seed_default_1, rand), kwargs = {})
#   %lt_3 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%inductor_random_default_1, 0.5), kwargs = {})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt_3, torch.float32), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%convert_element_type_1, 0.5), kwargs = {})
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_4, %div), kwargs = {})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit__to_copy_add_bernoulli_div_mul_view_0(in_ptr0, in_ptr1, out_ptr4, load_seed_offset, load_seed_offset1, ks2, ks3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 10
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = tl.load(in_ptr0 + load_seed_offset1)
    tmp4 = tl.rand(tmp3, (tmp1).to(tl.uint32))
    tmp20_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp20_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp20_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        r0_1 = r0_index
        tmp5 = tl.load(in_ptr1 + (r0_1 + ks2*ks3*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = 0.5
        tmp7 = tmp4 < tmp6
        tmp8 = tmp7.to(tl.float32)
        tmp9 = 0.8864048946659319
        tmp10 = tmp8 * tmp9
        tmp11 = tmp5 * tmp10
        tmp12 = -1.0
        tmp13 = tmp8 + tmp12
        tmp14 = 1.558387861036063
        tmp15 = tmp13 * tmp14
        tmp16 = 0.7791939305180315
        tmp17 = tmp15 + tmp16
        tmp18 = tmp11 + tmp17
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
        tmp20_mean_next, tmp20_m2_next, tmp20_weight_next = triton_helpers.welford_reduce(
            tmp19, tmp20_mean, tmp20_m2, tmp20_weight, roffset == 0
        )
        tmp20_mean = tl.where(r0_mask & xmask, tmp20_mean_next, tmp20_mean)
        tmp20_m2 = tl.where(r0_mask & xmask, tmp20_m2_next, tmp20_m2)
        tmp20_weight = tl.where(r0_mask & xmask, tmp20_weight_next, tmp20_weight)
    tmp23, tmp24, tmp25 = triton_helpers.welford(tmp20_mean, tmp20_m2, tmp20_weight, 1)
    tmp20 = tmp23[:, None]
    tmp21 = tmp24[:, None]
    tmp25[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        r0_1 = r0_index
        tmp26 = tl.load(in_ptr1 + (r0_1 + ks2*ks3*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp27 = 0.5
        tmp28 = tmp4 < tmp27
        tmp29 = tmp28.to(tl.float32)
        tmp30 = 0.8864048946659319
        tmp31 = tmp29 * tmp30
        tmp32 = tmp26 * tmp31
        tmp33 = -1.0
        tmp34 = tmp29 + tmp33
        tmp35 = 1.558387861036063
        tmp36 = tmp34 * tmp35
        tmp37 = 0.7791939305180315
        tmp38 = tmp36 + tmp37
        tmp39 = tmp32 + tmp38
        tmp40 = tmp39 - tmp20
        tmp41 = ks2*ks3
        tmp42 = tmp41.to(tl.float32)
        tmp43 = tmp21 / tmp42
        tmp44 = 1e-05
        tmp45 = tmp43 + tmp44
        tmp46 = libdevice.rsqrt(tmp45)
        tmp47 = tmp40 * tmp46
        tmp48 = tmp2 < tmp27
        tmp49 = tmp48.to(tl.float32)
        tmp50 = 2.0
        tmp51 = tmp49 * tmp50
        tmp52 = tmp47 * tmp51
        tl.store(out_ptr4 + (r0_1 + ks2*ks3*x0), tmp52, r0_mask & xmask)


# kernel path: /tmp/torchinductor_sahanp/ej/cej6vn5oua6bpktgmtmdnioxd4g6d7prmkiyyqbg6g42vcsc5bxx.py
# Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.bernoulli]
# Source node to ATen node mapping:
#   x_6 => inductor_lookup_seed_default_2, inductor_random_default
# Graph fragment:
#   %inductor_lookup_seed_default_2 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 2), kwargs = {})
#   %inductor_random_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 10, 1, 1, 1], %inductor_lookup_seed_default_2, rand), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_bernoulli_1(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (x0), tmp2, xmask)


# kernel path: /tmp/torchinductor_sahanp/nd/cndxvysg4vdolqxgttio4u5fdtv4zex2vv4q3ycgdlqdc3ulhdzj.py
# Topologically Sorted Source Nodes: [value, value_1, value_2, value_3, value_4], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   value => add_44
#   value_1 => add_45
#   value_2 => add_46
#   value_3 => add_47
#   value_4 => add_48
# Graph fragment:
#   %add_44 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_4, 0), kwargs = {})
#   %add_45 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_44, %primals_5), kwargs = {})
#   %add_46 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_45, %primals_6), kwargs = {})
#   %add_47 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_46, %primals_7), kwargs = {})
#   %add_48 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_47, %primals_8), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp5 = tl.load(in_ptr2 + (x0), xmask)
    tmp7 = tl.load(in_ptr3 + (x0), xmask)
    tmp9 = tl.load(in_ptr4 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tl.store(out_ptr0 + (x0), tmp10, xmask)


# kernel path: /tmp/torchinductor_sahanp/2e/c2evjcp4vk6ki2xqbof2gwprwibacutibqoet364v2eyzva7saof.py
# Topologically Sorted Source Nodes: [x_6, x_7], Original ATen: [aten.bernoulli, aten._to_copy, aten.add, aten.mul]
# Source node to ATen node mapping:
#   x_6 => add_41, add_42, add_43, convert_element_type_2, lt_4, mul_53, mul_54, mul_55
#   x_7 => add_49
# Graph fragment:
#   %lt_4 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%inductor_random_default, 0.5), kwargs = {})
#   %convert_element_type_2 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt_4, torch.float32), kwargs = {})
#   %add_41 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%convert_element_type_2, -1), kwargs = {})
#   %mul_53 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%add_41, 1.558387861036063), kwargs = {})
#   %add_42 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_53, 0.7791939305180315), kwargs = {})
#   %mul_54 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%convert_element_type_2, 0.8864048946659319), kwargs = {})
#   %mul_55 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_2, %mul_54), kwargs = {})
#   %add_43 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_55, %add_42), kwargs = {})
#   %add_49 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_43, %view_5), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_add_bernoulli_mul_3(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1250
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 125
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = 0.5
    tmp3 = tmp1 < tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 0.8864048946659319
    tmp6 = tmp4 * tmp5
    tmp7 = tmp0 * tmp6
    tmp8 = -1.0
    tmp9 = tmp4 + tmp8
    tmp10 = 1.558387861036063
    tmp11 = tmp9 * tmp10
    tmp12 = 0.7791939305180315
    tmp13 = tmp11 + tmp12
    tmp14 = tmp7 + tmp13
    tmp16 = tmp14 + tmp15
    tl.store(in_out_ptr0 + (x2), tmp16, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8 = args
    args.clear()
    s1 = primals_1
    s2 = primals_2
    assert_size_stride(primals_3, (1, 10, s1, s2), (10*s1*s2, s1*s2, s2, 1))
    assert_size_stride(primals_4, (10, ), (1, ))
    assert_size_stride(primals_5, (10, ), (1, ))
    assert_size_stride(primals_6, (10, ), (1, ))
    assert_size_stride(primals_7, (10, ), (1, ))
    assert_size_stride(primals_8, (10, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((3, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [3], out=buf0)
        buf6 = empty_strided_cuda((1, 10, s1, s2), (10*s1*s2, s1*s2, s2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_2, x_3], Original ATen: [aten.bernoulli, aten._to_copy, aten.add, aten.mul, aten.view, aten._native_batch_norm_legit, aten.div]
        s1*s2
        get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit__to_copy_add_bernoulli_div_mul_view_0[grid(10)](buf0, primals_3, buf6, 1, 0, 32, 32, 10, 1024, XBLOCK=1, R0_BLOCK=1024, num_warps=8, num_stages=1)
        del primals_3
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.adaptive_max_pool3d]
        buf7 = torch.ops.aten.adaptive_max_pool3d.default(reinterpret_tensor(buf6, (1, 10, 1, s1, s2), (10*s1*s2, s1*s2, s1*s2, s2, 1), 0), [5, 5, 5])
        del buf6
        buf8 = buf7[0]
        del buf7
        buf10 = empty_strided_cuda((1, 10, 1, 1, 1), (10, 1, 10, 10, 10), torch.float32)
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.bernoulli]
        get_raw_stream(0)
        triton_poi_fused_bernoulli_1[grid(10)](buf0, buf10, 2, 10, XBLOCK=16, num_warps=1, num_stages=1)
        del buf0
        buf11 = empty_strided_cuda((10, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [value, value_1, value_2, value_3, value_4], Original ATen: [aten.add]
        get_raw_stream(0)
        triton_poi_fused_add_2[grid(10)](primals_4, primals_5, primals_6, primals_7, primals_8, buf11, 10, XBLOCK=16, num_warps=1, num_stages=1)
        del primals_4
        del primals_5
        del primals_6
        del primals_7
        del primals_8
        buf12 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [x_6, x_7], Original ATen: [aten.bernoulli, aten._to_copy, aten.add, aten.mul]
        get_raw_stream(0)
        triton_poi_fused__to_copy_add_bernoulli_mul_3[grid(1250)](buf12, buf10, buf11, 1250, XBLOCK=128, num_warps=4, num_stages=1)
        del buf10
        del buf11
    return (buf12, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = 32
    primals_2 = 32
    primals_3 = rand_strided((1, 10, 32, 32), (10240, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
