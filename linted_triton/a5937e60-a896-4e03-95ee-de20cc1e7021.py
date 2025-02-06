# AOT ID: ['37_inference']
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


# kernel path: /tmp/torchinductor_sahanp/7k/c7kc2vp6syzi2cfcn56tigjkxo4lrg22fmub2dr4anm7vjz7dhph.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   x_1 => var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [0, 2, 3, 4]), kwargs = {correction: 0, keepdim: True})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_0(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(r0_mask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(r0_mask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(r0_mask & xmask, tmp2_weight_next, tmp2_weight)
    tmp5, tmp6, tmp7 = triton_helpers.welford(tmp2_mean, tmp2_m2, tmp2_weight, 1)
    tmp2 = tmp5[:, None]
    tmp3 = tmp6[:, None]
    tmp4 = tmp7[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tl.store(out_ptr1 + (x0), tmp3, xmask)
    tl.store(out_ptr2 + (x0), tmp4, xmask)


# kernel path: /tmp/torchinductor_sahanp/3t/c3tuvxj2g35kcg46pkv3x5aep5fcimfjpm53gbsoy7oqfmnyuir5.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   x_1 => var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [0, 2, 3, 4]), kwargs = {correction: 0, keepdim: True})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 64
    R0_BLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 4*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 4*x0), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r0_1 + 4*x0), xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp12[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)


# kernel path: /tmp/torchinductor_sahanp/kz/ckzmohkzhnwink2ez4mbi2ncamux3nbi7zi5ps7voeaqe3dyldq2.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   x_1 => add, mul, rsqrt, sub, var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [0, 2, 3, 4]), kwargs = {correction: 0, keepdim: True})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %getitem_1), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 32768
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 32768.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tl.store(out_ptr0 + (x2), tmp9, None)


# kernel path: /tmp/torchinductor_sahanp/7z/c7zljb744fb4d6xooml25jsxvyhhvvipl6fdr6vsalocrcahykgl.py
# Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.bernoulli]
# Source node to ATen node mapping:
#   x_4 => inductor_lookup_seed_default, inductor_random_default_1
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 64, 1], %inductor_lookup_seed_default, rand), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_bernoulli_3(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (x0), tmp2, xmask)


# kernel path: /tmp/torchinductor_sahanp/gt/cgt6mhta7evggcbyz2n5px3bubkxvznnsu7h4ealct34itpd22js.py
# Topologically Sorted Source Nodes: [x_6, x_7, x_8, x_9], Original ATen: [aten.leaky_relu, aten.bernoulli, aten._to_copy, aten.mul, aten.add, aten.abs, aten.le, aten.scalar_tensor, aten.where, aten.neg, aten._softmax]
# Source node to ATen node mapping:
#   x_6 => gt, mul_2, where
#   x_7 => add_1, add_2, add_3, convert_element_type_1, inductor_lookup_seed_default_1, inductor_random_default, lt_1, mul_3, mul_4, mul_5
#   x_8 => abs_1, full_default, le, where_1
#   x_9 => amax, div_1, exp, neg, sub_1, sum_1
# Graph fragment:
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%view_4, 0), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_4, 0.01), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %view_4, %mul_2), kwargs = {})
#   %inductor_lookup_seed_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 1), kwargs = {})
#   %inductor_random_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 8192], %inductor_lookup_seed_default_1, rand), kwargs = {})
#   %lt_1 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%inductor_random_default, 0.5), kwargs = {})
#   %convert_element_type_1 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt_1, torch.float32), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%convert_element_type_1, 0.8864048946659319), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where, %mul_4), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%convert_element_type_1, -1), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%add_1, 1.558387861036063), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_3, 0.7791939305180315), kwargs = {})
#   %add_3 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %add_2), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%add_3,), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%abs_1, 0.5), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le, %full_default, %add_3), kwargs = {})
#   %neg : [num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%where_1,), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%neg, [1], True), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%neg, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_1,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__softmax__to_copy_abs_add_bernoulli_le_leaky_relu_mul_neg_scalar_tensor_where_4(in_out_ptr0, in_ptr0, in_ptr1, load_seed_offset, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    _tmp33 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_0 = r0_index
        tmp3 = tl.load(in_out_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r0_0 // 128), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp0 = tl.load(in_ptr0 + load_seed_offset)
        tmp1 = r0_0
        tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
        tmp5 = 0.5
        tmp6 = tmp4 < tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = 2.0
        tmp9 = tmp7 * tmp8
        tmp10 = tmp3 * tmp9
        tmp11 = 0.0
        tmp12 = tmp10 > tmp11
        tmp13 = 0.01
        tmp14 = tmp10 * tmp13
        tmp15 = tl.where(tmp12, tmp10, tmp14)
        tmp16 = tmp2 < tmp5
        tmp17 = tmp16.to(tl.float32)
        tmp18 = 0.8864048946659319
        tmp19 = tmp17 * tmp18
        tmp20 = tmp15 * tmp19
        tmp21 = -1.0
        tmp22 = tmp17 + tmp21
        tmp23 = 1.558387861036063
        tmp24 = tmp22 * tmp23
        tmp25 = 0.7791939305180315
        tmp26 = tmp24 + tmp25
        tmp27 = tmp20 + tmp26
        tmp28 = tl_math.abs(tmp27)
        tmp29 = tmp28 <= tmp5
        tmp30 = tl.where(tmp29, tmp11, tmp27)
        tmp31 = -tmp30
        tmp32 = tl.broadcast_to(tmp31, [XBLOCK, R0_BLOCK])
        tmp34 = triton_helpers.maximum(_tmp33, tmp32)
        _tmp33 = tl.where(r0_mask, tmp34, _tmp33)
        tl.store(in_out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp31, r0_mask)
    tmp33 = triton_helpers.max2(_tmp33, 1)[:, None]
    _tmp39 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_0 = r0_index
        tmp35 = tl.load(in_out_ptr0 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp36 = tmp35 - tmp33
        tmp37 = tl_math.exp(tmp36)
        tmp38 = tl.broadcast_to(tmp37, [XBLOCK, R0_BLOCK])
        tmp40 = _tmp39 + tmp38
        _tmp39 = tl.where(r0_mask, tmp40, _tmp39)
    tmp39 = tl.sum(_tmp39, 1)[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_0 = r0_index
        tmp41 = tl.load(in_out_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp42 = tmp41 - tmp33
        tmp43 = tl_math.exp(tmp42)
        tmp44 = tmp43 / tmp39
        tl.store(in_out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp44, r0_mask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (1, 64, 32, 32, 32), (2097152, 32768, 1024, 32, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 64, 1, 1, 1, 4), (256, 4, 256, 256, 256, 1), torch.float32)
        buf1 = empty_strided_cuda((1, 64, 1, 1, 1, 4), (256, 4, 256, 256, 256, 1), torch.float32)
        buf2 = empty_strided_cuda((1, 64, 1, 1, 1, 4), (256, 4, 256, 256, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit]
        get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_0[grid(256)](arg0_1, buf0, buf1, buf2, 256, 8192, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        buf3 = empty_strided_cuda((1, 64, 1, 1, 1), (64, 1, 64, 64, 64), torch.float32)
        buf4 = empty_strided_cuda((1, 64, 1, 1, 1), (64, 1, 64, 64, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit]
        get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_1[grid(64)](buf0, buf1, buf2, buf3, buf4, 64, 4, XBLOCK=8, num_warps=2, num_stages=1)
        del buf0
        del buf1
        del buf2
        buf6 = empty_strided_cuda((1, 64, 32, 32, 32), (2097152, 32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit]
        get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_2[grid(2097152)](arg0_1, buf3, buf4, buf6, 2097152, XBLOCK=1024, num_warps=4, num_stages=1)
        del arg0_1
        del buf3
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.adaptive_max_pool2d]
        buf7 = torch.ops.aten.max_pool2d_with_indices.default(reinterpret_tensor(buf6, (1, 64, 1, 32768), (0, 32768, 0, 1), 0), [1, 256])
        del buf6
        buf8 = buf7[0]
        del buf7
        buf10 = empty_strided_cuda((2, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [2], out=buf10)
        buf11 = reinterpret_tensor(buf4, (1, 64, 1), (64, 1, 64), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.bernoulli]
        get_raw_stream(0)
        triton_poi_fused_bernoulli_3[grid(64)](buf10, buf11, 0, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf13 = reinterpret_tensor(buf8, (1, 8192), (8192, 1), 0); del buf8  # reuse
        buf16 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [x_6, x_7, x_8, x_9], Original ATen: [aten.leaky_relu, aten.bernoulli, aten._to_copy, aten.mul, aten.add, aten.abs, aten.le, aten.scalar_tensor, aten.where, aten.neg, aten._softmax]
        get_raw_stream(0)
        triton_red_fused__softmax__to_copy_abs_add_bernoulli_le_leaky_relu_mul_neg_scalar_tensor_where_4[grid(1)](buf16, buf10, buf11, 1, 1, 8192, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del buf10
        del buf11
    return (buf16, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 64, 32, 32, 32), (2097152, 32768, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
