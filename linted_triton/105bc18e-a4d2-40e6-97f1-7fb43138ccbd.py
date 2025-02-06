# AOT ID: ['48_inference']
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


# kernel path: /tmp/torchinductor_sahanp/5r/c5rbjah6y5ob2jexek2eua5vfetiyuaxw6zwnumtvbmm7cku2ceb.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.bernoulli]
# Source node to ATen node mapping:
#   x => inductor_lookup_seed_default, inductor_random_default
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, %arg0_1, 1, 1], %inductor_lookup_seed_default, rand), kwargs = {})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_bernoulli_0(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (x0), tmp2, xmask)


# kernel path: /tmp/torchinductor_sahanp/7v/c7vv24h43dnm3oavcoakeeazxt5bpgfokpa2fcx6qm7cjpqsfim2.py
# Topologically Sorted Source Nodes: [dist_pos, dist_neg], Original ATen: [aten.sub, aten.add, aten.norm]
# Source node to ATen node mapping:
#   dist_neg => add_45, pow_3, sub_21, sum_2
#   dist_pos => add_40, pow_1, sub_18, sum_1
# Graph fragment:
#   %sub_18 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %view), kwargs = {})
#   %add_40 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%sub_18, 1e-06), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_40, 2.0), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1]), kwargs = {})
#   %sub_21 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %view), kwargs = {})
#   %add_45 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%sub_21, 1e-06), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_45, 2.0), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_3, [1]), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_norm_sub_1(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    _tmp22 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp0 = r0_1 + x0*((1 + ks0*ks1*ks2) // 2)
        tmp1 = ks0*ks1*ks2
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (((r0_1 + x0*((1 + ks0*ks1*ks2) // 2)) % (ks0*ks1*ks2))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + ((((r0_1 + x0*((1 + ks0*ks1*ks2) // 2)) // (ks1*ks2)) % ks0)), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = 0.5
        tmp6 = tmp4 < tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = 2.0
        tmp9 = tmp7 * tmp8
        tmp10 = tmp3 * tmp9
        tmp11 = tl_math.abs(tmp10)
        tmp12 = tmp11 <= tmp5
        tmp13 = 0.0
        tmp14 = tl.where(tmp12, tmp13, tmp10)
        tmp15 = tmp14 - tmp14
        tmp16 = 1e-06
        tmp17 = tmp15 + tmp16
        tmp18 = tmp17 * tmp17
        tmp19 = tl.full(tmp18.shape, 0, tmp18.dtype)
        tmp20 = tl.where(tmp2, tmp18, tmp19)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(r0_mask & xmask, tmp23, _tmp22)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp22, xmask)
    tl.store(out_ptr1 + (x0), tmp22, xmask)


# kernel path: /tmp/torchinductor_sahanp/dq/cdqnhv6luqb3x7epo3vdjtf2p5gkpoubsrvvhspkye2fks37tgc5.py
# Topologically Sorted Source Nodes: [dist_pos, add, dist_neg, sub, loss, loss_1], Original ATen: [aten.sub, aten.add, aten.norm, aten.clamp_min, aten.mean]
# Source node to ATen node mapping:
#   add => add_48
#   dist_neg => add_45, pow_3, pow_4, sub_21, sum_2
#   dist_pos => add_40, pow_1, pow_2, sub_18, sum_1
#   loss => clamp_min
#   loss_1 => mean
#   sub => sub_24
# Graph fragment:
#   %sub_18 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %view), kwargs = {})
#   %add_40 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%sub_18, 1e-06), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_40, 2.0), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1]), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %add_48 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_2, 1.0), kwargs = {})
#   %sub_21 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %view), kwargs = {})
#   %add_45 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%sub_21, 1e-06), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_45, 2.0), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_3, [1]), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_2, 0.5), kwargs = {})
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_48, %pow_4), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_24, 0), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%clamp_min,), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_clamp_min_mean_norm_sub_2(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    R0_BLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), None)
    tmp4 = tl.load(in_ptr1 + (r0_0), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None]
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.sum(tmp5, 1)[:, None]
    tmp8 = libdevice.sqrt(tmp3)
    tmp9 = 1.0
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp7)
    tmp12 = tmp10 - tmp11
    tmp13 = 0.0
    tmp14 = triton_helpers.maximum(tmp12, tmp13)
    tmp15 = tmp14 / tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp15, None)


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
        buf1 = empty_strided_cuda((1, s0, 1, 1), (s0, 1, s0, s0), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.bernoulli]
        get_raw_stream(0)
        triton_poi_fused_bernoulli_0[grid(s0)](buf0, buf1, 0, 3, XBLOCK=4, num_warps=1, num_stages=1)
        del buf0
        buf2 = empty_strided_cuda((1, 2), (2, 1), torch.float32)
        buf4 = empty_strided_cuda((1, 2), (2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [dist_pos, dist_neg], Original ATen: [aten.sub, aten.add, aten.norm]
        (1 + s0*s1*s2) // 2
        get_raw_stream(0)
        triton_red_fused_add_norm_sub_1[grid(2)](arg3_1, buf1, buf2, buf4, 3, 64, 64, 2, 6144, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del arg3_1
        del buf1
        buf3 = empty_strided_cuda((1, ), (1, ), torch.float32)
        buf6 = reinterpret_tensor(buf3, (), (), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [dist_pos, add, dist_neg, sub, loss, loss_1], Original ATen: [aten.sub, aten.add, aten.norm, aten.clamp_min, aten.mean]
        get_raw_stream(0)
        triton_per_fused_add_clamp_min_mean_norm_sub_2[grid(1)](buf6, buf2, buf4, 1, 2, XBLOCK=1, num_warps=2, num_stages=1)
        del buf2
        del buf4
    return (buf6, )


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
