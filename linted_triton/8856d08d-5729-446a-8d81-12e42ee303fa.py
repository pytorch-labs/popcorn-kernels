# AOT ID: ['39_inference']
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


# kernel path: /tmp/torchinductor_sahanp/tq/ctqjxxl5ikxegiv5wa3eisurqpzaz3qtpu2pztpnhasouiv5235j.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   x => var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [0, 2]), kwargs = {correction: 0, keepdim: True})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_0(in_ptr0, out_ptr0, out_ptr1, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp7[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tl.store(out_ptr1 + (x0), tmp3, xmask)


# kernel path: /tmp/torchinductor_sahanp/ch/cchqju5nvoo2fxc6ljkhu4g6omi364v7sgimjmp5bxay43d2devv.py
# Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten._native_batch_norm_legit, aten.view, aten.glu]
# Source node to ATen node mapping:
#   x => add_3, mul_6, rsqrt, sub_1, var_mean, view_1
#   x_1 => glu
#   x_2 => var_mean_1
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %getitem_1), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_3,), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt), kwargs = {})
#   %view_1 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_6, [1, 128, %arg0_1]), kwargs = {})
#   %glu : [num_users=1] = call_function[target=torch.ops.aten.glu.default](args = (%view_1, 1), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_2, [0, 2]), kwargs = {correction: 0, keepdim: True})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_glu_view_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (64 + x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr2 + (64 + x0), xmask, eviction_policy='evict_last')
    tmp22_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp22_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp22_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.load(in_ptr0 + (r0_1 + 64*ks0 + ks0*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 - tmp1
        tmp4 = ks0
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 / tmp5
        tmp7 = 1e-05
        tmp8 = tmp6 + tmp7
        tmp9 = libdevice.rsqrt(tmp8)
        tmp10 = tmp2 * tmp9
        tmp13 = tmp11 - tmp12
        tmp15 = tmp14 / tmp5
        tmp16 = tmp15 + tmp7
        tmp17 = libdevice.rsqrt(tmp16)
        tmp18 = tmp13 * tmp17
        tmp19 = tl.sigmoid(tmp18)
        tmp20 = tmp10 * tmp19
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
        tmp22_mean_next, tmp22_m2_next, tmp22_weight_next = triton_helpers.welford_reduce(
            tmp21, tmp22_mean, tmp22_m2, tmp22_weight, roffset == 0
        )
        tmp22_mean = tl.where(r0_mask & xmask, tmp22_mean_next, tmp22_mean)
        tmp22_m2 = tl.where(r0_mask & xmask, tmp22_m2_next, tmp22_m2)
        tmp22_weight = tl.where(r0_mask & xmask, tmp22_weight_next, tmp22_weight)
        tl.store(out_ptr0 + (r0_1 + ks0*x0), tmp20, r0_mask & xmask)
    tmp25, tmp26, tmp27 = triton_helpers.welford(tmp22_mean, tmp22_m2, tmp22_weight, 1)
    tmp22 = tmp25[:, None]
    tmp23 = tmp26[:, None]
    tmp27[:, None]
    tl.store(out_ptr1 + (x0), tmp22, xmask)
    tl.store(out_ptr2 + (x0), tmp23, xmask)


# kernel path: /tmp/torchinductor_sahanp/3t/c3t7bwbncw37r6hgwzh2w6barkvs62bv7kpgf4w7l3yczk4urzdw.py
# Topologically Sorted Source Nodes: [target, loss], Original ATen: [aten.randint, aten.nll_loss2d_forward]
# Source node to ATen node mapping:
#   loss => convert_element_type, div, full_default_1, gather, ne_3, ne_4, neg, sum_1, sum_2, where_1
#   target => inductor_lookup_seed_default, inductor_randint_default
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_randint_default : [num_users=4] = call_function[target=torch.ops.prims.inductor_randint.default](args = (0, 32, [1, 1, %arg0_1], %inductor_lookup_seed_default), kwargs = {})
#   %ne_3 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%inductor_randint_default, -100), kwargs = {})
#   %gather : [num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%view_4, 1, %unsqueeze), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze_4,), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_3, %neg, %full_default_1), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_1,), kwargs = {})
#   %ne_4 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%inductor_randint_default, -100), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne_4,), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_1, torch.float32), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_2, %convert_element_type), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_nll_loss2d_forward_randint_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, load_seed_offset, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    _tmp38 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp42 = tl.full([XBLOCK, R0_BLOCK], 0, tl.int64)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + load_seed_offset)
        tmp1 = r0_0
        tmp2 = tl.full([1, 1], 0, tl.int64)
        tmp3 = tl.full([1, 1], 32, tl.int64)
        tmp4 = triton_helpers.randint64(tmp0, (tmp1).to(tl.uint32), tmp2, tmp3)
        tmp5 = tl.full([1, 1], -100, tl.int64)
        tmp6 = tmp4 != tmp5
        tmp7 = tl.where(tmp6, tmp4, tmp2)
        tmp8 = tl.full([XBLOCK, R0_BLOCK], 32, tl.int32)
        tmp9 = tmp7 + tmp8
        tmp10 = tmp7 < 0
        tmp11 = tl.where(tmp10, tmp9, tmp7)
        tl.device_assert(((0 <= tmp11) & (tmp11 < 32)) | ~(r0_mask), "index out of bounds: 0 <= tmp11 < 32")
        tmp13 = tl.load(in_ptr1 + (r0_0 + ks1*tmp11), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr2 + (tmp11), r0_mask, eviction_policy='evict_last')
        tmp15 = tmp13 - tmp14
        tmp16 = tl.load(in_ptr3 + (tmp11), r0_mask, eviction_policy='evict_last')
        tmp17 = ks1
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp16 / tmp18
        tmp20 = 1e-05
        tmp21 = tmp19 + tmp20
        tmp22 = libdevice.rsqrt(tmp21)
        tmp23 = tmp15 * tmp22
        tmp24 = tl.load(in_ptr1 + (r0_0 + 32*ks1 + ks1*tmp11), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp25 = tl.load(in_ptr2 + (32 + tmp11), r0_mask, eviction_policy='evict_last')
        tmp26 = tmp24 - tmp25
        tmp27 = tl.load(in_ptr3 + (32 + tmp11), r0_mask, eviction_policy='evict_last')
        tmp28 = tmp27 / tmp18
        tmp29 = tmp28 + tmp20
        tmp30 = libdevice.rsqrt(tmp29)
        tmp31 = tmp26 * tmp30
        tmp32 = tl.sigmoid(tmp31)
        tmp33 = tmp23 * tmp32
        tmp34 = -tmp33
        tmp35 = 0.0
        tmp36 = tl.where(tmp6, tmp34, tmp35)
        tmp37 = tl.broadcast_to(tmp36, [XBLOCK, R0_BLOCK])
        tmp39 = _tmp38 + tmp37
        _tmp38 = tl.where(r0_mask, tmp39, _tmp38)
        tmp40 = tmp6.to(tl.int64)
        tmp41 = tl.broadcast_to(tmp40, [XBLOCK, R0_BLOCK])
        tmp43 = _tmp42 + tmp41
        _tmp42 = tl.where(r0_mask, tmp43, _tmp42)
    tmp38 = tl.sum(_tmp38, 1)[:, None]
    tmp42 = tl.sum(_tmp42, 1)[:, None]
    tmp44 = tmp42.to(tl.float32)
    tmp45 = tmp38 / tmp44
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp45, None)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    s1 = arg0_1
    assert_size_stride(arg1_1, (1, 128, s1), (128*s1, s1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
        buf1 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten._native_batch_norm_legit]
        get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_0[grid(128)](arg1_1, buf0, buf1, 100, 128, 100, XBLOCK=1, R0_BLOCK=128, num_warps=2, num_stages=1)
        buf3 = empty_strided_cuda((1, 64, s1), (64*s1, s1, 1), torch.float32)
        buf4 = empty_strided_cuda((1, 64, 1), (64, 1, 64), torch.float32)
        buf5 = empty_strided_cuda((1, 64, 1), (64, 1, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten._native_batch_norm_legit, aten.view, aten.glu]
        get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_glu_view_1[grid(64)](arg1_1, buf0, buf1, buf3, buf4, buf5, 100, 64, 100, XBLOCK=1, R0_BLOCK=128, num_warps=2, num_stages=1)
        del arg1_1
        del buf0
        del buf1
        buf7 = empty_strided_cuda((1, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf7)
        buf9 = empty_strided_cuda((), (), torch.float32)
        buf11 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [target, loss], Original ATen: [aten.randint, aten.nll_loss2d_forward]
        get_raw_stream(0)
        triton_red_fused_nll_loss2d_forward_randint_2[grid(1)](buf11, buf7, buf3, buf4, buf5, 0, 100, 1, 100, XBLOCK=1, R0_BLOCK=128, num_warps=2, num_stages=1)
        del buf3
        del buf4
        del buf5
        del buf7
    return (buf11, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 100
    arg1_1 = rand_strided((1, 128, 100), (12800, 100, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
