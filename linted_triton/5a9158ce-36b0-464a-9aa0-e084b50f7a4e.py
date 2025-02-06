# AOT ID: ['131_inference']
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


# kernel path: /tmp/torchinductor_sahanp/io/ciobeqqjhgbuputek3ocfvikybzv67p4od6tkzm3pnd2kamtjlwy.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.max_unpool2d]
# Source node to ATen node mapping:
#   x_2 => full
# Graph fragment:
#   %full : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, %mul_16, %sub_16, 1], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_unpool2d_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)


# kernel path: /tmp/torchinductor_sahanp/jy/cjy3asimbt4omnoss3oiqs23o7s53ty2njh6uxtyed3mp7rfpd5c.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.max_unpool2d]
# Source node to ATen node mapping:
#   x_2 => index_put
# Graph fragment:
#   %index_put : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%view_5, [%view_4], %view_6), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_unpool2d_1(in_ptr0, out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*((x0 % (ks2 // 4))) + 2*(ks2 // 2)*((((((x0 // ((ks1 // 2)*(ks2 // 4))) % (4*ks0))) // 2) % 2)) + 4*(ks2 // 2)*(((x0 // (ks2 // 4)) % (ks1 // 2))) + 4*(ks1 // 2)*(ks2 // 2)*((((((x0 // ((ks1 // 2)*(ks2 // 4))) % (4*ks0))) // 4) % ks0)) + (((((x0 // ((ks1 // 2)*(ks2 // 4))) % (4*ks0))) % 2))), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (2 + 4*((x0 % (ks2 // 4))) + 2*(ks2 // 2)*((((((x0 // ((ks1 // 2)*(ks2 // 4))) % (4*ks0))) // 2) % 2)) + 4*(ks2 // 2)*(((x0 // (ks2 // 4)) % (ks1 // 2))) + 4*(ks1 // 2)*(ks2 // 2)*((((((x0 // ((ks1 // 2)*(ks2 // 4))) % (4*ks0))) // 4) % ks0)) + (((((x0 // ((ks1 // 2)*(ks2 // 4))) % (4*ks0))) % 2))), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 > tmp0
    tmp3 = tl.full([1], 1, tl.int8)
    tmp4 = tl.full([1], 0, tl.int8)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = triton_helpers.maximum(tmp1, tmp0)
    tmp7 = tl.full([1], 2, tl.int32)
    tmp8 = tl.where((tmp5 < 0) != (tmp7 < 0), tl.where(tmp5 % tmp7 != 0, tmp5 // tmp7 - 1, tmp5 // tmp7), tmp5 // tmp7)
    tmp9 = tmp8 * tmp7
    tmp10 = tmp5 - tmp9
    tmp11 = tl.full([1], 0, tl.int64)
    tmp12 = tmp11 + tmp8
    tmp13 = 2*((x0 % (ks2 // 4)))
    tmp14 = tmp13 + tmp10
    tmp15 = ks2 // 2
    tmp16 = tmp12 * tmp15
    tmp17 = tmp16 + tmp14
    tmp18 = 2*(ks2 // 4)*(triton_helpers.div_floor_integer(x0,  ks2 // 4))
    tmp19 = tmp17 + tmp18
    tmp20 = 2*ks0*(ks1 // 2)*(triton_helpers.div_floor_integer(ks1,  ks1 // 2))*(ks2 // 4)*(triton_helpers.div_floor_integer(ks2,  ks2 // 2))
    tmp21 = tmp19 + tmp20
    tmp22 = tmp19 < 0
    tmp23 = tl.where(tmp22, tmp21, tmp19)
    tl.device_assert(((0 <= tmp23) & (tmp23 < 2*ks0*(ks1 // 2)*(triton_helpers.div_floor_integer(ks1,  ks1 // 2))*(ks2 // 4)*(triton_helpers.div_floor_integer(ks2,  ks2 // 2)))) | ~(xmask), "index out of bounds: 0 <= tmp23 < 2*ks0*(ks1 // 2)*(triton_helpers.div_floor_integer(ks1,  ks1 // 2))*(ks2 // 4)*(triton_helpers.div_floor_integer(ks2,  ks2 // 2))")
    tl.store(out_ptr0 + (tl.broadcast_to((tmp23 % (2*ks0*(ks1 // 2)*(triton_helpers.div_floor_integer(ks1,  ks1 // 2))*(ks2 // 4)*(triton_helpers.div_floor_integer(ks2,  ks2 // 2)))), [XBLOCK])), tmp6, xmask)


# kernel path: /tmp/torchinductor_sahanp/o3/co33n3lkdyhrh4wxdtgvostdmk5tnlfueelz6twe5ft4onxmgapd.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten.arange, aten.ne, aten.gather, aten.rsub, aten.add, aten.clamp_min, aten.scalar_tensor, aten.where, aten.mean]
# Source node to ATen node mapping:
#   loss => add_58, clamp_min, full_default, gather, iota_1, mean, ne_13, sub_37, where
# Graph fragment:
#   %iota_1 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (%floordiv_7,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %ne_13 : [num_users=1] = call_function[target=torch.ops.aten.ne.Tensor](args = (%iota_1, %unsqueeze_3), kwargs = {})
#   %gather : [num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%view_9, 1, %unsqueeze_3), kwargs = {})
#   %sub_37 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %gather), kwargs = {})
#   %add_58 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub_37, %view_9), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_58, 0), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_13, %clamp_min, %full_default), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%where,), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_arange_clamp_min_gather_mean_ne_rsub_scalar_tensor_where_2(in_ptr0, in_ptr1, out_ptr1, load_seed_offset, ks1, ks2, ks3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    _tmp35 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp23 = tl.load(in_ptr1 + (2*(ks3 // 4)*((((2*(ks3 // 4)*((((r0_1 + ks1*x0*(ks2 // 2)*(triton_helpers.div_floor_integer(ks2,  ks2 // 2))*(ks3 // 4)*(triton_helpers.div_floor_integer(ks3,  ks3 // 2))) // (ks3 // 2)) % (ks2 // 2))) + 2*(ks2 // 2)*(ks3 // 4)*((((r0_1 + ks1*x0*(ks2 // 2)*(triton_helpers.div_floor_integer(ks2,  ks2 // 2))*(ks3 // 4)*(triton_helpers.div_floor_integer(ks3,  ks3 // 2))) // ((ks2 // 2)*(ks3 // 2))) % (ks1*(triton_helpers.div_floor_integer(ks2,  ks2 // 2))*(triton_helpers.div_floor_integer(ks3,  ks3 // 2))))) + (((r0_1 + ks1*x0*(ks2 // 2)*(triton_helpers.div_floor_integer(ks2,  ks2 // 2))*(ks3 // 4)*(triton_helpers.div_floor_integer(ks3,  ks3 // 2))) % (ks3 // 2)))) // (2*(ks3 // 4))) % (ks1*(ks2 // 2)*(triton_helpers.div_floor_integer(ks2,  ks2 // 2))*(triton_helpers.div_floor_integer(ks3,  ks3 // 2))))) + (((((r0_1 + ks1*x0*(ks2 // 2)*(triton_helpers.div_floor_integer(ks2,  ks2 // 2))*(ks3 // 4)*(triton_helpers.div_floor_integer(ks3,  ks3 // 2))) % (ks3 // 2))) % (2*(ks3 // 4))))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = tl.load(in_ptr0 + load_seed_offset)
        tmp1 = tl.full([1, 1], 0, tl.int32)
        tmp2 = tl.full([1, 1], 0, tl.int64)
        tmp3 = tl.full([1, 1], 10, tl.int64)
        tmp4 = triton_helpers.randint64(tmp0, (tmp1).to(tl.uint32), tmp2, tmp3)
        tmp5 = r0_1 + ks1*x0*(ks2 // 2)*(triton_helpers.div_floor_integer(ks2,  ks2 // 2))*(ks3 // 4)*(triton_helpers.div_floor_integer(ks3,  ks3 // 2))
        tmp6 = tmp5 != tmp4
        tmp7 = 2*ks1*(ks2 // 2)*(triton_helpers.div_floor_integer(ks2,  ks2 // 2))*(ks3 // 4)*(triton_helpers.div_floor_integer(ks3,  ks3 // 2))
        tmp8 = tmp4 + tmp7
        tmp9 = tmp4 < 0
        tmp10 = tl.where(tmp9, tmp8, tmp4)
        tl.device_assert((0 <= tmp10) & (tmp10 < 2*ks1*(ks2 // 2)*(triton_helpers.div_floor_integer(ks2,  ks2 // 2))*(ks3 // 4)*(triton_helpers.div_floor_integer(ks3,  ks3 // 2))), "index out of bounds: 0 <= tmp10 < 2*ks1*(ks2 // 2)*(triton_helpers.div_floor_integer(ks2,  ks2 // 2))*(ks3 // 4)*(triton_helpers.div_floor_integer(ks3,  ks3 // 2))")
        tmp12 = tl.load(in_ptr1 + (2*(ks3 // 4)*((((2*(ks3 // 4)*(((tmp10 // (ks3 // 2)) % (ks2 // 2))) + 2*(ks2 // 2)*(ks3 // 4)*(((tmp10 // ((ks2 // 2)*(ks3 // 2))) % (ks1*(triton_helpers.div_floor_integer(ks2,  ks2 // 2))*(triton_helpers.div_floor_integer(ks3,  ks3 // 2))))) + ((tmp10 % (ks3 // 2)))) // (2*(ks3 // 4))) % (ks1*(ks2 // 2)*(triton_helpers.div_floor_integer(ks2,  ks2 // 2))*(triton_helpers.div_floor_integer(ks3,  ks3 // 2))))) + ((((tmp10 % (ks3 // 2))) % (2*(ks3 // 4))))), None, eviction_policy='evict_last')
        tmp13 = 0.5
        tmp14 = tmp12 * tmp13
        tmp15 = 0.7071067811865476
        tmp16 = tmp12 * tmp15
        tmp17 = libdevice.erf(tmp16)
        tmp18 = 1.0
        tmp19 = tmp17 + tmp18
        tmp20 = tmp14 * tmp19
        tmp21 = tl.sigmoid(tmp20)
        tmp22 = tmp18 - tmp21
        tmp24 = tmp23 * tmp13
        tmp25 = tmp23 * tmp15
        tmp26 = libdevice.erf(tmp25)
        tmp27 = tmp26 + tmp18
        tmp28 = tmp24 * tmp27
        tmp29 = tl.sigmoid(tmp28)
        tmp30 = tmp22 + tmp29
        tmp31 = 0.0
        tmp32 = triton_helpers.maximum(tmp30, tmp31)
        tmp33 = tl.where(tmp6, tmp32, tmp31)
        tmp34 = tl.broadcast_to(tmp33, [XBLOCK, R0_BLOCK])
        tmp36 = _tmp35 + tmp34
        _tmp35 = tl.where(r0_mask & xmask, tmp36, _tmp35)
    tmp35 = tl.sum(_tmp35, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp35, xmask)


# kernel path: /tmp/torchinductor_sahanp/o7/co74pvlixysy25xq7byrt36y2aue6pogge54yb4a4gy4qb264xtg.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   loss => mean
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%where,), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_mean_3(in_out_ptr0, in_ptr0, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    R0_BLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None]
    tmp4 = 2*ks0*(ks1 // 2)*(triton_helpers.div_floor_integer(ks1,  ks1 // 2))*(ks2 // 4)*(triton_helpers.div_floor_integer(ks2,  ks2 // 2))
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp6, None)


def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, s0*(s1 // 2)*(s1 // (s1 // 2))*(s2 // (s2 // 2)), 2*(s2 // 4), 1), (2*s0*(s1 // 2)*(s1 // (s1 // 2))*(s2 // 4)*(s2 // (s2 // 2)), 2*(s2 // 4), 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.max_unpool2d]
        triton_poi_fused_max_unpool2d_0_xnumel = 2*s0*(s1 // 2)*(s1 // (s1 // 2))*(s2 // 4)*(s2 // (s2 // 2))
        get_raw_stream(0)
        triton_poi_fused_max_unpool2d_0[grid(triton_poi_fused_max_unpool2d_0_xnumel)](buf0, 12288, XBLOCK=256, num_warps=4, num_stages=1)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.max_unpool2d]
        triton_poi_fused_max_unpool2d_1_xnumel = s0*(s1 // 2)*(s1 // (s1 // 2))*(s2 // 4)*(s2 // (s2 // 2))
        get_raw_stream(0)
        triton_poi_fused_max_unpool2d_1[grid(triton_poi_fused_max_unpool2d_1_xnumel)](arg3_1, buf0, 3, 64, 64, 6144, XBLOCK=128, num_warps=4, num_stages=1)
        del arg3_1
        buf2 = empty_strided_cuda((1, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf2)
        buf4 = empty_strided_cuda((2, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [loss], Original ATen: [aten.arange, aten.ne, aten.gather, aten.rsub, aten.add, aten.clamp_min, aten.scalar_tensor, aten.where, aten.mean]
        s0*(s1 // 2)*(s1 // (s1 // 2))*(s2 // 4)*(s2 // (s2 // 2))
        get_raw_stream(0)
        triton_red_fused_add_arange_clamp_min_gather_mean_ne_rsub_scalar_tensor_where_2[grid(2)](buf2, buf0, buf4, 0, 3, 64, 64, 2, 6144, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del buf0
        del buf2
        buf5 = empty_strided_cuda((), (), torch.float32)
        buf6 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [loss], Original ATen: [aten.mean]
        get_raw_stream(0)
        triton_per_fused_mean_3[grid(1)](buf6, buf4, 3, 64, 64, 1, 2, XBLOCK=1, num_warps=2, num_stages=1)
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
