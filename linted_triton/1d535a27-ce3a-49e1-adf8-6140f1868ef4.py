# AOT ID: ['82_forward']
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


# kernel path: /tmp/torchinductor_sahanp/n4/cn44fgrsxpzkbyxq5aiod23j6oebqq7s4yfnvzlomhttir7su4lk.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   x => var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_group_norm_0(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 64
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


# kernel path: /tmp/torchinductor_sahanp/3e/c3el53hnrw6rkrh777j32yjl77qq2xidosiiuv5ufsbatzjqc3kl.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   x => add, rsqrt, var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_group_norm_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 4
    R0_BLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 16*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 16*x0), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r0_1 + 16*x0), xmask, other=0.0)
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
    tmp16 = 131072.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)


# kernel path: /tmp/torchinductor_sahanp/3v/c3vhp2jivoesmqpzfibopuojwbxamfdlhcm4yyhfk6xijldrir6f.py
# Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.native_group_norm, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x => add_1, mul_1
#   x_1 => constant_pad_nd
# Graph fragment:
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, %unsqueeze_7), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %unsqueeze_3), kwargs = {})
#   %constant_pad_nd : [num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%add_1, [1, 1, 1, 1, 1, 1], 0.0), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_constant_pad_nd_native_group_norm_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 628864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 1156) % 34)
    x1 = ((xindex // 34) % 34)
    x0 = (xindex % 34)
    x3 = xindex // 39304
    x4 = (xindex % 1156)
    x5 = xindex // 1156
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = (-1) + x0
    tmp9 = tmp8 >= tmp1
    tmp10 = tmp8 < tmp3
    tmp11 = tmp2 & tmp4
    tmp12 = tmp11 & tmp6
    tmp13 = tmp12 & tmp7
    tmp14 = tmp13 & tmp9
    tmp15 = tmp14 & tmp10
    tmp16 = tl.load(in_ptr0 + ((-1057) + x0 + 32*x1 + 1024*x2 + 32768*x3), tmp15 & xmask, other=0.0)
    tmp17 = tl.load(in_ptr1 + (x3 // 4), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 - tmp17
    tmp19 = tl.load(in_ptr2 + (x3 // 4), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = 131072.0
    tmp21 = tmp19 / tmp20
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp18 * tmp24
    tmp26 = tl.load(in_ptr3 + (x3), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 * tmp26
    tmp28 = tl.load(in_ptr4 + (x3), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tmp27 + tmp28
    tmp30 = tl.full(tmp29.shape, 0.0, tmp29.dtype)
    tmp31 = tl.where(tmp15, tmp29, tmp30)
    tl.store(out_ptr0 + (x4 + 1184*x5), tmp31, xmask)


# kernel path: /tmp/torchinductor_sahanp/27/c27urxijhvmhjuodtkvhv5c53gt7kujc7j2ooawjkvla2fvdzo7w.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.rand]
# Source node to ATen node mapping:
#   x_2 => inductor_lookup_seed_default, inductor_random_default_2
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default_2 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 16, 3], %inductor_lookup_seed_default, rand), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_rand_3(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 48
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (x0), tmp2, xmask)


# kernel path: /tmp/torchinductor_sahanp/35/c35z46tuyfmkyqc3ruzkdshd5owu6y3scl7fhl2zjpewo6bt7xe4.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   x_3 => add_2, rsqrt_1, var_mean_1
# Graph fragment:
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_2, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_group_norm_4(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 4
    r0_numel = 2048
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
        tmp0 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp8 = 2048.0
    tmp9 = tmp3 / tmp8
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tl.store(out_ptr2 + (x0), tmp12, xmask)


# kernel path: /tmp/torchinductor_sahanp/x7/cx7lepj4qmskf66runjo4of5tb2wcerthgub7zhziynj335a5ofw.py
# Topologically Sorted Source Nodes: [x_3, x_4], Original ATen: [aten.native_group_norm, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_3 => add_3, mul_3
#   x_4 => constant_pad_nd_1
# Graph fragment:
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_3, %unsqueeze_15), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %unsqueeze_11), kwargs = {})
#   %constant_pad_nd_1 : [num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%add_3, [1, 1, 1, 1, 1, 1], 0.0), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_constant_pad_nd_native_group_norm_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 100) % 10)
    x1 = ((xindex // 10) % 10)
    x0 = (xindex % 10)
    x3 = xindex // 1000
    x8 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = (-1) + x0
    tmp9 = tmp8 >= tmp1
    tmp10 = tmp8 < tmp3
    tmp11 = tmp2 & tmp4
    tmp12 = tmp11 & tmp6
    tmp13 = tmp12 & tmp7
    tmp14 = tmp13 & tmp9
    tmp15 = tmp14 & tmp10
    tmp16 = tl.load(in_ptr0 + ((-73) + x0 + 8*x1 + 64*x2 + 512*x3), tmp15 & xmask, other=0.0)
    tmp17 = tl.load(in_ptr1 + (x3 // 4), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 - tmp17
    tmp19 = tl.load(in_ptr2 + (x3 // 4), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = 2048.0
    tmp21 = tmp19 / tmp20
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp18 * tmp24
    tmp26 = tl.load(in_ptr3 + (x3), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 * tmp26
    tmp28 = tl.load(in_ptr4 + (x3), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tmp27 + tmp28
    tmp30 = tl.full(tmp29.shape, 0.0, tmp29.dtype)
    tmp31 = tl.where(tmp15, tmp29, tmp30)
    tl.store(out_ptr0 + (x8), tmp31, xmask)


# kernel path: /tmp/torchinductor_sahanp/4z/c4zsuaxc4t3plksai6vbivls2lgrhz2mxnrh724m2rv4b6h6txwa.py
# Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.rand]
# Source node to ATen node mapping:
#   x_5 => inductor_lookup_seed_default_1, inductor_random_default_1
# Graph fragment:
#   %inductor_lookup_seed_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 1), kwargs = {})
#   %inductor_random_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 16, 3], %inductor_lookup_seed_default_1, rand), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_rand_6(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 48
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (x0), tmp2, xmask)


# kernel path: /tmp/torchinductor_sahanp/we/cwewkxiod2burj5jvn6pek5orwnckfeeo6qt7h5lxzbvn5smzd3q.py
# Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   x_6 => add_4, rsqrt_2, var_mean_2
# Graph fragment:
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_4, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_8, 1e-05), kwargs = {})
#   %rsqrt_2 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_group_norm_7(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel):
    XBLOCK: tl.constexpr = 1
    R0_BLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    tl.full([R0_BLOCK], True, tl.int1)
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 256*x0), None)
    tmp1 = tl.broadcast_to(tmp0, [R0_BLOCK])
    tmp3 = tl.broadcast_to(tmp1, [R0_BLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tl.full([1], 256, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 / tmp7
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [R0_BLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 256.0
    tmp15 = tmp13 / tmp14
    tmp16 = 1e-05
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tl.store(out_ptr2 + (x0), tmp18, None)
    tl.store(out_ptr0 + (x0), tmp8, None)
    tl.store(out_ptr1 + (x0), tmp13, None)


# kernel path: /tmp/torchinductor_sahanp/ft/cfte5hu3qcszdswdh3357ngantj7ifvuurosbum6h7gavfuqedan.py
# Topologically Sorted Source Nodes: [x_6, x_7], Original ATen: [aten.native_group_norm, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_6 => add_5, mul_5
#   x_7 => constant_pad_nd_2
# Graph fragment:
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_5, %unsqueeze_23), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_19), kwargs = {})
#   %constant_pad_nd_2 : [num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%add_5, [1, 1, 1, 1, 1, 1], 0.0), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_constant_pad_nd_native_group_norm_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 36) % 6)
    x1 = ((xindex // 6) % 6)
    x0 = (xindex % 6)
    x3 = xindex // 216
    x8 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = (-1) + x0
    tmp9 = tmp8 >= tmp1
    tmp10 = tmp8 < tmp3
    tmp11 = tmp2 & tmp4
    tmp12 = tmp11 & tmp6
    tmp13 = tmp12 & tmp7
    tmp14 = tmp13 & tmp9
    tmp15 = tmp14 & tmp10
    tmp16 = tl.load(in_ptr0 + ((-21) + x0 + 4*x1 + 16*x2 + 64*x3), tmp15 & xmask, other=0.0)
    tmp17 = tl.load(in_ptr1 + (x3 // 4), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 - tmp17
    tmp19 = tl.load(in_ptr2 + (x3 // 4), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = 256.0
    tmp21 = tmp19 / tmp20
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp18 * tmp24
    tmp26 = tl.load(in_ptr3 + (x3), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 * tmp26
    tmp28 = tl.load(in_ptr4 + (x3), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tmp27 + tmp28
    tmp30 = tl.full(tmp29.shape, 0.0, tmp29.dtype)
    tmp31 = tl.where(tmp15, tmp29, tmp30)
    tl.store(out_ptr0 + (x8), tmp31, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7 = args
    args.clear()
    assert_size_stride(primals_1, (16, ), (1, ))
    assert_size_stride(primals_2, (16, ), (1, ))
    assert_size_stride(primals_3, (1, 16, 32, 32, 32), (524288, 32768, 1024, 32, 1))
    assert_size_stride(primals_4, (16, ), (1, ))
    assert_size_stride(primals_5, (16, ), (1, ))
    assert_size_stride(primals_6, (16, ), (1, ))
    assert_size_stride(primals_7, (16, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 4, 1, 1, 16), (64, 16, 64, 64, 1), torch.float32)
        buf1 = empty_strided_cuda((1, 4, 1, 1, 16), (64, 16, 64, 64, 1), torch.float32)
        buf2 = empty_strided_cuda((1, 4, 1, 1, 16), (64, 16, 64, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.native_group_norm]
        get_raw_stream(0)
        triton_red_fused_native_group_norm_0[grid(64)](primals_3, buf0, buf1, buf2, 64, 8192, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        buf3 = empty_strided_cuda((1, 4, 1, 1), (4, 1, 4, 4), torch.float32)
        buf4 = empty_strided_cuda((1, 4, 1, 1), (4, 1, 4, 4), torch.float32)
        buf30 = empty_strided_cuda((1, 4, 1, 1), (4, 1, 4, 4), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.native_group_norm]
        get_raw_stream(0)
        triton_per_fused_native_group_norm_1[grid(4)](buf0, buf1, buf2, buf3, buf4, buf30, 4, 16, XBLOCK=1, num_warps=2, num_stages=1)
        del buf0
        del buf1
        del buf2
        buf6 = empty_strided_cuda((1, 16, 34, 34, 34), (644096, 40256, 1184, 34, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.native_group_norm, aten.constant_pad_nd]
        get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_native_group_norm_2[grid(628864)](primals_3, buf3, buf4, primals_1, primals_2, buf6, 628864, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_1
        del primals_2
        buf7 = empty_strided_cuda((3, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [3], out=buf7)
        buf8 = empty_strided_cuda((1, 16, 3), (48, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.rand]
        get_raw_stream(0)
        triton_poi_fused_rand_3[grid(48)](buf7, buf8, 2, 48, XBLOCK=64, num_warps=1, num_stages=1)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.fractional_max_pool3d]
        buf9 = torch.ops.aten.fractional_max_pool3d.default(buf6, [2, 2, 2], [8, 8, 8], buf8)
        buf10 = buf9[0]
        buf11 = buf9[1]
        del buf9
        buf12 = buf4; del buf4  # reuse
        buf13 = empty_strided_cuda((1, 4, 1, 1), (4, 1, 4, 4), torch.float32)
        buf15 = empty_strided_cuda((1, 4, 1, 1), (4, 1, 4, 4), torch.float32)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.native_group_norm]
        get_raw_stream(0)
        triton_red_fused_native_group_norm_4[grid(4)](buf10, buf12, buf13, buf15, 4, 2048, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        buf16 = empty_strided_cuda((1, 16, 10, 10, 10), (16000, 1000, 100, 10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_3, x_4], Original ATen: [aten.native_group_norm, aten.constant_pad_nd]
        get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_native_group_norm_5[grid(16000)](buf10, buf12, buf13, primals_4, primals_5, buf16, 16000, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_5
        buf17 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.rand]
        get_raw_stream(0)
        triton_poi_fused_rand_6[grid(48)](buf7, buf17, 1, 48, XBLOCK=64, num_warps=1, num_stages=1)
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.fractional_max_pool3d]
        buf18 = torch.ops.aten.fractional_max_pool3d.default(buf16, [2, 2, 2], [4, 4, 4], buf17)
        buf19 = buf18[0]
        buf20 = buf18[1]
        del buf18
        buf21 = buf13; del buf13  # reuse
        buf22 = empty_strided_cuda((1, 4, 1, 1), (4, 1, 4, 4), torch.float32)
        buf24 = empty_strided_cuda((1, 4, 1, 1), (4, 1, 4, 4), torch.float32)
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.native_group_norm]
        get_raw_stream(0)
        triton_per_fused_native_group_norm_7[grid(4)](buf19, buf21, buf22, buf24, 4, 256, num_warps=2, num_stages=1)
        buf25 = empty_strided_cuda((1, 16, 6, 6, 6), (3456, 216, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_6, x_7], Original ATen: [aten.native_group_norm, aten.constant_pad_nd]
        get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_native_group_norm_8[grid(3456)](buf19, buf21, buf22, primals_6, primals_7, buf25, 3456, XBLOCK=256, num_warps=4, num_stages=1)
        del buf22
        del primals_7
        buf26 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten.rand]
        get_raw_stream(0)
        triton_poi_fused_rand_3[grid(48)](buf7, buf26, 2, 48, XBLOCK=64, num_warps=1, num_stages=1)
        del buf7
        # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten.fractional_max_pool3d]
        buf27 = torch.ops.aten.fractional_max_pool3d.default(buf25, [2, 2, 2], [2, 2, 2], buf26)
        del buf26
        buf28 = buf27[0]
        buf29 = buf27[1]
        del buf27
    return (buf28, primals_3, primals_4, primals_6, buf6, buf10, buf11, reinterpret_tensor(buf12, (1, 4), (4, 1), 0), reinterpret_tensor(buf15, (1, 4), (4, 1), 0), buf16, buf19, buf20, reinterpret_tensor(buf21, (1, 4), (4, 1), 0), reinterpret_tensor(buf24, (1, 4), (4, 1), 0), buf25, buf29, reinterpret_tensor(buf3, (1, 4, 1), (4, 1, 1), 0), reinterpret_tensor(buf30, (1, 4, 1), (4, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((1, 16, 32, 32, 32), (524288, 32768, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
