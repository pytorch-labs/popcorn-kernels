# AOT ID: ['23_forward']
import torch
from torch._inductor.select_algorithm import extern_kernels
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


# kernel path: /tmp/torchinductor_sahanp/2r/c2rojbqogblfyneyams2g77n3dgycftaowe4pbnj53iz7pw67ecy.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.copy]
# Source node to ATen node mapping:
#   x_1 => copy
# Graph fragment:
#   %copy : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_5, %slice_6), kwargs = {})
#   %slice_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_tensor_1, %copy, 2, 1, 15), kwargs = {})
#   %slice_scatter_default_1 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_tensor, %slice_scatter_default, 3, 1, 15), kwargs = {})
#   %slice_scatter_default_2 : [num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%empty, %slice_scatter_default_1, 4, 1, 15), kwargs = {})
#   %slice_scatter_default_3 : [num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_2, %slice_18, 4, 0, 1), kwargs = {})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_copy_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 16)
    x1 = ((xindex // 16) % 16)
    x2 = ((xindex // 256) % 16)
    x3 = xindex // 4096
    x5 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = 14 + x0
    tmp4 = tl.full([1], 1, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tl.full([1], 15, tl.int64)
    tmp7 = tmp3 < tmp6
    tmp8 = tmp5 & tmp7
    tmp9 = tmp8 & tmp2
    tmp10 = x1
    tmp11 = tl.full([1], 1, tl.int64)
    tmp12 = tmp10 >= tmp11
    tmp13 = tl.full([1], 15, tl.int64)
    tmp14 = tmp10 < tmp13
    tmp15 = tmp12 & tmp14
    tmp16 = tmp15 & tmp9
    tmp17 = x2
    tmp18 = tl.full([1], 1, tl.int64)
    tmp19 = tmp17 >= tmp18
    tmp20 = tl.full([1], 15, tl.int64)
    tmp21 = tmp17 < tmp20
    tmp22 = tmp19 & tmp21
    tmp23 = tmp22 & tmp16
    tmp24 = tl.load(in_ptr0 + ((-197) + x0 + 14*x1 + 196*x2 + 2744*x3), tmp23, other=0.0)
    tmp25 = tl.load(in_ptr1 + (x3), tmp23, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp24 + tmp25
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp23, tmp26, tmp27)
    tmp29 = tl.load(in_ptr2 + (14 + x5), tmp16, other=0.0)
    tmp30 = tl.where(tmp22, tmp28, tmp29)
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp16, tmp30, tmp31)
    tmp33 = tl.load(in_ptr2 + (14 + x5), tmp9, other=0.0)
    tmp34 = tl.where(tmp15, tmp32, tmp33)
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp9, tmp34, tmp35)
    tmp37 = float("nan")
    tmp38 = tl.where(tmp8, tmp36, tmp37)
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(tmp2, tmp38, tmp39)
    tmp41 = tmp0 >= tmp1
    tmp42 = tl.full([1], 15, tl.int64)
    tmp43 = tmp0 < tmp42
    tmp44 = tmp41 & tmp43
    tmp45 = x1
    tmp46 = tl.full([1], 1, tl.int64)
    tmp47 = tmp45 >= tmp46
    tmp48 = tl.full([1], 15, tl.int64)
    tmp49 = tmp45 < tmp48
    tmp50 = tmp47 & tmp49
    tmp51 = tmp50 & tmp44
    tmp52 = x2
    tmp53 = tl.full([1], 1, tl.int64)
    tmp54 = tmp52 >= tmp53
    tmp55 = tl.full([1], 15, tl.int64)
    tmp56 = tmp52 < tmp55
    tmp57 = tmp54 & tmp56
    tmp58 = tmp57 & tmp51
    tmp59 = tl.load(in_ptr0 + ((-211) + x0 + 14*x1 + 196*x2 + 2744*x3), tmp58, other=0.0)
    tmp60 = tl.load(in_ptr1 + (x3), tmp58, eviction_policy='evict_last', other=0.0)
    tmp61 = tmp59 + tmp60
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp58, tmp61, tmp62)
    tmp64 = tl.load(in_ptr2 + (x5), tmp51, other=0.0)
    tmp65 = tl.where(tmp57, tmp63, tmp64)
    tmp66 = tl.full(tmp65.shape, 0.0, tmp65.dtype)
    tmp67 = tl.where(tmp51, tmp65, tmp66)
    tmp68 = tl.load(in_ptr2 + (x5), tmp44, other=0.0)
    tmp69 = tl.where(tmp50, tmp67, tmp68)
    tmp70 = tl.full(tmp69.shape, 0.0, tmp69.dtype)
    tmp71 = tl.where(tmp44, tmp69, tmp70)
    tmp72 = float("nan")
    tmp73 = tl.where(tmp44, tmp71, tmp72)
    tmp74 = tl.where(tmp2, tmp40, tmp73)
    tl.store(out_ptr0 + (x5), tmp74, None)


# kernel path: /tmp/torchinductor_sahanp/vr/cvrtgy77jg6ud2pfx2xsu2axcmnagyhkcqtq3jdqa5zzsxaosbaf.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %slice_scatter_default_4 : [num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_3, %slice_24, 4, 15, 16), kwargs = {})
#   %slice_scatter_default_5 : [num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_4, %slice_30, 3, 0, 1), kwargs = {})
#   %slice_scatter_default_6 : [num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_5, %slice_36, 3, 15, 16), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x3 = xindex // 16
    x4 = xindex
    tmp40 = tl.load(in_ptr0 + (x4), None)
    tmp0 = x1
    tmp1 = tl.full([1], 15, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = (-14) + x1
    tmp4 = tl.full([1], 1, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = x0
    tmp8 = tl.full([1], 15, tl.int64)
    tmp9 = tmp7 >= tmp8
    tmp10 = tmp9 & tmp6
    tmp11 = tl.load(in_ptr0 + (1 + 16*x3), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr0 + (x4), tmp6, other=0.0)
    tmp13 = tl.where(tmp9, tmp11, tmp12)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = x0
    tmp17 = tl.full([1], 15, tl.int64)
    tmp18 = tmp16 >= tmp17
    tmp19 = tmp18 & tmp2
    tmp20 = tl.load(in_ptr0 + ((-223) + 16*x3), tmp19, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.load(in_ptr0 + ((-224) + x4), tmp2, other=0.0)
    tmp22 = tl.where(tmp18, tmp20, tmp21)
    tmp23 = tl.where(tmp5, tmp15, tmp22)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp2, tmp23, tmp24)
    tmp26 = tl.full([1], 1, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = x0
    tmp29 = tl.full([1], 15, tl.int64)
    tmp30 = tmp28 >= tmp29
    tmp31 = tmp30 & tmp27
    tmp32 = tl.load(in_ptr0 + (225 + 16*x3), tmp31, eviction_policy='evict_last', other=0.0)
    tmp33 = tl.load(in_ptr0 + (224 + x4), tmp27, other=0.0)
    tmp34 = tl.where(tmp30, tmp32, tmp33)
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp27, tmp34, tmp35)
    tmp37 = x0
    tmp38 = tmp37 >= tmp1
    tmp39 = tl.load(in_ptr0 + (1 + 16*x3), tmp38, eviction_policy='evict_last', other=0.0)
    tmp41 = tl.where(tmp38, tmp39, tmp40)
    tmp42 = tl.where(tmp27, tmp36, tmp41)
    tmp43 = tl.where(tmp2, tmp25, tmp42)
    tl.store(out_ptr0 + (x4), tmp43, None)


# kernel path: /tmp/torchinductor_sahanp/ur/curcb6phg2texb74q63v7nttrvmszguhgubfqoccwkgxqz2zcu7c.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %slice_scatter_default_7 : [num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_6, %slice_42, 2, 0, 1), kwargs = {})
#   %slice_scatter_default_8 : [num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default_7, %slice_48, 2, 15, 16), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 256) % 16)
    x0 = (xindex % 256)
    x2 = xindex // 4096
    x3 = xindex
    tmp15 = tl.load(in_ptr0 + (x3), None)
    tmp0 = x1
    tmp1 = tl.full([1], 15, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = (-14) + x1
    tmp4 = tl.full([1], 1, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + (3584 + x0 + 4096*x2), tmp6, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr0 + ((-3584) + x3), tmp2, other=0.0)
    tmp9 = tl.where(tmp5, tmp7, tmp8)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = tl.full([1], 1, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr0 + (3584 + x0 + 4096*x2), tmp13, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.where(tmp13, tmp14, tmp15)
    tmp17 = tl.where(tmp2, tmp11, tmp16)
    tl.store(out_ptr0 + (x3), tmp17, None)


# kernel path: /tmp/torchinductor_sahanp/7v/c7vbyjy32oc3vid2qeskzhtudn6rusfjszvikgbpo4t2rkyoj4y5.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.rand]
# Source node to ATen node mapping:
#   x_2 => inductor_lookup_seed_default, inductor_random_default
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 10, 3], %inductor_lookup_seed_default, rand), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_rand_3(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 30
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (x0), tmp2, xmask)


# kernel path: /tmp/torchinductor_sahanp/al/cal2cfa6errspg246zocu6s6xaa3x4zw766tlqnnptbh53s5bogp.py
# Topologically Sorted Source Nodes: [x_3, x_4], Original ATen: [aten.rrelu_with_noise_functional, aten.native_group_norm]
# Source node to ATen node mapping:
#   x_3 => le, mul, where
#   x_4 => add, rsqrt, var_mean
# Graph fragment:
#   %le : [num_users=2] = call_function[target=torch.ops.aten.le.Scalar](args = (%getitem, 0), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem, %uniform), kwargs = {})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le, %mul, %getitem), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_group_norm_rrelu_with_noise_functional_4(in_out_ptr0, in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2
    r0_numel = 5000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    tmp7_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp7_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp7_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        r0_1 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 5000*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr0 + (r0_1 + 5000*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tmp0 * tmp3
        tmp5 = tl.where(tmp2, tmp4, tmp0)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp7_mean_next, tmp7_m2_next, tmp7_weight_next = triton_helpers.welford_reduce(
            tmp6, tmp7_mean, tmp7_m2, tmp7_weight, roffset == 0
        )
        tmp7_mean = tl.where(r0_mask & xmask, tmp7_mean_next, tmp7_mean)
        tmp7_m2 = tl.where(r0_mask & xmask, tmp7_m2_next, tmp7_m2)
        tmp7_weight = tl.where(r0_mask & xmask, tmp7_weight_next, tmp7_weight)
        tl.store(out_ptr0 + (r0_1 + 5000*x0), tmp2, r0_mask & xmask)
        tl.store(in_out_ptr0 + (r0_1 + 5000*x0), tmp5, r0_mask & xmask)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7_mean, tmp7_m2, tmp7_weight, 1)
    tmp7 = tmp10[:, None]
    tmp8 = tmp11[:, None]
    tmp12[:, None]
    tl.store(out_ptr1 + (x0), tmp7, xmask)
    tl.store(out_ptr2 + (x0), tmp8, xmask)
    tmp13 = 5000.0
    tmp14 = tmp8 / tmp13
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = libdevice.rsqrt(tmp16)
    tl.store(out_ptr3 + (x0), tmp17, xmask)


# kernel path: /tmp/torchinductor_sahanp/ns/cns4fhodnrcitlrakq7wlcbgonf3i73cbzodjqgq2nnku7zknmg5.py
# Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.pixel_unshuffle]
# Source node to ATen node mapping:
#   x_6 => clone
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_pixel_unshuffle_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 40
    xnumel = 250
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex % 50)
    x4 = xindex // 50
    y0 = (yindex % 2)
    y1 = ((yindex // 2) % 2)
    y2 = yindex // 4
    x6 = xindex
    y7 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 2*x3 + 100*y1 + 200*x4 + 1000*y2), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y2 // 5), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y2 // 5), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y2), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y2), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 5000.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x6 + 250*y7), tmp13, xmask & ymask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (10, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(primals_2, (10, ), (1, ))
    assert_size_stride(primals_3, (1, 1, 16, 16, 16), (4096, 4096, 256, 16, 1))
    assert_size_stride(primals_4, (10, ), (1, ))
    assert_size_stride(primals_5, (10, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (1, 10, 14, 14, 14), (27440, 2744, 196, 14, 1))
        buf1 = empty_strided_cuda((1, 10, 16, 16, 16), (40960, 4096, 256, 16, 1), torch.float32)
        buf2 = empty_strided_cuda((1, 10, 16, 16, 16), (40960, 4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.copy]
        get_raw_stream(0)
        triton_poi_fused_copy_0[grid(40960)](buf0, primals_2, buf1, buf2, 40960, XBLOCK=256, num_warps=4, num_stages=1)
        del buf0
        del primals_2
        buf3 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        get_raw_stream(0)
        triton_poi_fused_1[grid(40960)](buf2, buf3, 40960, XBLOCK=256, num_warps=4, num_stages=1)
        buf4 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        get_raw_stream(0)
        triton_poi_fused_2[grid(40960)](buf3, buf4, 40960, XBLOCK=512, num_warps=4, num_stages=1)
        del buf3
        buf5 = empty_strided_cuda((1, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf5)
        buf6 = empty_strided_cuda((1, 10, 3), (30, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.rand]
        get_raw_stream(0)
        triton_poi_fused_rand_3[grid(30)](buf5, buf6, 0, 30, XBLOCK=32, num_warps=1, num_stages=1)
        del buf5
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.fractional_max_pool3d]
        buf7 = torch.ops.aten.fractional_max_pool3d.default(buf4, [2, 2, 2], [10, 10, 10], buf6)
        del buf6
        buf8 = buf7[0]
        buf9 = buf7[1]
        del buf7
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.rrelu_with_noise_functional]
        buf11 = torch.ops.aten.uniform.default(buf8, 0.125, 0.3333333333333333)
        buf12 = buf11
        del buf11
        buf10 = empty_strided_cuda((1, 10, 10, 10, 10), (10112, 1000, 100, 10, 1), torch.bool)
        buf13 = reinterpret_tensor(buf8, (1, 10, 10, 10, 10), (10016, 1000, 100, 10, 1), 0); del buf8  # reuse
        buf14 = empty_strided_cuda((1, 2, 1, 1), (2, 1, 2, 2), torch.float32)
        buf15 = empty_strided_cuda((1, 2, 1, 1), (2, 1, 2, 2), torch.float32)
        buf17 = empty_strided_cuda((1, 2, 1, 1), (2, 1, 2, 2), torch.float32)
        # Topologically Sorted Source Nodes: [x_3, x_4], Original ATen: [aten.rrelu_with_noise_functional, aten.native_group_norm]
        get_raw_stream(0)
        triton_red_fused_native_group_norm_rrelu_with_noise_functional_4[grid(2)](buf13, buf12, buf10, buf14, buf15, buf17, 2, 5000, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        buf18 = empty_strided_cuda((1, 10, 2, 2, 5, 50), (10000, 1000, 500, 250, 50, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.pixel_unshuffle]
        get_raw_stream(0)
        triton_poi_fused_pixel_unshuffle_5[grid(40, 250)](buf13, buf14, buf15, primals_4, primals_5, buf18, 40, 250, XBLOCK=256, YBLOCK=1, num_warps=4, num_stages=1)
        del buf15
        del primals_5
    return (reinterpret_tensor(buf18, (1, 40, 5, 50, 1), (10000, 250, 50, 1, 1), 0), primals_1, primals_3, primals_4, buf4, buf9, buf10, buf12, buf13, reinterpret_tensor(buf14, (1, 2), (2, 1), 0), reinterpret_tensor(buf17, (1, 2), (2, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((10, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((1, 1, 16, 16, 16), (4096, 4096, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([primals_1, primals_2, primals_3, primals_4, primals_5])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
