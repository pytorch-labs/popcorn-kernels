# AOT ID: ['110_inference']
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


# kernel path: /tmp/torchinductor_sahanp/g4/cg4kpa66apnziubrpi4ip6otheq5fvzltcbzpmagnjt5d67pyz33.py
# Topologically Sorted Source Nodes: [x, input_1, input_2, input_3, input_4], Original ATen: [aten.constant_pad_nd, aten.abs, aten.gt, aten.sign, aten.mul, aten.sub, aten.where, aten.sigmoid]
# Source node to ATen node mapping:
#   input_1 => abs_1, gt, mul_19, mul_30, sign, sub_20, where
#   input_2 => sigmoid
#   input_3 => abs_2, gt_1, mul_60, mul_71, sign_1, sub_53, where_1
#   input_4 => sigmoid_1
#   x => constant_pad_nd
# Graph fragment:
#   %constant_pad_nd : [num_users=4] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%arg4_1, [1, 1, 1, 1, 1, 1], 0.0), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%constant_pad_nd,), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%abs_1, 0.5), kwargs = {})
#   %sign : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%constant_pad_nd,), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign, 0.5), kwargs = {})
#   %sub_20 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%constant_pad_nd, %mul_19), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%constant_pad_nd, 0), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %sub_20, %mul_30), kwargs = {})
#   %sigmoid : [num_users=4] = call_function[target=torch.ops.aten.sigmoid.default](args = (%where,), kwargs = {})
#   %abs_2 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sigmoid,), kwargs = {})
#   %gt_1 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%abs_2, 0.5), kwargs = {})
#   %sign_1 : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%sigmoid,), kwargs = {})
#   %mul_60 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign_1, 0.5), kwargs = {})
#   %sub_53 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sigmoid, %mul_60), kwargs = {})
#   %mul_71 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid, 0), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %sub_53, %mul_71), kwargs = {})
#   %sigmoid_1 : [num_users=4] = call_function[target=torch.ops.aten.sigmoid.default](args = (%where_1,), kwargs = {})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_abs_constant_pad_nd_gt_mul_sigmoid_sign_sub_where_0(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, ks6, ks7, ks8, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x6 = ((xindex // ks0) % ks1)
    x1 = ((xindex // ks3) % ks4)
    x0 = (xindex % ks3)
    x2 = ((xindex // ks7) % ks1)
    x3 = xindex // ks8
    x10 = xindex
    tmp0 = (-1) + x6
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = ks2
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = ks5
    tmp8 = tmp5 < tmp7
    tmp9 = (-1) + x0
    tmp10 = tmp9 >= tmp1
    tmp11 = ks6
    tmp12 = tmp9 < tmp11
    tmp13 = tmp2 & tmp4
    tmp14 = tmp13 & tmp6
    tmp15 = tmp14 & tmp8
    tmp16 = tmp15 & tmp10
    tmp17 = tmp16 & tmp12
    tmp18 = tl.load(in_ptr0 + ((-1) + x0 + ((-1)*ks6) + ks6*x1 + ((-1)*ks5*ks6) + ks5*ks6*x2 + ks2*ks5*ks6*x3), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tl_math.abs(tmp18)
    tmp20 = 0.5
    tmp21 = tmp19 > tmp20
    tmp22 = tl.full([1], 0, tl.int32)
    tmp23 = tmp22 < tmp18
    tmp24 = tmp23.to(tl.int8)
    tmp25 = tmp18 < tmp22
    tmp26 = tmp25.to(tl.int8)
    tmp27 = tmp24 - tmp26
    tmp28 = tmp27.to(tmp18.dtype)
    tmp29 = tmp28 * tmp20
    tmp30 = tmp18 - tmp29
    tmp31 = 0.0
    tmp32 = tmp18 * tmp31
    tmp33 = tl.where(tmp21, tmp30, tmp32)
    tmp34 = tl.sigmoid(tmp33)
    tmp35 = tl_math.abs(tmp34)
    tmp36 = tmp35 > tmp20
    tmp37 = tmp22 < tmp34
    tmp38 = tmp37.to(tl.int8)
    tmp39 = tmp34 < tmp22
    tmp40 = tmp39.to(tl.int8)
    tmp41 = tmp38 - tmp40
    tmp42 = tmp41.to(tmp34.dtype)
    tmp43 = tmp42 * tmp20
    tmp44 = tmp34 - tmp43
    tmp45 = tmp34 * tmp31
    tmp46 = tl.where(tmp36, tmp44, tmp45)
    tmp47 = tl.sigmoid(tmp46)
    tl.store(out_ptr0 + (x10), tmp47, xmask)


# kernel path: /tmp/torchinductor_sahanp/fq/cfqtnevefpkd3ygprvns4cykil4jlipuin5bp7svtbogy6ukq5ug.py
# Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.abs, aten.gt, aten.sign, aten.mul, aten.sub, aten.where]
# Source node to ATen node mapping:
#   input_5 => abs_3, gt_2, mul_101, mul_112, sign_2, sub_86, where_2
# Graph fragment:
#   %abs_3 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sigmoid_1,), kwargs = {})
#   %gt_2 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%abs_3, 0.5), kwargs = {})
#   %sign_2 : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%sigmoid_1,), kwargs = {})
#   %mul_101 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign_2, 0.5), kwargs = {})
#   %sub_86 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sigmoid_1, %mul_101), kwargs = {})
#   %mul_112 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_1, 0), kwargs = {})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %sub_86, %mul_112), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_abs_gt_mul_sign_sub_where_1(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl_math.abs(tmp0)
    tmp2 = 0.5
    tmp3 = tmp1 > tmp2
    tmp4 = tl.full([1], 0, tl.int32)
    tmp5 = tmp4 < tmp0
    tmp6 = tmp5.to(tl.int8)
    tmp7 = tmp0 < tmp4
    tmp8 = tmp7.to(tl.int8)
    tmp9 = tmp6 - tmp8
    tmp10 = tmp9.to(tmp0.dtype)
    tmp11 = tmp10 * tmp2
    tmp12 = tmp0 - tmp11
    tmp13 = 0.0
    tmp14 = tmp0 * tmp13
    tmp15 = tl.where(tmp3, tmp12, tmp14)
    tl.store(in_out_ptr0 + (x0), tmp15, xmask)


# kernel path: /tmp/torchinductor_sahanp/5b/c5b3pjf3ff5e5psjqadeeguqxt3oio2aj5s7zotzlt34ufde3yhm.py
# Topologically Sorted Source Nodes: [input_5, x_1], Original ATen: [aten.abs, aten.gt, aten.sign, aten.mul, aten.sub, aten.where, aten.view]
# Source node to ATen node mapping:
#   input_5 => abs_3, gt_2, mul_101, mul_112, sign_2, sub_86, where_2
#   x_1 => view
# Graph fragment:
#   %abs_3 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sigmoid_1,), kwargs = {})
#   %gt_2 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%abs_3, 0.5), kwargs = {})
#   %sign_2 : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%sigmoid_1,), kwargs = {})
#   %mul_101 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign_2, 0.5), kwargs = {})
#   %sub_86 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sigmoid_1, %mul_101), kwargs = {})
#   %mul_112 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_1, 0), kwargs = {})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %sub_86, %mul_112), kwargs = {})
#   %view : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%where_2, [1, -1]), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_abs_gt_mul_sign_sub_view_where_2(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, ks6, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2*(((x0 // ks2) % ks3)) + 4*(((x0 // ks0) % ks1)) + 8*(x0 // (8 + 4*ks4 + 4*ks5 + 4*ks6 + 2*ks4*ks5 + 2*ks4*ks6 + 2*ks5*ks6 + ks4*ks5*ks6)) + ks6*(((x0 // ks2) % ks3)) + 2*ks5*(((x0 // ks0) % ks1)) + 2*ks6*(((x0 // ks0) % ks1)) + 4*ks4*(x0 // (8 + 4*ks4 + 4*ks5 + 4*ks6 + 2*ks4*ks5 + 2*ks4*ks6 + 2*ks5*ks6 + ks4*ks5*ks6)) + 4*ks5*(x0 // (8 + 4*ks4 + 4*ks5 + 4*ks6 + 2*ks4*ks5 + 2*ks4*ks6 + 2*ks5*ks6 + ks4*ks5*ks6)) + 4*ks6*(x0 // (8 + 4*ks4 + 4*ks5 + 4*ks6 + 2*ks4*ks5 + 2*ks4*ks6 + 2*ks5*ks6 + ks4*ks5*ks6)) + ks5*ks6*(((x0 // ks0) % ks1)) + 2*ks4*ks5*(x0 // (8 + 4*ks4 + 4*ks5 + 4*ks6 + 2*ks4*ks5 + 2*ks4*ks6 + 2*ks5*ks6 + ks4*ks5*ks6)) + 2*ks4*ks6*(x0 // (8 + 4*ks4 + 4*ks5 + 4*ks6 + 2*ks4*ks5 + 2*ks4*ks6 + 2*ks5*ks6 + ks4*ks5*ks6)) + 2*ks5*ks6*(x0 // (8 + 4*ks4 + 4*ks5 + 4*ks6 + 2*ks4*ks5 + 2*ks4*ks6 + 2*ks5*ks6 + ks4*ks5*ks6)) + ks4*ks5*ks6*(x0 // (8 + 4*ks4 + 4*ks5 + 4*ks6 + 2*ks4*ks5 + 2*ks4*ks6 + 2*ks5*ks6 + ks4*ks5*ks6)) + ((x0 % ks2))), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)


# kernel path: /tmp/torchinductor_sahanp/7r/c7r5mlvbb65x6gj5k7sq4pqueel3bp6mzhwq7cluhgzbsranxj4d.py
# Topologically Sorted Source Nodes: [target], Original ATen: [aten.zeros_like]
# Source node to ATen node mapping:
#   target => full
# Graph fragment:
#   %full : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, %sym_size_int_4], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_zeros_like_3(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)


# kernel path: /tmp/torchinductor_sahanp/g7/cg77gizyayt7cs2zpx2gdmw5g4pan4ujwunvaxrn4qwetxo642vj.py
# Topologically Sorted Source Nodes: [var], Original ATen: [aten.ones_like]
# Source node to ATen node mapping:
#   var => full_1
# Graph fragment:
#   %full_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, %sym_size_int_4], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_ones_like_4(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 1.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)


def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    s3 = arg3_1
    assert_size_stride(arg4_1, (1, s0, s1, s2, s3), (s0*s1*s2*s3, s1*s2*s3, s2*s3, s3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        4 + 2*s2 + 2*s3 + s2*s3
        2 + s1
        2 + s3
        2 + s2
        4 + 2*s2 + 2*s3 + s2*s3
        8 + 4*s1 + 4*s2 + 4*s3 + 2*s1*s2 + 2*s1*s3 + 2*s2*s3 + s1*s2*s3
        buf0 = empty_strided_cuda((1, s0, 2 + s1, 2 + s2, 2 + s3), (8*s0 + 4*s0*s1 + 4*s0*s2 + 4*s0*s3 + 2*s0*s1*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + s0*s1*s2*s3, 8 + 4*s1 + 4*s2 + 4*s3 + 2*s1*s2 + 2*s1*s3 + 2*s2*s3 + s1*s2*s3, 4 + 2*s2 + 2*s3 + s2*s3, 2 + s3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, input_1, input_2, input_3, input_4], Original ATen: [aten.constant_pad_nd, aten.abs, aten.gt, aten.sign, aten.mul, aten.sub, aten.where, aten.sigmoid]
        triton_poi_fused_abs_constant_pad_nd_gt_mul_sigmoid_sign_sub_where_0_xnumel = 8*s0 + 4*s0*s1 + 4*s0*s2 + 4*s0*s3 + 2*s0*s1*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + s0*s1*s2*s3
        get_raw_stream(0)
        triton_poi_fused_abs_constant_pad_nd_gt_mul_sigmoid_sign_sub_where_0[grid(triton_poi_fused_abs_constant_pad_nd_gt_mul_sigmoid_sign_sub_where_0_xnumel)](arg4_1, buf0, 4356, 66, 64, 66, 66, 64, 64, 4356, 287496, 862488, XBLOCK=1024, num_warps=4, num_stages=1)
        del arg4_1
        buf1 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.abs, aten.gt, aten.sign, aten.mul, aten.sub, aten.where]
        triton_poi_fused_abs_gt_mul_sign_sub_where_1_xnumel = 8*s0 + 4*s0*s1 + 4*s0*s2 + 4*s0*s3 + 2*s0*s1*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + s0*s1*s2*s3
        get_raw_stream(0)
        triton_poi_fused_abs_gt_mul_sign_sub_where_1[grid(triton_poi_fused_abs_gt_mul_sign_sub_where_1_xnumel)](buf1, 862488, XBLOCK=512, num_warps=8, num_stages=1)
        buf2 = empty_strided_cuda((1, 8*s0 + 4*s0*s1 + 4*s0*s2 + 4*s0*s3 + 2*s0*s1*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + s0*s1*s2*s3), (8*s0 + 4*s0*s1 + 4*s0*s2 + 4*s0*s3 + 2*s0*s1*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + s0*s1*s2*s3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_5, x_1], Original ATen: [aten.abs, aten.gt, aten.sign, aten.mul, aten.sub, aten.where, aten.view]
        triton_poi_fused_abs_gt_mul_sign_sub_view_where_2_xnumel = 8*s0 + 4*s0*s1 + 4*s0*s2 + 4*s0*s3 + 2*s0*s1*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + s0*s1*s2*s3
        get_raw_stream(0)
        triton_poi_fused_abs_gt_mul_sign_sub_view_where_2[grid(triton_poi_fused_abs_gt_mul_sign_sub_view_where_2_xnumel)](buf1, buf2, 4356, 66, 66, 66, 64, 64, 64, 862488, XBLOCK=512, num_warps=8, num_stages=1)
        buf3 = reinterpret_tensor(buf1, (1, 8*s0 + 4*s0*s1 + 4*s0*s2 + 4*s0*s3 + 2*s0*s1*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + s0*s1*s2*s3), (8*s0 + 4*s0*s1 + 4*s0*s2 + 4*s0*s3 + 2*s0*s1*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + s0*s1*s2*s3, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [target], Original ATen: [aten.zeros_like]
        triton_poi_fused_zeros_like_3_xnumel = 8*s0 + 4*s0*s1 + 4*s0*s2 + 4*s0*s3 + 2*s0*s1*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + s0*s1*s2*s3
        get_raw_stream(0)
        triton_poi_fused_zeros_like_3[grid(triton_poi_fused_zeros_like_3_xnumel)](buf3, 862488, XBLOCK=512, num_warps=8, num_stages=1)
        buf4 = empty_strided_cuda((1, 8*s0 + 4*s0*s1 + 4*s0*s2 + 4*s0*s3 + 2*s0*s1*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + s0*s1*s2*s3), (8*s0 + 4*s0*s1 + 4*s0*s2 + 4*s0*s3 + 2*s0*s1*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + s0*s1*s2*s3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [var], Original ATen: [aten.ones_like]
        triton_poi_fused_ones_like_4_xnumel = 8*s0 + 4*s0*s1 + 4*s0*s2 + 4*s0*s3 + 2*s0*s1*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + s0*s1*s2*s3
        get_raw_stream(0)
        triton_poi_fused_ones_like_4[grid(triton_poi_fused_ones_like_4_xnumel)](buf4, 862488, XBLOCK=512, num_warps=8, num_stages=1)
    return (buf2, buf3, buf4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 3
    arg1_1 = 64
    arg2_1 = 64
    arg3_1 = 64
    arg4_1 = rand_strided((1, 3, 64, 64, 64), (786432, 262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
