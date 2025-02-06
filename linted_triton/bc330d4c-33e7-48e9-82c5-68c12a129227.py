
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


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_bernoulli_div_mul_1(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*((x1 % (ks3 // 2))) + 2*(ks3 // 2)*(((x0 // 2) % 2)) + 4*(ks3 // 2)*(((x1 // (ks3 // 2)) % (ks2 // 2))) + 4*(ks2 // 2)*(ks3 // 2)*(((x0 // 4) % ks1)) + ((x0 % 2))), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 20.0
    tmp2 = tmp0 > tmp1
    tmp3 = tl_math.exp(tmp0)
    tmp4 = libdevice.log1p(tmp3)
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp6 = libdevice.tanh(tmp5)
    tmp7 = tmp0 * tmp6
    tmp9 = 0.5
    tmp10 = tmp8 < tmp9
    tmp11 = tmp10.to(tl.float32)
    tmp12 = 2.0
    tmp13 = tmp11 * tmp12
    tmp14 = tmp7 * tmp13
    tl.store(out_ptr0 + (x2), tmp14, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__adaptive_avg_pool2d__to_copy__unsafe_index_add_arange_celu_clamp_mul_pow_sub_2(in_out_ptr1, in_ptr0, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 32) % 32)
    x0 = (xindex % 32)
    x2 = xindex // 1024
    x4 = xindex
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.4838709677419355
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int64)
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full([1], 15, tl.int64)
    tmp10 = triton_helpers.minimum(tmp8, tmp9)
    tmp11 = x0
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp12 * tmp2
    tmp14 = triton_helpers.maximum(tmp13, tmp4)
    tmp15 = tmp14.to(tl.int64)
    tmp16 = tmp15 + tmp7
    tmp17 = triton_helpers.minimum(tmp16, tmp9)
    tmp18 = tl.load(in_ptr0 + (x2 + 2*ks0*tmp17*(triton_helpers.div_floor_integer(ks1,  ks1 // 2))*(triton_helpers.div_floor_integer(ks2,  ks2 // 2)) + 2*ks0*tmp10*(triton_helpers.div_floor_integer(ks1,  ks1 // 2))*(triton_helpers.div_floor_integer(ks2,  ks2 // 2))*libdevice.trunc(libdevice.pow(((ks1 // 2)*(ks2 // 2)).to(tl.float64), tl.full([], 0.500000000000000, tl.float64))).to(tl.int32)), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (ks3 + x2 + 2*ks0*tmp17*(triton_helpers.div_floor_integer(ks1,  ks1 // 2))*(triton_helpers.div_floor_integer(ks2,  ks2 // 2)) + 2*ks0*tmp10*(triton_helpers.div_floor_integer(ks1,  ks1 // 2))*(triton_helpers.div_floor_integer(ks2,  ks2 // 2))*libdevice.trunc(libdevice.pow(((ks1 // 2)*(ks2 // 2)).to(tl.float64), tl.full([], 0.500000000000000, tl.float64))).to(tl.int32)), xmask, eviction_policy='evict_last')
    tmp20 = tmp19 + tmp18
    tmp21 = tl.load(in_ptr0 + (x2 + ks0*(triton_helpers.div_floor_integer(ks1,  ks1 // 2))*(triton_helpers.div_floor_integer(ks2,  ks2 // 2))*libdevice.trunc(libdevice.pow(((ks1 // 2)*(ks2 // 2)).to(tl.float64), tl.full([], 0.500000000000000, tl.float64))).to(tl.int32) + 2*ks0*tmp17*(triton_helpers.div_floor_integer(ks1,  ks1 // 2))*(triton_helpers.div_floor_integer(ks2,  ks2 // 2)) + 2*ks0*tmp10*(triton_helpers.div_floor_integer(ks1,  ks1 // 2))*(triton_helpers.div_floor_integer(ks2,  ks2 // 2))*libdevice.trunc(libdevice.pow(((ks1 // 2)*(ks2 // 2)).to(tl.float64), tl.full([], 0.500000000000000, tl.float64))).to(tl.int32)), xmask, eviction_policy='evict_last')
    tmp22 = tmp21 + tmp20
    tmp23 = tl.load(in_ptr0 + (ks3 + x2 + ks0*(triton_helpers.div_floor_integer(ks1,  ks1 // 2))*(triton_helpers.div_floor_integer(ks2,  ks2 // 2))*libdevice.trunc(libdevice.pow(((ks1 // 2)*(ks2 // 2)).to(tl.float64), tl.full([], 0.500000000000000, tl.float64))).to(tl.int32) + 2*ks0*tmp17*(triton_helpers.div_floor_integer(ks1,  ks1 // 2))*(triton_helpers.div_floor_integer(ks2,  ks2 // 2)) + 2*ks0*tmp10*(triton_helpers.div_floor_integer(ks1,  ks1 // 2))*(triton_helpers.div_floor_integer(ks2,  ks2 // 2))*libdevice.trunc(libdevice.pow(((ks1 // 2)*(ks2 // 2)).to(tl.float64), tl.full([], 0.500000000000000, tl.float64))).to(tl.int32)), xmask, eviction_policy='evict_last')
    tmp24 = tmp23 + tmp22
    tmp25 = 0.25
    tmp26 = tmp24 * tmp25
    tmp27 = tl.load(in_ptr0 + (x2 + 2*ks0*tmp15*(triton_helpers.div_floor_integer(ks1,  ks1 // 2))*(triton_helpers.div_floor_integer(ks2,  ks2 // 2)) + 2*ks0*tmp10*(triton_helpers.div_floor_integer(ks1,  ks1 // 2))*(triton_helpers.div_floor_integer(ks2,  ks2 // 2))*libdevice.trunc(libdevice.pow(((ks1 // 2)*(ks2 // 2)).to(tl.float64), tl.full([], 0.500000000000000, tl.float64))).to(tl.int32)), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr0 + (ks3 + x2 + 2*ks0*tmp15*(triton_helpers.div_floor_integer(ks1,  ks1 // 2))*(triton_helpers.div_floor_integer(ks2,  ks2 // 2)) + 2*ks0*tmp10*(triton_helpers.div_floor_integer(ks1,  ks1 // 2))*(triton_helpers.div_floor_integer(ks2,  ks2 // 2))*libdevice.trunc(libdevice.pow(((ks1 // 2)*(ks2 // 2)).to(tl.float64), tl.full([], 0.500000000000000, tl.float64))).to(tl.int32)), xmask, eviction_policy='evict_last')
    tmp29 = tmp28 + tmp27
    tmp30 = tl.load(in_ptr0 + (x2 + ks0*(triton_helpers.div_floor_integer(ks1,  ks1 // 2))*(triton_helpers.div_floor_integer(ks2,  ks2 // 2))*libdevice.trunc(libdevice.pow(((ks1 // 2)*(ks2 // 2)).to(tl.float64), tl.full([], 0.500000000000000, tl.float64))).to(tl.int32) + 2*ks0*tmp15*(triton_helpers.div_floor_integer(ks1,  ks1 // 2))*(triton_helpers.div_floor_integer(ks2,  ks2 // 2)) + 2*ks0*tmp10*(triton_helpers.div_floor_integer(ks1,  ks1 // 2))*(triton_helpers.div_floor_integer(ks2,  ks2 // 2))*libdevice.trunc(libdevice.pow(((ks1 // 2)*(ks2 // 2)).to(tl.float64), tl.full([], 0.500000000000000, tl.float64))).to(tl.int32)), xmask, eviction_policy='evict_last')
    tmp31 = tmp30 + tmp29
    tmp32 = tl.load(in_ptr0 + (ks3 + x2 + ks0*(triton_helpers.div_floor_integer(ks1,  ks1 // 2))*(triton_helpers.div_floor_integer(ks2,  ks2 // 2))*libdevice.trunc(libdevice.pow(((ks1 // 2)*(ks2 // 2)).to(tl.float64), tl.full([], 0.500000000000000, tl.float64))).to(tl.int32) + 2*ks0*tmp15*(triton_helpers.div_floor_integer(ks1,  ks1 // 2))*(triton_helpers.div_floor_integer(ks2,  ks2 // 2)) + 2*ks0*tmp10*(triton_helpers.div_floor_integer(ks1,  ks1 // 2))*(triton_helpers.div_floor_integer(ks2,  ks2 // 2))*libdevice.trunc(libdevice.pow(((ks1 // 2)*(ks2 // 2)).to(tl.float64), tl.full([], 0.500000000000000, tl.float64))).to(tl.int32)), xmask, eviction_policy='evict_last')
    tmp33 = tmp32 + tmp31
    tmp34 = tmp33 * tmp25
    tmp35 = tmp26 - tmp34
    tmp36 = tl.load(in_ptr0 + (x2 + 2*ks0*tmp17*(triton_helpers.div_floor_integer(ks1,  ks1 // 2))*(triton_helpers.div_floor_integer(ks2,  ks2 // 2)) + 2*ks0*tmp6*(triton_helpers.div_floor_integer(ks1,  ks1 // 2))*(triton_helpers.div_floor_integer(ks2,  ks2 // 2))*libdevice.trunc(libdevice.pow(((ks1 // 2)*(ks2 // 2)).to(tl.float64), tl.full([], 0.500000000000000, tl.float64))).to(tl.int32)), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr0 + (ks3 + x2 + 2*ks0*tmp17*(triton_helpers.div_floor_integer(ks1,  ks1 // 2))*(triton_helpers.div_floor_integer(ks2,  ks2 // 2)) + 2*ks0*tmp6*(triton_helpers.div_floor_integer(ks1,  ks1 // 2))*(triton_helpers.div_floor_integer(ks2,  ks2 // 2))*libdevice.trunc(libdevice.pow(((ks1 // 2)*(ks2 // 2)).to(tl.float64), tl.full([], 0.500000000000000, tl.float64))).to(tl.int32)), xmask, eviction_policy='evict_last')
    tmp38 = tmp37 + tmp36
    tmp39 = tl.load(in_ptr0 + (x2 + ks0*(triton_helpers.div_floor_integer(ks1,  ks1 // 2))*(triton_helpers.div_floor_integer(ks2,  ks2 // 2))*libdevice.trunc(libdevice.pow(((ks1 // 2)*(ks2 // 2)).to(tl.float64), tl.full([], 0.500000000000000, tl.float64))).to(tl.int32) + 2*ks0*tmp17*(triton_helpers.div_floor_integer(ks1,  ks1 // 2))*(triton_helpers.div_floor_integer(ks2,  ks2 // 2)) + 2*ks0*tmp6*(triton_helpers.div_floor_integer(ks1,  ks1 // 2))*(triton_helpers.div_floor_integer(ks2,  ks2 // 2))*libdevice.trunc(libdevice.pow(((ks1 // 2)*(ks2 // 2)).to(tl.float64), tl.full([], 0.500000000000000, tl.float64))).to(tl.int32)), xmask, eviction_policy='evict_last')
    tmp40 = tmp39 + tmp38
    tmp41 = tl.load(in_ptr0 + (ks3 + x2 + ks0*(triton_helpers.div_floor_integer(ks1,  ks1 // 2))*(triton_helpers.div_floor_integer(ks2,  ks2 // 2))*libdevice.trunc(libdevice.pow(((ks1 // 2)*(ks2 // 2)).to(tl.float64), tl.full([], 0.500000000000000, tl.float64))).to(tl.int32) + 2*ks0*tmp17*(triton_helpers.div_floor_integer(ks1,  ks1 // 2))*(triton_helpers.div_floor_integer(ks2,  ks2 // 2)) + 2*ks0*tmp6*(triton_helpers.div_floor_integer(ks1,  ks1 // 2))*(triton_helpers.div_floor_integer(ks2,  ks2 // 2))*libdevice.trunc(libdevice.pow(((ks1 // 2)*(ks2 // 2)).to(tl.float64), tl.full([], 0.500000000000000, tl.float64))).to(tl.int32)), xmask, eviction_policy='evict_last')
    tmp42 = tmp41 + tmp40
    tmp43 = tmp42 * tmp25
    tmp44 = tl.load(in_ptr0 + (x2 + 2*ks0*tmp15*(triton_helpers.div_floor_integer(ks1,  ks1 // 2))*(triton_helpers.div_floor_integer(ks2,  ks2 // 2)) + 2*ks0*tmp6*(triton_helpers.div_floor_integer(ks1,  ks1 // 2))*(triton_helpers.div_floor_integer(ks2,  ks2 // 2))*libdevice.trunc(libdevice.pow(((ks1 // 2)*(ks2 // 2)).to(tl.float64), tl.full([], 0.500000000000000, tl.float64))).to(tl.int32)), xmask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr0 + (ks3 + x2 + 2*ks0*tmp15*(triton_helpers.div_floor_integer(ks1,  ks1 // 2))*(triton_helpers.div_floor_integer(ks2,  ks2 // 2)) + 2*ks0*tmp6*(triton_helpers.div_floor_integer(ks1,  ks1 // 2))*(triton_helpers.div_floor_integer(ks2,  ks2 // 2))*libdevice.trunc(libdevice.pow(((ks1 // 2)*(ks2 // 2)).to(tl.float64), tl.full([], 0.500000000000000, tl.float64))).to(tl.int32)), xmask, eviction_policy='evict_last')
    tmp46 = tmp45 + tmp44
    tmp47 = tl.load(in_ptr0 + (x2 + ks0*(triton_helpers.div_floor_integer(ks1,  ks1 // 2))*(triton_helpers.div_floor_integer(ks2,  ks2 // 2))*libdevice.trunc(libdevice.pow(((ks1 // 2)*(ks2 // 2)).to(tl.float64), tl.full([], 0.500000000000000, tl.float64))).to(tl.int32) + 2*ks0*tmp15*(triton_helpers.div_floor_integer(ks1,  ks1 // 2))*(triton_helpers.div_floor_integer(ks2,  ks2 // 2)) + 2*ks0*tmp6*(triton_helpers.div_floor_integer(ks1,  ks1 // 2))*(triton_helpers.div_floor_integer(ks2,  ks2 // 2))*libdevice.trunc(libdevice.pow(((ks1 // 2)*(ks2 // 2)).to(tl.float64), tl.full([], 0.500000000000000, tl.float64))).to(tl.int32)), xmask, eviction_policy='evict_last')
    tmp48 = tmp47 + tmp46
    tmp49 = tl.load(in_ptr0 + (ks3 + x2 + ks0*(triton_helpers.div_floor_integer(ks1,  ks1 // 2))*(triton_helpers.div_floor_integer(ks2,  ks2 // 2))*libdevice.trunc(libdevice.pow(((ks1 // 2)*(ks2 // 2)).to(tl.float64), tl.full([], 0.500000000000000, tl.float64))).to(tl.int32) + 2*ks0*tmp15*(triton_helpers.div_floor_integer(ks1,  ks1 // 2))*(triton_helpers.div_floor_integer(ks2,  ks2 // 2)) + 2*ks0*tmp6*(triton_helpers.div_floor_integer(ks1,  ks1 // 2))*(triton_helpers.div_floor_integer(ks2,  ks2 // 2))*libdevice.trunc(libdevice.pow(((ks1 // 2)*(ks2 // 2)).to(tl.float64), tl.full([], 0.500000000000000, tl.float64))).to(tl.int32)), xmask, eviction_policy='evict_last')
    tmp50 = tmp49 + tmp48
    tmp51 = tmp50 * tmp25
    tmp52 = tmp43 - tmp51
    tmp53 = tmp14.to(tl.int32)
    tmp54 = tmp53.to(tl.float32)
    tmp55 = tmp14 - tmp54
    tmp56 = triton_helpers.maximum(tmp55, tmp4)
    tmp57 = 1.0
    tmp58 = triton_helpers.minimum(tmp56, tmp57)
    tmp59 = tmp35 * tmp58
    tmp60 = tmp34 + tmp59
    tmp61 = tmp52 * tmp58
    tmp62 = tmp51 + tmp61
    tmp63 = tmp60 - tmp62
    tmp64 = tmp5.to(tl.int32)
    tmp65 = tmp64.to(tl.float32)
    tmp66 = tmp5 - tmp65
    tmp67 = triton_helpers.maximum(tmp66, tmp4)
    tmp68 = triton_helpers.minimum(tmp67, tmp57)
    tmp69 = tmp63 * tmp68
    tmp70 = tmp62 + tmp69
    tmp71 = tmp70 > tmp4
    tmp72 = libdevice.expm1(tmp70)
    tmp73 = tl.where(tmp71, tmp70, tmp72)
    tmp74 = tmp73 * tmp73
    tl.store(in_out_ptr1 + (x4), tmp74, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_abs_add_avg_pool2d_celu_clamp_mul_pow_relu_sign_sub_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 64*x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 64*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (32 + 2*x0 + 64*x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (33 + 2*x0 + 64*x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tmp9 = tl.full([1], 0, tl.int32)
    tmp10 = tmp9 < tmp8
    tmp11 = tmp10.to(tl.int8)
    tmp12 = tmp8 < tmp9
    tmp13 = tmp12.to(tl.int8)
    tmp14 = tmp11 - tmp13
    tmp15 = tmp14.to(tmp8.dtype)
    tmp16 = tl_math.abs(tmp8)
    tmp17 = triton_helpers.maximum(tmp9, tmp16)
    tmp18 = tmp15 * tmp17
    tmp19 = 4.0
    tmp20 = tmp18 * tmp19
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = 0.0
    tmp23 = tmp21 > tmp22
    tmp24 = libdevice.expm1(tmp21)
    tmp25 = tl.where(tmp23, tmp21, tmp24)
    tmp26 = tmp25 * tmp25
    tl.store(out_ptr0 + (x2), tmp26, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_abs_add_avg_pool2d_celu_clamp_mul_pow_relu_sign_sub_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 8)
    x1 = xindex // 8
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 32*x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 32*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (16 + 2*x0 + 32*x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (17 + 2*x0 + 32*x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tmp9 = tl.full([1], 0, tl.int32)
    tmp10 = tmp9 < tmp8
    tmp11 = tmp10.to(tl.int8)
    tmp12 = tmp8 < tmp9
    tmp13 = tmp12.to(tl.int8)
    tmp14 = tmp11 - tmp13
    tmp15 = tmp14.to(tmp8.dtype)
    tmp16 = tl_math.abs(tmp8)
    tmp17 = triton_helpers.maximum(tmp9, tmp16)
    tmp18 = tmp15 * tmp17
    tmp19 = 4.0
    tmp20 = tmp18 * tmp19
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = 0.0
    tmp23 = tmp21 > tmp22
    tmp24 = libdevice.expm1(tmp21)
    tmp25 = tl.where(tmp23, tmp21, tmp24)
    tmp26 = tmp25 * tmp25
    tl.store(out_ptr0 + (x2), tmp26, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_abs_add_avg_pool2d_celu_clamp_mul_pow_relu_sign_sub_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = xindex // 4
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 16*x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 16*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (8 + 2*x0 + 16*x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (9 + 2*x0 + 16*x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tmp9 = tl.full([1], 0, tl.int32)
    tmp10 = tmp9 < tmp8
    tmp11 = tmp10.to(tl.int8)
    tmp12 = tmp8 < tmp9
    tmp13 = tmp12.to(tl.int8)
    tmp14 = tmp11 - tmp13
    tmp15 = tmp14.to(tmp8.dtype)
    tmp16 = tl_math.abs(tmp8)
    tmp17 = triton_helpers.maximum(tmp9, tmp16)
    tmp18 = tmp15 * tmp17
    tmp19 = 4.0
    tmp20 = tmp18 * tmp19
    tmp21 = libdevice.sqrt(tmp20)
    tl.store(out_ptr0 + (x2), tmp21, xmask)


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

        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf0)
        buf1 = empty_strided_cuda((1, s0*(s1 // (s1 // 2))*(s2 // (s2 // 2)), 1), (s0*(s1 // (s1 // 2))*(s2 // (s2 // 2)), 1, s0*(s1 // (s1 // 2))*(s2 // (s2 // 2))), torch.float32)

        triton_poi_fused_bernoulli_0_xnumel = s0*(s1 // (s1 // 2))*(s2 // (s2 // 2))
        get_raw_stream(0)
        triton_poi_fused_bernoulli_0[grid(triton_poi_fused_bernoulli_0_xnumel)](buf0, buf1, 0, 12, XBLOCK=16, num_warps=1, num_stages=1)
        del buf0
        s0*(s1 // (s1 // 2))*(s2 // (s2 // 2))
        buf2 = empty_strided_cuda((1, s0*(s1 // (s1 // 2))*(s2 // (s2 // 2)), (4*(s1 // 2)*(s2 // 2)) // ((s1 // (s1 // 2))*(s2 // (s2 // 2)))), (s0*(s1 // (s1 // 2))*(s2 // (s2 // 2))*((4*(s1 // 2)*(s2 // 2)) // ((s1 // (s1 // 2))*(s2 // (s2 // 2)))), 1, s0*(s1 // (s1 // 2))*(s2 // (s2 // 2))), torch.float32)

        triton_poi_fused__to_copy_bernoulli_div_mul_1_xnumel = s0*(s1 // (s1 // 2))*(s2 // (s2 // 2))*((4*(s1 // 2)*(s2 // 2)) // ((s1 // (s1 // 2))*(s2 // (s2 // 2))))
        get_raw_stream(0)
        triton_poi_fused__to_copy_bernoulli_div_mul_1[grid(triton_poi_fused__to_copy_bernoulli_div_mul_1_xnumel)](arg3_1, buf1, buf2, 12, 3, 64, 64, 12288, XBLOCK=128, num_warps=4, num_stages=1)
        del arg3_1
        del buf1
        buf5 = empty_strided_cuda((1, s0*(s1 // (s1 // 2))*(s2 // (s2 // 2)), 32, 32), (1024*s0*(s1 // (s1 // 2))*(s2 // (s2 // 2)), 1024, 32, 1), torch.float32)
        buf6 = buf5; del buf5
        buf7 = buf6; del buf6

        triton_poi_fused__adaptive_avg_pool2d__to_copy__unsafe_index_add_arange_celu_clamp_mul_pow_sub_2_xnumel = 1024*s0*(s1 // (s1 // 2))*(s2 // (s2 // 2))
        get_raw_stream(0)
        triton_poi_fused__adaptive_avg_pool2d__to_copy__unsafe_index_add_arange_celu_clamp_mul_pow_sub_2[grid(triton_poi_fused__adaptive_avg_pool2d__to_copy__unsafe_index_add_arange_celu_clamp_mul_pow_sub_2_xnumel)](buf7, buf2, 3, 64, 64, 12, 12288, XBLOCK=128, num_warps=4, num_stages=1)
        del buf2
        buf8 = empty_strided_cuda((1, s0*(s1 // (s1 // 2))*(s2 // (s2 // 2)), 16, 16), (256*s0*(s1 // (s1 // 2))*(s2 // (s2 // 2)), 256, 16, 1), torch.float32)

        triton_poi_fused__to_copy_abs_add_avg_pool2d_celu_clamp_mul_pow_relu_sign_sub_3_xnumel = 256*s0*(s1 // (s1 // 2))*(s2 // (s2 // 2))
        get_raw_stream(0)
        triton_poi_fused__to_copy_abs_add_avg_pool2d_celu_clamp_mul_pow_relu_sign_sub_3[grid(triton_poi_fused__to_copy_abs_add_avg_pool2d_celu_clamp_mul_pow_relu_sign_sub_3_xnumel)](buf7, buf8, 3072, XBLOCK=256, num_warps=4, num_stages=1)
        del buf7
        buf9 = empty_strided_cuda((1, s0*(s1 // (s1 // 2))*(s2 // (s2 // 2)), 8, 8), (64*s0*(s1 // (s1 // 2))*(s2 // (s2 // 2)), 64, 8, 1), torch.float32)

        triton_poi_fused__to_copy_abs_add_avg_pool2d_celu_clamp_mul_pow_relu_sign_sub_4_xnumel = 64*s0*(s1 // (s1 // 2))*(s2 // (s2 // 2))
        get_raw_stream(0)
        triton_poi_fused__to_copy_abs_add_avg_pool2d_celu_clamp_mul_pow_relu_sign_sub_4[grid(triton_poi_fused__to_copy_abs_add_avg_pool2d_celu_clamp_mul_pow_relu_sign_sub_4_xnumel)](buf8, buf9, 768, XBLOCK=128, num_warps=4, num_stages=1)
        del buf8
        buf10 = empty_strided_cuda((1, s0*(s1 // (s1 // 2))*(s2 // (s2 // 2)), 4, 4), (16*s0*(s1 // (s1 // 2))*(s2 // (s2 // 2)), 16, 4, 1), torch.float32)

        triton_poi_fused__to_copy_abs_add_avg_pool2d_celu_clamp_mul_pow_relu_sign_sub_5_xnumel = 16*s0*(s1 // (s1 // 2))*(s2 // (s2 // 2))
        get_raw_stream(0)
        triton_poi_fused__to_copy_abs_add_avg_pool2d_celu_clamp_mul_pow_relu_sign_sub_5[grid(triton_poi_fused__to_copy_abs_add_avg_pool2d_celu_clamp_mul_pow_relu_sign_sub_5_xnumel)](buf9, buf10, 192, XBLOCK=256, num_warps=4, num_stages=1)
        del buf9
    return (buf10, )


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
