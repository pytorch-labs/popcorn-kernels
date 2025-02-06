
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


from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 1024)
    x2 = xindex // 65536
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x2 + 192*x1), None)
    tmp1 = tl.load(in_ptr1 + (x0 + 64*x2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr4, out_ptr5, out_ptr6, load_seed_offset, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1024
    R0_BLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_1 = r0_index
    x0 = xindex
    tmp5 = tl.load(in_ptr1 + (x0 + 1024*r0_1), xmask, other=0.0)
    tmp7 = tl.load(in_ptr2 + (r0_1 + 64*x0), xmask, other=0.0)
    tmp8 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr4 + (r0_1), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr5 + (r0_1), None, eviction_policy='evict_last')
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = r0_1 + 64*x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp6 = tmp4.to(tl.float32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp5 + tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tl.where(xmask, tmp14, 0)
    tmp17 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp20 / tmp22
    tmp24 = tmp14 - tmp23
    tmp25 = tmp24 * tmp24
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
    tmp28 = tl.where(xmask, tmp26, 0)
    tmp29 = tl.sum(tmp28, 1)[:, None]
    tmp30 = tmp13 - tmp23
    tmp31 = 64.0
    tmp32 = tmp29 / tmp31
    tmp33 = 1e-05
    tmp34 = tmp32 + tmp33
    tmp35 = libdevice.rsqrt(tmp34)
    tmp36 = tmp30 * tmp35
    tmp38 = tmp36 * tmp37
    tmp40 = tmp38 + tmp39
    tmp41 = 0.015625
    tmp42 = tmp35 * tmp41
    tl.store(out_ptr1 + (r0_1 + 64*x0), tmp4, xmask)
    tl.store(out_ptr4 + (r0_1 + 64*x0), tmp40, xmask)
    tl.store(out_ptr5 + (x0 + 1024*r0_1), tmp36, xmask)
    tl.store(out_ptr6 + (x0), tmp42, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_relu_threshold_backward_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 2048)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(out_ptr0 + (x2), tmp6, None)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_dropout_relu_3(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    x1 = (xindex % 2048)
    tmp6 = tl.load(in_out_ptr0 + (x0), None)
    tmp7 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full([1], 0, tl.int32)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp11 = tmp5 * tmp10
    tmp12 = 1.1111111111111112
    tmp13 = tmp11 * tmp12
    tl.store(out_ptr1 + (x0), tmp4, None)
    tl.store(in_out_ptr0 + (x0), tmp13, None)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr4, out_ptr5, load_seed_offset, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1024
    R0_BLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_1 = r0_index
    x0 = xindex
    tmp5 = tl.load(in_ptr1 + (r0_1 + 64*x0), xmask, other=0.0)
    tmp7 = tl.load(in_out_ptr0 + (r0_1 + 64*x0), xmask, other=0.0)
    tmp8 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr4 + (r0_1), None, eviction_policy='evict_last')
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = r0_1 + 64*x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp6 = tmp4.to(tl.float32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp5 + tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tl.where(xmask, tmp14, 0)
    tmp17 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp20 / tmp22
    tmp24 = tmp14 - tmp23
    tmp25 = tmp24 * tmp24
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
    tmp28 = tl.where(xmask, tmp26, 0)
    tmp29 = tl.sum(tmp28, 1)[:, None]
    tmp30 = tmp13 - tmp23
    tmp31 = 64.0
    tmp32 = tmp29 / tmp31
    tmp33 = 1e-05
    tmp34 = tmp32 + tmp33
    tmp35 = libdevice.rsqrt(tmp34)
    tmp36 = tmp30 * tmp35
    tmp38 = tmp36 * tmp37
    tmp40 = tmp38 + tmp39
    tmp41 = 0.015625
    tmp42 = tmp35 * tmp41
    tl.store(out_ptr1 + (r0_1 + 64*x0), tmp4, xmask)
    tl.store(in_out_ptr0 + (r0_1 + 64*x0), tmp36, xmask)
    tl.store(out_ptr4 + (r0_1 + 64*x0), tmp40, xmask)
    tl.store(out_ptr5 + (x0), tmp42, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_dropout_relu_threshold_backward_5(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    x1 = (xindex % 2048)
    tmp6 = tl.load(in_ptr1 + (x0), None)
    tmp7 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full([1], 0, tl.int32)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp11 = tmp5 * tmp10
    tmp12 = 1.1111111111111112
    tmp13 = tmp11 * tmp12
    tmp14 = 0.0
    tmp15 = tmp10 <= tmp14
    tl.store(out_ptr1 + (x0), tmp4, None)
    tl.store(out_ptr2 + (x0), tmp13, None)
    tl.store(out_ptr3 + (x0), tmp15, None)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr4, load_seed_offset, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1024
    R0_BLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_1 = r0_index
    x0 = xindex
    tmp5 = tl.load(in_ptr1 + (r0_1 + 64*x0), xmask, other=0.0)
    tmp7 = tl.load(in_out_ptr0 + (r0_1 + 64*x0), xmask, other=0.0)
    tmp8 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = r0_1 + 64*x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp6 = tmp4.to(tl.float32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp5 + tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tl.where(xmask, tmp14, 0)
    tmp17 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp20 / tmp22
    tmp24 = tmp14 - tmp23
    tmp25 = tmp24 * tmp24
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
    tmp28 = tl.where(xmask, tmp26, 0)
    tmp29 = tl.sum(tmp28, 1)[:, None]
    tmp30 = tmp13 - tmp23
    tmp31 = 64.0
    tmp32 = tmp29 / tmp31
    tmp33 = 1e-05
    tmp34 = tmp32 + tmp33
    tmp35 = libdevice.rsqrt(tmp34)
    tmp36 = tmp30 * tmp35
    tmp37 = 0.015625
    tmp38 = tmp35 * tmp37
    tl.store(out_ptr1 + (r0_1 + 64*x0), tmp4, xmask)
    tl.store(in_out_ptr0 + (r0_1 + 64*x0), tmp36, xmask)
    tl.store(out_ptr4 + (x0), tmp38, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_norm_sub_7(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 128
    R0_BLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_2 = r0_index
    x0 = (xindex % 2)
    x1 = xindex // 2
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (32*x0 + 64*((r0_2 % 16)) + 2048*((((r0_2 + 128*x1) // 16) % 16)) + ((r0_2 + 128*x1) // 256)), xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (32*x0 + ((r0_2 + 128*x1) // 256)), xmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (32*x0 + ((r0_2 + 128*x1) // 256)), xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr0 + (1024 + 32*x0 + 64*((r0_2 % 16)) + 2048*((((r0_2 + 128*x1) // 16) % 16)) + ((r0_2 + 128*x1) // 256)), xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr0 + (32768 + 32*x0 + 64*((r0_2 % 16)) + 2048*((((r0_2 + 128*x1) // 16) % 16)) + ((r0_2 + 128*x1) // 256)), xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp5 * tmp1
    tmp7 = tmp6 + tmp3
    tmp8 = tmp4 - tmp7
    tmp9 = 1e-06
    tmp10 = tmp8 + tmp9
    tmp11 = tmp10 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
    tmp14 = tl.where(xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp17 = tmp16 * tmp1
    tmp18 = tmp17 + tmp3
    tmp19 = tmp4 - tmp18
    tmp20 = tmp19 + tmp9
    tmp21 = tmp20 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, R0_BLOCK])
    tmp24 = tl.where(xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
    tl.store(out_ptr1 + (x3), tmp25, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_norm_sub_8(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 2
    R0_BLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 2*r0_1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_clamp_min_mean_norm_sub_9(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = libdevice.sqrt(tmp3)
    tmp10 = 1.0
    tmp11 = tmp8 + tmp10
    tmp12 = tmp11 - tmp9
    tmp13 = 0.0
    tmp14 = triton_helpers.maximum(tmp12, tmp13)
    tmp15 = tmp14 / tmp10
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp8, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp9, None)
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp15, None)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37 = args
    args.clear()
    assert_size_stride(primals_1, (1, 64, 32, 32), (65536, 1024, 32, 1))
    assert_size_stride(primals_2, (192, ), (1, ))
    assert_size_stride(primals_3, (192, 64), (64, 1))
    assert_size_stride(primals_4, (64, 64), (64, 1))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_7, (64, ), (1, ))
    assert_size_stride(primals_8, (2048, 64), (64, 1))
    assert_size_stride(primals_9, (2048, ), (1, ))
    assert_size_stride(primals_10, (64, 2048), (2048, 1))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_12, (64, ), (1, ))
    assert_size_stride(primals_13, (64, ), (1, ))
    assert_size_stride(primals_14, (192, ), (1, ))
    assert_size_stride(primals_15, (192, 64), (64, 1))
    assert_size_stride(primals_16, (64, 64), (64, 1))
    assert_size_stride(primals_17, (64, ), (1, ))
    assert_size_stride(primals_18, (64, ), (1, ))
    assert_size_stride(primals_19, (64, ), (1, ))
    assert_size_stride(primals_20, (2048, 64), (64, 1))
    assert_size_stride(primals_21, (2048, ), (1, ))
    assert_size_stride(primals_22, (64, 2048), (2048, 1))
    assert_size_stride(primals_23, (64, ), (1, ))
    assert_size_stride(primals_24, (64, ), (1, ))
    assert_size_stride(primals_25, (64, ), (1, ))
    assert_size_stride(primals_26, (192, ), (1, ))
    assert_size_stride(primals_27, (192, 64), (64, 1))
    assert_size_stride(primals_28, (64, 64), (64, 1))
    assert_size_stride(primals_29, (64, ), (1, ))
    assert_size_stride(primals_30, (64, ), (1, ))
    assert_size_stride(primals_31, (64, ), (1, ))
    assert_size_stride(primals_32, (2048, 64), (64, 1))
    assert_size_stride(primals_33, (2048, ), (1, ))
    assert_size_stride(primals_34, (64, 2048), (2048, 1))
    assert_size_stride(primals_35, (64, ), (1, ))
    assert_size_stride(primals_36, (64, ), (1, ))
    assert_size_stride(primals_37, (64, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf8 = empty_strided_cuda((9, ), (1, ), torch.int64)

        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [9], out=buf8)
        buf0 = empty_strided_cuda((1024, 192), (192, 1), torch.float32)

        extern_kernels.mm(reinterpret_tensor(primals_1, (1024, 64), (1, 1024), 0), reinterpret_tensor(primals_3, (64, 192), (1, 64), 0), out=buf0)
        del primals_3
        buf1 = empty_strided_cuda((3, 1024, 1, 64), (65536, 64, 64, 1), torch.float32)

        get_raw_stream(0)
        triton_poi_fused_clone_0[grid(196608)](buf0, primals_2, buf1, 196608, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_2

        buf2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf1, (1, 8, 1024, 8), (64, 8, 64, 1), 0), reinterpret_tensor(buf1, (1, 8, 1024, 8), (64, 8, 64, 1), 65536), reinterpret_tensor(buf1, (1, 8, 1024, 8), (64, 8, 64, 1), 131072), None, True, 0.1)
        buf3 = buf2[0]
        buf4 = buf2[1]
        buf5 = buf2[2]
        buf6 = buf2[3]
        del buf2
        buf7 = empty_strided_cuda((1024, 64), (64, 1), torch.float32)

        extern_kernels.mm(reinterpret_tensor(buf3, (1024, 64), (64, 1), 0), reinterpret_tensor(primals_4, (64, 64), (1, 64), 0), out=buf7)
        buf10 = empty_strided_cuda((1024, 1, 64), (64, 64, 1), torch.bool)
        buf14 = empty_strided_cuda((1024, 1, 64), (64, 64, 1), torch.float32)
        buf96 = empty_strided_cuda((1024, 1, 64), (1, 65536, 1024), torch.float32)
        buf97 = empty_strided_cuda((1024, 1, 1), (1, 1, 1), torch.float32)

        get_raw_stream(0)
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_1[grid(1024)](buf8, primals_1, buf7, primals_5, primals_6, primals_7, buf10, buf14, buf96, buf97, 0, 1024, 64, XBLOCK=8, num_warps=4, num_stages=1)
        del primals_5
        del primals_7
        buf15 = empty_strided_cuda((1024, 2048), (2048, 1), torch.float32)

        extern_kernels.mm(reinterpret_tensor(buf14, (1024, 64), (64, 1), 0), reinterpret_tensor(primals_8, (64, 2048), (1, 64), 0), out=buf15)
        buf95 = empty_strided_cuda((1024, 1, 2048), (2048, 2048, 1), torch.bool)

        get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_2[grid(2097152)](buf15, primals_9, buf95, 2097152, XBLOCK=1024, num_warps=4, num_stages=1)
        buf17 = empty_strided_cuda((1024, 1, 2048), (2048, 2048, 1), torch.bool)
        buf18 = reinterpret_tensor(buf15, (1024, 1, 2048), (2048, 2048, 1), 0); del buf15

        get_raw_stream(0)
        triton_poi_fused_native_dropout_relu_3[grid(2097152)](buf18, buf8, primals_9, buf17, 1, 2097152, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_9
        buf19 = buf7; del buf7

        extern_kernels.mm(reinterpret_tensor(buf18, (1024, 2048), (2048, 1), 0), reinterpret_tensor(primals_10, (2048, 64), (1, 2048), 0), out=buf19)
        buf21 = empty_strided_cuda((1024, 1, 64), (64, 64, 1), torch.bool)
        buf25 = reinterpret_tensor(buf19, (1024, 1, 64), (64, 64, 1), 0); del buf19
        buf26 = empty_strided_cuda((1024, 1, 64), (64, 64, 1), torch.float32)
        buf94 = empty_strided_cuda((1024, 1, 1), (1, 1, 1), torch.float32)

        get_raw_stream(0)
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_4[grid(1024)](buf25, buf8, buf14, primals_11, primals_12, primals_13, buf21, buf26, buf94, 6, 1024, 64, XBLOCK=1, num_warps=2, num_stages=1)
        del primals_11
        del primals_13
        buf27 = buf0; del buf0

        extern_kernels.mm(reinterpret_tensor(buf26, (1024, 64), (64, 1), 0), reinterpret_tensor(primals_15, (64, 192), (1, 64), 0), out=buf27)
        buf28 = empty_strided_cuda((3, 1024, 1, 64), (65536, 64, 64, 1), torch.float32)

        get_raw_stream(0)
        triton_poi_fused_clone_0[grid(196608)](buf27, primals_14, buf28, 196608, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_14

        buf29 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf28, (1, 8, 1024, 8), (64, 8, 64, 1), 0), reinterpret_tensor(buf28, (1, 8, 1024, 8), (64, 8, 64, 1), 65536), reinterpret_tensor(buf28, (1, 8, 1024, 8), (64, 8, 64, 1), 131072), None, True, 0.1)
        buf30 = buf29[0]
        buf31 = buf29[1]
        buf32 = buf29[2]
        buf33 = buf29[3]
        del buf29
        buf34 = empty_strided_cuda((1024, 64), (64, 1), torch.float32)

        extern_kernels.mm(reinterpret_tensor(buf30, (1024, 64), (64, 1), 0), reinterpret_tensor(primals_16, (64, 64), (1, 64), 0), out=buf34)
        buf36 = empty_strided_cuda((1024, 1, 64), (64, 64, 1), torch.bool)
        buf40 = reinterpret_tensor(buf34, (1024, 1, 64), (64, 64, 1), 0); del buf34
        buf41 = empty_strided_cuda((1024, 1, 64), (64, 64, 1), torch.float32)
        buf93 = empty_strided_cuda((1024, 1, 1), (1, 1, 1), torch.float32)

        get_raw_stream(0)
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_4[grid(1024)](buf40, buf8, buf26, primals_17, primals_18, primals_19, buf36, buf41, buf93, 6, 1024, 64, XBLOCK=1, num_warps=2, num_stages=1)
        del primals_17
        del primals_19
        buf42 = empty_strided_cuda((1024, 2048), (2048, 1), torch.float32)

        extern_kernels.mm(reinterpret_tensor(buf41, (1024, 64), (64, 1), 0), reinterpret_tensor(primals_20, (64, 2048), (1, 64), 0), out=buf42)
        buf44 = empty_strided_cuda((1024, 1, 2048), (2048, 2048, 1), torch.bool)
        buf45 = empty_strided_cuda((1024, 1, 2048), (2048, 2048, 1), torch.float32)
        buf92 = empty_strided_cuda((1024, 1, 2048), (2048, 2048, 1), torch.bool)

        get_raw_stream(0)
        triton_poi_fused_native_dropout_relu_threshold_backward_5[grid(2097152)](buf8, buf42, primals_21, buf44, buf45, buf92, 7, 2097152, XBLOCK=512, num_warps=8, num_stages=1)
        del primals_21
        buf46 = empty_strided_cuda((1024, 64), (64, 1), torch.float32)

        extern_kernels.mm(reinterpret_tensor(buf45, (1024, 2048), (2048, 1), 0), reinterpret_tensor(primals_22, (2048, 64), (1, 2048), 0), out=buf46)
        buf48 = empty_strided_cuda((1024, 1, 64), (64, 64, 1), torch.bool)
        buf52 = reinterpret_tensor(buf46, (1024, 1, 64), (64, 64, 1), 0); del buf46
        buf53 = empty_strided_cuda((1024, 1, 64), (64, 64, 1), torch.float32)
        buf91 = empty_strided_cuda((1024, 1, 1), (1, 1, 1), torch.float32)

        get_raw_stream(0)
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_4[grid(1024)](buf52, buf8, buf41, primals_23, primals_24, primals_25, buf48, buf53, buf91, 6, 1024, 64, XBLOCK=1, num_warps=2, num_stages=1)
        del primals_23
        del primals_25
        buf54 = buf27; del buf27

        extern_kernels.mm(reinterpret_tensor(buf53, (1024, 64), (64, 1), 0), reinterpret_tensor(primals_27, (64, 192), (1, 64), 0), out=buf54)
        buf55 = empty_strided_cuda((3, 1024, 1, 64), (65536, 64, 64, 1), torch.float32)

        get_raw_stream(0)
        triton_poi_fused_clone_0[grid(196608)](buf54, primals_26, buf55, 196608, XBLOCK=1024, num_warps=4, num_stages=1)
        del buf54
        del primals_26

        buf56 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf55, (1, 8, 1024, 8), (64, 8, 64, 1), 0), reinterpret_tensor(buf55, (1, 8, 1024, 8), (64, 8, 64, 1), 65536), reinterpret_tensor(buf55, (1, 8, 1024, 8), (64, 8, 64, 1), 131072), None, True, 0.1)
        buf57 = buf56[0]
        buf58 = buf56[1]
        buf59 = buf56[2]
        buf60 = buf56[3]
        del buf56
        buf61 = empty_strided_cuda((1024, 64), (64, 1), torch.float32)

        extern_kernels.mm(reinterpret_tensor(buf57, (1024, 64), (64, 1), 0), reinterpret_tensor(primals_28, (64, 64), (1, 64), 0), out=buf61)
        buf63 = empty_strided_cuda((1024, 1, 64), (64, 64, 1), torch.bool)
        buf67 = reinterpret_tensor(buf61, (1024, 1, 64), (64, 64, 1), 0); del buf61
        buf68 = empty_strided_cuda((1024, 1, 64), (64, 64, 1), torch.float32)
        buf90 = empty_strided_cuda((1024, 1, 1), (1, 1, 1), torch.float32)

        get_raw_stream(0)
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_4[grid(1024)](buf67, buf8, buf53, primals_29, primals_30, primals_31, buf63, buf68, buf90, 6, 1024, 64, XBLOCK=1, num_warps=2, num_stages=1)
        del primals_29
        del primals_31
        buf69 = buf42; del buf42

        extern_kernels.mm(reinterpret_tensor(buf68, (1024, 64), (64, 1), 0), reinterpret_tensor(primals_32, (64, 2048), (1, 64), 0), out=buf69)
        buf71 = empty_strided_cuda((1024, 1, 2048), (2048, 2048, 1), torch.bool)
        buf72 = empty_strided_cuda((1024, 1, 2048), (2048, 2048, 1), torch.float32)
        buf89 = empty_strided_cuda((1024, 1, 2048), (2048, 2048, 1), torch.bool)

        get_raw_stream(0)
        triton_poi_fused_native_dropout_relu_threshold_backward_5[grid(2097152)](buf8, buf69, primals_33, buf71, buf72, buf89, 7, 2097152, XBLOCK=512, num_warps=8, num_stages=1)
        del buf69
        del primals_33
        buf73 = empty_strided_cuda((1024, 64), (64, 1), torch.float32)

        extern_kernels.mm(reinterpret_tensor(buf72, (1024, 2048), (2048, 1), 0), reinterpret_tensor(primals_34, (2048, 64), (1, 2048), 0), out=buf73)
        buf75 = empty_strided_cuda((1024, 1, 64), (64, 64, 1), torch.bool)
        buf79 = reinterpret_tensor(buf73, (1024, 1, 64), (64, 64, 1), 0); del buf73
        buf88 = empty_strided_cuda((1024, 1, 1), (1, 1, 1), torch.float32)

        get_raw_stream(0)
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_6[grid(1024)](buf79, buf8, buf68, primals_35, buf75, buf88, 8, 1024, 64, XBLOCK=1, num_warps=2, num_stages=1)
        del buf8
        del primals_35
        buf80 = empty_strided_cuda((1, 2, 64), (128, 1, 2), torch.float32)
        buf84 = empty_strided_cuda((1, 2, 64), (128, 1, 2), torch.float32)

        get_raw_stream(0)
        triton_per_fused_add_norm_sub_7[grid(128)](buf79, primals_36, primals_37, buf80, buf84, 128, 128, XBLOCK=1, num_warps=2, num_stages=1)
        buf81 = empty_strided_cuda((1, 2), (2, 1), torch.float32)

        get_raw_stream(0)
        triton_per_fused_add_norm_sub_8[grid(2)](buf80, buf81, 2, 64, XBLOCK=1, num_warps=2, num_stages=1)
        del buf80
        buf85 = empty_strided_cuda((1, 2), (2, 1), torch.float32)

        get_raw_stream(0)
        triton_per_fused_add_norm_sub_8[grid(2)](buf84, buf85, 2, 64, XBLOCK=1, num_warps=2, num_stages=1)
        del buf84
        buf86 = empty_strided_cuda((1, ), (1, ), torch.float32)
        buf82 = empty_strided_cuda((1, ), (1, ), torch.float32)
        buf83 = buf82; del buf82
        buf87 = buf86; del buf86
        buf98 = empty_strided_cuda((), (), torch.float32)

        get_raw_stream(0)
        triton_per_fused_add_clamp_min_mean_norm_sub_9[grid(1)](buf83, buf87, buf85, buf81, buf98, 1, 2, XBLOCK=1, num_warps=2, num_stages=1)
        del buf81
        del buf85
    return (buf98, primals_6, primals_12, primals_18, primals_24, primals_30, primals_36, primals_37, reinterpret_tensor(primals_1, (1024, 64), (1, 1024), 0), reinterpret_tensor(buf1, (1, 8, 1024, 8), (64, 8, 64, 1), 0), reinterpret_tensor(buf1, (1, 8, 1024, 8), (64, 8, 64, 1), 65536), reinterpret_tensor(buf1, (1, 8, 1024, 8), (64, 8, 64, 1), 131072), buf3, buf4, buf5, buf6, buf10, reinterpret_tensor(buf14, (1024, 64), (64, 1), 0), buf17, reinterpret_tensor(buf18, (1024, 2048), (2048, 1), 0), buf21, buf25, reinterpret_tensor(buf26, (1024, 64), (64, 1), 0), reinterpret_tensor(buf28, (1, 8, 1024, 8), (64, 8, 64, 1), 0), reinterpret_tensor(buf28, (1, 8, 1024, 8), (64, 8, 64, 1), 65536), reinterpret_tensor(buf28, (1, 8, 1024, 8), (64, 8, 64, 1), 131072), buf30, buf31, buf32, buf33, buf36, buf40, reinterpret_tensor(buf41, (1024, 64), (64, 1), 0), buf44, reinterpret_tensor(buf45, (1024, 2048), (2048, 1), 0), buf48, buf52, reinterpret_tensor(buf53, (1024, 64), (64, 1), 0), reinterpret_tensor(buf55, (1, 8, 1024, 8), (64, 8, 64, 1), 0), reinterpret_tensor(buf55, (1, 8, 1024, 8), (64, 8, 64, 1), 65536), reinterpret_tensor(buf55, (1, 8, 1024, 8), (64, 8, 64, 1), 131072), buf57, buf58, buf59, buf60, buf63, buf67, reinterpret_tensor(buf68, (1024, 64), (64, 1), 0), buf71, reinterpret_tensor(buf72, (1024, 2048), (2048, 1), 0), buf75, buf79, buf83, buf87, buf88, primals_34, buf89, primals_32, buf90, primals_28, primals_27, buf91, primals_22, buf92, primals_20, buf93, primals_16, primals_15, buf94, primals_10, buf95, primals_8, buf96, buf97, primals_4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 64, 32, 32), (65536, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((192, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((2048, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((192, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((2048, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((64, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((192, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((2048, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((64, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
