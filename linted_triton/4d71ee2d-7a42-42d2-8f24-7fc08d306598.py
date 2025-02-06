
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
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_mean_mul_pow_rsqrt_2(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel):
    XBLOCK: tl.constexpr = 1
    R0_BLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    tl.full([1], xoffset, tl.int32)
    tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    tl.full([R0_BLOCK], True, tl.int1)
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), None)
    tmp11 = tl.load(in_ptr1 + (r0_0), None)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [R0_BLOCK])
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp2, 0))
    tmp5 = 512.0
    tmp6 = tmp4 / tmp5
    tmp7 = 1.1920928955078125e-07
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp0 * tmp9
    tmp12 = tmp10 * tmp11
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([1], 0, tl.int32)), tmp9, None)
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [R0_BLOCK])), tmp12, None)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__log_softmax_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tl.load(in_ptr0 + (1))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp6 = tl.load(in_ptr0 + (2))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp9 = tl.load(in_ptr0 + (3))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK])
    tmp5 = triton_helpers.maximum(tmp2, tmp4)
    tmp8 = triton_helpers.maximum(tmp5, tmp7)
    tmp11 = triton_helpers.maximum(tmp8, tmp10)
    tmp12 = tmp0 - tmp11
    tl.store(out_ptr0 + (x0), tmp12, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__log_softmax_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_ptr0 + (1))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp8 = tl.load(in_ptr0 + (2))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp12 = tl.load(in_ptr0 + (3))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK])
    tmp3 = tl_math.exp(tmp2)
    tmp6 = tl_math.exp(tmp5)
    tmp7 = tmp3 + tmp6
    tmp10 = tl_math.exp(tmp9)
    tmp11 = tmp7 + tmp10
    tmp14 = tl_math.exp(tmp13)
    tmp15 = tmp11 + tmp14
    tmp16 = tl_math.log(tmp15)
    tmp17 = tmp0 - tmp16
    tl.store(out_ptr0 + (x0), tmp17, xmask)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__log_softmax_5(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.load(in_ptr0 + (1))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp5 = tl.load(in_ptr0 + (2))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp8 = tl.load(in_ptr0 + (3))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp11 = tl.load(in_ptr0 + (4))
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK])
    tmp14 = tl.load(in_ptr0 + (5))
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK])
    tmp4 = triton_helpers.maximum(tmp1, tmp3)
    tmp7 = triton_helpers.maximum(tmp4, tmp6)
    tmp10 = triton_helpers.maximum(tmp7, tmp9)
    tmp13 = triton_helpers.maximum(tmp10, tmp12)
    tmp16 = triton_helpers.maximum(tmp13, tmp15)
    tmp17 = tmp1 - tmp16
    tmp18 = tl_math.exp(tmp17)
    tmp19 = tmp3 - tmp16
    tmp20 = tl_math.exp(tmp19)
    tmp21 = tmp18 + tmp20
    tmp22 = tmp6 - tmp16
    tmp23 = tl_math.exp(tmp22)
    tmp24 = tmp21 + tmp23
    tmp25 = tmp9 - tmp16
    tmp26 = tl_math.exp(tmp25)
    tmp27 = tmp24 + tmp26
    tmp28 = tmp12 - tmp16
    tmp29 = tl_math.exp(tmp28)
    tmp30 = tmp27 + tmp29
    tmp31 = tmp15 - tmp16
    tmp32 = tl_math.exp(tmp31)
    tmp33 = tmp30 + tmp32
    tmp34 = tl_math.log(tmp33)
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp16, None)
    tl.store(out_ptr1 + (tl.full([XBLOCK], 0, tl.int32)), tmp34, None)


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__log_softmax_add_copy_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + ((-2) + x0), tmp5 & xmask, other=0.0)
    tmp7 = tl.load(in_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp5, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr0 + (tl.full([XBLOCK], 1, tl.int32)), tmp5, eviction_policy='evict_last', other=0.0)
    tmp9 = triton_helpers.maximum(tmp7, tmp8)
    tmp10 = tmp6 - tmp9
    tmp11 = tmp7 - tmp9
    tmp12 = tl_math.exp(tmp11)
    tmp13 = tmp8 - tmp9
    tmp14 = tl_math.exp(tmp13)
    tmp15 = tmp12 + tmp14
    tmp16 = tl_math.log(tmp15)
    tmp17 = tmp10 - tmp16
    tmp18 = tl.load(in_ptr1 + (tl.full([XBLOCK], 2, tl.int32)), tmp5, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp5, tmp19, tmp20)
    tmp22 = tmp0 < tmp1
    tmp23 = tl.load(in_ptr1 + (x0), tmp22 & xmask, other=0.0)
    tmp24 = float("nan")
    tmp25 = tl.where(tmp22, tmp23, tmp24)
    tmp26 = tl.where(tmp5, tmp21, tmp25)
    tmp27 = tmp0 >= tmp3
    tmp28 = tl.load(in_ptr2 + ((-4) + x0), tmp27 & xmask, other=0.0)
    tmp29 = tl.load(in_ptr3 + (tl.full([XBLOCK], 0, tl.int32)), tmp27, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 - tmp29
    tmp31 = tl.load(in_ptr4 + (tl.full([XBLOCK], 0, tl.int32)), tmp27, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp30 - tmp31
    tmp33 = tl.load(in_ptr1 + (tl.full([XBLOCK], 3, tl.int32)), tmp27, eviction_policy='evict_last', other=0.0)
    tmp34 = tmp32 + tmp33
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp27, tmp34, tmp35)
    tmp37 = tl.where(tmp27, tmp36, tmp26)
    tl.store(in_out_ptr0 + (x0), tmp37, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15 = args
    args.clear()
    assert_size_stride(primals_1, (1, 10, 128), (1280, 128, 1))
    assert_size_stride(primals_2, (768, 128), (128, 1))
    assert_size_stride(primals_3, (768, 256), (256, 1))
    assert_size_stride(primals_4, (768, ), (1, ))
    assert_size_stride(primals_5, (768, ), (1, ))
    assert_size_stride(primals_6, (1536, 256), (256, 1))
    assert_size_stride(primals_7, (1536, 512), (512, 1))
    assert_size_stride(primals_8, (1536, ), (1, ))
    assert_size_stride(primals_9, (1536, ), (1, ))
    assert_size_stride(primals_10, (512, ), (1, ))
    assert_size_stride(primals_11, (4, 512), (512, 1))
    assert_size_stride(primals_12, (128, 512), (512, 1))
    assert_size_stride(primals_13, (2, 128), (128, 1))
    assert_size_stride(primals_14, (32, 512), (512, 1))
    assert_size_stride(primals_15, (6, 32), (32, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 256), (256, 1), torch.float32)

        get_raw_stream(0)
        triton_poi_fused__to_copy_0[grid(256)](buf0, 256, XBLOCK=128, num_warps=4, num_stages=1)
        buf1 = empty_strided_cuda((1, 512), (512, 1), torch.float32)

        get_raw_stream(0)
        triton_poi_fused__to_copy_1[grid(512)](buf1, 512, XBLOCK=128, num_warps=4, num_stages=1)
        buf2 = empty_strided_cuda((1, 768), (768, 1), torch.float32)

        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 128), (128, 1), 0), reinterpret_tensor(primals_2, (128, 768), (1, 128), 0), out=buf2)
        buf3 = empty_strided_cuda((1, 768), (768, 1), torch.float32)

        extern_kernels.mm(buf0, reinterpret_tensor(primals_3, (256, 768), (1, 256), 0), out=buf3)

        buf4 = torch.ops.aten._thnn_fused_gru_cell.default(buf2, buf3, buf0, primals_4, primals_5)
        buf5 = buf4[0]
        buf6 = buf4[1]
        del buf4
        buf7 = empty_strided_cuda((1, 1536), (1536, 1), torch.float32)

        extern_kernels.mm(buf5, reinterpret_tensor(primals_6, (256, 1536), (1, 256), 0), out=buf7)
        buf8 = empty_strided_cuda((1, 1536), (1536, 1), torch.float32)

        extern_kernels.mm(buf1, reinterpret_tensor(primals_7, (512, 1536), (1, 512), 0), out=buf8)

        buf9 = torch.ops.aten._thnn_fused_gru_cell.default(buf7, buf8, buf1, primals_8, primals_9)
        buf10 = buf9[0]
        buf11 = buf9[1]
        del buf9
        buf12 = buf3; del buf3

        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 128), (128, 1), 128), reinterpret_tensor(primals_2, (128, 768), (1, 128), 0), out=buf12)
        buf13 = buf2; del buf2

        extern_kernels.mm(buf5, reinterpret_tensor(primals_3, (256, 768), (1, 256), 0), out=buf13)

        buf14 = torch.ops.aten._thnn_fused_gru_cell.default(buf12, buf13, buf5, primals_4, primals_5)
        buf15 = buf14[0]
        buf16 = buf14[1]
        del buf14
        buf17 = buf8; del buf8

        extern_kernels.mm(buf15, reinterpret_tensor(primals_6, (256, 1536), (1, 256), 0), out=buf17)
        buf18 = buf7; del buf7

        extern_kernels.mm(buf10, reinterpret_tensor(primals_7, (512, 1536), (1, 512), 0), out=buf18)

        buf19 = torch.ops.aten._thnn_fused_gru_cell.default(buf17, buf18, buf10, primals_8, primals_9)
        buf20 = buf19[0]
        buf21 = buf19[1]
        del buf19
        buf22 = buf13; del buf13

        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 128), (128, 1), 256), reinterpret_tensor(primals_2, (128, 768), (1, 128), 0), out=buf22)
        buf23 = buf12; del buf12

        extern_kernels.mm(buf15, reinterpret_tensor(primals_3, (256, 768), (1, 256), 0), out=buf23)

        buf24 = torch.ops.aten._thnn_fused_gru_cell.default(buf22, buf23, buf15, primals_4, primals_5)
        buf25 = buf24[0]
        buf26 = buf24[1]
        del buf24
        buf27 = buf18; del buf18

        extern_kernels.mm(buf25, reinterpret_tensor(primals_6, (256, 1536), (1, 256), 0), out=buf27)
        buf28 = buf17; del buf17

        extern_kernels.mm(buf20, reinterpret_tensor(primals_7, (512, 1536), (1, 512), 0), out=buf28)

        buf29 = torch.ops.aten._thnn_fused_gru_cell.default(buf27, buf28, buf20, primals_8, primals_9)
        buf30 = buf29[0]
        buf31 = buf29[1]
        del buf29
        buf32 = buf23; del buf23

        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 128), (128, 1), 384), reinterpret_tensor(primals_2, (128, 768), (1, 128), 0), out=buf32)
        buf33 = buf22; del buf22

        extern_kernels.mm(buf25, reinterpret_tensor(primals_3, (256, 768), (1, 256), 0), out=buf33)

        buf34 = torch.ops.aten._thnn_fused_gru_cell.default(buf32, buf33, buf25, primals_4, primals_5)
        buf35 = buf34[0]
        buf36 = buf34[1]
        del buf34
        buf37 = buf28; del buf28

        extern_kernels.mm(buf35, reinterpret_tensor(primals_6, (256, 1536), (1, 256), 0), out=buf37)
        buf38 = buf27; del buf27

        extern_kernels.mm(buf30, reinterpret_tensor(primals_7, (512, 1536), (1, 512), 0), out=buf38)

        buf39 = torch.ops.aten._thnn_fused_gru_cell.default(buf37, buf38, buf30, primals_8, primals_9)
        buf40 = buf39[0]
        buf41 = buf39[1]
        del buf39
        buf42 = buf33; del buf33

        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 128), (128, 1), 512), reinterpret_tensor(primals_2, (128, 768), (1, 128), 0), out=buf42)
        buf43 = buf32; del buf32

        extern_kernels.mm(buf35, reinterpret_tensor(primals_3, (256, 768), (1, 256), 0), out=buf43)

        buf44 = torch.ops.aten._thnn_fused_gru_cell.default(buf42, buf43, buf35, primals_4, primals_5)
        buf45 = buf44[0]
        buf46 = buf44[1]
        del buf44
        buf47 = buf38; del buf38

        extern_kernels.mm(buf45, reinterpret_tensor(primals_6, (256, 1536), (1, 256), 0), out=buf47)
        buf48 = buf37; del buf37

        extern_kernels.mm(buf40, reinterpret_tensor(primals_7, (512, 1536), (1, 512), 0), out=buf48)

        buf49 = torch.ops.aten._thnn_fused_gru_cell.default(buf47, buf48, buf40, primals_8, primals_9)
        buf50 = buf49[0]
        buf51 = buf49[1]
        del buf49
        buf52 = buf43; del buf43

        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 128), (128, 1), 640), reinterpret_tensor(primals_2, (128, 768), (1, 128), 0), out=buf52)
        buf53 = buf42; del buf42

        extern_kernels.mm(buf45, reinterpret_tensor(primals_3, (256, 768), (1, 256), 0), out=buf53)

        buf54 = torch.ops.aten._thnn_fused_gru_cell.default(buf52, buf53, buf45, primals_4, primals_5)
        buf55 = buf54[0]
        buf56 = buf54[1]
        del buf54
        buf57 = buf48; del buf48

        extern_kernels.mm(buf55, reinterpret_tensor(primals_6, (256, 1536), (1, 256), 0), out=buf57)
        buf58 = buf47; del buf47

        extern_kernels.mm(buf50, reinterpret_tensor(primals_7, (512, 1536), (1, 512), 0), out=buf58)

        buf59 = torch.ops.aten._thnn_fused_gru_cell.default(buf57, buf58, buf50, primals_8, primals_9)
        buf60 = buf59[0]
        buf61 = buf59[1]
        del buf59
        buf62 = buf53; del buf53

        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 128), (128, 1), 768), reinterpret_tensor(primals_2, (128, 768), (1, 128), 0), out=buf62)
        buf63 = buf52; del buf52

        extern_kernels.mm(buf55, reinterpret_tensor(primals_3, (256, 768), (1, 256), 0), out=buf63)

        buf64 = torch.ops.aten._thnn_fused_gru_cell.default(buf62, buf63, buf55, primals_4, primals_5)
        buf65 = buf64[0]
        buf66 = buf64[1]
        del buf64
        buf67 = buf58; del buf58

        extern_kernels.mm(buf65, reinterpret_tensor(primals_6, (256, 1536), (1, 256), 0), out=buf67)
        buf68 = buf57; del buf57

        extern_kernels.mm(buf60, reinterpret_tensor(primals_7, (512, 1536), (1, 512), 0), out=buf68)

        buf69 = torch.ops.aten._thnn_fused_gru_cell.default(buf67, buf68, buf60, primals_8, primals_9)
        buf70 = buf69[0]
        buf71 = buf69[1]
        del buf69
        buf72 = buf63; del buf63

        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 128), (128, 1), 896), reinterpret_tensor(primals_2, (128, 768), (1, 128), 0), out=buf72)
        buf73 = buf62; del buf62

        extern_kernels.mm(buf65, reinterpret_tensor(primals_3, (256, 768), (1, 256), 0), out=buf73)

        buf74 = torch.ops.aten._thnn_fused_gru_cell.default(buf72, buf73, buf65, primals_4, primals_5)
        buf75 = buf74[0]
        buf76 = buf74[1]
        del buf74
        buf77 = buf68; del buf68

        extern_kernels.mm(buf75, reinterpret_tensor(primals_6, (256, 1536), (1, 256), 0), out=buf77)
        buf78 = buf67; del buf67

        extern_kernels.mm(buf70, reinterpret_tensor(primals_7, (512, 1536), (1, 512), 0), out=buf78)

        buf79 = torch.ops.aten._thnn_fused_gru_cell.default(buf77, buf78, buf70, primals_8, primals_9)
        buf80 = buf79[0]
        buf81 = buf79[1]
        del buf79
        buf82 = buf73; del buf73

        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 128), (128, 1), 1024), reinterpret_tensor(primals_2, (128, 768), (1, 128), 0), out=buf82)
        buf83 = buf72; del buf72

        extern_kernels.mm(buf75, reinterpret_tensor(primals_3, (256, 768), (1, 256), 0), out=buf83)

        buf84 = torch.ops.aten._thnn_fused_gru_cell.default(buf82, buf83, buf75, primals_4, primals_5)
        buf85 = buf84[0]
        buf86 = buf84[1]
        del buf84
        buf87 = buf78; del buf78

        extern_kernels.mm(buf85, reinterpret_tensor(primals_6, (256, 1536), (1, 256), 0), out=buf87)
        buf88 = buf77; del buf77

        extern_kernels.mm(buf80, reinterpret_tensor(primals_7, (512, 1536), (1, 512), 0), out=buf88)

        buf89 = torch.ops.aten._thnn_fused_gru_cell.default(buf87, buf88, buf80, primals_8, primals_9)
        buf90 = buf89[0]
        buf91 = buf89[1]
        del buf89
        buf92 = buf83; del buf83

        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 128), (128, 1), 1152), reinterpret_tensor(primals_2, (128, 768), (1, 128), 0), out=buf92)
        del primals_2
        buf93 = buf82; del buf82

        extern_kernels.mm(buf85, reinterpret_tensor(primals_3, (256, 768), (1, 256), 0), out=buf93)

        buf94 = torch.ops.aten._thnn_fused_gru_cell.default(buf92, buf93, buf85, primals_4, primals_5)
        del buf92
        del buf93
        del primals_4
        del primals_5
        buf95 = buf94[0]
        buf96 = buf94[1]
        del buf94
        buf97 = buf88; del buf88

        extern_kernels.mm(buf95, reinterpret_tensor(primals_6, (256, 1536), (1, 256), 0), out=buf97)
        buf98 = buf87; del buf87

        extern_kernels.mm(buf90, reinterpret_tensor(primals_7, (512, 1536), (1, 512), 0), out=buf98)

        buf99 = torch.ops.aten._thnn_fused_gru_cell.default(buf97, buf98, buf90, primals_8, primals_9)
        del buf97
        del buf98
        del primals_8
        del primals_9
        buf100 = buf99[0]
        buf101 = buf99[1]
        del buf99
        buf102 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        buf103 = buf102; del buf102
        buf104 = empty_strided_cuda((1, 512), (512, 1), torch.float32)

        get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_2[grid(1)](buf103, buf100, primals_10, buf104, 1, 512, num_warps=4, num_stages=1)

        buf105 = torch.ops.aten.adaptive_max_pool3d.default(reinterpret_tensor(buf104, (1, 512, 1, 1, 1), (0, 1, 0, 0, 0), 0), [1, 1, 1])
        buf106 = buf105[0]
        buf107 = buf105[1]
        del buf105
        buf108 = empty_strided_cuda((1, 4), (4, 1), torch.float32)

        extern_kernels.mm(reinterpret_tensor(buf106, (1, 512), (512, 1), 0), reinterpret_tensor(primals_11, (512, 4), (1, 512), 0), out=buf108)
        buf110 = empty_strided_cuda((1, 4), (4, 1), torch.float32)

        get_raw_stream(0)
        triton_poi_fused__log_softmax_3[grid(4)](buf108, buf110, 4, XBLOCK=4, num_warps=1, num_stages=1)
        buf111 = empty_strided_cuda((1, 4), (4, 1), torch.float32)

        get_raw_stream(0)
        triton_poi_fused__log_softmax_4[grid(4)](buf110, buf111, 4, XBLOCK=4, num_warps=1, num_stages=1)
        del buf110
        buf112 = empty_strided_cuda((1, 128), (128, 1), torch.float32)

        extern_kernels.mm(reinterpret_tensor(buf106, (1, 512), (512, 1), 0), reinterpret_tensor(primals_12, (512, 128), (1, 512), 0), out=buf112)
        buf113 = empty_strided_cuda((1, 2), (2, 1), torch.float32)

        extern_kernels.mm(buf112, reinterpret_tensor(primals_13, (128, 2), (1, 128), 0), out=buf113)
        buf115 = empty_strided_cuda((1, 32), (32, 1), torch.float32)

        extern_kernels.mm(reinterpret_tensor(buf106, (1, 512), (512, 1), 0), reinterpret_tensor(primals_14, (512, 32), (1, 512), 0), out=buf115)
        buf116 = empty_strided_cuda((1, 6), (6, 1), torch.float32)

        extern_kernels.mm(buf115, reinterpret_tensor(primals_15, (32, 6), (1, 32), 0), out=buf116)
        buf117 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        buf118 = empty_strided_cuda((1, 1), (1, 1), torch.float32)

        get_raw_stream(0)
        triton_poi_fused__log_softmax_5[grid(1)](buf116, buf117, buf118, 1, XBLOCK=1, num_warps=1, num_stages=1)
        buf114 = empty_strided_cuda((1, 10), (10, 1), torch.float32)
        buf119 = buf114; del buf114

        get_raw_stream(0)
        triton_poi_fused__log_softmax_add_copy_6[grid(10)](buf119, buf113, buf111, buf116, buf117, buf118, 10, XBLOCK=16, num_warps=1, num_stages=1)
        del buf111
    return (buf119, primals_10, buf0, buf1, reinterpret_tensor(primals_1, (1, 128), (1280, 1), 0), buf5, buf6, buf10, buf11, reinterpret_tensor(primals_1, (1, 128), (1280, 1), 128), buf15, buf16, buf20, buf21, reinterpret_tensor(primals_1, (1, 128), (1280, 1), 256), buf25, buf26, buf30, buf31, reinterpret_tensor(primals_1, (1, 128), (1280, 1), 384), buf35, buf36, buf40, buf41, reinterpret_tensor(primals_1, (1, 128), (1280, 1), 512), buf45, buf46, buf50, buf51, reinterpret_tensor(primals_1, (1, 128), (1280, 1), 640), buf55, buf56, buf60, buf61, reinterpret_tensor(primals_1, (1, 128), (1280, 1), 768), buf65, buf66, buf70, buf71, reinterpret_tensor(primals_1, (1, 128), (1280, 1), 896), buf75, buf76, buf80, buf81, reinterpret_tensor(primals_1, (1, 128), (1280, 1), 1024), buf85, buf86, buf90, buf91, reinterpret_tensor(primals_1, (1, 128), (1280, 1), 1152), buf95, buf96, buf100, buf101, buf103, reinterpret_tensor(buf104, (1, 512, 1, 1, 1), (512, 1, 1, 1, 1), 0), buf107, reinterpret_tensor(buf106, (1, 512), (512, 1), 0), buf108, buf112, buf113, buf115, buf116, buf117, buf118, primals_15, primals_14, primals_13, primals_12, primals_11, primals_7, primals_6, primals_3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 10, 128), (1280, 128, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((768, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((4, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((2, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((32, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((6, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
