# AOT ID: ['166_forward']
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


# kernel path: /tmp/torchinductor_sahanp/na/cnattssr7vg3bdv5r5ykt4mtxcukggbmcsqwc7ysxp2nmar7fkz3.py
# Topologically Sorted Source Nodes: [hx], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   hx => full_default
# Graph fragment:
#   %full_default : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([1, 128], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)


# kernel path: /tmp/torchinductor_sahanp/lg/clgz5s264ibffmfutvcv6r43wfh57eih2l3mua4mqmmxhxe7wyze.py
# Topologically Sorted Source Nodes: [ret, x], Original ATen: [aten.addmm, aten.add, aten.tanh, aten.stack]
# Source node to ATen node mapping:
#   ret => add, add_tensor_18, add_tensor_19, tanh
#   x => cat
# Graph fragment:
#   %add_tensor_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_19, %primals_5), kwargs = {})
#   %add_tensor_18 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_18, %primals_4), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_tensor_19, %add_tensor_18), kwargs = {})
#   %tanh : [num_users=3] = call_function[target=torch.ops.aten.tanh.default](args = (%add,), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%tanh, %tanh_1, %tanh_2, %tanh_3, %tanh_4, %tanh_5, %tanh_6, %tanh_7, %tanh_8, %tanh_9], 1), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_addmm_stack_tanh_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp4 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = libdevice.tanh(tmp6)
    tl.store(in_out_ptr0 + (x0), tmp7, xmask)
    tl.store(out_ptr0 + (x0), tmp7, xmask)


# kernel path: /tmp/torchinductor_sahanp/vk/cvkcrc27aln5xdaqfeaefknv4ydwwzh74fcd6zopejbixart5eo3.py
# Topologically Sorted Source Nodes: [ret_9], Original ATen: [aten.addmm, aten.add, aten.tanh, aten.tanh_backward]
# Source node to ATen node mapping:
#   ret_9 => add_9, add_tensor, add_tensor_1, tanh_9
# Graph fragment:
#   %add_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_1, %primals_5), kwargs = {})
#   %add_tensor : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default, %primals_4), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_tensor_1, %add_tensor), kwargs = {})
#   %tanh_9 : [num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%add_9,), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh_9, %tanh_9), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %mul_4), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_addmm_tanh_tanh_backward_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp3 = tl.load(in_ptr2 + (x0), xmask)
    tmp4 = tl.load(in_ptr3 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = libdevice.tanh(tmp6)
    tmp8 = tmp7 * tmp7
    tmp9 = 1.0
    tmp10 = tmp9 - tmp8
    tl.store(out_ptr0 + (x0), tmp7, xmask)
    tl.store(out_ptr1 + (x0), tmp10, xmask)


# kernel path: /tmp/torchinductor_sahanp/si/csiiksbckmkttcbora4bmtygawj5nbei3jespwrip5kfnrec3ml2.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.abs, aten.gt, aten.sign, aten.mul, aten.sub, aten.where]
# Source node to ATen node mapping:
#   x_1 => abs_1, gt, mul, mul_1, sign, sub, where
# Graph fragment:
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%view,), kwargs = {})
#   %gt : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%abs_1, 0.5), kwargs = {})
#   %sign : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%view,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign, 0.5), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %mul), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, 0), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %sub, %mul_1), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_abs_gt_mul_sign_sub_where_3(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
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
    tl.store(out_ptr0 + (x0), tmp3, xmask)
    tl.store(out_ptr1 + (x0), tmp15, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (1, 10, 64), (640, 64, 1))
    assert_size_stride(primals_2, (128, 64), (64, 1))
    assert_size_stride(primals_3, (128, 128), (128, 1))
    assert_size_stride(primals_4, (128, ), (1, ))
    assert_size_stride(primals_5, (128, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx], Original ATen: [aten._to_copy]
        get_raw_stream(0)
        triton_poi_fused__to_copy_0[grid(128)](buf0, 128, XBLOCK=128, num_warps=4, num_stages=1)
        buf1 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret], Original ATen: [aten.addmm]
        extern_kernels.mm(buf0, reinterpret_tensor(primals_3, (128, 128), (1, 128), 0), out=buf1)
        buf2 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 64), (64, 1), 0), reinterpret_tensor(primals_2, (64, 128), (1, 64), 0), out=buf2)
        buf3 = buf1; del buf1  # reuse
        buf40 = empty_strided_cuda((1, 1280), (1280, 1), torch.float32)
        buf30 = reinterpret_tensor(buf40, (1, 128), (1280, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [ret, x], Original ATen: [aten.addmm, aten.add, aten.tanh, aten.stack]
        get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_1[grid(128)](buf3, primals_5, buf2, primals_4, buf30, 128, XBLOCK=128, num_warps=4, num_stages=1)
        buf4 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [ret_1], Original ATen: [aten.addmm]
        extern_kernels.mm(buf3, reinterpret_tensor(primals_3, (128, 128), (1, 128), 0), out=buf4)
        buf5 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_1], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 64), (64, 1), 64), reinterpret_tensor(primals_2, (64, 128), (1, 64), 0), out=buf5)
        buf6 = buf4; del buf4  # reuse
        buf31 = reinterpret_tensor(buf40, (1, 128), (1280, 1), 128)  # alias
        # Topologically Sorted Source Nodes: [ret_1, x], Original ATen: [aten.addmm, aten.add, aten.tanh, aten.stack]
        get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_1[grid(128)](buf6, primals_5, buf5, primals_4, buf31, 128, XBLOCK=128, num_warps=4, num_stages=1)
        buf7 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [ret_2], Original ATen: [aten.addmm]
        extern_kernels.mm(buf6, reinterpret_tensor(primals_3, (128, 128), (1, 128), 0), out=buf7)
        buf8 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_2], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 64), (64, 1), 128), reinterpret_tensor(primals_2, (64, 128), (1, 64), 0), out=buf8)
        buf9 = buf7; del buf7  # reuse
        buf32 = reinterpret_tensor(buf40, (1, 128), (1280, 1), 256)  # alias
        # Topologically Sorted Source Nodes: [ret_2, x], Original ATen: [aten.addmm, aten.add, aten.tanh, aten.stack]
        get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_1[grid(128)](buf9, primals_5, buf8, primals_4, buf32, 128, XBLOCK=128, num_warps=4, num_stages=1)
        buf10 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [ret_3], Original ATen: [aten.addmm]
        extern_kernels.mm(buf9, reinterpret_tensor(primals_3, (128, 128), (1, 128), 0), out=buf10)
        buf11 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_3], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 64), (64, 1), 192), reinterpret_tensor(primals_2, (64, 128), (1, 64), 0), out=buf11)
        buf12 = buf10; del buf10  # reuse
        buf33 = reinterpret_tensor(buf40, (1, 128), (1280, 1), 384)  # alias
        # Topologically Sorted Source Nodes: [ret_3, x], Original ATen: [aten.addmm, aten.add, aten.tanh, aten.stack]
        get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_1[grid(128)](buf12, primals_5, buf11, primals_4, buf33, 128, XBLOCK=128, num_warps=4, num_stages=1)
        buf13 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [ret_4], Original ATen: [aten.addmm]
        extern_kernels.mm(buf12, reinterpret_tensor(primals_3, (128, 128), (1, 128), 0), out=buf13)
        buf14 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_4], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 64), (64, 1), 256), reinterpret_tensor(primals_2, (64, 128), (1, 64), 0), out=buf14)
        buf15 = buf13; del buf13  # reuse
        buf34 = reinterpret_tensor(buf40, (1, 128), (1280, 1), 512)  # alias
        # Topologically Sorted Source Nodes: [ret_4, x], Original ATen: [aten.addmm, aten.add, aten.tanh, aten.stack]
        get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_1[grid(128)](buf15, primals_5, buf14, primals_4, buf34, 128, XBLOCK=128, num_warps=4, num_stages=1)
        buf16 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [ret_5], Original ATen: [aten.addmm]
        extern_kernels.mm(buf15, reinterpret_tensor(primals_3, (128, 128), (1, 128), 0), out=buf16)
        buf17 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_5], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 64), (64, 1), 320), reinterpret_tensor(primals_2, (64, 128), (1, 64), 0), out=buf17)
        buf18 = buf16; del buf16  # reuse
        buf35 = reinterpret_tensor(buf40, (1, 128), (1280, 1), 640)  # alias
        # Topologically Sorted Source Nodes: [ret_5, x], Original ATen: [aten.addmm, aten.add, aten.tanh, aten.stack]
        get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_1[grid(128)](buf18, primals_5, buf17, primals_4, buf35, 128, XBLOCK=128, num_warps=4, num_stages=1)
        buf19 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [ret_6], Original ATen: [aten.addmm]
        extern_kernels.mm(buf18, reinterpret_tensor(primals_3, (128, 128), (1, 128), 0), out=buf19)
        buf20 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_6], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 64), (64, 1), 384), reinterpret_tensor(primals_2, (64, 128), (1, 64), 0), out=buf20)
        buf21 = buf19; del buf19  # reuse
        buf36 = reinterpret_tensor(buf40, (1, 128), (1280, 1), 768)  # alias
        # Topologically Sorted Source Nodes: [ret_6, x], Original ATen: [aten.addmm, aten.add, aten.tanh, aten.stack]
        get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_1[grid(128)](buf21, primals_5, buf20, primals_4, buf36, 128, XBLOCK=128, num_warps=4, num_stages=1)
        buf22 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [ret_7], Original ATen: [aten.addmm]
        extern_kernels.mm(buf21, reinterpret_tensor(primals_3, (128, 128), (1, 128), 0), out=buf22)
        buf23 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_7], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 64), (64, 1), 448), reinterpret_tensor(primals_2, (64, 128), (1, 64), 0), out=buf23)
        buf24 = buf22; del buf22  # reuse
        buf37 = reinterpret_tensor(buf40, (1, 128), (1280, 1), 896)  # alias
        # Topologically Sorted Source Nodes: [ret_7, x], Original ATen: [aten.addmm, aten.add, aten.tanh, aten.stack]
        get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_1[grid(128)](buf24, primals_5, buf23, primals_4, buf37, 128, XBLOCK=128, num_warps=4, num_stages=1)
        buf25 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [ret_8], Original ATen: [aten.addmm]
        extern_kernels.mm(buf24, reinterpret_tensor(primals_3, (128, 128), (1, 128), 0), out=buf25)
        buf26 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_8], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 64), (64, 1), 512), reinterpret_tensor(primals_2, (64, 128), (1, 64), 0), out=buf26)
        buf27 = buf25; del buf25  # reuse
        buf38 = reinterpret_tensor(buf40, (1, 128), (1280, 1), 1024)  # alias
        # Topologically Sorted Source Nodes: [ret_8, x], Original ATen: [aten.addmm, aten.add, aten.tanh, aten.stack]
        get_raw_stream(0)
        triton_poi_fused_add_addmm_stack_tanh_1[grid(128)](buf27, primals_5, buf26, primals_4, buf38, 128, XBLOCK=128, num_warps=4, num_stages=1)
        buf28 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [ret_9], Original ATen: [aten.addmm]
        extern_kernels.mm(buf27, reinterpret_tensor(primals_3, (128, 128), (1, 128), 0), out=buf28)
        buf29 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_9], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 64), (64, 1), 576), reinterpret_tensor(primals_2, (64, 128), (1, 64), 0), out=buf29)
        del primals_2
        buf39 = reinterpret_tensor(buf40, (1, 128), (1280, 1), 1152)  # alias
        buf43 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_9], Original ATen: [aten.addmm, aten.add, aten.tanh, aten.tanh_backward]
        get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_tanh_backward_2[grid(128)](buf28, primals_5, buf29, primals_4, buf39, buf43, 128, XBLOCK=128, num_warps=4, num_stages=1)
        del buf28
        del buf29
        del primals_4
        del primals_5
        buf41 = empty_strided_cuda((1, 10, 128), (1280, 128, 1), torch.bool)
        buf42 = empty_strided_cuda((1, 10, 128), (1280, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.abs, aten.gt, aten.sign, aten.mul, aten.sub, aten.where]
        get_raw_stream(0)
        triton_poi_fused_abs_gt_mul_sign_sub_where_3[grid(1280)](buf40, buf41, buf42, 1280, XBLOCK=256, num_warps=4, num_stages=1)
        del buf30
        del buf31
        del buf32
        del buf33
        del buf34
        del buf35
        del buf36
        del buf37
        del buf38
        del buf39
        del buf40
    return (buf42, buf0, reinterpret_tensor(primals_1, (1, 64), (640, 1), 0), buf3, reinterpret_tensor(primals_1, (1, 64), (640, 1), 64), buf6, reinterpret_tensor(primals_1, (1, 64), (640, 1), 128), buf9, reinterpret_tensor(primals_1, (1, 64), (640, 1), 192), buf12, reinterpret_tensor(primals_1, (1, 64), (640, 1), 256), buf15, reinterpret_tensor(primals_1, (1, 64), (640, 1), 320), buf18, reinterpret_tensor(primals_1, (1, 64), (640, 1), 384), buf21, reinterpret_tensor(primals_1, (1, 64), (640, 1), 448), buf24, reinterpret_tensor(primals_1, (1, 64), (640, 1), 512), buf27, reinterpret_tensor(primals_1, (1, 64), (640, 1), 576), buf41, buf43, primals_3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 10, 64), (640, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((128, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([primals_1, primals_2, primals_3, primals_4, primals_5])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
