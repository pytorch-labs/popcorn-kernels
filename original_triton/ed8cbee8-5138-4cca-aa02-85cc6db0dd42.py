# AOT ID: ['16_forward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import (
    grid,
    split_scan_grid,
    grid_combo_kernels,
    start_graph,
    end_graph,
    cooperative_reduction_grid,
)
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
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
# Topologically Sorted Source Nodes: [h1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   h1 => full_default
# Graph fragment:
#   %full_default : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([1, 128], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
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




# kernel path: /tmp/torchinductor_sahanp/rk/crkaeu57d4cbcxkuthfyx6xd3akutco76cryxfvnrl2mr5d3kvxr.py
# Topologically Sorted Source Nodes: [h2], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   h2 => full_default_1
# Graph fragment:
#   %full_default_1 : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([1, 256], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/bm/cbmb6oqgawc3glelohgek3hbp2ll7xanmj3iclodh5fliylnxnis.py
# Topologically Sorted Source Nodes: [ret], Original ATen: [aten.addmm, aten.add, aten.tanh]
# Source node to ATen node mapping:
#   ret => add, add_tensor_38, add_tensor_39, tanh
# Graph fragment:
#   %add_tensor_39 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_39, %primals_5), kwargs = {})
#   %add_tensor_38 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_38, %primals_4), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_tensor_39, %add_tensor_38), kwargs = {})
#   %tanh : [num_users=3] = call_function[target=torch.ops.aten.tanh.default](args = (%add,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_addmm_tanh_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
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




# kernel path: /tmp/torchinductor_sahanp/34/c34rgxq2zdesudysdl4xpuue46czdjg76ffld2ie3yk6moggbuw4.py
# Topologically Sorted Source Nodes: [ret_1], Original ATen: [aten.addmm, aten.add, aten.tanh]
# Source node to ATen node mapping:
#   ret_1 => add_1, add_tensor_36, add_tensor_37, tanh_1
# Graph fragment:
#   %add_tensor_37 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_37, %primals_9), kwargs = {})
#   %add_tensor_36 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_36, %primals_8), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_tensor_37, %add_tensor_36), kwargs = {})
#   %tanh_1 : [num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%add_1,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_addmm_tanh_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
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




# kernel path: /tmp/torchinductor_sahanp/n4/cn4gq4gw55q63yoa5mudq6neqrwda4ooiv3rnir3jq6fdoynybdx.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   x_2 => add_20, add_21, convert_element_type_2, convert_element_type_3, iota, mul, mul_1
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (32,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, 1), kwargs = {})
#   %add_20 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, 0), kwargs = {})
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_20, torch.float32), kwargs = {})
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_2, 0.0), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_21, 0.5), kwargs = {})
#   %convert_element_type_3 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1, torch.int64), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_4(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)




# kernel path: /tmp/torchinductor_sahanp/ji/cji6vlvcmi6oemiguyw2gx32c33by46a5xhgiokz25ewq66wuox2.py
# Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_4 => mean
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%unsqueeze_2, [-1, -2, -3], True), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_mean_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    r0_numel = 1024
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index // 32
    r0_0 = (r0_index % 32)
    tmp0 = tl.load(in_ptr0 + (r0_1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (r0_0), None, eviction_policy='evict_last')
    tmp1 = tl.full([R0_BLOCK], 16, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr1 + (tmp8 + 16*tmp4), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (tmp8 + 16*tmp4), None, eviction_policy='evict_last')
    tmp11 = tmp9 + tmp10
    tmp12 = tl.load(in_ptr3 + (tmp8 + 16*tmp4), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (tmp8 + 16*tmp4), None, eviction_policy='evict_last')
    tmp14 = tmp12 + tmp13
    tmp15 = tmp11 + tmp14
    tmp16 = libdevice.tanh(tmp15)
    tmp17 = tl.broadcast_to(tmp16, [R0_BLOCK])
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp20 = 1024.0
    tmp21 = tmp19 / tmp20
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([1], 0, tl.int32)), tmp21, None)




# kernel path: /tmp/torchinductor_sahanp/ah/cah3gycj6zlzwg6dfyx4dih6yjjojyidp277c2utsbllp26zowaa.py
# Topologically Sorted Source Nodes: [ret_19], Original ATen: [aten.addmm, aten.add, aten.tanh, aten.tanh_backward]
# Source node to ATen node mapping:
#   ret_19 => add_19, add_tensor, add_tensor_1, tanh_19
# Graph fragment:
#   %add_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_1, %primals_9), kwargs = {})
#   %add_tensor : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default, %primals_8), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_tensor_1, %add_tensor), kwargs = {})
#   %tanh_19 : [num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%add_19,), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh_19, %tanh_19), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %mul_4), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_addmm_tanh_tanh_backward_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
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
    tmp8 = tmp7 * tmp7
    tmp9 = 1.0
    tmp10 = tmp9 - tmp8
    tl.store(in_out_ptr0 + (x0), tmp10, xmask)







def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9 = args
    args.clear()
    assert_size_stride(primals_1, (1, 10, 64), (640, 64, 1))
    assert_size_stride(primals_2, (128, 64), (64, 1))
    assert_size_stride(primals_3, (128, 128), (128, 1))
    assert_size_stride(primals_4, (128, ), (1, ))
    assert_size_stride(primals_5, (128, ), (1, ))
    assert_size_stride(primals_6, (256, 128), (128, 1))
    assert_size_stride(primals_7, (256, 256), (256, 1))
    assert_size_stride(primals_8, (256, ), (1, ))
    assert_size_stride(primals_9, (256, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [h1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_0[grid(128)](buf0, 128, XBLOCK=128, num_warps=4, num_stages=1)
        buf1 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [h2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1[grid(256)](buf1, 256, XBLOCK=256, num_warps=4, num_stages=1)
        buf2 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret], Original ATen: [aten.addmm]
        extern_kernels.mm(buf0, reinterpret_tensor(primals_3, (128, 128), (1, 128), 0), out=buf2)
        buf3 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 64), (64, 1), 0), reinterpret_tensor(primals_2, (64, 128), (1, 64), 0), out=buf3)
        buf4 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [ret], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_2[grid(128)](buf4, primals_5, buf3, primals_4, 128, XBLOCK=128, num_warps=4, num_stages=1)
        buf5 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_1], Original ATen: [aten.addmm]
        extern_kernels.mm(buf1, reinterpret_tensor(primals_7, (256, 256), (1, 256), 0), out=buf5)
        buf6 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_1], Original ATen: [aten.addmm]
        extern_kernels.mm(buf4, reinterpret_tensor(primals_6, (128, 256), (1, 128), 0), out=buf6)
        buf7 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [ret_1], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_3[grid(256)](buf7, primals_9, buf6, primals_8, 256, XBLOCK=256, num_warps=4, num_stages=1)
        buf8 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [ret_2], Original ATen: [aten.addmm]
        extern_kernels.mm(buf4, reinterpret_tensor(primals_3, (128, 128), (1, 128), 0), out=buf8)
        buf9 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_2], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 64), (64, 1), 64), reinterpret_tensor(primals_2, (64, 128), (1, 64), 0), out=buf9)
        buf10 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [ret_2], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_2[grid(128)](buf10, primals_5, buf9, primals_4, 128, XBLOCK=128, num_warps=4, num_stages=1)
        buf11 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [ret_3], Original ATen: [aten.addmm]
        extern_kernels.mm(buf7, reinterpret_tensor(primals_7, (256, 256), (1, 256), 0), out=buf11)
        buf12 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_3], Original ATen: [aten.addmm]
        extern_kernels.mm(buf10, reinterpret_tensor(primals_6, (128, 256), (1, 128), 0), out=buf12)
        buf13 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [ret_3], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_3[grid(256)](buf13, primals_9, buf12, primals_8, 256, XBLOCK=256, num_warps=4, num_stages=1)
        buf14 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [ret_4], Original ATen: [aten.addmm]
        extern_kernels.mm(buf10, reinterpret_tensor(primals_3, (128, 128), (1, 128), 0), out=buf14)
        buf15 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_4], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 64), (64, 1), 128), reinterpret_tensor(primals_2, (64, 128), (1, 64), 0), out=buf15)
        buf16 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [ret_4], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_2[grid(128)](buf16, primals_5, buf15, primals_4, 128, XBLOCK=128, num_warps=4, num_stages=1)
        buf17 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [ret_5], Original ATen: [aten.addmm]
        extern_kernels.mm(buf13, reinterpret_tensor(primals_7, (256, 256), (1, 256), 0), out=buf17)
        buf18 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_5], Original ATen: [aten.addmm]
        extern_kernels.mm(buf16, reinterpret_tensor(primals_6, (128, 256), (1, 128), 0), out=buf18)
        buf19 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [ret_5], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_3[grid(256)](buf19, primals_9, buf18, primals_8, 256, XBLOCK=256, num_warps=4, num_stages=1)
        buf20 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [ret_6], Original ATen: [aten.addmm]
        extern_kernels.mm(buf16, reinterpret_tensor(primals_3, (128, 128), (1, 128), 0), out=buf20)
        buf21 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_6], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 64), (64, 1), 192), reinterpret_tensor(primals_2, (64, 128), (1, 64), 0), out=buf21)
        buf22 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [ret_6], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_2[grid(128)](buf22, primals_5, buf21, primals_4, 128, XBLOCK=128, num_warps=4, num_stages=1)
        buf23 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [ret_7], Original ATen: [aten.addmm]
        extern_kernels.mm(buf19, reinterpret_tensor(primals_7, (256, 256), (1, 256), 0), out=buf23)
        buf24 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_7], Original ATen: [aten.addmm]
        extern_kernels.mm(buf22, reinterpret_tensor(primals_6, (128, 256), (1, 128), 0), out=buf24)
        buf25 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [ret_7], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_3[grid(256)](buf25, primals_9, buf24, primals_8, 256, XBLOCK=256, num_warps=4, num_stages=1)
        buf26 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [ret_8], Original ATen: [aten.addmm]
        extern_kernels.mm(buf22, reinterpret_tensor(primals_3, (128, 128), (1, 128), 0), out=buf26)
        buf27 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_8], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 64), (64, 1), 256), reinterpret_tensor(primals_2, (64, 128), (1, 64), 0), out=buf27)
        buf28 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [ret_8], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_2[grid(128)](buf28, primals_5, buf27, primals_4, 128, XBLOCK=128, num_warps=4, num_stages=1)
        buf29 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [ret_9], Original ATen: [aten.addmm]
        extern_kernels.mm(buf25, reinterpret_tensor(primals_7, (256, 256), (1, 256), 0), out=buf29)
        buf30 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_9], Original ATen: [aten.addmm]
        extern_kernels.mm(buf28, reinterpret_tensor(primals_6, (128, 256), (1, 128), 0), out=buf30)
        buf31 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [ret_9], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_3[grid(256)](buf31, primals_9, buf30, primals_8, 256, XBLOCK=256, num_warps=4, num_stages=1)
        buf32 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [ret_10], Original ATen: [aten.addmm]
        extern_kernels.mm(buf28, reinterpret_tensor(primals_3, (128, 128), (1, 128), 0), out=buf32)
        buf33 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_10], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 64), (64, 1), 320), reinterpret_tensor(primals_2, (64, 128), (1, 64), 0), out=buf33)
        buf34 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [ret_10], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_2[grid(128)](buf34, primals_5, buf33, primals_4, 128, XBLOCK=128, num_warps=4, num_stages=1)
        buf35 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [ret_11], Original ATen: [aten.addmm]
        extern_kernels.mm(buf31, reinterpret_tensor(primals_7, (256, 256), (1, 256), 0), out=buf35)
        buf36 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_11], Original ATen: [aten.addmm]
        extern_kernels.mm(buf34, reinterpret_tensor(primals_6, (128, 256), (1, 128), 0), out=buf36)
        buf37 = buf35; del buf35  # reuse
        # Topologically Sorted Source Nodes: [ret_11], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_3[grid(256)](buf37, primals_9, buf36, primals_8, 256, XBLOCK=256, num_warps=4, num_stages=1)
        buf38 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [ret_12], Original ATen: [aten.addmm]
        extern_kernels.mm(buf34, reinterpret_tensor(primals_3, (128, 128), (1, 128), 0), out=buf38)
        buf39 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_12], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 64), (64, 1), 384), reinterpret_tensor(primals_2, (64, 128), (1, 64), 0), out=buf39)
        buf40 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [ret_12], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_2[grid(128)](buf40, primals_5, buf39, primals_4, 128, XBLOCK=128, num_warps=4, num_stages=1)
        buf41 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [ret_13], Original ATen: [aten.addmm]
        extern_kernels.mm(buf37, reinterpret_tensor(primals_7, (256, 256), (1, 256), 0), out=buf41)
        buf42 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_13], Original ATen: [aten.addmm]
        extern_kernels.mm(buf40, reinterpret_tensor(primals_6, (128, 256), (1, 128), 0), out=buf42)
        buf43 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [ret_13], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_3[grid(256)](buf43, primals_9, buf42, primals_8, 256, XBLOCK=256, num_warps=4, num_stages=1)
        buf44 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [ret_14], Original ATen: [aten.addmm]
        extern_kernels.mm(buf40, reinterpret_tensor(primals_3, (128, 128), (1, 128), 0), out=buf44)
        buf45 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_14], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 64), (64, 1), 448), reinterpret_tensor(primals_2, (64, 128), (1, 64), 0), out=buf45)
        buf46 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [ret_14], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_2[grid(128)](buf46, primals_5, buf45, primals_4, 128, XBLOCK=128, num_warps=4, num_stages=1)
        buf47 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [ret_15], Original ATen: [aten.addmm]
        extern_kernels.mm(buf43, reinterpret_tensor(primals_7, (256, 256), (1, 256), 0), out=buf47)
        buf48 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_15], Original ATen: [aten.addmm]
        extern_kernels.mm(buf46, reinterpret_tensor(primals_6, (128, 256), (1, 128), 0), out=buf48)
        buf49 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [ret_15], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_3[grid(256)](buf49, primals_9, buf48, primals_8, 256, XBLOCK=256, num_warps=4, num_stages=1)
        buf50 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [ret_16], Original ATen: [aten.addmm]
        extern_kernels.mm(buf46, reinterpret_tensor(primals_3, (128, 128), (1, 128), 0), out=buf50)
        buf51 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_16], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 64), (64, 1), 512), reinterpret_tensor(primals_2, (64, 128), (1, 64), 0), out=buf51)
        buf52 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [ret_16], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_2[grid(128)](buf52, primals_5, buf51, primals_4, 128, XBLOCK=128, num_warps=4, num_stages=1)
        buf53 = buf48; del buf48  # reuse
        # Topologically Sorted Source Nodes: [ret_17], Original ATen: [aten.addmm]
        extern_kernels.mm(buf49, reinterpret_tensor(primals_7, (256, 256), (1, 256), 0), out=buf53)
        buf54 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_17], Original ATen: [aten.addmm]
        extern_kernels.mm(buf52, reinterpret_tensor(primals_6, (128, 256), (1, 128), 0), out=buf54)
        buf55 = buf53; del buf53  # reuse
        # Topologically Sorted Source Nodes: [ret_17], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_3[grid(256)](buf55, primals_9, buf54, primals_8, 256, XBLOCK=256, num_warps=4, num_stages=1)
        buf56 = buf51; del buf51  # reuse
        # Topologically Sorted Source Nodes: [ret_18], Original ATen: [aten.addmm]
        extern_kernels.mm(buf52, reinterpret_tensor(primals_3, (128, 128), (1, 128), 0), out=buf56)
        buf57 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_18], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 64), (64, 1), 576), reinterpret_tensor(primals_2, (64, 128), (1, 64), 0), out=buf57)
        del primals_2
        buf58 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [ret_18], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_2[grid(128)](buf58, primals_5, buf57, primals_4, 128, XBLOCK=128, num_warps=4, num_stages=1)
        del buf57
        del primals_4
        del primals_5
        buf59 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [ret_19], Original ATen: [aten.addmm]
        extern_kernels.mm(buf55, reinterpret_tensor(primals_7, (256, 256), (1, 256), 0), out=buf59)
        buf60 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_19], Original ATen: [aten.addmm]
        extern_kernels.mm(buf58, reinterpret_tensor(primals_6, (128, 256), (1, 128), 0), out=buf60)
        buf61 = empty_strided_cuda((32, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_4[grid(32)](buf61, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf62 = empty_strided_cuda((1, 1, 1, 1, 1), (1, 1, 1, 1, 1), torch.float32)
        buf63 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_5[grid(1)](buf63, buf61, buf59, primals_9, buf60, primals_8, 1, 1024, num_warps=8, num_stages=1)
        buf64 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [ret_19], Original ATen: [aten.addmm, aten.add, aten.tanh, aten.tanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_tanh_backward_6[grid(256)](buf64, primals_9, buf60, primals_8, 256, XBLOCK=128, num_warps=4, num_stages=1)
        del buf60
        del primals_8
        del primals_9
    return (reinterpret_tensor(buf63, (1, 1), (1, 1), 0), buf0, buf1, reinterpret_tensor(primals_1, (1, 64), (640, 1), 0), buf4, buf7, reinterpret_tensor(primals_1, (1, 64), (640, 1), 64), buf10, buf13, reinterpret_tensor(primals_1, (1, 64), (640, 1), 128), buf16, buf19, reinterpret_tensor(primals_1, (1, 64), (640, 1), 192), buf22, buf25, reinterpret_tensor(primals_1, (1, 64), (640, 1), 256), buf28, buf31, reinterpret_tensor(primals_1, (1, 64), (640, 1), 320), buf34, buf37, reinterpret_tensor(primals_1, (1, 64), (640, 1), 384), buf40, buf43, reinterpret_tensor(primals_1, (1, 64), (640, 1), 448), buf46, buf49, reinterpret_tensor(primals_1, (1, 64), (640, 1), 512), buf52, buf55, reinterpret_tensor(primals_1, (1, 64), (640, 1), 576), buf58, buf61, buf64, primals_6, primals_7, primals_3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 10, 64), (640, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((128, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
