# AOT ID: ['39_forward']
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


# kernel path: /tmp/torchinductor_sahanp/qc/cqcelt6apon7fqz47767yi5fttpnwlir7efkugzl5kfuezqagvq7.py
# Topologically Sorted Source Nodes: [hx], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   hx => full_default
# Graph fragment:
#   %full_default : [num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([1, 64], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)




#include "/tmp/torchinductor_sahanp/3b/c3bi5gk6mslf6u4iaqafhxm64z6u65e3eain4xlary5blqnvv6xx.h"
extern "C"  void kernel(const int64_t* in_ptr0,
                       int64_t* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(64L); x0+=static_cast<int64_t>(1L))
        {
            {
                {
                    auto tmp0 = in_ptr0[static_cast<int64_t>(0L)];
                    auto tmp1 = x0;
                    auto tmp2 = c10::convert<int32_t>(tmp1);
                    auto tmp3 = static_cast<int64_t>(0);
                    auto tmp4 = static_cast<int64_t>(1);
                    auto tmp5 = randint64_cpu(tmp0, tmp2, tmp3, tmp4);
                    out_ptr0[static_cast<int64_t>(x0)] = tmp5;
                }
            }
        }
    }
}
''')


# kernel path: /tmp/torchinductor_sahanp/ij/cijjnnhwsjd4eeejuaw3xehkejunnawoy6n5fj5k75my45tmwu2i.py
# Topologically Sorted Source Nodes: [cosine_loss, nll_loss], Original ATen: [aten.mul, aten.sum, aten.add, aten.sqrt, aten.div, aten.zeros_like, aten.fill, aten.sub, aten.clamp_min, aten.eq, aten.where, aten.mean, aten.nll_loss2d_forward, aten.nll_loss2d_backward]
# Source node to ATen node mapping:
#   cosine_loss => add, add_2, clamp_min, div, full_default_2, full_default_3, full_default_4, full_default_5, mean, mul, mul_3, sqrt, sub, sub_1, sum_1, where, where_1
#   nll_loss => convert_element_type_4, div_1, full_default_6, full_default_7, ne, neg, sum_5, sum_6, where_3
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_57, %getitem_57), kwargs = {})
#   %sum_1 : [num_users=3] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul, [1]), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%sum_1, 9.999999960041972e-13), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %add), kwargs = {})
#   %sqrt : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%mul_3,), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_1, %sqrt), kwargs = {})
#   %full_default_2 : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([1], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default_3, %div), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Scalar](args = (%div, 0.0), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_1, 0), kwargs = {})
#   %full_default_4 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%full_default_4, %sub, %full_default_2), kwargs = {})
#   %full_default_5 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], False), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%full_default_5, %clamp_min, %full_default_2), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where, %where_1), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%add_2,), kwargs = {})
#   %ne : [num_users=3] = call_function[target=torch.ops.aten.ne.Scalar](args = (%device_put_3, -100), kwargs = {})
#   %full_default_6 : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_7 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_3 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne, %neg, %full_default_7), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne,), kwargs = {})
#   %convert_element_type_4 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_5, torch.float32), kwargs = {})
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_3,), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_6, %convert_element_type_4), kwargs = {})
#   %ne_3 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_1, -100), kwargs = {})
#   %where_4 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_3, %unsqueeze_1, %full_default_6), kwargs = {})
#   %scatter_upon_const_tensor : [num_users=1] = call_function[target=torch._inductor.fx_passes.post_grad.scatter_upon_const_tensor](args = (), kwargs = {shape: [1, 1, 8, 8], background_val: 0, dtype: torch.float32, dim: 1, selector: %where_4, val: -1.0})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_clamp_min_div_eq_fill_mean_mul_nll_loss2d_backward_nll_loss2d_forward_sqrt_sub_sum_where_zeros_like_2(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), None)
    tmp5 = tl.load(in_ptr1 + (r0_0), None)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.sum(tmp2, 1)[:, None]
    tmp6 = tl.full([1, 1], -100, tl.int64)
    tmp7 = tmp5 != tmp6
    tmp8 = tmp7.to(tl.int64)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp11 = tl.sum(tmp9, 1)[:, None]
    tmp12 = tl.full([1, 1], 0, tl.int64)
    tmp13 = tl.where(tmp7, tmp5, tmp12)
    tmp14 = tl.full([XBLOCK, R0_BLOCK], 1, tl.int32)
    tmp15 = tmp13 + tmp14
    tmp16 = tmp13 < 0
    tmp17 = tl.where(tmp16, tmp15, tmp13)
    tl.device_assert((0 <= tmp17) & (tmp17 < 1), "index out of bounds: 0 <= tmp17 < 1")
    tmp19 = tmp0 - tmp0
    tmp20 = tl_math.exp(tmp19)
    tmp21 = tl_math.log(tmp20)
    tmp22 = tmp19 - tmp21
    tmp23 = -tmp22
    tmp24 = 0.0
    tmp25 = tl.where(tmp7, tmp23, tmp24)
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
    tmp28 = tl.sum(tmp26, 1)[:, None]
    tmp29 = tmp13 == tmp12
    tmp30 = -1.0
    tmp31 = tl.where(tmp29, tmp30, tmp24)
    tmp32 = tmp11.to(tl.float32)
    tmp33 = tmp28 / tmp32
    tmp34 = 9.999999960041972e-13
    tmp35 = tmp4 + tmp34
    tmp36 = tmp35 * tmp35
    tmp37 = libdevice.sqrt(tmp36)
    tmp38 = tmp4 / tmp37
    tmp39 = 1.0
    tmp40 = tmp39 - tmp38
    tmp41 = tl.full([1, 1], True, tl.int1)
    tmp42 = tl.where(tmp41, tmp40, tmp24)
    tmp43 = tmp38 - tmp24
    tmp44 = triton_helpers.maximum(tmp43, tmp24)
    tmp45 = tl.full([1, 1], False, tl.int1)
    tmp46 = tl.where(tmp45, tmp44, tmp24)
    tmp47 = tmp42 + tmp46
    tmp48 = tmp47 / tmp39
    tl.store(out_ptr2 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp7, None)
    tl.store(out_ptr3 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp31, None)
    tl.store(out_ptr4 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp32, None)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp33, None)
    tl.store(out_ptr5 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp48, None)
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp4, None)







def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9 = args
    args.clear()
    assert_size_stride(primals_1, (1, 10, 128), (1280, 128, 1))
    assert_size_stride(primals_2, (256, 128), (128, 1))
    assert_size_stride(primals_3, (256, 64), (64, 1))
    assert_size_stride(primals_4, (256, ), (1, ))
    assert_size_stride(primals_5, (256, ), (1, ))
    assert_size_stride(primals_6, (256, 64), (64, 1))
    assert_size_stride(primals_7, (256, 64), (64, 1))
    assert_size_stride(primals_8, (256, ), (1, ))
    assert_size_stride(primals_9, (256, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_0[grid(64)](buf0, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf1 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [lstm_cell], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 128), (128, 1), 0), reinterpret_tensor(primals_2, (128, 256), (1, 128), 0), out=buf1)
        buf2 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [lstm_cell], Original ATen: [aten.mm]
        extern_kernels.mm(buf0, reinterpret_tensor(primals_3, (64, 256), (1, 64), 0), out=buf2)
        # Topologically Sorted Source Nodes: [lstm_cell], Original ATen: [aten._thnn_fused_lstm_cell]
        buf3 = torch.ops.aten._thnn_fused_lstm_cell.default(buf1, buf2, buf0, primals_4, primals_5)
        buf4 = buf3[0]
        buf5 = buf3[1]
        buf6 = buf3[2]
        del buf3
        buf7 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf4, reinterpret_tensor(primals_6, (64, 256), (1, 64), 0), out=buf7)
        buf8 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf4, reinterpret_tensor(primals_7, (64, 256), (1, 64), 0), out=buf8)
        # Topologically Sorted Source Nodes: [lstm_cell_1], Original ATen: [aten._thnn_fused_lstm_cell]
        buf9 = torch.ops.aten._thnn_fused_lstm_cell.default(buf7, buf8, buf5, primals_8, primals_9)
        buf10 = buf9[0]
        buf11 = buf9[1]
        buf12 = buf9[2]
        del buf9
        buf13 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 128), (128, 1), 128), reinterpret_tensor(primals_2, (128, 256), (1, 128), 0), out=buf13)
        buf14 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_2], Original ATen: [aten.mm]
        extern_kernels.mm(buf10, reinterpret_tensor(primals_3, (64, 256), (1, 64), 0), out=buf14)
        # Topologically Sorted Source Nodes: [lstm_cell_2], Original ATen: [aten._thnn_fused_lstm_cell]
        buf15 = torch.ops.aten._thnn_fused_lstm_cell.default(buf13, buf14, buf11, primals_4, primals_5)
        buf16 = buf15[0]
        buf17 = buf15[1]
        buf18 = buf15[2]
        del buf15
        buf19 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_3], Original ATen: [aten.mm]
        extern_kernels.mm(buf16, reinterpret_tensor(primals_6, (64, 256), (1, 64), 0), out=buf19)
        buf20 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_3], Original ATen: [aten.mm]
        extern_kernels.mm(buf16, reinterpret_tensor(primals_7, (64, 256), (1, 64), 0), out=buf20)
        # Topologically Sorted Source Nodes: [lstm_cell_3], Original ATen: [aten._thnn_fused_lstm_cell]
        buf21 = torch.ops.aten._thnn_fused_lstm_cell.default(buf19, buf20, buf17, primals_8, primals_9)
        buf22 = buf21[0]
        buf23 = buf21[1]
        buf24 = buf21[2]
        del buf21
        buf25 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 128), (128, 1), 256), reinterpret_tensor(primals_2, (128, 256), (1, 128), 0), out=buf25)
        buf26 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_4], Original ATen: [aten.mm]
        extern_kernels.mm(buf22, reinterpret_tensor(primals_3, (64, 256), (1, 64), 0), out=buf26)
        # Topologically Sorted Source Nodes: [lstm_cell_4], Original ATen: [aten._thnn_fused_lstm_cell]
        buf27 = torch.ops.aten._thnn_fused_lstm_cell.default(buf25, buf26, buf23, primals_4, primals_5)
        buf28 = buf27[0]
        buf29 = buf27[1]
        buf30 = buf27[2]
        del buf27
        buf31 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_5], Original ATen: [aten.mm]
        extern_kernels.mm(buf28, reinterpret_tensor(primals_6, (64, 256), (1, 64), 0), out=buf31)
        buf32 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_5], Original ATen: [aten.mm]
        extern_kernels.mm(buf28, reinterpret_tensor(primals_7, (64, 256), (1, 64), 0), out=buf32)
        # Topologically Sorted Source Nodes: [lstm_cell_5], Original ATen: [aten._thnn_fused_lstm_cell]
        buf33 = torch.ops.aten._thnn_fused_lstm_cell.default(buf31, buf32, buf29, primals_8, primals_9)
        buf34 = buf33[0]
        buf35 = buf33[1]
        buf36 = buf33[2]
        del buf33
        buf37 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 128), (128, 1), 384), reinterpret_tensor(primals_2, (128, 256), (1, 128), 0), out=buf37)
        buf38 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_6], Original ATen: [aten.mm]
        extern_kernels.mm(buf34, reinterpret_tensor(primals_3, (64, 256), (1, 64), 0), out=buf38)
        # Topologically Sorted Source Nodes: [lstm_cell_6], Original ATen: [aten._thnn_fused_lstm_cell]
        buf39 = torch.ops.aten._thnn_fused_lstm_cell.default(buf37, buf38, buf35, primals_4, primals_5)
        buf40 = buf39[0]
        buf41 = buf39[1]
        buf42 = buf39[2]
        del buf39
        buf43 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_7], Original ATen: [aten.mm]
        extern_kernels.mm(buf40, reinterpret_tensor(primals_6, (64, 256), (1, 64), 0), out=buf43)
        buf44 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_7], Original ATen: [aten.mm]
        extern_kernels.mm(buf40, reinterpret_tensor(primals_7, (64, 256), (1, 64), 0), out=buf44)
        # Topologically Sorted Source Nodes: [lstm_cell_7], Original ATen: [aten._thnn_fused_lstm_cell]
        buf45 = torch.ops.aten._thnn_fused_lstm_cell.default(buf43, buf44, buf41, primals_8, primals_9)
        buf46 = buf45[0]
        buf47 = buf45[1]
        buf48 = buf45[2]
        del buf45
        buf49 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_8], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 128), (128, 1), 512), reinterpret_tensor(primals_2, (128, 256), (1, 128), 0), out=buf49)
        buf50 = buf43; del buf43  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_8], Original ATen: [aten.mm]
        extern_kernels.mm(buf46, reinterpret_tensor(primals_3, (64, 256), (1, 64), 0), out=buf50)
        # Topologically Sorted Source Nodes: [lstm_cell_8], Original ATen: [aten._thnn_fused_lstm_cell]
        buf51 = torch.ops.aten._thnn_fused_lstm_cell.default(buf49, buf50, buf47, primals_4, primals_5)
        buf52 = buf51[0]
        buf53 = buf51[1]
        buf54 = buf51[2]
        del buf51
        buf55 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_9], Original ATen: [aten.mm]
        extern_kernels.mm(buf52, reinterpret_tensor(primals_6, (64, 256), (1, 64), 0), out=buf55)
        buf56 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_9], Original ATen: [aten.mm]
        extern_kernels.mm(buf52, reinterpret_tensor(primals_7, (64, 256), (1, 64), 0), out=buf56)
        # Topologically Sorted Source Nodes: [lstm_cell_9], Original ATen: [aten._thnn_fused_lstm_cell]
        buf57 = torch.ops.aten._thnn_fused_lstm_cell.default(buf55, buf56, buf53, primals_8, primals_9)
        buf58 = buf57[0]
        buf59 = buf57[1]
        buf60 = buf57[2]
        del buf57
        buf61 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_10], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 128), (128, 1), 640), reinterpret_tensor(primals_2, (128, 256), (1, 128), 0), out=buf61)
        buf62 = buf55; del buf55  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_10], Original ATen: [aten.mm]
        extern_kernels.mm(buf58, reinterpret_tensor(primals_3, (64, 256), (1, 64), 0), out=buf62)
        # Topologically Sorted Source Nodes: [lstm_cell_10], Original ATen: [aten._thnn_fused_lstm_cell]
        buf63 = torch.ops.aten._thnn_fused_lstm_cell.default(buf61, buf62, buf59, primals_4, primals_5)
        buf64 = buf63[0]
        buf65 = buf63[1]
        buf66 = buf63[2]
        del buf63
        buf67 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_11], Original ATen: [aten.mm]
        extern_kernels.mm(buf64, reinterpret_tensor(primals_6, (64, 256), (1, 64), 0), out=buf67)
        buf68 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_11], Original ATen: [aten.mm]
        extern_kernels.mm(buf64, reinterpret_tensor(primals_7, (64, 256), (1, 64), 0), out=buf68)
        # Topologically Sorted Source Nodes: [lstm_cell_11], Original ATen: [aten._thnn_fused_lstm_cell]
        buf69 = torch.ops.aten._thnn_fused_lstm_cell.default(buf67, buf68, buf65, primals_8, primals_9)
        buf70 = buf69[0]
        buf71 = buf69[1]
        buf72 = buf69[2]
        del buf69
        buf73 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_12], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 128), (128, 1), 768), reinterpret_tensor(primals_2, (128, 256), (1, 128), 0), out=buf73)
        buf74 = buf67; del buf67  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_12], Original ATen: [aten.mm]
        extern_kernels.mm(buf70, reinterpret_tensor(primals_3, (64, 256), (1, 64), 0), out=buf74)
        # Topologically Sorted Source Nodes: [lstm_cell_12], Original ATen: [aten._thnn_fused_lstm_cell]
        buf75 = torch.ops.aten._thnn_fused_lstm_cell.default(buf73, buf74, buf71, primals_4, primals_5)
        buf76 = buf75[0]
        buf77 = buf75[1]
        buf78 = buf75[2]
        del buf75
        buf79 = buf74; del buf74  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_13], Original ATen: [aten.mm]
        extern_kernels.mm(buf76, reinterpret_tensor(primals_6, (64, 256), (1, 64), 0), out=buf79)
        buf80 = buf73; del buf73  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_13], Original ATen: [aten.mm]
        extern_kernels.mm(buf76, reinterpret_tensor(primals_7, (64, 256), (1, 64), 0), out=buf80)
        # Topologically Sorted Source Nodes: [lstm_cell_13], Original ATen: [aten._thnn_fused_lstm_cell]
        buf81 = torch.ops.aten._thnn_fused_lstm_cell.default(buf79, buf80, buf77, primals_8, primals_9)
        buf82 = buf81[0]
        buf83 = buf81[1]
        buf84 = buf81[2]
        del buf81
        buf85 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_14], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 128), (128, 1), 896), reinterpret_tensor(primals_2, (128, 256), (1, 128), 0), out=buf85)
        buf86 = buf79; del buf79  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_14], Original ATen: [aten.mm]
        extern_kernels.mm(buf82, reinterpret_tensor(primals_3, (64, 256), (1, 64), 0), out=buf86)
        # Topologically Sorted Source Nodes: [lstm_cell_14], Original ATen: [aten._thnn_fused_lstm_cell]
        buf87 = torch.ops.aten._thnn_fused_lstm_cell.default(buf85, buf86, buf83, primals_4, primals_5)
        buf88 = buf87[0]
        buf89 = buf87[1]
        buf90 = buf87[2]
        del buf87
        buf91 = buf86; del buf86  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_15], Original ATen: [aten.mm]
        extern_kernels.mm(buf88, reinterpret_tensor(primals_6, (64, 256), (1, 64), 0), out=buf91)
        buf92 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_15], Original ATen: [aten.mm]
        extern_kernels.mm(buf88, reinterpret_tensor(primals_7, (64, 256), (1, 64), 0), out=buf92)
        # Topologically Sorted Source Nodes: [lstm_cell_15], Original ATen: [aten._thnn_fused_lstm_cell]
        buf93 = torch.ops.aten._thnn_fused_lstm_cell.default(buf91, buf92, buf89, primals_8, primals_9)
        buf94 = buf93[0]
        buf95 = buf93[1]
        buf96 = buf93[2]
        del buf93
        buf97 = buf92; del buf92  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_16], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 128), (128, 1), 1024), reinterpret_tensor(primals_2, (128, 256), (1, 128), 0), out=buf97)
        buf98 = buf91; del buf91  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_16], Original ATen: [aten.mm]
        extern_kernels.mm(buf94, reinterpret_tensor(primals_3, (64, 256), (1, 64), 0), out=buf98)
        # Topologically Sorted Source Nodes: [lstm_cell_16], Original ATen: [aten._thnn_fused_lstm_cell]
        buf99 = torch.ops.aten._thnn_fused_lstm_cell.default(buf97, buf98, buf95, primals_4, primals_5)
        buf100 = buf99[0]
        buf101 = buf99[1]
        buf102 = buf99[2]
        del buf99
        buf103 = buf98; del buf98  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_17], Original ATen: [aten.mm]
        extern_kernels.mm(buf100, reinterpret_tensor(primals_6, (64, 256), (1, 64), 0), out=buf103)
        buf104 = buf97; del buf97  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_17], Original ATen: [aten.mm]
        extern_kernels.mm(buf100, reinterpret_tensor(primals_7, (64, 256), (1, 64), 0), out=buf104)
        # Topologically Sorted Source Nodes: [lstm_cell_17], Original ATen: [aten._thnn_fused_lstm_cell]
        buf105 = torch.ops.aten._thnn_fused_lstm_cell.default(buf103, buf104, buf101, primals_8, primals_9)
        buf106 = buf105[0]
        buf107 = buf105[1]
        buf108 = buf105[2]
        del buf105
        buf109 = buf104; del buf104  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_18], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 128), (128, 1), 1152), reinterpret_tensor(primals_2, (128, 256), (1, 128), 0), out=buf109)
        del primals_2
        buf110 = buf103; del buf103  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_18], Original ATen: [aten.mm]
        extern_kernels.mm(buf106, reinterpret_tensor(primals_3, (64, 256), (1, 64), 0), out=buf110)
        # Topologically Sorted Source Nodes: [lstm_cell_18], Original ATen: [aten._thnn_fused_lstm_cell]
        buf111 = torch.ops.aten._thnn_fused_lstm_cell.default(buf109, buf110, buf107, primals_4, primals_5)
        del primals_4
        del primals_5
        buf112 = buf111[0]
        buf113 = buf111[1]
        buf114 = buf111[2]
        del buf111
        buf115 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_19], Original ATen: [aten.mm]
        extern_kernels.mm(buf112, reinterpret_tensor(primals_6, (64, 256), (1, 64), 0), out=buf115)
        buf116 = buf109; del buf109  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_19], Original ATen: [aten.mm]
        extern_kernels.mm(buf112, reinterpret_tensor(primals_7, (64, 256), (1, 64), 0), out=buf116)
        # Topologically Sorted Source Nodes: [lstm_cell_19], Original ATen: [aten._thnn_fused_lstm_cell]
        buf117 = torch.ops.aten._thnn_fused_lstm_cell.default(buf115, buf116, buf113, primals_8, primals_9)
        del buf115
        del buf116
        del primals_8
        del primals_9
        buf118 = buf117[0]
        buf119 = buf117[1]
        buf120 = buf117[2]
        del buf117
    buf122 = empty_strided_cpu((1, ), (1, ), torch.int64)
    # Topologically Sorted Source Nodes: [], Original ATen: []
    aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf122)
    buf123 = empty_strided_cpu((1, 8, 8), (64, 8, 1), torch.int64)
    cpp_fused_randint_1(buf122, buf123)
    del buf122
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf124 = empty_strided_cuda((1, 8, 8), (64, 8, 1), torch.int64)
        buf124.copy_(buf123, False)
        del buf123
        buf121 = empty_strided_cuda((1, ), (1, ), torch.float32)
        buf127 = empty_strided_cuda((), (), torch.float32)
        buf128 = empty_strided_cuda((1, 1, 8, 8), (64, 64, 8, 1), torch.bool)
        buf129 = empty_strided_cuda((1, 1, 8, 8), (64, 64, 8, 1), torch.float32)
        buf126 = empty_strided_cuda((), (), torch.float32)
        buf131 = buf127; del buf127  # reuse
        buf130 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [cosine_loss, nll_loss], Original ATen: [aten.mul, aten.sum, aten.add, aten.sqrt, aten.div, aten.zeros_like, aten.fill, aten.sub, aten.clamp_min, aten.eq, aten.where, aten.mean, aten.nll_loss2d_forward, aten.nll_loss2d_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clamp_min_div_eq_fill_mean_mul_nll_loss2d_backward_nll_loss2d_forward_sqrt_sub_sum_where_zeros_like_2[grid(1)](buf131, buf118, buf124, buf121, buf128, buf129, buf126, buf130, 1, 64, XBLOCK=1, num_warps=2, num_stages=1)
        del buf124
    return (buf130, buf131, buf0, reinterpret_tensor(primals_1, (1, 128), (1280, 1), 0), buf4, buf5, buf6, buf10, buf11, buf12, reinterpret_tensor(primals_1, (1, 128), (1280, 1), 128), buf16, buf17, buf18, buf22, buf23, buf24, reinterpret_tensor(primals_1, (1, 128), (1280, 1), 256), buf28, buf29, buf30, buf34, buf35, buf36, reinterpret_tensor(primals_1, (1, 128), (1280, 1), 384), buf40, buf41, buf42, buf46, buf47, buf48, reinterpret_tensor(primals_1, (1, 128), (1280, 1), 512), buf52, buf53, buf54, buf58, buf59, buf60, reinterpret_tensor(primals_1, (1, 128), (1280, 1), 640), buf64, buf65, buf66, buf70, buf71, buf72, reinterpret_tensor(primals_1, (1, 128), (1280, 1), 768), buf76, buf77, buf78, buf82, buf83, buf84, reinterpret_tensor(primals_1, (1, 128), (1280, 1), 896), buf88, buf89, buf90, buf94, buf95, buf96, reinterpret_tensor(primals_1, (1, 128), (1280, 1), 1024), buf100, buf101, buf102, buf106, buf107, buf108, reinterpret_tensor(primals_1, (1, 128), (1280, 1), 1152), buf112, buf113, buf114, buf118, buf119, buf120, buf121, buf126, buf128, buf129, primals_7, primals_6, primals_3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 10, 128), (1280, 128, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((256, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((256, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((256, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
