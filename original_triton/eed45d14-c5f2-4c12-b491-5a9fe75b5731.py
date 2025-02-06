# AOT ID: ['21_forward']
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
# Topologically Sorted Source Nodes: [h_t], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   h_t => full_default
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




# kernel path: /tmp/torchinductor_sahanp/i2/ci2qscjviy7jwwl5sidil226ftcws5aqwgnwmsosa2s4lkprtj6p.py
# Topologically Sorted Source Nodes: [randn_like, target_distribution, log_softmax, kl_loss], Original ATen: [aten.randn_like, aten._softmax, aten._log_softmax, aten.mul, aten.xlogy, aten.sub, aten.sum, aten.div, aten._log_softmax_backward_data]
# Source node to ATen node mapping:
#   kl_loss => div_1, eq, full_default_2, full_default_3, isnan, log_1, mul, mul_1, sub_3, sum_3, where, where_1
#   log_softmax => amax_1, exp_1, log, sub_1, sub_2, sum_2
#   randn_like => inductor_lookup_seed_default, inductor_random_default_2
#   target_distribution => amax, div, exp, sub, sum_1
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default_2 : [num_users=3] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 64], %inductor_lookup_seed_default, randn), kwargs = {})
#   %amax : [num_users=2] = call_function[target=torch.ops.aten.amax.default](args = (%inductor_random_default_2, [1], True), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%inductor_random_default_2, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %sum_1 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
#   %div : [num_users=5] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
#   %amax_1 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%getitem_147, [1], True), kwargs = {})
#   %sub_1 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%getitem_147, %amax_1), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_1,), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_1, [1], True), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_2,), kwargs = {})
#   %sub_2 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_1, %log), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %sub_2), kwargs = {})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%div, 0), kwargs = {})
#   %log_1 : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%div,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %log_1), kwargs = {})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default_2, %mul_1), kwargs = {})
#   %isnan : [num_users=1] = call_function[target=torch.ops.aten.isnan.default](args = (%div,), kwargs = {})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], nan), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%isnan, %full_default_3, %where), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_1, %mul), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%sub_3,), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_3, 1), kwargs = {})
#   %exp_2 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_2,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__log_softmax__log_softmax_backward_data__softmax_div_mul_randn_like_sub_sum_xlogy_1(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr2, out_ptr3, out_ptr5, load_seed_offset, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp3 = tl.load(in_ptr1 + (r0_0), None)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = r0_0
    tmp2 = tl.randn(tmp0, (tmp1).to(tl.uint32))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = triton_helpers.max2(tmp4, 1)[:, None]
    tmp7 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp9 = triton_helpers.max2(tmp7, 1)[:, None]
    tmp10 = tmp2 - tmp9
    tmp11 = tl_math.exp(tmp10)
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
    tmp14 = tl.sum(tmp12, 1)[:, None]
    tmp15 = tmp3 - tmp6
    tmp16 = tl_math.exp(tmp15)
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.sum(tmp17, 1)[:, None]
    tmp20 = tmp11 / tmp14
    tmp21 = libdevice.isnan(tmp20).to(tl.int1)
    tmp22 = 0.0
    tmp23 = tmp20 == tmp22
    tmp24 = tl_math.log(tmp20)
    tmp25 = tmp20 * tmp24
    tmp26 = tl.where(tmp23, tmp22, tmp25)
    tmp27 = float("nan")
    tmp28 = tl.where(tmp21, tmp27, tmp26)
    tmp29 = tl_math.log(tmp19)
    tmp30 = tmp15 - tmp29
    tmp31 = tmp20 * tmp30
    tmp32 = tmp28 - tmp31
    tmp33 = tl.broadcast_to(tmp32, [XBLOCK, R0_BLOCK])
    tmp35 = tl.sum(tmp33, 1)[:, None]
    tmp36 = tl_math.exp(tmp30)
    tmp37 = 1.0
    tmp38 = tmp35 * tmp37
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp2, None)
    tl.store(out_ptr5 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp36, None)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp38, None)
    tl.store(out_ptr2 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp9, None)
    tl.store(out_ptr3 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp14, None)




#include "/tmp/torchinductor_sahanp/3b/c3bi5gk6mslf6u4iaqafhxm64z6u65e3eain4xlary5blqnvv6xx.h"
extern "C"  void kernel(const int64_t* in_ptr0,
                       float* out_ptr0)
{
    {
        {
            {
                auto tmp0 = in_ptr0[static_cast<int64_t>(0L)];
                auto tmp1 = static_cast<int32_t>(0);
                auto tmp2 = randn_cpu(tmp0, tmp1);
                out_ptr0[static_cast<int64_t>(0L)] = tmp2;
            }
        }
    }
}
''')


#include "/tmp/torchinductor_sahanp/3b/c3bi5gk6mslf6u4iaqafhxm64z6u65e3eain4xlary5blqnvv6xx.h"
extern "C"  void kernel(const int64_t* in_ptr0,
                       float* out_ptr0)
{
    {
        {
            {
                auto tmp0 = in_ptr0[static_cast<int64_t>(1L)];
                auto tmp1 = static_cast<int32_t>(0);
                auto tmp2 = randn_cpu(tmp0, tmp1);
                out_ptr0[static_cast<int64_t>(0L)] = tmp2;
            }
        }
    }
}
''')


#include "/tmp/torchinductor_sahanp/3b/c3bi5gk6mslf6u4iaqafhxm64z6u65e3eain4xlary5blqnvv6xx.h"
extern "C"  void kernel(const int64_t* in_ptr0,
                       int64_t* out_ptr0)
{
    {
        {
            {
                auto tmp0 = in_ptr0[static_cast<int64_t>(2L)];
                auto tmp1 = static_cast<int32_t>(0);
                auto tmp2 = static_cast<int64_t>(0);
                auto tmp3 = static_cast<int64_t>(2);
                auto tmp4 = randint64_cpu(tmp0, tmp1, tmp2, tmp3);
                out_ptr0[static_cast<int64_t>(0L)] = tmp4;
            }
        }
    }
}
''')


# kernel path: /tmp/torchinductor_sahanp/lp/clpmszv3ve7evglheodfflwctr65oyyam2rynxxtolhwnhgpntgs.py
# Topologically Sorted Source Nodes: [float_1, mul, target, margin_loss], Original ATen: [aten._to_copy, aten.mul, aten.sub, aten.neg, aten.add, aten.clamp_min, aten.mean]
# Source node to ATen node mapping:
#   float_1 => convert_element_type_5
#   margin_loss => add, clamp_min, mean, mul_3, neg, sub_5
#   mul => mul_2
#   target => sub_4
# Graph fragment:
#   %convert_element_type_5 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%device_put_4, torch.float32), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_5, 2), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_2, 1), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%device_put_2, %device_put_3), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%sub_4,), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg, %sub_5), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_3, 1.0), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add, 0), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%clamp_min,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_add_clamp_min_mean_mul_neg_sub_5(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp8 = tl.load(in_out_ptr0 + (0))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp10 = tl.load(in_ptr1 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 2.0
    tmp4 = tmp2 * tmp3
    tmp5 = 1.0
    tmp6 = tmp4 - tmp5
    tmp7 = -tmp6
    tmp12 = tmp9 - tmp11
    tmp13 = tmp7 * tmp12
    tmp14 = tmp13 + tmp5
    tmp15 = 0.0
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = tmp16 / tmp5
    tl.store(in_out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp17, None)







def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21 = args
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
    assert_size_stride(primals_10, (256, 64), (64, 1))
    assert_size_stride(primals_11, (256, 64), (64, 1))
    assert_size_stride(primals_12, (256, ), (1, ))
    assert_size_stride(primals_13, (256, ), (1, ))
    assert_size_stride(primals_14, (256, 64), (64, 1))
    assert_size_stride(primals_15, (256, 64), (64, 1))
    assert_size_stride(primals_16, (256, ), (1, ))
    assert_size_stride(primals_17, (256, ), (1, ))
    assert_size_stride(primals_18, (256, 64), (64, 1))
    assert_size_stride(primals_19, (256, 64), (64, 1))
    assert_size_stride(primals_20, (256, ), (1, ))
    assert_size_stride(primals_21, (256, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [h_t], Original ATen: [aten._to_copy]
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
        extern_kernels.mm(buf10, reinterpret_tensor(primals_10, (64, 256), (1, 64), 0), out=buf13)
        buf14 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_2], Original ATen: [aten.mm]
        extern_kernels.mm(buf10, reinterpret_tensor(primals_11, (64, 256), (1, 64), 0), out=buf14)
        # Topologically Sorted Source Nodes: [lstm_cell_2], Original ATen: [aten._thnn_fused_lstm_cell]
        buf15 = torch.ops.aten._thnn_fused_lstm_cell.default(buf13, buf14, buf11, primals_12, primals_13)
        buf16 = buf15[0]
        buf17 = buf15[1]
        buf18 = buf15[2]
        del buf15
        buf19 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_3], Original ATen: [aten.mm]
        extern_kernels.mm(buf16, reinterpret_tensor(primals_14, (64, 256), (1, 64), 0), out=buf19)
        buf20 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_3], Original ATen: [aten.mm]
        extern_kernels.mm(buf16, reinterpret_tensor(primals_15, (64, 256), (1, 64), 0), out=buf20)
        # Topologically Sorted Source Nodes: [lstm_cell_3], Original ATen: [aten._thnn_fused_lstm_cell]
        buf21 = torch.ops.aten._thnn_fused_lstm_cell.default(buf19, buf20, buf17, primals_16, primals_17)
        buf22 = buf21[0]
        buf23 = buf21[1]
        buf24 = buf21[2]
        del buf21
        buf25 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_4], Original ATen: [aten.mm]
        extern_kernels.mm(buf22, reinterpret_tensor(primals_18, (64, 256), (1, 64), 0), out=buf25)
        buf26 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_4], Original ATen: [aten.mm]
        extern_kernels.mm(buf22, reinterpret_tensor(primals_19, (64, 256), (1, 64), 0), out=buf26)
        # Topologically Sorted Source Nodes: [lstm_cell_4], Original ATen: [aten._thnn_fused_lstm_cell]
        buf27 = torch.ops.aten._thnn_fused_lstm_cell.default(buf25, buf26, buf23, primals_20, primals_21)
        buf28 = buf27[0]
        buf29 = buf27[1]
        buf30 = buf27[2]
        del buf27
        buf31 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 128), (128, 1), 128), reinterpret_tensor(primals_2, (128, 256), (1, 128), 0), out=buf31)
        buf32 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_5], Original ATen: [aten.mm]
        extern_kernels.mm(buf28, reinterpret_tensor(primals_3, (64, 256), (1, 64), 0), out=buf32)
        # Topologically Sorted Source Nodes: [lstm_cell_5], Original ATen: [aten._thnn_fused_lstm_cell]
        buf33 = torch.ops.aten._thnn_fused_lstm_cell.default(buf31, buf32, buf29, primals_4, primals_5)
        buf34 = buf33[0]
        buf35 = buf33[1]
        buf36 = buf33[2]
        del buf33
        buf37 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_6], Original ATen: [aten.mm]
        extern_kernels.mm(buf34, reinterpret_tensor(primals_6, (64, 256), (1, 64), 0), out=buf37)
        buf38 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_6], Original ATen: [aten.mm]
        extern_kernels.mm(buf34, reinterpret_tensor(primals_7, (64, 256), (1, 64), 0), out=buf38)
        # Topologically Sorted Source Nodes: [lstm_cell_6], Original ATen: [aten._thnn_fused_lstm_cell]
        buf39 = torch.ops.aten._thnn_fused_lstm_cell.default(buf37, buf38, buf35, primals_8, primals_9)
        buf40 = buf39[0]
        buf41 = buf39[1]
        buf42 = buf39[2]
        del buf39
        buf43 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_7], Original ATen: [aten.mm]
        extern_kernels.mm(buf40, reinterpret_tensor(primals_10, (64, 256), (1, 64), 0), out=buf43)
        buf44 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_7], Original ATen: [aten.mm]
        extern_kernels.mm(buf40, reinterpret_tensor(primals_11, (64, 256), (1, 64), 0), out=buf44)
        # Topologically Sorted Source Nodes: [lstm_cell_7], Original ATen: [aten._thnn_fused_lstm_cell]
        buf45 = torch.ops.aten._thnn_fused_lstm_cell.default(buf43, buf44, buf41, primals_12, primals_13)
        buf46 = buf45[0]
        buf47 = buf45[1]
        buf48 = buf45[2]
        del buf45
        buf49 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_8], Original ATen: [aten.mm]
        extern_kernels.mm(buf46, reinterpret_tensor(primals_14, (64, 256), (1, 64), 0), out=buf49)
        buf50 = buf43; del buf43  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_8], Original ATen: [aten.mm]
        extern_kernels.mm(buf46, reinterpret_tensor(primals_15, (64, 256), (1, 64), 0), out=buf50)
        # Topologically Sorted Source Nodes: [lstm_cell_8], Original ATen: [aten._thnn_fused_lstm_cell]
        buf51 = torch.ops.aten._thnn_fused_lstm_cell.default(buf49, buf50, buf47, primals_16, primals_17)
        buf52 = buf51[0]
        buf53 = buf51[1]
        buf54 = buf51[2]
        del buf51
        buf55 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_9], Original ATen: [aten.mm]
        extern_kernels.mm(buf52, reinterpret_tensor(primals_18, (64, 256), (1, 64), 0), out=buf55)
        buf56 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_9], Original ATen: [aten.mm]
        extern_kernels.mm(buf52, reinterpret_tensor(primals_19, (64, 256), (1, 64), 0), out=buf56)
        # Topologically Sorted Source Nodes: [lstm_cell_9], Original ATen: [aten._thnn_fused_lstm_cell]
        buf57 = torch.ops.aten._thnn_fused_lstm_cell.default(buf55, buf56, buf53, primals_20, primals_21)
        buf58 = buf57[0]
        buf59 = buf57[1]
        buf60 = buf57[2]
        del buf57
        buf61 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_10], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 128), (128, 1), 256), reinterpret_tensor(primals_2, (128, 256), (1, 128), 0), out=buf61)
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
        extern_kernels.mm(buf70, reinterpret_tensor(primals_10, (64, 256), (1, 64), 0), out=buf73)
        buf74 = buf67; del buf67  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_12], Original ATen: [aten.mm]
        extern_kernels.mm(buf70, reinterpret_tensor(primals_11, (64, 256), (1, 64), 0), out=buf74)
        # Topologically Sorted Source Nodes: [lstm_cell_12], Original ATen: [aten._thnn_fused_lstm_cell]
        buf75 = torch.ops.aten._thnn_fused_lstm_cell.default(buf73, buf74, buf71, primals_12, primals_13)
        buf76 = buf75[0]
        buf77 = buf75[1]
        buf78 = buf75[2]
        del buf75
        buf79 = buf74; del buf74  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_13], Original ATen: [aten.mm]
        extern_kernels.mm(buf76, reinterpret_tensor(primals_14, (64, 256), (1, 64), 0), out=buf79)
        buf80 = buf73; del buf73  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_13], Original ATen: [aten.mm]
        extern_kernels.mm(buf76, reinterpret_tensor(primals_15, (64, 256), (1, 64), 0), out=buf80)
        # Topologically Sorted Source Nodes: [lstm_cell_13], Original ATen: [aten._thnn_fused_lstm_cell]
        buf81 = torch.ops.aten._thnn_fused_lstm_cell.default(buf79, buf80, buf77, primals_16, primals_17)
        buf82 = buf81[0]
        buf83 = buf81[1]
        buf84 = buf81[2]
        del buf81
        buf85 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_14], Original ATen: [aten.mm]
        extern_kernels.mm(buf82, reinterpret_tensor(primals_18, (64, 256), (1, 64), 0), out=buf85)
        buf86 = buf79; del buf79  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_14], Original ATen: [aten.mm]
        extern_kernels.mm(buf82, reinterpret_tensor(primals_19, (64, 256), (1, 64), 0), out=buf86)
        # Topologically Sorted Source Nodes: [lstm_cell_14], Original ATen: [aten._thnn_fused_lstm_cell]
        buf87 = torch.ops.aten._thnn_fused_lstm_cell.default(buf85, buf86, buf83, primals_20, primals_21)
        buf88 = buf87[0]
        buf89 = buf87[1]
        buf90 = buf87[2]
        del buf87
        buf91 = buf86; del buf86  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_15], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 128), (128, 1), 384), reinterpret_tensor(primals_2, (128, 256), (1, 128), 0), out=buf91)
        buf92 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_15], Original ATen: [aten.mm]
        extern_kernels.mm(buf88, reinterpret_tensor(primals_3, (64, 256), (1, 64), 0), out=buf92)
        # Topologically Sorted Source Nodes: [lstm_cell_15], Original ATen: [aten._thnn_fused_lstm_cell]
        buf93 = torch.ops.aten._thnn_fused_lstm_cell.default(buf91, buf92, buf89, primals_4, primals_5)
        buf94 = buf93[0]
        buf95 = buf93[1]
        buf96 = buf93[2]
        del buf93
        buf97 = buf92; del buf92  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_16], Original ATen: [aten.mm]
        extern_kernels.mm(buf94, reinterpret_tensor(primals_6, (64, 256), (1, 64), 0), out=buf97)
        buf98 = buf91; del buf91  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_16], Original ATen: [aten.mm]
        extern_kernels.mm(buf94, reinterpret_tensor(primals_7, (64, 256), (1, 64), 0), out=buf98)
        # Topologically Sorted Source Nodes: [lstm_cell_16], Original ATen: [aten._thnn_fused_lstm_cell]
        buf99 = torch.ops.aten._thnn_fused_lstm_cell.default(buf97, buf98, buf95, primals_8, primals_9)
        buf100 = buf99[0]
        buf101 = buf99[1]
        buf102 = buf99[2]
        del buf99
        buf103 = buf98; del buf98  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_17], Original ATen: [aten.mm]
        extern_kernels.mm(buf100, reinterpret_tensor(primals_10, (64, 256), (1, 64), 0), out=buf103)
        buf104 = buf97; del buf97  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_17], Original ATen: [aten.mm]
        extern_kernels.mm(buf100, reinterpret_tensor(primals_11, (64, 256), (1, 64), 0), out=buf104)
        # Topologically Sorted Source Nodes: [lstm_cell_17], Original ATen: [aten._thnn_fused_lstm_cell]
        buf105 = torch.ops.aten._thnn_fused_lstm_cell.default(buf103, buf104, buf101, primals_12, primals_13)
        buf106 = buf105[0]
        buf107 = buf105[1]
        buf108 = buf105[2]
        del buf105
        buf109 = buf104; del buf104  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_18], Original ATen: [aten.mm]
        extern_kernels.mm(buf106, reinterpret_tensor(primals_14, (64, 256), (1, 64), 0), out=buf109)
        buf110 = buf103; del buf103  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_18], Original ATen: [aten.mm]
        extern_kernels.mm(buf106, reinterpret_tensor(primals_15, (64, 256), (1, 64), 0), out=buf110)
        # Topologically Sorted Source Nodes: [lstm_cell_18], Original ATen: [aten._thnn_fused_lstm_cell]
        buf111 = torch.ops.aten._thnn_fused_lstm_cell.default(buf109, buf110, buf107, primals_16, primals_17)
        buf112 = buf111[0]
        buf113 = buf111[1]
        buf114 = buf111[2]
        del buf111
        buf115 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_19], Original ATen: [aten.mm]
        extern_kernels.mm(buf112, reinterpret_tensor(primals_18, (64, 256), (1, 64), 0), out=buf115)
        buf116 = buf109; del buf109  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_19], Original ATen: [aten.mm]
        extern_kernels.mm(buf112, reinterpret_tensor(primals_19, (64, 256), (1, 64), 0), out=buf116)
        # Topologically Sorted Source Nodes: [lstm_cell_19], Original ATen: [aten._thnn_fused_lstm_cell]
        buf117 = torch.ops.aten._thnn_fused_lstm_cell.default(buf115, buf116, buf113, primals_20, primals_21)
        buf118 = buf117[0]
        buf119 = buf117[1]
        buf120 = buf117[2]
        del buf117
        buf121 = buf116; del buf116  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_20], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 128), (128, 1), 512), reinterpret_tensor(primals_2, (128, 256), (1, 128), 0), out=buf121)
        buf122 = buf115; del buf115  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_20], Original ATen: [aten.mm]
        extern_kernels.mm(buf118, reinterpret_tensor(primals_3, (64, 256), (1, 64), 0), out=buf122)
        # Topologically Sorted Source Nodes: [lstm_cell_20], Original ATen: [aten._thnn_fused_lstm_cell]
        buf123 = torch.ops.aten._thnn_fused_lstm_cell.default(buf121, buf122, buf119, primals_4, primals_5)
        buf124 = buf123[0]
        buf125 = buf123[1]
        buf126 = buf123[2]
        del buf123
        buf127 = buf122; del buf122  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_21], Original ATen: [aten.mm]
        extern_kernels.mm(buf124, reinterpret_tensor(primals_6, (64, 256), (1, 64), 0), out=buf127)
        buf128 = buf121; del buf121  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_21], Original ATen: [aten.mm]
        extern_kernels.mm(buf124, reinterpret_tensor(primals_7, (64, 256), (1, 64), 0), out=buf128)
        # Topologically Sorted Source Nodes: [lstm_cell_21], Original ATen: [aten._thnn_fused_lstm_cell]
        buf129 = torch.ops.aten._thnn_fused_lstm_cell.default(buf127, buf128, buf125, primals_8, primals_9)
        buf130 = buf129[0]
        buf131 = buf129[1]
        buf132 = buf129[2]
        del buf129
        buf133 = buf128; del buf128  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_22], Original ATen: [aten.mm]
        extern_kernels.mm(buf130, reinterpret_tensor(primals_10, (64, 256), (1, 64), 0), out=buf133)
        buf134 = buf127; del buf127  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_22], Original ATen: [aten.mm]
        extern_kernels.mm(buf130, reinterpret_tensor(primals_11, (64, 256), (1, 64), 0), out=buf134)
        # Topologically Sorted Source Nodes: [lstm_cell_22], Original ATen: [aten._thnn_fused_lstm_cell]
        buf135 = torch.ops.aten._thnn_fused_lstm_cell.default(buf133, buf134, buf131, primals_12, primals_13)
        buf136 = buf135[0]
        buf137 = buf135[1]
        buf138 = buf135[2]
        del buf135
        buf139 = buf134; del buf134  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_23], Original ATen: [aten.mm]
        extern_kernels.mm(buf136, reinterpret_tensor(primals_14, (64, 256), (1, 64), 0), out=buf139)
        buf140 = buf133; del buf133  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_23], Original ATen: [aten.mm]
        extern_kernels.mm(buf136, reinterpret_tensor(primals_15, (64, 256), (1, 64), 0), out=buf140)
        # Topologically Sorted Source Nodes: [lstm_cell_23], Original ATen: [aten._thnn_fused_lstm_cell]
        buf141 = torch.ops.aten._thnn_fused_lstm_cell.default(buf139, buf140, buf137, primals_16, primals_17)
        buf142 = buf141[0]
        buf143 = buf141[1]
        buf144 = buf141[2]
        del buf141
        buf145 = buf140; del buf140  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_24], Original ATen: [aten.mm]
        extern_kernels.mm(buf142, reinterpret_tensor(primals_18, (64, 256), (1, 64), 0), out=buf145)
        buf146 = buf139; del buf139  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_24], Original ATen: [aten.mm]
        extern_kernels.mm(buf142, reinterpret_tensor(primals_19, (64, 256), (1, 64), 0), out=buf146)
        # Topologically Sorted Source Nodes: [lstm_cell_24], Original ATen: [aten._thnn_fused_lstm_cell]
        buf147 = torch.ops.aten._thnn_fused_lstm_cell.default(buf145, buf146, buf143, primals_20, primals_21)
        buf148 = buf147[0]
        buf149 = buf147[1]
        buf150 = buf147[2]
        del buf147
        buf151 = buf146; del buf146  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_25], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 128), (128, 1), 640), reinterpret_tensor(primals_2, (128, 256), (1, 128), 0), out=buf151)
        buf152 = buf145; del buf145  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_25], Original ATen: [aten.mm]
        extern_kernels.mm(buf148, reinterpret_tensor(primals_3, (64, 256), (1, 64), 0), out=buf152)
        # Topologically Sorted Source Nodes: [lstm_cell_25], Original ATen: [aten._thnn_fused_lstm_cell]
        buf153 = torch.ops.aten._thnn_fused_lstm_cell.default(buf151, buf152, buf149, primals_4, primals_5)
        buf154 = buf153[0]
        buf155 = buf153[1]
        buf156 = buf153[2]
        del buf153
        buf157 = buf152; del buf152  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_26], Original ATen: [aten.mm]
        extern_kernels.mm(buf154, reinterpret_tensor(primals_6, (64, 256), (1, 64), 0), out=buf157)
        buf158 = buf151; del buf151  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_26], Original ATen: [aten.mm]
        extern_kernels.mm(buf154, reinterpret_tensor(primals_7, (64, 256), (1, 64), 0), out=buf158)
        # Topologically Sorted Source Nodes: [lstm_cell_26], Original ATen: [aten._thnn_fused_lstm_cell]
        buf159 = torch.ops.aten._thnn_fused_lstm_cell.default(buf157, buf158, buf155, primals_8, primals_9)
        buf160 = buf159[0]
        buf161 = buf159[1]
        buf162 = buf159[2]
        del buf159
        buf163 = buf158; del buf158  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_27], Original ATen: [aten.mm]
        extern_kernels.mm(buf160, reinterpret_tensor(primals_10, (64, 256), (1, 64), 0), out=buf163)
        buf164 = buf157; del buf157  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_27], Original ATen: [aten.mm]
        extern_kernels.mm(buf160, reinterpret_tensor(primals_11, (64, 256), (1, 64), 0), out=buf164)
        # Topologically Sorted Source Nodes: [lstm_cell_27], Original ATen: [aten._thnn_fused_lstm_cell]
        buf165 = torch.ops.aten._thnn_fused_lstm_cell.default(buf163, buf164, buf161, primals_12, primals_13)
        buf166 = buf165[0]
        buf167 = buf165[1]
        buf168 = buf165[2]
        del buf165
        buf169 = buf164; del buf164  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_28], Original ATen: [aten.mm]
        extern_kernels.mm(buf166, reinterpret_tensor(primals_14, (64, 256), (1, 64), 0), out=buf169)
        buf170 = buf163; del buf163  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_28], Original ATen: [aten.mm]
        extern_kernels.mm(buf166, reinterpret_tensor(primals_15, (64, 256), (1, 64), 0), out=buf170)
        # Topologically Sorted Source Nodes: [lstm_cell_28], Original ATen: [aten._thnn_fused_lstm_cell]
        buf171 = torch.ops.aten._thnn_fused_lstm_cell.default(buf169, buf170, buf167, primals_16, primals_17)
        buf172 = buf171[0]
        buf173 = buf171[1]
        buf174 = buf171[2]
        del buf171
        buf175 = buf170; del buf170  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_29], Original ATen: [aten.mm]
        extern_kernels.mm(buf172, reinterpret_tensor(primals_18, (64, 256), (1, 64), 0), out=buf175)
        buf176 = buf169; del buf169  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_29], Original ATen: [aten.mm]
        extern_kernels.mm(buf172, reinterpret_tensor(primals_19, (64, 256), (1, 64), 0), out=buf176)
        # Topologically Sorted Source Nodes: [lstm_cell_29], Original ATen: [aten._thnn_fused_lstm_cell]
        buf177 = torch.ops.aten._thnn_fused_lstm_cell.default(buf175, buf176, buf173, primals_20, primals_21)
        buf178 = buf177[0]
        buf179 = buf177[1]
        buf180 = buf177[2]
        del buf177
        buf181 = buf176; del buf176  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_30], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 128), (128, 1), 768), reinterpret_tensor(primals_2, (128, 256), (1, 128), 0), out=buf181)
        buf182 = buf175; del buf175  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_30], Original ATen: [aten.mm]
        extern_kernels.mm(buf178, reinterpret_tensor(primals_3, (64, 256), (1, 64), 0), out=buf182)
        # Topologically Sorted Source Nodes: [lstm_cell_30], Original ATen: [aten._thnn_fused_lstm_cell]
        buf183 = torch.ops.aten._thnn_fused_lstm_cell.default(buf181, buf182, buf179, primals_4, primals_5)
        buf184 = buf183[0]
        buf185 = buf183[1]
        buf186 = buf183[2]
        del buf183
        buf187 = buf182; del buf182  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_31], Original ATen: [aten.mm]
        extern_kernels.mm(buf184, reinterpret_tensor(primals_6, (64, 256), (1, 64), 0), out=buf187)
        buf188 = buf181; del buf181  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_31], Original ATen: [aten.mm]
        extern_kernels.mm(buf184, reinterpret_tensor(primals_7, (64, 256), (1, 64), 0), out=buf188)
        # Topologically Sorted Source Nodes: [lstm_cell_31], Original ATen: [aten._thnn_fused_lstm_cell]
        buf189 = torch.ops.aten._thnn_fused_lstm_cell.default(buf187, buf188, buf185, primals_8, primals_9)
        buf190 = buf189[0]
        buf191 = buf189[1]
        buf192 = buf189[2]
        del buf189
        buf193 = buf188; del buf188  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_32], Original ATen: [aten.mm]
        extern_kernels.mm(buf190, reinterpret_tensor(primals_10, (64, 256), (1, 64), 0), out=buf193)
        buf194 = buf187; del buf187  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_32], Original ATen: [aten.mm]
        extern_kernels.mm(buf190, reinterpret_tensor(primals_11, (64, 256), (1, 64), 0), out=buf194)
        # Topologically Sorted Source Nodes: [lstm_cell_32], Original ATen: [aten._thnn_fused_lstm_cell]
        buf195 = torch.ops.aten._thnn_fused_lstm_cell.default(buf193, buf194, buf191, primals_12, primals_13)
        buf196 = buf195[0]
        buf197 = buf195[1]
        buf198 = buf195[2]
        del buf195
        buf199 = buf194; del buf194  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_33], Original ATen: [aten.mm]
        extern_kernels.mm(buf196, reinterpret_tensor(primals_14, (64, 256), (1, 64), 0), out=buf199)
        buf200 = buf193; del buf193  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_33], Original ATen: [aten.mm]
        extern_kernels.mm(buf196, reinterpret_tensor(primals_15, (64, 256), (1, 64), 0), out=buf200)
        # Topologically Sorted Source Nodes: [lstm_cell_33], Original ATen: [aten._thnn_fused_lstm_cell]
        buf201 = torch.ops.aten._thnn_fused_lstm_cell.default(buf199, buf200, buf197, primals_16, primals_17)
        buf202 = buf201[0]
        buf203 = buf201[1]
        buf204 = buf201[2]
        del buf201
        buf205 = buf200; del buf200  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_34], Original ATen: [aten.mm]
        extern_kernels.mm(buf202, reinterpret_tensor(primals_18, (64, 256), (1, 64), 0), out=buf205)
        buf206 = buf199; del buf199  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_34], Original ATen: [aten.mm]
        extern_kernels.mm(buf202, reinterpret_tensor(primals_19, (64, 256), (1, 64), 0), out=buf206)
        # Topologically Sorted Source Nodes: [lstm_cell_34], Original ATen: [aten._thnn_fused_lstm_cell]
        buf207 = torch.ops.aten._thnn_fused_lstm_cell.default(buf205, buf206, buf203, primals_20, primals_21)
        buf208 = buf207[0]
        buf209 = buf207[1]
        buf210 = buf207[2]
        del buf207
        buf211 = buf206; del buf206  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_35], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 128), (128, 1), 896), reinterpret_tensor(primals_2, (128, 256), (1, 128), 0), out=buf211)
        buf212 = buf205; del buf205  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_35], Original ATen: [aten.mm]
        extern_kernels.mm(buf208, reinterpret_tensor(primals_3, (64, 256), (1, 64), 0), out=buf212)
        # Topologically Sorted Source Nodes: [lstm_cell_35], Original ATen: [aten._thnn_fused_lstm_cell]
        buf213 = torch.ops.aten._thnn_fused_lstm_cell.default(buf211, buf212, buf209, primals_4, primals_5)
        buf214 = buf213[0]
        buf215 = buf213[1]
        buf216 = buf213[2]
        del buf213
        buf217 = buf212; del buf212  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_36], Original ATen: [aten.mm]
        extern_kernels.mm(buf214, reinterpret_tensor(primals_6, (64, 256), (1, 64), 0), out=buf217)
        buf218 = buf211; del buf211  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_36], Original ATen: [aten.mm]
        extern_kernels.mm(buf214, reinterpret_tensor(primals_7, (64, 256), (1, 64), 0), out=buf218)
        # Topologically Sorted Source Nodes: [lstm_cell_36], Original ATen: [aten._thnn_fused_lstm_cell]
        buf219 = torch.ops.aten._thnn_fused_lstm_cell.default(buf217, buf218, buf215, primals_8, primals_9)
        buf220 = buf219[0]
        buf221 = buf219[1]
        buf222 = buf219[2]
        del buf219
        buf223 = buf218; del buf218  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_37], Original ATen: [aten.mm]
        extern_kernels.mm(buf220, reinterpret_tensor(primals_10, (64, 256), (1, 64), 0), out=buf223)
        buf224 = buf217; del buf217  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_37], Original ATen: [aten.mm]
        extern_kernels.mm(buf220, reinterpret_tensor(primals_11, (64, 256), (1, 64), 0), out=buf224)
        # Topologically Sorted Source Nodes: [lstm_cell_37], Original ATen: [aten._thnn_fused_lstm_cell]
        buf225 = torch.ops.aten._thnn_fused_lstm_cell.default(buf223, buf224, buf221, primals_12, primals_13)
        buf226 = buf225[0]
        buf227 = buf225[1]
        buf228 = buf225[2]
        del buf225
        buf229 = buf224; del buf224  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_38], Original ATen: [aten.mm]
        extern_kernels.mm(buf226, reinterpret_tensor(primals_14, (64, 256), (1, 64), 0), out=buf229)
        buf230 = buf223; del buf223  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_38], Original ATen: [aten.mm]
        extern_kernels.mm(buf226, reinterpret_tensor(primals_15, (64, 256), (1, 64), 0), out=buf230)
        # Topologically Sorted Source Nodes: [lstm_cell_38], Original ATen: [aten._thnn_fused_lstm_cell]
        buf231 = torch.ops.aten._thnn_fused_lstm_cell.default(buf229, buf230, buf227, primals_16, primals_17)
        buf232 = buf231[0]
        buf233 = buf231[1]
        buf234 = buf231[2]
        del buf231
        buf235 = buf230; del buf230  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_39], Original ATen: [aten.mm]
        extern_kernels.mm(buf232, reinterpret_tensor(primals_18, (64, 256), (1, 64), 0), out=buf235)
        buf236 = buf229; del buf229  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_39], Original ATen: [aten.mm]
        extern_kernels.mm(buf232, reinterpret_tensor(primals_19, (64, 256), (1, 64), 0), out=buf236)
        # Topologically Sorted Source Nodes: [lstm_cell_39], Original ATen: [aten._thnn_fused_lstm_cell]
        buf237 = torch.ops.aten._thnn_fused_lstm_cell.default(buf235, buf236, buf233, primals_20, primals_21)
        buf238 = buf237[0]
        buf239 = buf237[1]
        buf240 = buf237[2]
        del buf237
        buf241 = buf236; del buf236  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_40], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 128), (128, 1), 1024), reinterpret_tensor(primals_2, (128, 256), (1, 128), 0), out=buf241)
        buf242 = buf235; del buf235  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_40], Original ATen: [aten.mm]
        extern_kernels.mm(buf238, reinterpret_tensor(primals_3, (64, 256), (1, 64), 0), out=buf242)
        # Topologically Sorted Source Nodes: [lstm_cell_40], Original ATen: [aten._thnn_fused_lstm_cell]
        buf243 = torch.ops.aten._thnn_fused_lstm_cell.default(buf241, buf242, buf239, primals_4, primals_5)
        buf244 = buf243[0]
        buf245 = buf243[1]
        buf246 = buf243[2]
        del buf243
        buf247 = buf242; del buf242  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_41], Original ATen: [aten.mm]
        extern_kernels.mm(buf244, reinterpret_tensor(primals_6, (64, 256), (1, 64), 0), out=buf247)
        buf248 = buf241; del buf241  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_41], Original ATen: [aten.mm]
        extern_kernels.mm(buf244, reinterpret_tensor(primals_7, (64, 256), (1, 64), 0), out=buf248)
        # Topologically Sorted Source Nodes: [lstm_cell_41], Original ATen: [aten._thnn_fused_lstm_cell]
        buf249 = torch.ops.aten._thnn_fused_lstm_cell.default(buf247, buf248, buf245, primals_8, primals_9)
        buf250 = buf249[0]
        buf251 = buf249[1]
        buf252 = buf249[2]
        del buf249
        buf253 = buf248; del buf248  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_42], Original ATen: [aten.mm]
        extern_kernels.mm(buf250, reinterpret_tensor(primals_10, (64, 256), (1, 64), 0), out=buf253)
        buf254 = buf247; del buf247  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_42], Original ATen: [aten.mm]
        extern_kernels.mm(buf250, reinterpret_tensor(primals_11, (64, 256), (1, 64), 0), out=buf254)
        # Topologically Sorted Source Nodes: [lstm_cell_42], Original ATen: [aten._thnn_fused_lstm_cell]
        buf255 = torch.ops.aten._thnn_fused_lstm_cell.default(buf253, buf254, buf251, primals_12, primals_13)
        buf256 = buf255[0]
        buf257 = buf255[1]
        buf258 = buf255[2]
        del buf255
        buf259 = buf254; del buf254  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_43], Original ATen: [aten.mm]
        extern_kernels.mm(buf256, reinterpret_tensor(primals_14, (64, 256), (1, 64), 0), out=buf259)
        buf260 = buf253; del buf253  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_43], Original ATen: [aten.mm]
        extern_kernels.mm(buf256, reinterpret_tensor(primals_15, (64, 256), (1, 64), 0), out=buf260)
        # Topologically Sorted Source Nodes: [lstm_cell_43], Original ATen: [aten._thnn_fused_lstm_cell]
        buf261 = torch.ops.aten._thnn_fused_lstm_cell.default(buf259, buf260, buf257, primals_16, primals_17)
        buf262 = buf261[0]
        buf263 = buf261[1]
        buf264 = buf261[2]
        del buf261
        buf265 = buf260; del buf260  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_44], Original ATen: [aten.mm]
        extern_kernels.mm(buf262, reinterpret_tensor(primals_18, (64, 256), (1, 64), 0), out=buf265)
        buf266 = buf259; del buf259  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_44], Original ATen: [aten.mm]
        extern_kernels.mm(buf262, reinterpret_tensor(primals_19, (64, 256), (1, 64), 0), out=buf266)
        # Topologically Sorted Source Nodes: [lstm_cell_44], Original ATen: [aten._thnn_fused_lstm_cell]
        buf267 = torch.ops.aten._thnn_fused_lstm_cell.default(buf265, buf266, buf263, primals_20, primals_21)
        buf268 = buf267[0]
        buf269 = buf267[1]
        buf270 = buf267[2]
        del buf267
        buf271 = buf266; del buf266  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_45], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 128), (128, 1), 1152), reinterpret_tensor(primals_2, (128, 256), (1, 128), 0), out=buf271)
        del primals_2
        buf272 = buf265; del buf265  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_45], Original ATen: [aten.mm]
        extern_kernels.mm(buf268, reinterpret_tensor(primals_3, (64, 256), (1, 64), 0), out=buf272)
        # Topologically Sorted Source Nodes: [lstm_cell_45], Original ATen: [aten._thnn_fused_lstm_cell]
        buf273 = torch.ops.aten._thnn_fused_lstm_cell.default(buf271, buf272, buf269, primals_4, primals_5)
        del primals_4
        del primals_5
        buf274 = buf273[0]
        buf275 = buf273[1]
        buf276 = buf273[2]
        del buf273
        buf277 = buf272; del buf272  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_46], Original ATen: [aten.mm]
        extern_kernels.mm(buf274, reinterpret_tensor(primals_6, (64, 256), (1, 64), 0), out=buf277)
        buf278 = buf271; del buf271  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_46], Original ATen: [aten.mm]
        extern_kernels.mm(buf274, reinterpret_tensor(primals_7, (64, 256), (1, 64), 0), out=buf278)
        # Topologically Sorted Source Nodes: [lstm_cell_46], Original ATen: [aten._thnn_fused_lstm_cell]
        buf279 = torch.ops.aten._thnn_fused_lstm_cell.default(buf277, buf278, buf275, primals_8, primals_9)
        del primals_8
        del primals_9
        buf280 = buf279[0]
        buf281 = buf279[1]
        buf282 = buf279[2]
        del buf279
        buf283 = buf278; del buf278  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_47], Original ATen: [aten.mm]
        extern_kernels.mm(buf280, reinterpret_tensor(primals_10, (64, 256), (1, 64), 0), out=buf283)
        buf284 = buf277; del buf277  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_47], Original ATen: [aten.mm]
        extern_kernels.mm(buf280, reinterpret_tensor(primals_11, (64, 256), (1, 64), 0), out=buf284)
        # Topologically Sorted Source Nodes: [lstm_cell_47], Original ATen: [aten._thnn_fused_lstm_cell]
        buf285 = torch.ops.aten._thnn_fused_lstm_cell.default(buf283, buf284, buf281, primals_12, primals_13)
        del primals_12
        del primals_13
        buf286 = buf285[0]
        buf287 = buf285[1]
        buf288 = buf285[2]
        del buf285
        buf289 = buf284; del buf284  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_48], Original ATen: [aten.mm]
        extern_kernels.mm(buf286, reinterpret_tensor(primals_14, (64, 256), (1, 64), 0), out=buf289)
        buf290 = buf283; del buf283  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_48], Original ATen: [aten.mm]
        extern_kernels.mm(buf286, reinterpret_tensor(primals_15, (64, 256), (1, 64), 0), out=buf290)
        # Topologically Sorted Source Nodes: [lstm_cell_48], Original ATen: [aten._thnn_fused_lstm_cell]
        buf291 = torch.ops.aten._thnn_fused_lstm_cell.default(buf289, buf290, buf287, primals_16, primals_17)
        del primals_16
        del primals_17
        buf292 = buf291[0]
        buf293 = buf291[1]
        buf294 = buf291[2]
        del buf291
        buf295 = buf290; del buf290  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_49], Original ATen: [aten.mm]
        extern_kernels.mm(buf292, reinterpret_tensor(primals_18, (64, 256), (1, 64), 0), out=buf295)
        buf296 = buf289; del buf289  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_49], Original ATen: [aten.mm]
        extern_kernels.mm(buf292, reinterpret_tensor(primals_19, (64, 256), (1, 64), 0), out=buf296)
        # Topologically Sorted Source Nodes: [lstm_cell_49], Original ATen: [aten._thnn_fused_lstm_cell]
        buf297 = torch.ops.aten._thnn_fused_lstm_cell.default(buf295, buf296, buf293, primals_20, primals_21)
        del buf295
        del buf296
        del primals_20
        del primals_21
        buf298 = buf297[0]
        buf299 = buf297[1]
        buf300 = buf297[2]
        del buf297
        buf301 = empty_strided_cuda((1, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf301)
        buf302 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
        buf303 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        buf304 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        buf307 = empty_strided_cuda((), (), torch.float32)
        buf315 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
        buf316 = buf307; del buf307  # reuse
        # Topologically Sorted Source Nodes: [randn_like, target_distribution, log_softmax, kl_loss], Original ATen: [aten.randn_like, aten._softmax, aten._log_softmax, aten.mul, aten.xlogy, aten.sub, aten.sum, aten.div, aten._log_softmax_backward_data]
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax__log_softmax_backward_data__softmax_div_mul_randn_like_sub_sum_xlogy_1[grid(1)](buf316, buf301, buf298, buf302, buf303, buf304, buf315, 0, 1, 64, XBLOCK=1, num_warps=2, num_stages=1)
        del buf298
    buf308 = empty_strided_cpu((3, ), (1, ), torch.int64)
    # Topologically Sorted Source Nodes: [], Original ATen: []
    aten.randint.low_out(-9223372036854775808, 9223372036854775807, [3], out=buf308)
    buf309 = empty_strided_cpu((1, 1), (1, 1), torch.float32)
    cpp_fused_randn_2(buf308, buf309)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf310 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        buf310.copy_(buf309, False)
    buf311 = buf309; del buf309  # reuse
    cpp_fused_randn_3(buf308, buf311)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf312 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        buf312.copy_(buf311, False)
        del buf311
    buf313 = empty_strided_cpu((1, 1), (1, 1), torch.int64)
    cpp_fused_randint_4(buf308, buf313)
    del buf308
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf314 = reinterpret_tensor(buf301, (1, 1), (1, 1), 0); del buf301  # reuse
        buf314.copy_(buf313, False)
        del buf313
        buf317 = reinterpret_tensor(buf310, (), (), 0); del buf310  # reuse
        # Topologically Sorted Source Nodes: [float_1, mul, target, margin_loss], Original ATen: [aten._to_copy, aten.mul, aten.sub, aten.neg, aten.add, aten.clamp_min, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clamp_min_mean_mul_neg_sub_5[grid(1)](buf317, buf314, buf312, 1, XBLOCK=1, num_warps=1, num_stages=1)
        del buf312
        del buf314
    return (buf316, buf317, buf0, reinterpret_tensor(primals_1, (1, 128), (1280, 1), 0), buf4, buf5, buf6, buf10, buf11, buf12, buf16, buf17, buf18, buf22, buf23, buf24, buf28, buf29, buf30, reinterpret_tensor(primals_1, (1, 128), (1280, 1), 128), buf34, buf35, buf36, buf40, buf41, buf42, buf46, buf47, buf48, buf52, buf53, buf54, buf58, buf59, buf60, reinterpret_tensor(primals_1, (1, 128), (1280, 1), 256), buf64, buf65, buf66, buf70, buf71, buf72, buf76, buf77, buf78, buf82, buf83, buf84, buf88, buf89, buf90, reinterpret_tensor(primals_1, (1, 128), (1280, 1), 384), buf94, buf95, buf96, buf100, buf101, buf102, buf106, buf107, buf108, buf112, buf113, buf114, buf118, buf119, buf120, reinterpret_tensor(primals_1, (1, 128), (1280, 1), 512), buf124, buf125, buf126, buf130, buf131, buf132, buf136, buf137, buf138, buf142, buf143, buf144, buf148, buf149, buf150, reinterpret_tensor(primals_1, (1, 128), (1280, 1), 640), buf154, buf155, buf156, buf160, buf161, buf162, buf166, buf167, buf168, buf172, buf173, buf174, buf178, buf179, buf180, reinterpret_tensor(primals_1, (1, 128), (1280, 1), 768), buf184, buf185, buf186, buf190, buf191, buf192, buf196, buf197, buf198, buf202, buf203, buf204, buf208, buf209, buf210, reinterpret_tensor(primals_1, (1, 128), (1280, 1), 896), buf214, buf215, buf216, buf220, buf221, buf222, buf226, buf227, buf228, buf232, buf233, buf234, buf238, buf239, buf240, reinterpret_tensor(primals_1, (1, 128), (1280, 1), 1024), buf244, buf245, buf246, buf250, buf251, buf252, buf256, buf257, buf258, buf262, buf263, buf264, buf268, buf269, buf270, reinterpret_tensor(primals_1, (1, 128), (1280, 1), 1152), buf274, buf275, buf276, buf280, buf281, buf282, buf286, buf287, buf288, buf292, buf293, buf294, buf299, buf300, buf302, buf303, buf304, buf315, primals_19, primals_18, primals_15, primals_14, primals_11, primals_10, primals_7, primals_6, primals_3, )


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
    primals_10 = rand_strided((256, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((256, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((256, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((256, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((256, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((256, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
