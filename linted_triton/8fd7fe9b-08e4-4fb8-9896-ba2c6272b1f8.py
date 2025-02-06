# AOT ID: ['61_forward']
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


# kernel path: /tmp/torchinductor_sahanp/wo/cwofv3pqrnlxa7rtrfhpwzbmhzay7ovw6otnj6fv2oizj62udpkw.py
# Topologically Sorted Source Nodes: [h1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   h1 => full_default
# Graph fragment:
#   %full_default : [num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([1, 256], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
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




# kernel path: /tmp/torchinductor_sahanp/mp/cmpk7xwnjs4yaqrin6l6zms35cbaf5lnosqbtnq4qcrurol2tzjs.py
# Topologically Sorted Source Nodes: [h2], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   h2 => full_default_1
# Graph fragment:
#   %full_default_1 : [num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([1, 128], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/se/cseiimvohjtyumo6kdmg3znezqng2dzmphhmj232md6nwmhe62ws.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.bernoulli, aten._to_copy, aten.add, aten.mul]
# Source node to ATen node mapping:
#   x => add, add_1, add_2, convert_element_type_2, inductor_lookup_seed_default, inductor_random_default, lt, mul, mul_1, mul_2
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 128], %inductor_lookup_seed_default, rand), kwargs = {})
#   %lt : [num_users=2] = call_function[target=torch.ops.aten.lt.Scalar](args = (%inductor_random_default, 0.5), kwargs = {})
#   %convert_element_type_2 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt, torch.float32), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%convert_element_type_2, -1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%add, 1.558387861036063), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul, 0.7791939305180315), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%convert_element_type_2, 0.8864048946659319), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_38, %mul_1), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %add_1), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_add_bernoulli_mul_2(in_out_ptr0, in_ptr0, out_ptr1, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp5 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.5
    tmp4 = tmp2 < tmp3
    tmp6 = tmp4.to(tl.float32)
    tmp7 = 0.8864048946659319
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = -1.0
    tmp11 = tmp6 + tmp10
    tmp12 = 1.558387861036063
    tmp13 = tmp11 * tmp12
    tmp14 = 0.7791939305180315
    tmp15 = tmp13 + tmp14
    tmp16 = tmp9 + tmp15
    tl.store(out_ptr1 + (x0), tmp4, xmask)
    tl.store(in_out_ptr0 + (x0), tmp16, xmask)




#include "/tmp/torchinductor_sahanp/3b/c3bi5gk6mslf6u4iaqafhxm64z6u65e3eain4xlary5blqnvv6xx.h"
extern "C"  void kernel(int64_t* in_out_ptr0)
{
    {
        {
            {
                auto tmp0 = in_out_ptr0[static_cast<int64_t>(0L)];
                auto tmp1 = static_cast<int32_t>(0);
                auto tmp2 = static_cast<int64_t>(0);
                auto tmp3 = static_cast<int64_t>(10);
                auto tmp4 = randint64_cpu(tmp0, tmp1, tmp2, tmp3);
                in_out_ptr0[static_cast<int64_t>(0L)] = tmp4;
            }
        }
    }
}
''')


# kernel path: /tmp/torchinductor_sahanp/ea/ceakjoi65tr5262t2jmt53ccv6so3ukpcqbusq4adgiakinf7ruf.py
# Topologically Sorted Source Nodes: [x_4, loss], Original ATen: [aten._log_softmax, aten.mul, aten.exp, aten.sub, aten.mean]
# Source node to ATen node mapping:
#   loss => exp_1, mean, mul_3, sub_2
#   x_4 => amax, exp, log, sub, sub_1, sum_1
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_2, [1], True), kwargs = {})
#   %sub : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_2, %amax), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_1,), kwargs = {})
#   %sub_1 : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub, %log), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%device_put_2, %sub_1), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_1,), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%exp_1, %mul_3), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sub_2,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__log_softmax_exp_mean_mul_sub_4(in_out_ptr0, in_ptr0, in_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 240
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (16*((7) * ((7) <= (((0) * ((0) >= ((-2) + (r0_0 // 20))) + ((-2) + (r0_0 // 20)) * (((-2) + (r0_0 // 20)) > (0))))) + (((0) * ((0) >= ((-2) + (r0_0 // 20))) + ((-2) + (r0_0 // 20)) * (((-2) + (r0_0 // 20)) > (0)))) * ((((0) * ((0) >= ((-2) + (r0_0 // 20))) + ((-2) + (r0_0 // 20)) * (((-2) + (r0_0 // 20)) > (0)))) < (7))) + ((15) * ((15) <= (((0) * ((0) >= ((-2) + ((r0_0 % 20)))) + ((-2) + ((r0_0 % 20))) * (((-2) + ((r0_0 % 20))) > (0))))) + (((0) * ((0) >= ((-2) + ((r0_0 % 20)))) + ((-2) + ((r0_0 % 20))) * (((-2) + ((r0_0 % 20))) > (0)))) * ((((0) * ((0) >= ((-2) + ((r0_0 % 20)))) + ((-2) + ((r0_0 % 20))) * (((-2) + ((r0_0 % 20))) > (0)))) < (15)))), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(in_ptr1 + (0))
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tmp5 = tmp0 - tmp4
    tmp6 = tl_math.exp(tmp5)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(r0_mask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp11 = tl_math.log(tmp10)
    tmp12 = tmp5 - tmp11
    tmp13 = tl_math.exp(tmp12)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp16 * tmp12
    tmp18 = tmp13 - tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
    tmp21 = tl.where(r0_mask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None]
    tmp23 = 240.0
    tmp24 = tmp22 / tmp23
    tl.store(out_ptr2 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp12, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp24, None)







def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9 = args
    args.clear()
    assert_size_stride(primals_1, (1, 10, 128), (1280, 128, 1))
    assert_size_stride(primals_2, (768, 128), (128, 1))
    assert_size_stride(primals_3, (768, 256), (256, 1))
    assert_size_stride(primals_4, (768, ), (1, ))
    assert_size_stride(primals_5, (768, ), (1, ))
    assert_size_stride(primals_6, (384, 256), (256, 1))
    assert_size_stride(primals_7, (384, 128), (128, 1))
    assert_size_stride(primals_8, (384, ), (1, ))
    assert_size_stride(primals_9, (384, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [h1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_0[grid(256)](buf0, 256, XBLOCK=128, num_warps=4, num_stages=1)
        buf1 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [h2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1[grid(128)](buf1, 128, XBLOCK=128, num_warps=4, num_stages=1)
        buf2 = empty_strided_cuda((1, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 128), (128, 1), 0), reinterpret_tensor(primals_2, (128, 768), (1, 128), 0), out=buf2)
        buf3 = empty_strided_cuda((1, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret], Original ATen: [aten.mm]
        extern_kernels.mm(buf0, reinterpret_tensor(primals_3, (256, 768), (1, 256), 0), out=buf3)
        # Topologically Sorted Source Nodes: [ret], Original ATen: [aten._thnn_fused_gru_cell]
        buf4 = torch.ops.aten._thnn_fused_gru_cell.default(buf2, buf3, buf0, primals_4, primals_5)
        buf5 = buf4[0]
        buf6 = buf4[1]
        del buf4
        buf7 = empty_strided_cuda((1, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf5, reinterpret_tensor(primals_6, (256, 384), (1, 256), 0), out=buf7)
        buf8 = empty_strided_cuda((1, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf1, reinterpret_tensor(primals_7, (128, 384), (1, 128), 0), out=buf8)
        # Topologically Sorted Source Nodes: [ret_1], Original ATen: [aten._thnn_fused_gru_cell]
        buf9 = torch.ops.aten._thnn_fused_gru_cell.default(buf7, buf8, buf1, primals_8, primals_9)
        buf10 = buf9[0]
        buf11 = buf9[1]
        del buf9
        buf12 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [ret_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 128), (128, 1), 128), reinterpret_tensor(primals_2, (128, 768), (1, 128), 0), out=buf12)
        buf13 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [ret_2], Original ATen: [aten.mm]
        extern_kernels.mm(buf5, reinterpret_tensor(primals_3, (256, 768), (1, 256), 0), out=buf13)
        # Topologically Sorted Source Nodes: [ret_2], Original ATen: [aten._thnn_fused_gru_cell]
        buf14 = torch.ops.aten._thnn_fused_gru_cell.default(buf12, buf13, buf5, primals_4, primals_5)
        buf15 = buf14[0]
        buf16 = buf14[1]
        del buf14
        buf17 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [ret_3], Original ATen: [aten.mm]
        extern_kernels.mm(buf15, reinterpret_tensor(primals_6, (256, 384), (1, 256), 0), out=buf17)
        buf18 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [ret_3], Original ATen: [aten.mm]
        extern_kernels.mm(buf10, reinterpret_tensor(primals_7, (128, 384), (1, 128), 0), out=buf18)
        # Topologically Sorted Source Nodes: [ret_3], Original ATen: [aten._thnn_fused_gru_cell]
        buf19 = torch.ops.aten._thnn_fused_gru_cell.default(buf17, buf18, buf10, primals_8, primals_9)
        buf20 = buf19[0]
        buf21 = buf19[1]
        del buf19
        buf22 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [ret_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 128), (128, 1), 256), reinterpret_tensor(primals_2, (128, 768), (1, 128), 0), out=buf22)
        buf23 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [ret_4], Original ATen: [aten.mm]
        extern_kernels.mm(buf15, reinterpret_tensor(primals_3, (256, 768), (1, 256), 0), out=buf23)
        # Topologically Sorted Source Nodes: [ret_4], Original ATen: [aten._thnn_fused_gru_cell]
        buf24 = torch.ops.aten._thnn_fused_gru_cell.default(buf22, buf23, buf15, primals_4, primals_5)
        buf25 = buf24[0]
        buf26 = buf24[1]
        del buf24
        buf27 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [ret_5], Original ATen: [aten.mm]
        extern_kernels.mm(buf25, reinterpret_tensor(primals_6, (256, 384), (1, 256), 0), out=buf27)
        buf28 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [ret_5], Original ATen: [aten.mm]
        extern_kernels.mm(buf20, reinterpret_tensor(primals_7, (128, 384), (1, 128), 0), out=buf28)
        # Topologically Sorted Source Nodes: [ret_5], Original ATen: [aten._thnn_fused_gru_cell]
        buf29 = torch.ops.aten._thnn_fused_gru_cell.default(buf27, buf28, buf20, primals_8, primals_9)
        buf30 = buf29[0]
        buf31 = buf29[1]
        del buf29
        buf32 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [ret_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 128), (128, 1), 384), reinterpret_tensor(primals_2, (128, 768), (1, 128), 0), out=buf32)
        buf33 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [ret_6], Original ATen: [aten.mm]
        extern_kernels.mm(buf25, reinterpret_tensor(primals_3, (256, 768), (1, 256), 0), out=buf33)
        # Topologically Sorted Source Nodes: [ret_6], Original ATen: [aten._thnn_fused_gru_cell]
        buf34 = torch.ops.aten._thnn_fused_gru_cell.default(buf32, buf33, buf25, primals_4, primals_5)
        buf35 = buf34[0]
        buf36 = buf34[1]
        del buf34
        buf37 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [ret_7], Original ATen: [aten.mm]
        extern_kernels.mm(buf35, reinterpret_tensor(primals_6, (256, 384), (1, 256), 0), out=buf37)
        buf38 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [ret_7], Original ATen: [aten.mm]
        extern_kernels.mm(buf30, reinterpret_tensor(primals_7, (128, 384), (1, 128), 0), out=buf38)
        # Topologically Sorted Source Nodes: [ret_7], Original ATen: [aten._thnn_fused_gru_cell]
        buf39 = torch.ops.aten._thnn_fused_gru_cell.default(buf37, buf38, buf30, primals_8, primals_9)
        buf40 = buf39[0]
        buf41 = buf39[1]
        del buf39
        buf42 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [ret_8], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 128), (128, 1), 512), reinterpret_tensor(primals_2, (128, 768), (1, 128), 0), out=buf42)
        buf43 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [ret_8], Original ATen: [aten.mm]
        extern_kernels.mm(buf35, reinterpret_tensor(primals_3, (256, 768), (1, 256), 0), out=buf43)
        # Topologically Sorted Source Nodes: [ret_8], Original ATen: [aten._thnn_fused_gru_cell]
        buf44 = torch.ops.aten._thnn_fused_gru_cell.default(buf42, buf43, buf35, primals_4, primals_5)
        buf45 = buf44[0]
        buf46 = buf44[1]
        del buf44
        buf47 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [ret_9], Original ATen: [aten.mm]
        extern_kernels.mm(buf45, reinterpret_tensor(primals_6, (256, 384), (1, 256), 0), out=buf47)
        buf48 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [ret_9], Original ATen: [aten.mm]
        extern_kernels.mm(buf40, reinterpret_tensor(primals_7, (128, 384), (1, 128), 0), out=buf48)
        # Topologically Sorted Source Nodes: [ret_9], Original ATen: [aten._thnn_fused_gru_cell]
        buf49 = torch.ops.aten._thnn_fused_gru_cell.default(buf47, buf48, buf40, primals_8, primals_9)
        buf50 = buf49[0]
        buf51 = buf49[1]
        del buf49
        buf52 = buf43; del buf43  # reuse
        # Topologically Sorted Source Nodes: [ret_10], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 128), (128, 1), 640), reinterpret_tensor(primals_2, (128, 768), (1, 128), 0), out=buf52)
        buf53 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [ret_10], Original ATen: [aten.mm]
        extern_kernels.mm(buf45, reinterpret_tensor(primals_3, (256, 768), (1, 256), 0), out=buf53)
        # Topologically Sorted Source Nodes: [ret_10], Original ATen: [aten._thnn_fused_gru_cell]
        buf54 = torch.ops.aten._thnn_fused_gru_cell.default(buf52, buf53, buf45, primals_4, primals_5)
        buf55 = buf54[0]
        buf56 = buf54[1]
        del buf54
        buf57 = buf48; del buf48  # reuse
        # Topologically Sorted Source Nodes: [ret_11], Original ATen: [aten.mm]
        extern_kernels.mm(buf55, reinterpret_tensor(primals_6, (256, 384), (1, 256), 0), out=buf57)
        buf58 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [ret_11], Original ATen: [aten.mm]
        extern_kernels.mm(buf50, reinterpret_tensor(primals_7, (128, 384), (1, 128), 0), out=buf58)
        # Topologically Sorted Source Nodes: [ret_11], Original ATen: [aten._thnn_fused_gru_cell]
        buf59 = torch.ops.aten._thnn_fused_gru_cell.default(buf57, buf58, buf50, primals_8, primals_9)
        buf60 = buf59[0]
        buf61 = buf59[1]
        del buf59
        buf62 = buf53; del buf53  # reuse
        # Topologically Sorted Source Nodes: [ret_12], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 128), (128, 1), 768), reinterpret_tensor(primals_2, (128, 768), (1, 128), 0), out=buf62)
        buf63 = buf52; del buf52  # reuse
        # Topologically Sorted Source Nodes: [ret_12], Original ATen: [aten.mm]
        extern_kernels.mm(buf55, reinterpret_tensor(primals_3, (256, 768), (1, 256), 0), out=buf63)
        # Topologically Sorted Source Nodes: [ret_12], Original ATen: [aten._thnn_fused_gru_cell]
        buf64 = torch.ops.aten._thnn_fused_gru_cell.default(buf62, buf63, buf55, primals_4, primals_5)
        buf65 = buf64[0]
        buf66 = buf64[1]
        del buf64
        buf67 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [ret_13], Original ATen: [aten.mm]
        extern_kernels.mm(buf65, reinterpret_tensor(primals_6, (256, 384), (1, 256), 0), out=buf67)
        buf68 = buf57; del buf57  # reuse
        # Topologically Sorted Source Nodes: [ret_13], Original ATen: [aten.mm]
        extern_kernels.mm(buf60, reinterpret_tensor(primals_7, (128, 384), (1, 128), 0), out=buf68)
        # Topologically Sorted Source Nodes: [ret_13], Original ATen: [aten._thnn_fused_gru_cell]
        buf69 = torch.ops.aten._thnn_fused_gru_cell.default(buf67, buf68, buf60, primals_8, primals_9)
        buf70 = buf69[0]
        buf71 = buf69[1]
        del buf69
        buf72 = buf63; del buf63  # reuse
        # Topologically Sorted Source Nodes: [ret_14], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 128), (128, 1), 896), reinterpret_tensor(primals_2, (128, 768), (1, 128), 0), out=buf72)
        buf73 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [ret_14], Original ATen: [aten.mm]
        extern_kernels.mm(buf65, reinterpret_tensor(primals_3, (256, 768), (1, 256), 0), out=buf73)
        # Topologically Sorted Source Nodes: [ret_14], Original ATen: [aten._thnn_fused_gru_cell]
        buf74 = torch.ops.aten._thnn_fused_gru_cell.default(buf72, buf73, buf65, primals_4, primals_5)
        buf75 = buf74[0]
        buf76 = buf74[1]
        del buf74
        buf77 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [ret_15], Original ATen: [aten.mm]
        extern_kernels.mm(buf75, reinterpret_tensor(primals_6, (256, 384), (1, 256), 0), out=buf77)
        buf78 = buf67; del buf67  # reuse
        # Topologically Sorted Source Nodes: [ret_15], Original ATen: [aten.mm]
        extern_kernels.mm(buf70, reinterpret_tensor(primals_7, (128, 384), (1, 128), 0), out=buf78)
        # Topologically Sorted Source Nodes: [ret_15], Original ATen: [aten._thnn_fused_gru_cell]
        buf79 = torch.ops.aten._thnn_fused_gru_cell.default(buf77, buf78, buf70, primals_8, primals_9)
        buf80 = buf79[0]
        buf81 = buf79[1]
        del buf79
        buf82 = buf73; del buf73  # reuse
        # Topologically Sorted Source Nodes: [ret_16], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 128), (128, 1), 1024), reinterpret_tensor(primals_2, (128, 768), (1, 128), 0), out=buf82)
        buf83 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [ret_16], Original ATen: [aten.mm]
        extern_kernels.mm(buf75, reinterpret_tensor(primals_3, (256, 768), (1, 256), 0), out=buf83)
        # Topologically Sorted Source Nodes: [ret_16], Original ATen: [aten._thnn_fused_gru_cell]
        buf84 = torch.ops.aten._thnn_fused_gru_cell.default(buf82, buf83, buf75, primals_4, primals_5)
        buf85 = buf84[0]
        buf86 = buf84[1]
        del buf84
        buf87 = buf78; del buf78  # reuse
        # Topologically Sorted Source Nodes: [ret_17], Original ATen: [aten.mm]
        extern_kernels.mm(buf85, reinterpret_tensor(primals_6, (256, 384), (1, 256), 0), out=buf87)
        buf88 = buf77; del buf77  # reuse
        # Topologically Sorted Source Nodes: [ret_17], Original ATen: [aten.mm]
        extern_kernels.mm(buf80, reinterpret_tensor(primals_7, (128, 384), (1, 128), 0), out=buf88)
        # Topologically Sorted Source Nodes: [ret_17], Original ATen: [aten._thnn_fused_gru_cell]
        buf89 = torch.ops.aten._thnn_fused_gru_cell.default(buf87, buf88, buf80, primals_8, primals_9)
        buf90 = buf89[0]
        buf91 = buf89[1]
        del buf89
        buf92 = buf83; del buf83  # reuse
        # Topologically Sorted Source Nodes: [ret_18], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 128), (128, 1), 1152), reinterpret_tensor(primals_2, (128, 768), (1, 128), 0), out=buf92)
        del primals_2
        buf93 = buf82; del buf82  # reuse
        # Topologically Sorted Source Nodes: [ret_18], Original ATen: [aten.mm]
        extern_kernels.mm(buf85, reinterpret_tensor(primals_3, (256, 768), (1, 256), 0), out=buf93)
        # Topologically Sorted Source Nodes: [ret_18], Original ATen: [aten._thnn_fused_gru_cell]
        buf94 = torch.ops.aten._thnn_fused_gru_cell.default(buf92, buf93, buf85, primals_4, primals_5)
        del buf92
        del buf93
        del primals_4
        del primals_5
        buf95 = buf94[0]
        buf96 = buf94[1]
        del buf94
        buf97 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [ret_19], Original ATen: [aten.mm]
        extern_kernels.mm(buf95, reinterpret_tensor(primals_6, (256, 384), (1, 256), 0), out=buf97)
        buf98 = buf87; del buf87  # reuse
        # Topologically Sorted Source Nodes: [ret_19], Original ATen: [aten.mm]
        extern_kernels.mm(buf90, reinterpret_tensor(primals_7, (128, 384), (1, 128), 0), out=buf98)
        # Topologically Sorted Source Nodes: [ret_19], Original ATen: [aten._thnn_fused_gru_cell]
        buf99 = torch.ops.aten._thnn_fused_gru_cell.default(buf97, buf98, buf90, primals_8, primals_9)
        del buf97
        del buf98
        del primals_8
        del primals_9
        buf100 = buf99[0]
        buf101 = buf99[1]
        del buf99
        buf102 = empty_strided_cuda((1, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf102)
        buf104 = empty_strided_cuda((1, 128), (128, 1), torch.bool)
        buf105 = buf100; del buf100  # reuse
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.bernoulli, aten._to_copy, aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_bernoulli_mul_2[grid(128)](buf105, buf102, buf104, 0, 128, XBLOCK=128, num_warps=4, num_stages=1)
    buf109 = empty_strided_cpu((1, ), (1, ), torch.int64)
    # Topologically Sorted Source Nodes: [], Original ATen: []
    aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf109)
    buf110 = buf109; del buf109  # reuse
    cpp_fused_randint_3(buf110)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf111 = buf102; del buf102  # reuse
        buf111.copy_(buf110, False)
        del buf110
        buf108 = empty_strided_cuda((1, 240), (240, 1), torch.float32)
        buf112 = empty_strided_cuda((), (), torch.float32)
        buf113 = buf112; del buf112  # reuse
        # Topologically Sorted Source Nodes: [x_4, loss], Original ATen: [aten._log_softmax, aten.mul, aten.exp, aten.sub, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax_exp_mean_mul_sub_4[grid(1)](buf113, buf105, buf111, buf108, 1, 240, XBLOCK=1, num_warps=2, num_stages=1)
    return (buf108, buf113, buf0, buf1, reinterpret_tensor(primals_1, (1, 128), (1280, 1), 0), buf5, buf6, buf10, buf11, reinterpret_tensor(primals_1, (1, 128), (1280, 1), 128), buf15, buf16, buf20, buf21, reinterpret_tensor(primals_1, (1, 128), (1280, 1), 256), buf25, buf26, buf30, buf31, reinterpret_tensor(primals_1, (1, 128), (1280, 1), 384), buf35, buf36, buf40, buf41, reinterpret_tensor(primals_1, (1, 128), (1280, 1), 512), buf45, buf46, buf50, buf51, reinterpret_tensor(primals_1, (1, 128), (1280, 1), 640), buf55, buf56, buf60, buf61, reinterpret_tensor(primals_1, (1, 128), (1280, 1), 768), buf65, buf66, buf70, buf71, reinterpret_tensor(primals_1, (1, 128), (1280, 1), 896), buf75, buf76, buf80, buf81, reinterpret_tensor(primals_1, (1, 128), (1280, 1), 1024), buf85, buf86, buf90, buf91, reinterpret_tensor(primals_1, (1, 128), (1280, 1), 1152), buf95, buf96, buf101, buf104, reinterpret_tensor(buf105, (1, 1, 8, 16), (128, 128, 16, 1), 0), buf108, buf111, primals_7, primals_6, primals_3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 10, 128), (1280, 128, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((768, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((384, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
