# AOT ID: ['97_inference']
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


# kernel path: /tmp/torchinductor_sahanp/gd/cgd7inwufhlmpkvrkfarryxbcgxncgkacqkf2tt44vv4phb2jz73.py
# Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.avg_pool2d, aten.softplus]
# Source node to ATen node mapping:
#   x => avg_pool2d
#   x_1 => div, exp, gt, log1p, mul_4, where
# Graph fragment:
#   %avg_pool2d : [num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%arg3_1, [2, 2], [2, 2]), kwargs = {})
#   %mul_4 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%avg_pool2d, 1.0), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%mul_4, 20.0), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_4,), kwargs = {})
#   %log1p : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%log1p, 1.0), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %avg_pool2d, %div), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_avg_pool2d_softplus_0(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks1)
    x2 = xindex // ks2
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 2*ks4*x1 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 2*ks4*x1 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (ks4 + 2*x0 + 2*ks4*x1 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (1 + ks4 + 2*x0 + 2*ks4*x1 + ks3*ks4*x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = 20.0
    tmp12 = tmp10 > tmp11
    tmp13 = tl_math.exp(tmp10)
    tmp14 = libdevice.log1p(tmp13)
    tmp15 = tmp14 * tmp9
    tmp16 = tl.where(tmp12, tmp8, tmp15)
    tl.store(out_ptr0 + (x3), tmp16, xmask)




# kernel path: /tmp/torchinductor_sahanp/if/ciffw7dcsariwfukdieddtc47qhuv6tyfydrn2glrj7d7qtzgf2c.py
# Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.col2im]
# Source node to ATen node mapping:
#   x_4 => full_default
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, %arg0_1, 8, 8], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_col2im_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/n7/cn7mzvruvgjkspnlho3amxmgevhkrthi4zhpfiytx5ugsueibzxo.py
# Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.col2im]
# Source node to ATen node mapping:
#   x_4 => full_default, index_put
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, %arg0_1, 8, 8], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default, [None, None, %unsqueeze_3, %add_16], %permute, True), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_col2im_2(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    xnumel = 16
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex % 4)
    x4 = xindex // 4
    y0 = (yindex % 2)
    y1 = ((yindex // 2) % 2)
    y2 = yindex // 4
    tmp0 = tl.load(in_ptr0 + (2*(((x3 + 4*x4 + y0*(triton_helpers.div_floor_integer((ks3 // 4)*(ks4 // 4),  4)) + 2*y1*(triton_helpers.div_floor_integer((ks3 // 4)*(ks4 // 4),  4)) + 4*y2*(triton_helpers.div_floor_integer((ks3 // 4)*(ks4 // 4),  4))) % (ks4 // 4))) + 2*ks0*((((x3 + 4*x4 + y0*(triton_helpers.div_floor_integer((ks3 // 4)*(ks4 // 4),  4)) + 2*y1*(triton_helpers.div_floor_integer((ks3 // 4)*(ks4 // 4),  4)) + 4*y2*(triton_helpers.div_floor_integer((ks3 // 4)*(ks4 // 4),  4))) // (ks4 // 4)) % (ks3 // 4))) + ks0*ks1*((((x3 + 4*x4 + y0*(triton_helpers.div_floor_integer((ks3 // 4)*(ks4 // 4),  4)) + 2*y1*(triton_helpers.div_floor_integer((ks3 // 4)*(ks4 // 4),  4)) + 4*y2*(triton_helpers.div_floor_integer((ks3 // 4)*(ks4 // 4),  4))) // ((ks3 // 4)*(ks4 // 4))) % ks2))), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*(((x3 + 4*x4 + y0*(triton_helpers.div_floor_integer((ks3 // 4)*(ks4 // 4),  4)) + 2*y1*(triton_helpers.div_floor_integer((ks3 // 4)*(ks4 // 4),  4)) + 4*y2*(triton_helpers.div_floor_integer((ks3 // 4)*(ks4 // 4),  4))) % (ks4 // 4))) + 2*ks0*((((x3 + 4*x4 + y0*(triton_helpers.div_floor_integer((ks3 // 4)*(ks4 // 4),  4)) + 2*y1*(triton_helpers.div_floor_integer((ks3 // 4)*(ks4 // 4),  4)) + 4*y2*(triton_helpers.div_floor_integer((ks3 // 4)*(ks4 // 4),  4))) // (ks4 // 4)) % (ks3 // 4))) + ks0*ks1*((((x3 + 4*x4 + y0*(triton_helpers.div_floor_integer((ks3 // 4)*(ks4 // 4),  4)) + 2*y1*(triton_helpers.div_floor_integer((ks3 // 4)*(ks4 // 4),  4)) + 4*y2*(triton_helpers.div_floor_integer((ks3 // 4)*(ks4 // 4),  4))) // ((ks3 // 4)*(ks4 // 4))) % ks2))), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (ks0 + 2*(((x3 + 4*x4 + y0*(triton_helpers.div_floor_integer((ks3 // 4)*(ks4 // 4),  4)) + 2*y1*(triton_helpers.div_floor_integer((ks3 // 4)*(ks4 // 4),  4)) + 4*y2*(triton_helpers.div_floor_integer((ks3 // 4)*(ks4 // 4),  4))) % (ks4 // 4))) + 2*ks0*((((x3 + 4*x4 + y0*(triton_helpers.div_floor_integer((ks3 // 4)*(ks4 // 4),  4)) + 2*y1*(triton_helpers.div_floor_integer((ks3 // 4)*(ks4 // 4),  4)) + 4*y2*(triton_helpers.div_floor_integer((ks3 // 4)*(ks4 // 4),  4))) // (ks4 // 4)) % (ks3 // 4))) + ks0*ks1*((((x3 + 4*x4 + y0*(triton_helpers.div_floor_integer((ks3 // 4)*(ks4 // 4),  4)) + 2*y1*(triton_helpers.div_floor_integer((ks3 // 4)*(ks4 // 4),  4)) + 4*y2*(triton_helpers.div_floor_integer((ks3 // 4)*(ks4 // 4),  4))) // ((ks3 // 4)*(ks4 // 4))) % ks2))), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (1 + ks0 + 2*(((x3 + 4*x4 + y0*(triton_helpers.div_floor_integer((ks3 // 4)*(ks4 // 4),  4)) + 2*y1*(triton_helpers.div_floor_integer((ks3 // 4)*(ks4 // 4),  4)) + 4*y2*(triton_helpers.div_floor_integer((ks3 // 4)*(ks4 // 4),  4))) % (ks4 // 4))) + 2*ks0*((((x3 + 4*x4 + y0*(triton_helpers.div_floor_integer((ks3 // 4)*(ks4 // 4),  4)) + 2*y1*(triton_helpers.div_floor_integer((ks3 // 4)*(ks4 // 4),  4)) + 4*y2*(triton_helpers.div_floor_integer((ks3 // 4)*(ks4 // 4),  4))) // (ks4 // 4)) % (ks3 // 4))) + ks0*ks1*((((x3 + 4*x4 + y0*(triton_helpers.div_floor_integer((ks3 // 4)*(ks4 // 4),  4)) + 2*y1*(triton_helpers.div_floor_integer((ks3 // 4)*(ks4 // 4),  4)) + 4*y2*(triton_helpers.div_floor_integer((ks3 // 4)*(ks4 // 4),  4))) // ((ks3 // 4)*(ks4 // 4))) % ks2))), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.atomic_add(out_ptr0 + (y0 + 2*x3 + 8*y1 + 16*x4 + 64*y2), tmp8, xmask & ymask, sem='relaxed')







def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        ps0 = s2 // 2
        ps1 = s1 // 2
        ps2 = (s1 // 2)*(s2 // 2)
        buf0 = empty_strided_cuda((1, s0, s1 // 2, s2 // 2), (s0*(s1 // 2)*(s2 // 2), (s1 // 2)*(s2 // 2), s2 // 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.avg_pool2d, aten.softplus]
        triton_poi_fused_avg_pool2d_softplus_0_xnumel = s0*(s1 // 2)*(s2 // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_softplus_0[grid(triton_poi_fused_avg_pool2d_softplus_0_xnumel)](arg3_1, buf0, 16, 16, 256, 32, 32, 768, XBLOCK=256, num_warps=4, num_stages=1)
        del arg3_1
        buf1 = empty_strided_cuda((1, s0, 8, 8), (64*s0, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_1_xnumel = 64*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_col2im_1[grid(triton_poi_fused_col2im_1_xnumel)](buf1, 192, XBLOCK=256, num_warps=4, num_stages=1)
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_2_ynumel = 4*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_col2im_2[grid(triton_poi_fused_col2im_2_ynumel, 16)](buf0, buf1, 16, 16, 3, 32, 32, 12, 16, XBLOCK=16, YBLOCK=16, num_warps=4, num_stages=1)
        del buf0
    return (buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 3
    arg1_1 = 32
    arg2_1 = 32
    arg3_1 = rand_strided((1, 3, 32, 32), (3072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
