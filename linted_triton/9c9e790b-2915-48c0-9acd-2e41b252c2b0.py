# AOT ID: ['1_inference']
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


# kernel path: /tmp/torchinductor_sahanp/dc/cdcxgn3wltruitttrezameosmnm7i22vqnq4ljxtqpqttdm7c5qd.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   x_1 => avg_pool2d
# Graph fragment:
#   %avg_pool2d : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%unsqueeze, [1, 2], [1, 2]), kwargs = {})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_avg_pool2d_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 34)
    x1 = xindex // 34
    x2 = xindex
    tmp0 = (-2) + 2*x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + ((-2) + 2*x0 + 64*x1), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = (-1) + 2*x0
    tmp8 = tmp7 >= tmp1
    tmp9 = tmp7 < tmp3
    tmp10 = tmp8 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-1) + 2*x0 + 64*x1), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 + tmp6
    tmp13 = 0.5
    tmp14 = tmp12 * tmp13
    tl.store(out_ptr0 + (x2), tmp14, xmask)


# kernel path: /tmp/torchinductor_sahanp/63/c632ewifx43ywpjnveyjcjcluqppcdssimfqee33o6fyazql74fi.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.rrelu_with_noise_functional]
# Source node to ATen node mapping:
#   x_2 => le, mul, where
# Graph fragment:
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%squeeze, 0), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze, %uniform), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le, %mul, %squeeze), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_rrelu_with_noise_functional_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp0 * tmp3
    tmp5 = tl.where(tmp2, tmp4, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp5, xmask)


# kernel path: /tmp/torchinductor_sahanp/oz/cozep7gklg5fuowqbpy5qndlnhh4h5laoto2tifrpdhypyy4t3hm.py
# Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.abs, aten.le, aten.scalar_tensor, aten.where]
# Source node to ATen node mapping:
#   x_7 => abs_1, full_default, le_1, where_2
# Graph fragment:
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%squeeze_1,), kwargs = {})
#   %le_1 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%abs_1, 0.5), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_1, %full_default, %squeeze_1), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_abs_le_scalar_tensor_where_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4
    xnumel = 126
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (64*(x1 // 63) + 64*(y0 // 2) + ((x1 % 63)) + ((y0 % 2))), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl_math.abs(tmp0)
    tmp2 = 0.5
    tmp3 = tmp1 <= tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp3, tmp4, tmp0)
    tl.store(out_ptr0 + (x1 + 126*y0), tmp5, xmask & ymask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (1, 3, 64), (192, 64, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 3, 1, 34), (102, 34, 34, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.avg_pool2d]
        get_raw_stream(0)
        triton_poi_fused_avg_pool2d_0[grid(102)](arg0_1, buf0, 102, XBLOCK=128, num_warps=4, num_stages=1)
        del arg0_1
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.rrelu_with_noise_functional]
        buf1 = torch.ops.aten.uniform.default(reinterpret_tensor(buf0, (1, 3, 34), (0, 34, 1), 0), 0.125, 0.3333333333333333)
        buf2 = buf1
        del buf1
        buf3 = reinterpret_tensor(buf0, (1, 3, 34), (102, 34, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.rrelu_with_noise_functional]
        get_raw_stream(0)
        triton_poi_fused_rrelu_with_noise_functional_1[grid(102)](buf3, buf2, 102, XBLOCK=128, num_warps=4, num_stages=1)
        del buf2
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.adaptive_max_pool3d]
        buf4 = torch.ops.aten.adaptive_max_pool3d.default(reinterpret_tensor(buf3, (1, 3, 1, 1, 34), (0, 34, 0, 0, 1), 0), [4, 4, 4])
        del buf3
        buf5 = buf4[0]
        del buf4
        buf7 = empty_strided_cuda((4, 126), (126, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.abs, aten.le, aten.scalar_tensor, aten.where]
        get_raw_stream(0)
        triton_poi_fused_abs_le_scalar_tensor_where_2[grid(4, 126)](buf5, buf7, 4, 126, XBLOCK=32, YBLOCK=4, num_warps=4, num_stages=1)
        del buf5
    return (buf7, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 3, 64), (192, 64, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
