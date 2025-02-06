# AOT ID: ['39_inference']
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


# kernel path: /tmp/torchinductor_sahanp/bo/cbojkgg7gnylmould7pvbhf4lcmsshmomipubpttitrxb4eglu2x.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.adaptive_max_pool2d]
# Source node to ATen node mapping:
#   x_2 => adaptive_max_pool2d
# Graph fragment:
#   %adaptive_max_pool2d : [num_users=1] = call_function[target=torch.ops.aten.adaptive_max_pool2d.default](args = (%unsqueeze, [1, 10]), kwargs = {})

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_adaptive_max_pool2d_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 10)
    x1 = xindex // 10
    x2 = xindex
    tmp0 = tl.full([1], 0, tl.int64)
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = (6*x0) // 5
    tmp4 = (21 + 12*x0) // 10
    tmp5 = tmp3 < tmp4
    tmp6 = tmp2 & tmp5
    tmp7 = (-1) + (((x1 // 12) % 12))
    tmp8 = tl.full([1], 0, tl.int64)
    tmp9 = tmp7 >= tmp8
    tmp10 = tl.full([1], 10, tl.int64)
    tmp11 = tmp7 < tmp10
    tmp12 = (-1) + ((x1 % 12))
    tmp13 = tmp12 >= tmp8
    tmp14 = tmp12 < tmp10
    tmp15 = (-1) + ((6*x0) // 5)
    tmp16 = tmp15 >= tmp8
    tmp17 = tmp15 < tmp10
    tmp18 = tmp9 & tmp11
    tmp19 = tmp18 & tmp13
    tmp20 = tmp19 & tmp14
    tmp21 = tmp20 & tmp16
    tmp22 = tmp21 & tmp17
    tmp23 = tmp22 & tmp6
    tmp24 = tl.load(in_ptr0 + ((-111) + 10*((x1 % 12)) + 100*(((x1 // 12) % 12)) + 1000*(x1 // 144) + ((6*x0) // 5)), tmp23 & xmask, other=0.0)
    tmp25 = tl.full(tmp24.shape, float("-inf"), tmp24.dtype)
    tmp26 = tl.where(tmp6, tmp24, tmp25)
    tmp27 = 1 + ((6*x0) // 5)
    tmp28 = tmp27 < tmp4
    tmp29 = tmp2 & tmp28
    tmp30 = (-1) + (((x1 // 12) % 12))
    tmp31 = tl.full([1], 0, tl.int64)
    tmp32 = tmp30 >= tmp31
    tmp33 = tl.full([1], 10, tl.int64)
    tmp34 = tmp30 < tmp33
    tmp35 = (-1) + ((x1 % 12))
    tmp36 = tmp35 >= tmp31
    tmp37 = tmp35 < tmp33
    tmp38 = (6*x0) // 5
    tmp39 = tmp38 >= tmp31
    tmp40 = tmp38 < tmp33
    tmp41 = tmp32 & tmp34
    tmp42 = tmp41 & tmp36
    tmp43 = tmp42 & tmp37
    tmp44 = tmp43 & tmp39
    tmp45 = tmp44 & tmp40
    tmp46 = tmp45 & tmp29
    tmp47 = tl.load(in_ptr0 + ((-110) + 10*((x1 % 12)) + 100*(((x1 // 12) % 12)) + 1000*(x1 // 144) + ((6*x0) // 5)), tmp46 & xmask, other=0.0)
    tmp48 = tl.full(tmp47.shape, float("-inf"), tmp47.dtype)
    tmp49 = tl.where(tmp29, tmp47, tmp48)
    tmp50 = triton_helpers.maximum(tmp49, tmp26)
    tmp51 = 2 + ((6*x0) // 5)
    tmp52 = tmp51 < tmp4
    tmp53 = tmp2 & tmp52
    tmp54 = (-1) + (((x1 // 12) % 12))
    tmp55 = tl.full([1], 0, tl.int64)
    tmp56 = tmp54 >= tmp55
    tmp57 = tl.full([1], 10, tl.int64)
    tmp58 = tmp54 < tmp57
    tmp59 = (-1) + ((x1 % 12))
    tmp60 = tmp59 >= tmp55
    tmp61 = tmp59 < tmp57
    tmp62 = 1 + ((6*x0) // 5)
    tmp63 = tmp62 >= tmp55
    tmp64 = tmp62 < tmp57
    tmp65 = tmp56 & tmp58
    tmp66 = tmp65 & tmp60
    tmp67 = tmp66 & tmp61
    tmp68 = tmp67 & tmp63
    tmp69 = tmp68 & tmp64
    tmp70 = tmp69 & tmp53
    tmp71 = tl.load(in_ptr0 + ((-109) + 10*((x1 % 12)) + 100*(((x1 // 12) % 12)) + 1000*(x1 // 144) + ((6*x0) // 5)), tmp70 & xmask, other=0.0)
    tmp72 = tl.full(tmp71.shape, float("-inf"), tmp71.dtype)
    tmp73 = tl.where(tmp53, tmp71, tmp72)
    tmp74 = triton_helpers.maximum(tmp73, tmp50)
    tl.store(out_ptr0 + (x2), tmp74, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (1, 3, 10, 10, 10), (3000, 1000, 100, 10, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((432, 1, 10), (10, 10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.adaptive_max_pool2d]
        get_raw_stream(0)
        triton_poi_fused_adaptive_max_pool2d_0[grid(4320)](arg0_1, buf0, 4320, XBLOCK=256, num_warps=4, num_stages=1)
        del arg0_1
    return (reinterpret_tensor(buf0, (1, 3, 12, 12, 10), (4320, 1440, 120, 10, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 3, 10, 10, 10), (3000, 1000, 100, 10, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
