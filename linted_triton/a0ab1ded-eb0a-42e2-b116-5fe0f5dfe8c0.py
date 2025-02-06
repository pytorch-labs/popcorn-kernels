# AOT ID: ['19_inference']
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


# kernel path: /tmp/torchinductor_sahanp/qp/cqp2lxytsc6rsg6uff5uzafcf2apf7qg32lgj7q4bx3ghzguod43.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.max_unpool2d]
# Source node to ATen node mapping:
#   x => full
# Graph fragment:
#   %full : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, %arg0_1, %sub_11, 1], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_unpool2d_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)


# kernel path: /tmp/torchinductor_sahanp/js/cjsffqabn5fgee4xjdycf4vpk7bl3nijdsmwkkmvqjmyeymmxrc4.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.max_unpool2d]
# Source node to ATen node mapping:
#   x => index_put
# Graph fragment:
#   %index_put : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%view_2, [%view_1], %view_3), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_unpool2d_1(in_ptr0, out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2*((x0 % (ks0 // 2))) + ks0*(triton_helpers.div_floor_integer(x0,  ks0 // 2))), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*((x0 % (ks0 // 2))) + ks0*(triton_helpers.div_floor_integer(x0,  ks0 // 2))), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 > tmp0
    tmp3 = tl.full([1], 1, tl.int8)
    tmp4 = tl.full([1], 0, tl.int8)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = triton_helpers.maximum(tmp1, tmp0)
    tmp7 = tl.full([1], 2, tl.int32)
    tmp8 = tl.where((tmp5 < 0) != (tmp7 < 0), tl.where(tmp5 % tmp7 != 0, tmp5 // tmp7 - 1, tmp5 // tmp7), tmp5 // tmp7)
    tmp9 = tmp8 * tmp7
    tmp10 = tmp5 - tmp9
    tmp11 = tl.full([1], 0, tl.int64)
    tmp12 = tmp11 + tmp8
    tmp13 = 2*((x0 % (ks0 // 2)))
    tmp14 = tmp13 + tmp10
    tmp15 = ks0
    tmp16 = tmp12 * tmp15
    tmp17 = tmp16 + tmp14
    tmp18 = 2*(ks0 // 2)*(triton_helpers.div_floor_integer(x0,  ks0 // 2))
    tmp19 = tmp17 + tmp18
    tmp20 = 2*ks1*(ks0 // 2)
    tmp21 = tmp19 + tmp20
    tmp22 = tmp19 < 0
    tmp23 = tl.where(tmp22, tmp21, tmp19)
    tl.device_assert(((0 <= tmp23) & (tmp23 < 2*ks1*(ks0 // 2))) | ~(xmask), "index out of bounds: 0 <= tmp23 < 2*ks1*(ks0 // 2)")
    tl.store(out_ptr0 + (tl.broadcast_to((tmp23 % (2*ks1*(ks0 // 2))), [XBLOCK])), tmp6, xmask)


# kernel path: /tmp/torchinductor_sahanp/eb/ceb4j2q4fqux7cuexqglujvvczp2jufgieuhsqvpxzuvkshak5ia.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.permute]
# Source node to ATen node mapping:
#   x_1 => permute
# Graph fragment:
#   %permute : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%squeeze_2, [0, 2, 1]), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_permute_2(in_ptr0, out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 2*(ks2 // 2)*((((x0 + 2*x1*(ks2 // 2)) // (2*(ks2 // 2))) % ks1))), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp0, xmask)


def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    assert_size_stride(arg2_1, (1, s0, s1), (s0*s1, s1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, s0, 2*(s1 // 2), 1), (2*s0*(s1 // 2), 2*(s1 // 2), 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.max_unpool2d]
        triton_poi_fused_max_unpool2d_0_xnumel = 2*s0*(s1 // 2)
        get_raw_stream(0)
        triton_poi_fused_max_unpool2d_0[grid(triton_poi_fused_max_unpool2d_0_xnumel)](buf0, 8192, XBLOCK=256, num_warps=4, num_stages=1)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.max_unpool2d]
        triton_poi_fused_max_unpool2d_1_xnumel = s0*(s1 // 2)
        get_raw_stream(0)
        triton_poi_fused_max_unpool2d_1[grid(triton_poi_fused_max_unpool2d_1_xnumel)](arg2_1, buf0, 64, 128, 4096, XBLOCK=128, num_warps=4, num_stages=1)
        del arg2_1
        2*(s1 // 2)
        buf2 = empty_strided_cuda((1, 2*(s1 // 2), s0), (2*s0*(s1 // 2), 1, 2*(s1 // 2)), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.permute]
        triton_poi_fused_permute_2_xnumel = 2*s0*(s1 // 2)
        get_raw_stream(0)
        triton_poi_fused_permute_2[grid(triton_poi_fused_permute_2_xnumel)](buf0, buf2, 64, 128, 64, 8192, XBLOCK=256, num_warps=4, num_stages=1)
        del buf0
    return (buf2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 128
    arg1_1 = 64
    arg2_1 = rand_strided((1, 128, 64), (8192, 64, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
