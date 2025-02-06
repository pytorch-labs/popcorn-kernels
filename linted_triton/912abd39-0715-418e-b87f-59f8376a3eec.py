# AOT ID: ['65_inference']
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


# kernel path: /tmp/torchinductor_sahanp/p4/cp45mfwilwtwhghc43cre73xipta5opvvemqxyf7yi7m3rgq65nh.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.bernoulli]
# Source node to ATen node mapping:
#   x_1 => inductor_lookup_seed_default, inductor_random_default
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, %mul, 1], %inductor_lookup_seed_default, rand), kwargs = {})

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_bernoulli_0(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (x0), tmp2, xmask)


# kernel path: /tmp/torchinductor_sahanp/oo/coo6myy5o2qhvb5cvrcltofkmiuu26ppzr2u4xxo5mjntt5f5fcr.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.leaky_relu]
# Source node to ATen node mapping:
#   x_3 => gt, mul_23, where
# Graph fragment:
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%view_1, 0), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, 0.1), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %view_1, %mul_23), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_leaky_relu_1(in_ptr0, in_ptr1, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // ks0
    tmp0 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = 0.5
    tmp3 = tmp1 < tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 2.0
    tmp6 = tmp4 * tmp5
    tmp7 = tmp0 * tmp6
    tmp8 = 0.0
    tmp9 = tmp7 > tmp8
    tmp10 = 0.1
    tmp11 = tmp7 * tmp10
    tmp12 = tl.where(tmp9, tmp7, tmp11)
    tl.store(out_ptr0 + (x2), tmp12, xmask)


# kernel path: /tmp/torchinductor_sahanp/qg/cqgmw7yyvboxkwe23fix65hicsx25bdp56u3x6mrqd3ng22apziv.py
# Topologically Sorted Source Nodes: [x_3, x_4], Original ATen: [aten.leaky_relu, aten.view]
# Source node to ATen node mapping:
#   x_3 => gt, mul_23, where
#   x_4 => view_2
# Graph fragment:
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%view_1, 0), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, 0.1), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %view_1, %mul_23), kwargs = {})
#   %view_2 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%where, [1, %arg1_1, %mul_28]), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_leaky_relu_view_2(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (ks3*((((x0 + ks1*ks3*x1) // ks3) % (ks1*ks2))) + ((x0 % ks3))), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp0, xmask)


def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf0)
        buf1 = empty_strided_cuda((1, s0*s1, 1), (s0*s1, 1, s0*s1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_0_xnumel = s0*s1
        get_raw_stream(0)
        triton_poi_fused_bernoulli_0[grid(triton_poi_fused_bernoulli_0_xnumel)](buf0, buf1, 0, 96, XBLOCK=128, num_warps=4, num_stages=1)
        del buf0
        buf2 = empty_strided_cuda((1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.leaky_relu]
        triton_poi_fused_leaky_relu_1_xnumel = s0*s1*s2
        get_raw_stream(0)
        triton_poi_fused_leaky_relu_1[grid(triton_poi_fused_leaky_relu_1_xnumel)](arg3_1, buf1, buf2, 32, 3072, XBLOCK=256, num_warps=4, num_stages=1)
        del arg3_1
        del buf1
        s0*s2
        buf3 = empty_strided_cuda((1, s1, s0*s2), (s0*s1*s2, s0*s2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_3, x_4], Original ATen: [aten.leaky_relu, aten.view]
        triton_poi_fused_leaky_relu_view_2_xnumel = s0*s1*s2
        get_raw_stream(0)
        triton_poi_fused_leaky_relu_view_2[grid(triton_poi_fused_leaky_relu_view_2_xnumel)](buf2, buf3, 96, 3, 32, 32, 3072, XBLOCK=256, num_warps=4, num_stages=1)
        del buf2
    return (buf3, s1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 3
    arg1_1 = 32
    arg2_1 = 32
    arg3_1 = rand_strided((1, 3, 32, 32), (3072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
