# AOT ID: ['6_forward']
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


# kernel path: /tmp/torchinductor_sahanp/xd/cxdsj33bvhidgky34pg3ka7m323l6kwxiwp7jm5qliom5c62f5ox.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.bernoulli]
# Source node to ATen node mapping:
#   x_2 => inductor_lookup_seed_default, inductor_random_default, lt
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 10, 1, 1], %inductor_lookup_seed_default, rand), kwargs = {})
#   %lt : [num_users=2] = call_function[target=torch.ops.aten.lt.Scalar](args = (%inductor_random_default, 0.5), kwargs = {})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_bernoulli_0(in_ptr0, out_ptr1, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.5
    tmp4 = tmp2 < tmp3
    tl.store(out_ptr1 + (x0), tmp4, xmask)


# kernel path: /tmp/torchinductor_sahanp/eu/ceugddwhhe36i5fxy6wahyvey4ajm7xmfnpsjsexs5cwtj57dhgu.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten._to_copy, aten.div, aten.mul]
# Source node to ATen node mapping:
#   x_2 => convert_element_type, div, mul
# Graph fragment:
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt, torch.float32), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%convert_element_type, 0.5), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze, %div), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_div_mul_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 60
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 2.0
    tmp6 = tmp4 * tmp5
    tmp7 = tmp2 * tmp6
    tl.store(in_out_ptr0 + (x2), tmp7, xmask)


# kernel path: /tmp/torchinductor_sahanp/om/comiamjfb4xkphw4uaghr4icu3gl5x5ypv3bb66a32kqdb4vkuct.py
# Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.adaptive_max_pool2d]
# Source node to ATen node mapping:
#   x_4 => _low_memory_max_pool2d_with_offsets, getitem_1
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%unsqueeze_1, [1, 6], [1, 6], [0, 0], [1, 1], False), kwargs = {})
#   %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 1), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_adaptive_max_pool2d_2(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (6*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 6*x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (2 + 6*x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (3 + 6*x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (4 + 6*x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr0 + (5 + 6*x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 > tmp0
    tmp3 = tl.full([1], 1, tl.int8)
    tmp4 = tl.full([1], 0, tl.int8)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = triton_helpers.maximum(tmp1, tmp0)
    tmp8 = tmp7 > tmp6
    tmp9 = tl.full([1], 2, tl.int8)
    tmp10 = tl.where(tmp8, tmp9, tmp5)
    tmp11 = triton_helpers.maximum(tmp7, tmp6)
    tmp13 = tmp12 > tmp11
    tmp14 = tl.full([1], 3, tl.int8)
    tmp15 = tl.where(tmp13, tmp14, tmp10)
    tmp16 = triton_helpers.maximum(tmp12, tmp11)
    tmp18 = tmp17 > tmp16
    tmp19 = tl.full([1], 4, tl.int8)
    tmp20 = tl.where(tmp18, tmp19, tmp15)
    tmp21 = triton_helpers.maximum(tmp17, tmp16)
    tmp23 = tmp22 > tmp21
    tmp24 = tl.full([1], 5, tl.int8)
    tmp25 = tl.where(tmp23, tmp24, tmp20)
    tmp26 = triton_helpers.maximum(tmp22, tmp21)
    tl.store(out_ptr0 + (x0), tmp25, xmask)
    tl.store(out_ptr1 + (x0), tmp26, xmask)


# kernel path: /tmp/torchinductor_sahanp/6a/c6amnhaabx2ubma4ihnj7jy5gf7kyj4wtsynfphtk5mih7b7cxrr.py
# Topologically Sorted Source Nodes: [x_5, x_6], Original ATen: [aten.convolution, aten.softplus]
# Source node to ATen node mapping:
#   x_5 => convolution_1
#   x_6 => div_1, exp, gt, log1p, mul_1, where
# Graph fragment:
#   %convolution_1 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%squeeze_2, %primals_4, %primals_5, [1], [0], [1], False, [0], 1), kwargs = {})
#   %mul_1 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_1, 1.0), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_1,), kwargs = {})
#   %log1p : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp,), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%log1p, 1.0), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%mul_1, 20.0), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %convolution_1, %div_1), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_softplus_3(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 6
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 1.0
    tmp4 = tmp2 * tmp3
    tmp5 = 20.0
    tmp6 = tmp4 > tmp5
    tmp7 = tl_math.exp(tmp4)
    tmp8 = libdevice.log1p(tmp7)
    tmp9 = tmp8 * tmp3
    tmp10 = tl.where(tmp6, tmp2, tmp9)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp10, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (10, 1, 5), (5, 5, 1))
    assert_size_stride(primals_2, (10, ), (1, ))
    assert_size_stride(primals_3, (1, 1, 64), (64, 64, 1))
    assert_size_stride(primals_4, (20, 10, 5), (50, 5, 1))
    assert_size_stride(primals_5, (20, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf0, (1, 10, 60), (600, 60, 1))
        buf1 = empty_strided_cuda((1, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf1)
        buf3 = empty_strided_cuda((1, 10, 1, 1), (10, 1, 1, 1), torch.bool)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.bernoulli]
        get_raw_stream(0)
        triton_poi_fused_bernoulli_0[grid(10)](buf1, buf3, 0, 10, XBLOCK=16, num_warps=1, num_stages=1)
        del buf1
        buf4 = reinterpret_tensor(buf0, (1, 10, 1, 60), (600, 60, 60, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten._to_copy, aten.div, aten.mul]
        get_raw_stream(0)
        triton_poi_fused__to_copy_div_mul_1[grid(600)](buf4, primals_2, buf3, 600, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_2
        buf5 = empty_strided_cuda((1, 10, 1, 10), (100, 10, 10, 1), torch.int8)
        buf6 = empty_strided_cuda((1, 10, 1, 10), (100, 10, 10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.adaptive_max_pool2d]
        get_raw_stream(0)
        triton_poi_fused_adaptive_max_pool2d_2[grid(100)](buf4, buf5, buf6, 100, XBLOCK=128, num_warps=4, num_stages=1)
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(reinterpret_tensor(buf6, (1, 10, 10), (0, 10, 1), 0), primals_4, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf7, (1, 20, 6), (120, 6, 1))
        buf8 = buf7; del buf7  # reuse
        buf9 = empty_strided_cuda((1, 20, 6), (120, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_5, x_6], Original ATen: [aten.convolution, aten.softplus]
        get_raw_stream(0)
        triton_poi_fused_convolution_softplus_3[grid(120)](buf8, primals_5, buf9, 120, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_5
    return (buf9, primals_1, primals_3, primals_4, buf3, buf4, buf5, reinterpret_tensor(buf6, (1, 10, 10), (100, 10, 1), 0), buf8, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((10, 1, 5), (5, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((1, 1, 64), (64, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((20, 10, 5), (50, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([primals_1, primals_2, primals_3, primals_4, primals_5])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
