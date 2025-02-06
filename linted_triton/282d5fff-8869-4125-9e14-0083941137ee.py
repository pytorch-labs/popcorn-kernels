# AOT ID: ['0_forward']
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


# kernel path: /tmp/torchinductor_sahanp/rb/crbvvvfxrc77kwkrapqlvomk7o3zbeyabw5degot2cvcixa754ci.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x => convolution
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_1, %primals_2, [1, 1, 1], [1, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 4096
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)


# kernel path: /tmp/torchinductor_sahanp/ze/czebxp6qzcxrorg4fvx6idmsnijtjhtwzgf75phxlhwtxonki2lz.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.rand]
# Source node to ATen node mapping:
#   x_1 => inductor_lookup_seed_default, inductor_random_default_1
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 10, 3], %inductor_lookup_seed_default, rand), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_rand_1(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 30
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (x0), tmp2, xmask)


# kernel path: /tmp/torchinductor_sahanp/zs/czszjaroigk4wt5gvilzpiveu5mx7nkjy662f6pcm4y23fumkbkj.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_2 => constant_pad_nd
# Graph fragment:
#   %constant_pad_nd : [num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view, [0, 0, 0, 0, 2, 2], 0.0), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_constant_pad_nd_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 512
    x2 = xindex
    tmp0 = (-2) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 10, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + ((-1024) + x2), tmp5 & xmask, other=0.0)
    tmp7 = tmp6 * tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp5, tmp7, tmp8)
    tl.store(out_ptr0 + (x2), tmp9, xmask)


# kernel path: /tmp/torchinductor_sahanp/6q/c6qydskpt6ou5bwdjbixjsxeyyxyagyf34exvnwyd72f7jsvtcwh.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.bernoulli]
# Source node to ATen node mapping:
#   x_3 => inductor_lookup_seed_default_1, inductor_random_default, lt
# Graph fragment:
#   %inductor_lookup_seed_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 1), kwargs = {})
#   %inductor_random_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 10, 1, 1, 1], %inductor_lookup_seed_default_1, rand), kwargs = {})
#   %lt : [num_users=2] = call_function[target=torch.ops.aten.lt.Scalar](args = (%inductor_random_default, 0.5), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_bernoulli_3(in_ptr0, out_ptr1, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/aq/caqkwedmvk4wc2xg26x6kcdpjk4nd7t5mvnmoexrojh2lodgdlnr.py
# Topologically Sorted Source Nodes: [x_2, x_3], Original ATen: [aten.avg_pool3d, aten.mul, aten.add, aten.pow, aten.div, aten._to_copy]
# Source node to ATen node mapping:
#   x_2 => add, avg_pool3d, div, mul_1, pow_1
#   x_3 => convert_element_type, div_1, mul_2
# Graph fragment:
#   %avg_pool3d : [num_users=2] = call_function[target=torch.ops.aten.avg_pool3d.default](args = (%constant_pad_nd, [5, 1, 1], [1, 1, 1]), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, 0.0001), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, 1.0), kwargs = {})
#   %pow_1 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add, 0.75), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%getitem, %pow_1), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt, torch.float32), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%convert_element_type, 0.5), kwargs = {})
#   %mul_2 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %div_1), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div, %pow_1), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_add_avg_pool3d_div_mul_pow_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x2 = xindex // 512
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (512 + x0), xmask)
    tmp3 = tl.load(in_ptr0 + (1024 + x0), xmask)
    tmp5 = tl.load(in_ptr0 + (1536 + x0), xmask)
    tmp7 = tl.load(in_ptr0 + (2048 + x0), xmask)
    tmp11 = tl.load(in_ptr1 + (x0), xmask)
    tmp19 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp9 = 0.2
    tmp10 = tmp8 * tmp9
    tmp12 = 0.0001
    tmp13 = tmp10 * tmp12
    tmp14 = 1.0
    tmp15 = tmp13 + tmp14
    tmp16 = 0.75
    tmp17 = libdevice.pow(tmp15, tmp16)
    tmp18 = tmp11 / tmp17
    tmp20 = tmp19.to(tl.float32)
    tmp21 = 2.0
    tmp22 = tmp20 * tmp21
    tmp23 = tmp18 * tmp22
    tmp24 = tmp18 / tmp17
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp23, xmask)
    tl.store(out_ptr2 + (x0), tmp24, xmask)


# kernel path: /tmp/torchinductor_sahanp/7h/c7hv5vjwj5u63ahaqbvsopj4azqjvwuxe3vuq6tpvldnmyk623na.py
# Topologically Sorted Source Nodes: [x_4, x_5], Original ATen: [aten.convolution, aten.tanh]
# Source node to ATen node mapping:
#   x_4 => convolution_1
#   x_5 => tanh
# Graph fragment:
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_2, %primals_4, %primals_5, [1, 1, 1], [1, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%convolution_1,), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_tanh_5(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 512
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = libdevice.tanh(tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (10, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(primals_2, (10, ), (1, ))
    assert_size_stride(primals_3, (1, 1, 16, 16, 16), (4096, 4096, 256, 16, 1))
    assert_size_stride(primals_4, (20, 10, 3, 3, 3), (270, 27, 9, 3, 1))
    assert_size_stride(primals_5, (20, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (1, 10, 16, 16, 16), (40960, 4096, 256, 16, 1))
        buf1 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(40960)](buf1, primals_2, 40960, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_2
        buf2 = empty_strided_cuda((2, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [2], out=buf2)
        buf3 = empty_strided_cuda((1, 10, 3), (30, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.rand]
        get_raw_stream(0)
        triton_poi_fused_rand_1[grid(30)](buf2, buf3, 0, 30, XBLOCK=32, num_warps=1, num_stages=1)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.fractional_max_pool3d]
        buf4 = torch.ops.aten.fractional_max_pool3d.default(buf1, [2, 2, 2], [8, 8, 8], buf3)
        del buf3
        buf5 = buf4[0]
        buf6 = buf4[1]
        del buf4
        buf7 = empty_strided_cuda((1, 1, 14, 8, 64), (7168, 7168, 512, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.constant_pad_nd]
        get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_2[grid(7168)](buf5, buf7, 7168, XBLOCK=256, num_warps=4, num_stages=1)
        buf10 = empty_strided_cuda((1, 10, 1, 1, 1), (10, 1, 1, 1, 1), torch.bool)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.bernoulli]
        get_raw_stream(0)
        triton_poi_fused_bernoulli_3[grid(10)](buf2, buf10, 1, 10, XBLOCK=16, num_warps=1, num_stages=1)
        del buf2
        buf8 = empty_strided_cuda((1, 1, 10, 8, 64), (5120, 5120, 512, 64, 1), torch.float32)
        buf11 = empty_strided_cuda((1, 10, 8, 8, 8), (5120, 512, 64, 8, 1), torch.float32)
        buf14 = empty_strided_cuda((1, 10, 8, 8, 8), (5120, 512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2, x_3], Original ATen: [aten.avg_pool3d, aten.mul, aten.add, aten.pow, aten.div, aten._to_copy]
        get_raw_stream(0)
        triton_poi_fused__to_copy_add_avg_pool3d_div_mul_pow_4[grid(5120)](buf7, buf5, buf10, buf8, buf11, buf14, 5120, XBLOCK=128, num_warps=4, num_stages=1)
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, primals_4, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (1, 20, 8, 8, 8), (10240, 512, 64, 8, 1))
        buf13 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [x_4, x_5], Original ATen: [aten.convolution, aten.tanh]
        get_raw_stream(0)
        triton_poi_fused_convolution_tanh_5[grid(10240)](buf13, primals_5, 10240, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_5
    return (buf13, primals_1, primals_3, primals_4, buf1, buf5, buf6, buf7, buf8, buf10, buf11, buf13, buf14, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((10, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((1, 1, 16, 16, 16), (4096, 4096, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((20, 10, 3, 3, 3), (270, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([primals_1, primals_2, primals_3, primals_4, primals_5])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
