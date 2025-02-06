# AOT ID: ['1_forward']
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


# kernel path: /tmp/torchinductor_sahanp/rl/crlknfatwq2hikyvc5k6n2rbimeztxwgfgfhlr2zwv3d3vywv23c.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x => _low_memory_max_pool2d_with_offsets
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%primals_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 32)
    x1 = xindex // 32
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 128*x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 128*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (64 + 2*x0 + 128*x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (65 + 2*x0 + 128*x1), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (x2), tmp6, xmask)


# kernel path: /tmp/torchinductor_sahanp/pj/cpjoyouz5ti72dmmpewmwxdzxb7a7te27nsodo4dqfrnarvjtsho.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_2 => convolution
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%unsqueeze, %primals_2, %primals_3, [2, 2, 2], [0, 0, 0], [1, 1, 1], True, [0, 0, 0], 1), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 295750
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 29575
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)


# kernel path: /tmp/torchinductor_sahanp/er/cerqvojzaobezkhxazcmy5tiv6t3tp57eswyaoc5sn4eqjfq7hlb.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.bernoulli]
# Source node to ATen node mapping:
#   x_3 => inductor_lookup_seed_default, inductor_random_default, lt
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 10, 1, 1, 1], %inductor_lookup_seed_default, rand), kwargs = {})
#   %lt : [num_users=2] = call_function[target=torch.ops.aten.lt.Scalar](args = (%inductor_random_default, 0.5), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_bernoulli_2(in_ptr0, out_ptr1, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/fp/cfpmvlmfzxvtxf2zfh6tlkhmp2yq6p4eeqgxwuvtfubzuotv6a3w.py
# Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   x_5 => var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_1, [2, 3]), kwargs = {correction: 0, keepdim: True})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_group_norm_3(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 38
    r0_numel = 7783
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = (xindex % 19)
    x1 = xindex // 19
    tmp20_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp20_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp20_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_2 = r0_index
        tmp0 = r0_2 + 7783*x0
        tmp1 = tl.full([1, 1], 147875, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (147875*x1 + (((r0_2 + 7783*x0) % 147875))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (5*x1 + ((((r0_2 + 7783*x0) // 29575) % 5))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = 2.0
        tmp7 = tmp5 * tmp6
        tmp8 = tmp3 * tmp7
        tmp9 = tl.full(tmp8.shape, 0, tmp8.dtype)
        tmp10 = tl.where(tmp2, tmp8, tmp9)
        tmp11 = 0.0
        tmp12 = tl.full(tmp11.shape, 0, tmp11.dtype)
        tmp13 = tl.where(tmp2, tmp11, tmp12)
        tmp14 = 1.0
        tmp15 = tl.full(tmp14.shape, 0, tmp14.dtype)
        tmp16 = tl.where(tmp2, tmp14, tmp15)
        tmp17 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp18 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
        tmp19 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
        tmp20_mean_next, tmp20_m2_next, tmp20_weight_next = triton_helpers.welford_combine(
            tmp20_mean, tmp20_m2, tmp20_weight,
            tmp17, tmp18, tmp19
        )
        tmp20_mean = tl.where(r0_mask & xmask, tmp20_mean_next, tmp20_mean)
        tmp20_m2 = tl.where(r0_mask & xmask, tmp20_m2_next, tmp20_m2)
        tmp20_weight = tl.where(r0_mask & xmask, tmp20_weight_next, tmp20_weight)
    tmp23, tmp24, tmp25 = triton_helpers.welford(tmp20_mean, tmp20_m2, tmp20_weight, 1)
    tmp20 = tmp23[:, None]
    tmp21 = tmp24[:, None]
    tmp22 = tmp25[:, None]
    tl.store(out_ptr0 + (x3), tmp20, xmask)
    tl.store(out_ptr1 + (x3), tmp21, xmask)
    tl.store(out_ptr2 + (x3), tmp22, xmask)


# kernel path: /tmp/torchinductor_sahanp/ex/cex7s4cskpetm3oft5wd37zr5viqpgraork4zc2bnlx76jiganpf.py
# Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   x_5 => add, rsqrt, var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_1, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_group_norm_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 2
    r0_numel = 19
    R0_BLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_mask = r0_index < r0_numel
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 19*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 19*x0), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r0_1 + 19*x0), r0_mask & xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp3, 0)
    tmp8 = tl.where(r0_mask & xmask, tmp4, 0)
    tmp9 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp12[:, None]
    tmp16 = 147875.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)


# kernel path: /tmp/torchinductor_sahanp/vh/cvha7yxqrldbm3n4dso44rw4y3ihbopvcpiiv2oc57bt7aaxkjos.py
# Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   x_5 => add_1, mul_2
# Graph fragment:
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_2, %unsqueeze_4), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_2), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_group_norm_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 295750
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 29575
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp6 = tl.load(in_ptr2 + (x1 // 5), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1 // 5), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 2.0
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp5 - tmp6
    tmp9 = 147875.0
    tmp10 = tmp8 / tmp9
    tmp11 = 1e-05
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.rsqrt(tmp12)
    tmp14 = tmp7 * tmp13
    tmp16 = tmp14 * tmp15
    tmp18 = tmp16 + tmp17
    tl.store(out_ptr0 + (x2), tmp18, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (1, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_2, (1, 10, 3, 3, 3), (270, 27, 9, 3, 1))
    assert_size_stride(primals_3, (10, ), (1, ))
    assert_size_stride(primals_4, (10, ), (1, ))
    assert_size_stride(primals_5, (10, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 3, 32, 32), (3072, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.max_pool2d_with_indices]
        get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_0[grid(3072)](primals_1, buf0, 3072, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_1
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(reinterpret_tensor(buf0, (1, 1, 3, 32, 32), (0, 0, 1024, 32, 1), 0), primals_2, stride=(2, 2, 2), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=True, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (1, 10, 7, 65, 65), (295750, 29575, 4225, 65, 1))
        buf2 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.convolution]
        get_raw_stream(0)
        triton_poi_fused_convolution_1[grid(295750)](buf2, primals_3, 295750, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_3
        buf3 = empty_strided_cuda((1, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf3)
        buf5 = empty_strided_cuda((1, 10, 1, 1, 1), (10, 1, 1, 1, 1), torch.bool)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.bernoulli]
        get_raw_stream(0)
        triton_poi_fused_bernoulli_2[grid(10)](buf3, buf5, 0, 10, XBLOCK=16, num_warps=1, num_stages=1)
        del buf3
        buf6 = empty_strided_cuda((1, 2, 1, 1, 19), (38, 19, 38, 38, 1), torch.float32)
        buf7 = empty_strided_cuda((1, 2, 1, 1, 19), (38, 19, 38, 38, 1), torch.float32)
        buf8 = empty_strided_cuda((1, 2, 1, 1, 19), (38, 19, 38, 38, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.native_group_norm]
        get_raw_stream(0)
        triton_red_fused_native_group_norm_3[grid(38)](buf2, buf5, buf6, buf7, buf8, 38, 7783, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        buf9 = empty_strided_cuda((1, 2, 1, 1), (2, 1, 2, 2), torch.float32)
        buf10 = empty_strided_cuda((1, 2, 1, 1), (2, 1, 2, 2), torch.float32)
        buf12 = empty_strided_cuda((1, 2, 1, 1), (2, 1, 2, 2), torch.float32)
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.native_group_norm]
        get_raw_stream(0)
        triton_per_fused_native_group_norm_4[grid(2)](buf6, buf7, buf8, buf9, buf10, buf12, 2, 19, XBLOCK=1, num_warps=2, num_stages=1)
        del buf6
        del buf7
        del buf8
        buf13 = empty_strided_cuda((1, 10, 29575), (295750, 29575, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.native_group_norm]
        get_raw_stream(0)
        triton_poi_fused_native_group_norm_5[grid(295750)](buf2, buf5, buf9, buf10, primals_4, primals_5, buf13, 295750, XBLOCK=512, num_warps=8, num_stages=1)
        del buf10
        del primals_5
    return (reinterpret_tensor(buf13, (1, 29575, 10), (295750, 1, 29575), 0), primals_2, primals_4, reinterpret_tensor(buf0, (1, 1, 3, 32, 32), (3072, 3072, 1024, 32, 1), 0), buf2, buf5, reinterpret_tensor(buf9, (1, 2), (2, 1), 0), reinterpret_tensor(buf12, (1, 2), (2, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((1, 10, 3, 3, 3), (270, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([primals_1, primals_2, primals_3, primals_4, primals_5])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
