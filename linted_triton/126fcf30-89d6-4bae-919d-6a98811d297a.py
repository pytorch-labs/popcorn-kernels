# AOT ID: ['55_inference']
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


# kernel path: /tmp/torchinductor_sahanp/27/c27x4i4r34skysvb7kugt5lnwjderxsgm7wkrunlxjy4fn3pba5i.py
# Topologically Sorted Source Nodes: [x_3, pow_1], Original ATen: [aten.clone, aten.pow]
# Source node to ATen node mapping:
#   pow_1 => pow_1
#   x_3 => clone
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%clone, 2.0), kwargs = {})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_pow_0(in_ptr0, out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    x2 = xindex
    tmp0 = (-2) + x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = (ks1 // 2)*(ks2 // 2)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + ((-2) + x0 + x1*(ks1 // 2)*(ks2 // 2)), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = (-1) + x0
    tmp8 = tmp7 >= tmp1
    tmp9 = tmp7 < tmp3
    tmp10 = tmp8 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-1) + x0 + x1*(ks1 // 2)*(ks2 // 2)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 + tmp6
    tmp13 = x0
    tmp14 = tmp13 >= tmp1
    tmp15 = tmp13 < tmp3
    tmp16 = tmp14 & tmp15
    tmp17 = tl.load(in_ptr0 + (x0 + x1*(ks1 // 2)*(ks2 // 2)), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17 + tmp12
    tmp19 = 0.3333333333333333
    tmp20 = tmp18 * tmp19
    tmp21 = tmp20 * tmp20
    tl.store(out_ptr0 + (x2), tmp21, xmask)


# kernel path: /tmp/torchinductor_sahanp/gh/cghg4d5xinkbhlb2buobvajxy3o57a2kerhekm5qx3doljppnqf6.py
# Topologically Sorted Source Nodes: [x_3, pow_1, out], Original ATen: [aten.clone, aten.pow, aten.avg_pool3d]
# Source node to ATen node mapping:
#   out => avg_pool3d
#   pow_1 => pow_1
#   x_3 => clone
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%clone, 2.0), kwargs = {})
#   %avg_pool3d : [num_users=2] = call_function[target=torch.ops.aten.avg_pool3d.default](args = (%pow_1, [2, 2, 2], [2, 2, 2]), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_avg_pool3d_clone_pow_1(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks1)
    x2 = xindex // ks2
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 4*x1 + 4*ks3*x2 + 2*x1*(ks4 // 2)*(ks5 // 2) + 2*ks3*x2*(ks4 // 2)*(ks5 // 2)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 4*x1 + 4*ks3*x2 + 2*x1*(ks4 // 2)*(ks5 // 2) + 2*ks3*x2*(ks4 // 2)*(ks5 // 2)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2*ks3 + 2*x0 + 4*x1 + 4*ks3*x2 + ks3*(ks4 // 2)*(ks5 // 2) + 2*x1*(ks4 // 2)*(ks5 // 2) + 2*ks3*x2*(ks4 // 2)*(ks5 // 2)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (1 + 2*ks3 + 2*x0 + 4*x1 + 4*ks3*x2 + ks3*(ks4 // 2)*(ks5 // 2) + 2*x1*(ks4 // 2)*(ks5 // 2) + 2*ks3*x2*(ks4 // 2)*(ks5 // 2)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (2 + 2*x0 + 4*x1 + (ks4 // 2)*(ks5 // 2) + 4*ks3*x2 + 2*x1*(ks4 // 2)*(ks5 // 2) + 2*ks3*x2*(ks4 // 2)*(ks5 // 2)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (3 + 2*x0 + 4*x1 + (ks4 // 2)*(ks5 // 2) + 4*ks3*x2 + 2*x1*(ks4 // 2)*(ks5 // 2) + 2*ks3*x2*(ks4 // 2)*(ks5 // 2)), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (2 + 2*ks3 + 2*x0 + 4*x1 + (ks4 // 2)*(ks5 // 2) + 4*ks3*x2 + ks3*(ks4 // 2)*(ks5 // 2) + 2*x1*(ks4 // 2)*(ks5 // 2) + 2*ks3*x2*(ks4 // 2)*(ks5 // 2)), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (3 + 2*ks3 + 2*x0 + 4*x1 + (ks4 // 2)*(ks5 // 2) + 4*ks3*x2 + ks3*(ks4 // 2)*(ks5 // 2) + 2*x1*(ks4 // 2)*(ks5 // 2) + 2*ks3*x2*(ks4 // 2)*(ks5 // 2)), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp15 = 0.125
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (x3), tmp16, xmask)


# kernel path: /tmp/torchinductor_sahanp/i7/ci7egcjuf4vqmguiwumljbkmb4fcqhrbhmvjdfieyfzg3bkzqvvi.py
# Topologically Sorted Source Nodes: [sign, abs_1, relu, mul_2, mul_3, x_4], Original ATen: [aten.sign, aten.abs, aten.relu, aten.mul, aten.pow]
# Source node to ATen node mapping:
#   abs_1 => abs_1
#   mul_2 => mul_64
#   mul_3 => mul_69
#   relu => relu
#   sign => sign
#   x_4 => pow_2
# Graph fragment:
#   %sign : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%avg_pool3d,), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%avg_pool3d,), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%abs_1,), kwargs = {})
#   %mul_64 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign, %relu), kwargs = {})
#   %mul_69 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_64, 8), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_69, 0.5), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_abs_mul_pow_relu_sign_2(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks1)
    x2 = xindex // ks2
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + x2 + ks3*x1 + x2*(triton_helpers.div_floor_integer((ks4 // 2)*(ks5 // 2),  2)) + ks3*x1*(triton_helpers.div_floor_integer((ks4 // 2)*(ks5 // 2),  2))), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp1 < tmp0
    tmp3 = tmp2.to(tl.int8)
    tmp4 = tmp0 < tmp1
    tmp5 = tmp4.to(tl.int8)
    tmp6 = tmp3 - tmp5
    tmp7 = tmp6.to(tmp0.dtype)
    tmp8 = tl_math.abs(tmp0)
    tmp9 = triton_helpers.maximum(tmp1, tmp8)
    tmp10 = tmp7 * tmp9
    tmp11 = 8.0
    tmp12 = tmp10 * tmp11
    tmp13 = libdevice.sqrt(tmp12)
    tl.store(out_ptr0 + (x3), tmp13, xmask)


def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    s3 = arg3_1
    assert_size_stride(arg4_1, (1, s0, s1, s2, s3), (s0*s1*s2*s3, s1*s2*s3, s2*s3, s3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.max_pool3d_with_indices]
        buf0 = torch.ops.aten.max_pool3d_with_indices.default(arg4_1, [2, 2, 2], [2, 2, 2])
        del arg4_1
        buf1 = buf0[0]
        del buf0
        2 + (s2 // 2)*(s3 // 2)
        buf3 = empty_strided_cuda((1, s0, s1 // 2, 2 + (s2 // 2)*(s3 // 2)), (2*s0*(s1 // 2) + s0*(s1 // 2)*(s2 // 2)*(s3 // 2), 2 + (s2 // 2)*(s3 // 2), 2*s0 + s0*(s2 // 2)*(s3 // 2), 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_3, pow_1], Original ATen: [aten.clone, aten.pow]
        triton_poi_fused_clone_pow_0_xnumel = 2*s0*(s1 // 2) + s0*(s1 // 2)*(s2 // 2)*(s3 // 2)
        get_raw_stream(0)
        triton_poi_fused_clone_pow_0[grid(triton_poi_fused_clone_pow_0_xnumel)](buf1, buf3, 1026, 64, 64, 49248, XBLOCK=256, num_warps=4, num_stages=1)
        del buf1
        1 + (((s2 // 2)*(s3 // 2)) // 2)
        s0 // 2
        (s0 // 2)*(((s2 // 2)*(s3 // 2)) // 2) + (s0 // 2)
        buf4 = empty_strided_cuda((1, s0 // 2, s1 // 4, 1 + (((s2 // 2)*(s3 // 2)) // 2)), ((s0 // 2)*(s1 // 4) + (s0 // 2)*(s1 // 4)*(((s2 // 2)*(s3 // 2)) // 2), 1 + (((s2 // 2)*(s3 // 2)) // 2), (s0 // 2)*(((s2 // 2)*(s3 // 2)) // 2) + (s0 // 2), 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_3, pow_1, out], Original ATen: [aten.clone, aten.pow, aten.avg_pool3d]
        triton_poi_fused_avg_pool3d_clone_pow_1_xnumel = (s0 // 2)*(s1 // 4) + (s0 // 2)*(s1 // 4)*(((s2 // 2)*(s3 // 2)) // 2)
        get_raw_stream(0)
        triton_poi_fused_avg_pool3d_clone_pow_1[grid(triton_poi_fused_avg_pool3d_clone_pow_1_xnumel)](buf3, buf4, 513, 1, 513, 3, 64, 64, 4104, XBLOCK=128, num_warps=4, num_stages=1)
        del buf3
        s1 // 4
        (s1 // 4)*(((s2 // 2)*(s3 // 2)) // 2) + (s1 // 4)
        buf5 = empty_strided_cuda((1, s0 // 2, s1 // 4, 1 + (((s2 // 2)*(s3 // 2)) // 2)), ((s1 // 4)*(((s2 // 2)*(s3 // 2)) // 2) + (s1 // 4), (s1 // 4)*(((s2 // 2)*(s3 // 2)) // 2) + (s1 // 4), 1 + (((s2 // 2)*(s3 // 2)) // 2), 1), torch.float32)
        # Topologically Sorted Source Nodes: [sign, abs_1, relu, mul_2, mul_3, x_4], Original ATen: [aten.sign, aten.abs, aten.relu, aten.mul, aten.pow]
        triton_poi_fused_abs_mul_pow_relu_sign_2_xnumel = (s0 // 2)*(s1 // 4) + (s0 // 2)*(s1 // 4)*(((s2 // 2)*(s3 // 2)) // 2)
        get_raw_stream(0)
        triton_poi_fused_abs_mul_pow_relu_sign_2[grid(triton_poi_fused_abs_mul_pow_relu_sign_2_xnumel)](buf4, buf5, 513, 8, 4104, 1, 64, 64, 4104, XBLOCK=128, num_warps=4, num_stages=1)
        del buf4
    return (buf5, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 3
    arg1_1 = 32
    arg2_1 = 64
    arg3_1 = 64
    arg4_1 = rand_strided((1, 3, 32, 64, 64), (393216, 131072, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
