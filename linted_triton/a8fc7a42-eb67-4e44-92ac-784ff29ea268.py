# AOT ID: ['48_forward']
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


# kernel path: /tmp/torchinductor_sahanp/lc/clcb7s57hxkhiu7sfilceaqun32jctv6cep3nn6ejqwmva76rixx.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x => convolution
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_5, %primals_1, %primals_2, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12544
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 784
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)


# kernel path: /tmp/torchinductor_sahanp/7q/c7qcmrutnhl7a5st4suumrxq7bgo6cgtn2sh7qy6l5fbhl4gwwqh.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.rand]
# Source node to ATen node mapping:
#   x_1 => inductor_lookup_seed_default, inductor_random_default
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 16, 2], %inductor_lookup_seed_default, rand), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_rand_1(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tl.store(out_ptr0 + (x0), tmp2, xmask)


# kernel path: /tmp/torchinductor_sahanp/46/c46fovwpna6uv7s5b2sdk7smz2pubwllkzpuuifpryrgyxui5kuy.py
# Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten.fractional_max_pool2d, aten._native_batch_norm_legit_functional, aten.mean, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   x_1 => fractional_max_pool2d, getitem_1
#   x_2 => add_4, add_7, mean, mean_1, mul_10, mul_4, rsqrt, sub_2, var_mean
# Graph fragment:
#   %fractional_max_pool2d : [num_users=2] = call_function[target=torch.ops.aten.fractional_max_pool2d.default](args = (%convolution, [2, 2], [14, 14], %inductor_random_default), kwargs = {})
#   %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%fractional_max_pool2d, 1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%getitem, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%getitem, %getitem_3), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_1), kwargs = {})
#   %add_7 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, %unsqueeze_3), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_2, [0]), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_4, [0]), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%getitem, %unsqueeze_6), kwargs = {})
#   %copy_ : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_6, %mean), kwargs = {})
#   %copy__1 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_7, %mean_1), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_fractional_max_pool2d_mean_native_batch_norm_backward_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr4, out_ptr5, out_ptr6, out_ptr8, out_ptr10, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 16
    r0_numel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr0 + (1 + 2*x0), xmask, eviction_policy='evict_last')
    tmp64_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp64_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp64_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        r0_2 = r0_index // 14
        r0_1 = (r0_index % 14)
        r0_3 = r0_index
        tmp1 = r0_2
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tmp2 + tmp0
        tmp4 = 2.0
        tmp5 = tmp3 * tmp4
        tmp6 = libdevice.floor(tmp5)
        tmp7 = tmp0 * tmp4
        tmp8 = libdevice.floor(tmp7)
        tmp9 = tmp6 - tmp8
        tmp10 = tmp9.to(tl.int64)
        tmp11 = tl.full([1, 1], 13, tl.int64)
        tmp12 = tmp2 < tmp11
        tmp13 = tl.full([1, 1], 26, tl.int64)
        tmp14 = tl.where(tmp12, tmp10, tmp13)
        tmp15 = tl.full([XBLOCK, R0_BLOCK], 28, tl.int32)
        tmp16 = tmp14 + tmp15
        tmp17 = tmp14 < 0
        tmp18 = tl.where(tmp17, tmp16, tmp14)
        tl.device_assert(((0 <= tmp18) & (tmp18 < 28)) | ~(r0_mask & xmask), "index out of bounds: 0 <= tmp18 < 28")
        tmp21 = r0_1
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp22 + tmp20
        tmp24 = tmp23 * tmp4
        tmp25 = libdevice.floor(tmp24)
        tmp26 = tmp20 * tmp4
        tmp27 = libdevice.floor(tmp26)
        tmp28 = tmp25 - tmp27
        tmp29 = tmp28.to(tl.int64)
        tmp30 = tmp22 < tmp11
        tmp31 = tl.where(tmp30, tmp29, tmp13)
        tmp32 = tmp31 + tmp15
        tmp33 = tmp31 < 0
        tmp34 = tl.where(tmp33, tmp32, tmp31)
        tl.device_assert(((0 <= tmp34) & (tmp34 < 28)) | ~(r0_mask & xmask), "index out of bounds: 0 <= tmp34 < 28")
        tmp36 = tl.load(in_ptr1 + (tmp34 + 28*tmp18 + 784*x0), r0_mask & xmask, eviction_policy='evict_last')
        tmp37 = tl.load(in_ptr1 + (1 + tmp34 + 28*tmp18 + 784*x0), r0_mask & xmask, eviction_policy='evict_last')
        tmp38 = triton_helpers.maximum(tmp37, tmp36)
        tmp39 = tl.load(in_ptr1 + (28 + tmp34 + 28*tmp18 + 784*x0), r0_mask & xmask, eviction_policy='evict_last')
        tmp40 = triton_helpers.maximum(tmp39, tmp38)
        tmp41 = tl.load(in_ptr1 + (29 + tmp34 + 28*tmp18 + 784*x0), r0_mask & xmask, eviction_policy='evict_last')
        tmp42 = triton_helpers.maximum(tmp41, tmp40)
        tmp43 = tmp37 > tmp36
        tmp44 = libdevice.isnan(tmp37).to(tl.int1)
        tmp45 = tmp43 | tmp44
        tmp46 = 1 + tmp34 + 28*tmp18
        tmp47 = tmp46.to(tl.int32)
        tmp48 = tmp34 + 28*tmp18
        tmp49 = tmp48.to(tl.int32)
        tmp50 = tl.where(tmp45, tmp47, tmp49)
        tmp51 = tmp39 > tmp38
        tmp52 = libdevice.isnan(tmp39).to(tl.int1)
        tmp53 = tmp51 | tmp52
        tmp54 = 28 + tmp34 + 28*tmp18
        tmp55 = tmp54.to(tl.int32)
        tmp56 = tl.where(tmp53, tmp55, tmp50)
        tmp57 = tmp41 > tmp40
        tmp58 = libdevice.isnan(tmp41).to(tl.int1)
        tmp59 = tmp57 | tmp58
        tmp60 = 29 + tmp34 + 28*tmp18
        tmp61 = tmp60.to(tl.int32)
        tmp62 = tl.where(tmp59, tmp61, tmp56)
        tmp63 = tl.broadcast_to(tmp42, [XBLOCK, R0_BLOCK])
        tmp64_mean_next, tmp64_m2_next, tmp64_weight_next = triton_helpers.welford_reduce(
            tmp63, tmp64_mean, tmp64_m2, tmp64_weight, roffset == 0
        )
        tmp64_mean = tl.where(r0_mask & xmask, tmp64_mean_next, tmp64_mean)
        tmp64_m2 = tl.where(r0_mask & xmask, tmp64_m2_next, tmp64_m2)
        tmp64_weight = tl.where(r0_mask & xmask, tmp64_weight_next, tmp64_weight)
        tl.store(out_ptr0 + (r0_3 + 196*x0), tmp42, r0_mask & xmask)
        tl.store(out_ptr1 + (r0_3 + 196*x0), tmp62, r0_mask & xmask)
    tmp67, tmp68, tmp69 = triton_helpers.welford(tmp64_mean, tmp64_m2, tmp64_weight, 1)
    tmp64 = tmp67[:, None]
    tmp65 = tmp68[:, None]
    tmp69[:, None]
    tmp78 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp80 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        r0_3 = r0_index
        tmp70 = tl.load(out_ptr0 + (r0_3 + 196*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp71 = tmp70 - tmp64
        tmp72 = 196.0
        tmp73 = tmp65 / tmp72
        tmp74 = 1e-05
        tmp75 = tmp73 + tmp74
        tmp76 = libdevice.rsqrt(tmp75)
        tmp77 = tmp71 * tmp76
        tmp79 = tmp77 * tmp78
        tmp81 = tmp79 + tmp80
        tl.store(out_ptr4 + (r0_3 + 196*x0), tmp81, r0_mask & xmask)
        tl.store(out_ptr5 + (r0_3 + 196*x0), tmp71, r0_mask & xmask)
    tmp91 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp98 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp82 = 196.0
    tmp83 = tmp65 / tmp82
    tmp84 = 1e-05
    tmp85 = tmp83 + tmp84
    tmp86 = libdevice.rsqrt(tmp85)
    tmp87 = 1.005128205128205
    tmp88 = tmp83 * tmp87
    tmp89 = 0.1
    tmp90 = tmp88 * tmp89
    tmp92 = 0.9
    tmp93 = tmp91 * tmp92
    tmp94 = tmp90 + tmp93
    tmp95 = 1.0
    tmp96 = tmp94 / tmp95
    tmp97 = tmp64 * tmp89
    tmp99 = tmp98 * tmp92
    tmp100 = tmp97 + tmp99
    tmp101 = tmp100 / tmp95
    tl.store(out_ptr6 + (x0), tmp86, xmask)
    tl.store(out_ptr8 + (x0), tmp96, xmask)
    tl.store(out_ptr10 + (x0), tmp101, xmask)


# kernel path: /tmp/torchinductor_sahanp/3n/c3nfomrv6kyqyekqavvyciew3smj3d2iqe5ac3vgfep6mzl4dbp4.py
# Topologically Sorted Source Nodes: [x_3, x_4], Original ATen: [aten.convolution, aten.abs, aten.gt, aten.sign, aten.mul, aten.sub, aten.where]
# Source node to ATen node mapping:
#   x_3 => convolution_1
#   x_4 => abs_1, gt, mul_11, mul_12, sign, sub_3, where
# Graph fragment:
#   %convolution_1 : [num_users=4] = call_function[target=torch.ops.aten.convolution.default](args = (%add_7, %primals_10, %primals_11, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%convolution_1,), kwargs = {})
#   %gt : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%abs_1, 0.5), kwargs = {})
#   %sign : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%convolution_1,), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sign, 0.5), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %mul_11), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_1, 0), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %sub_3, %mul_12), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_abs_convolution_gt_mul_sign_sub_where_3(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 196
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl_math.abs(tmp2)
    tmp4 = 0.5
    tmp5 = tmp3 > tmp4
    tmp6 = tl.full([1], 0, tl.int32)
    tmp7 = tmp6 < tmp2
    tmp8 = tmp7.to(tl.int8)
    tmp9 = tmp2 < tmp6
    tmp10 = tmp9.to(tl.int8)
    tmp11 = tmp8 - tmp10
    tmp12 = tmp11.to(tmp2.dtype)
    tmp13 = tmp12 * tmp4
    tmp14 = tmp2 - tmp13
    tmp15 = 0.0
    tmp16 = tmp2 * tmp15
    tmp17 = tl.where(tmp5, tmp14, tmp16)
    tl.store(out_ptr0 + (x2), tmp5, xmask)
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)


def call(args):
    primals_1, primals_2, _primals_3, _primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11 = args
    args.clear()
    assert_size_stride(primals_1, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (16, ), (1, ))
    assert_size_stride(primals_5, (1, 3, 28, 28), (2352, 784, 28, 1))
    assert_size_stride(primals_6, (16, ), (1, ))
    assert_size_stride(primals_7, (16, ), (1, ))
    assert_size_stride(primals_8, (16, ), (1, ))
    assert_size_stride(primals_9, (16, ), (1, ))
    assert_size_stride(primals_10, (32, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_11, (32, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_5, primals_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (1, 16, 28, 28), (12544, 784, 28, 1))
        buf1 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(12544)](buf1, primals_2, 12544, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_2
        buf2 = empty_strided_cuda((1, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf2)
        buf3 = empty_strided_cuda((1, 16, 2), (32, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.rand]
        get_raw_stream(0)
        triton_poi_fused_rand_1[grid(32)](buf2, buf3, 0, 32, XBLOCK=32, num_warps=1, num_stages=1)
        del buf2
        buf4 = empty_strided_cuda((1, 16, 14, 14), (3136, 196, 14, 1), torch.float32)
        buf5 = empty_strided_cuda((1, 16, 14, 14), (3136, 196, 14, 1), torch.int64)
        buf10 = empty_strided_cuda((1, 16, 14, 14), (3136, 196, 14, 1), torch.float32)
        buf14 = empty_strided_cuda((1, 16, 14, 14), (3136, 196, 14, 1), torch.float32)
        buf9 = empty_strided_cuda((1, 16, 1, 1), (16, 1, 16, 16), torch.float32)
        # Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten.fractional_max_pool2d, aten._native_batch_norm_legit_functional, aten.mean, aten.native_batch_norm_backward]
        get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_fractional_max_pool2d_mean_native_batch_norm_backward_2[grid(16)](buf3, buf1, primals_8, primals_9, primals_7, primals_6, buf4, buf5, buf10, buf14, buf9, primals_7, primals_6, 16, 196, XBLOCK=2, R0_BLOCK=256, num_warps=4, num_stages=1)
        del buf3
        del buf4
        del primals_6
        del primals_7
        del primals_9
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(buf10, primals_10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (1, 32, 14, 14), (6272, 196, 14, 1))
        buf12 = empty_strided_cuda((1, 32, 14, 14), (6272, 196, 14, 1), torch.bool)
        buf13 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [x_3, x_4], Original ATen: [aten.convolution, aten.abs, aten.gt, aten.sign, aten.mul, aten.sub, aten.where]
        get_raw_stream(0)
        triton_poi_fused_abs_convolution_gt_mul_sign_sub_where_3[grid(6272)](buf13, primals_11, buf12, 6272, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_11
    return (reinterpret_tensor(buf13, (1, 196, 32), (6272, 1, 196), 0), primals_1, primals_5, primals_8, primals_10, buf1, buf5, reinterpret_tensor(buf9, (16, ), (1, ), 0), buf10, buf12, buf14, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = 28
    primals_4 = 28
    primals_5 = rand_strided((1, 3, 28, 28), (2352, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((32, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
