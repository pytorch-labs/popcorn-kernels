# AOT ID: ['86_inference']
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


# kernel path: /tmp/torchinductor_sahanp/73/c73he5bw5ynh5pzgtog7qwntvlzvymxnu6tnilmcniicbs4bkewk.py
# Topologically Sorted Source Nodes: [x, x_1, loss], Original ATen: [aten.replication_pad3d, aten.hardswish, aten.huber_loss]
# Source node to ATen node mapping:
#   loss => abs_1, lt_8, mean, mul_24, mul_25, mul_26, sigmoid, sub_26, where
#   x => _unsafe_index, _unsafe_index_1, _unsafe_index_2
#   x_1 => add_11, clamp_max_3, clamp_min_3, div, mul_6
# Graph fragment:
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg4_1, [None, None, %clamp_max, None, None]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index, [None, None, None, %clamp_max_1, None]), kwargs = {})
#   %_unsafe_index_2 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_1, [None, None, None, None, %clamp_max_2]), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_2, 3), kwargs = {})
#   %clamp_min_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_11, 0), kwargs = {})
#   %clamp_max_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_3, 6), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_2, %clamp_max_3), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_6, 6), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%div,), kwargs = {})
#   %abs_1 : [num_users=4] = call_function[target=torch.ops.aten.abs.default](args = (%sigmoid,), kwargs = {})
#   %lt_8 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%abs_1, 1.0), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%abs_1, 0.5), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_24, %abs_1), kwargs = {})
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%abs_1, 0.5), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_26, 1.0), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%lt_8, %mul_25, %mul_26), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%where,), kwargs = {})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_hardswish_huber_loss_replication_pad3d_0(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 15
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    _tmp26 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp0 = r0_1 + x0*((14 + 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 15)
        tmp1 = 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (ks3*(((-1) + ks2) * (((-1) + ks2) <= (((0) * ((0) >= ((-1) + ((((r0_1 + x0*((14 + 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 15)) // (2 + ks3)) % (2 + ks2))))) + ((-1) + ((((r0_1 + x0*((14 + 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 15)) // (2 + ks3)) % (2 + ks2)))) * (((-1) + ((((r0_1 + x0*((14 + 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 15)) // (2 + ks3)) % (2 + ks2)))) > (0))))) + (((0) * ((0) >= ((-1) + ((((r0_1 + x0*((14 + 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 15)) // (2 + ks3)) % (2 + ks2))))) + ((-1) + ((((r0_1 + x0*((14 + 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 15)) // (2 + ks3)) % (2 + ks2)))) * (((-1) + ((((r0_1 + x0*((14 + 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 15)) // (2 + ks3)) % (2 + ks2)))) > (0)))) * ((((0) * ((0) >= ((-1) + ((((r0_1 + x0*((14 + 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 15)) // (2 + ks3)) % (2 + ks2))))) + ((-1) + ((((r0_1 + x0*((14 + 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 15)) // (2 + ks3)) % (2 + ks2)))) * (((-1) + ((((r0_1 + x0*((14 + 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 15)) // (2 + ks3)) % (2 + ks2)))) > (0)))) < ((-1) + ks2))) + ks2*ks3*(((-1) + ks1) * (((-1) + ks1) <= (((0) * ((0) >= ((-1) + ((((r0_1 + x0*((14 + 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 15)) // (4 + 2*ks2 + 2*ks3 + ks2*ks3)) % (2 + ks1))))) + ((-1) + ((((r0_1 + x0*((14 + 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 15)) // (4 + 2*ks2 + 2*ks3 + ks2*ks3)) % (2 + ks1)))) * (((-1) + ((((r0_1 + x0*((14 + 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 15)) // (4 + 2*ks2 + 2*ks3 + ks2*ks3)) % (2 + ks1)))) > (0))))) + (((0) * ((0) >= ((-1) + ((((r0_1 + x0*((14 + 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 15)) // (4 + 2*ks2 + 2*ks3 + ks2*ks3)) % (2 + ks1))))) + ((-1) + ((((r0_1 + x0*((14 + 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 15)) // (4 + 2*ks2 + 2*ks3 + ks2*ks3)) % (2 + ks1)))) * (((-1) + ((((r0_1 + x0*((14 + 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 15)) // (4 + 2*ks2 + 2*ks3 + ks2*ks3)) % (2 + ks1)))) > (0)))) * ((((0) * ((0) >= ((-1) + ((((r0_1 + x0*((14 + 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 15)) // (4 + 2*ks2 + 2*ks3 + ks2*ks3)) % (2 + ks1))))) + ((-1) + ((((r0_1 + x0*((14 + 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 15)) // (4 + 2*ks2 + 2*ks3 + ks2*ks3)) % (2 + ks1)))) * (((-1) + ((((r0_1 + x0*((14 + 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 15)) // (4 + 2*ks2 + 2*ks3 + ks2*ks3)) % (2 + ks1)))) > (0)))) < ((-1) + ks1))) + ks1*ks2*ks3*((((r0_1 + x0*((14 + 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 15)) // (8 + 4*ks1 + 4*ks2 + 4*ks3 + 2*ks1*ks2 + 2*ks1*ks3 + 2*ks2*ks3 + ks1*ks2*ks3)) % ks0)) + (((-1) + ks3) * (((-1) + ks3) <= (((0) * ((0) >= ((-1) + (((r0_1 + x0*((14 + 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 15)) % (2 + ks3))))) + ((-1) + (((r0_1 + x0*((14 + 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 15)) % (2 + ks3)))) * (((-1) + (((r0_1 + x0*((14 + 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 15)) % (2 + ks3)))) > (0))))) + (((0) * ((0) >= ((-1) + (((r0_1 + x0*((14 + 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 15)) % (2 + ks3))))) + ((-1) + (((r0_1 + x0*((14 + 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 15)) % (2 + ks3)))) * (((-1) + (((r0_1 + x0*((14 + 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 15)) % (2 + ks3)))) > (0)))) * ((((0) * ((0) >= ((-1) + (((r0_1 + x0*((14 + 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 15)) % (2 + ks3))))) + ((-1) + (((r0_1 + x0*((14 + 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 15)) % (2 + ks3)))) * (((-1) + (((r0_1 + x0*((14 + 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3) // 15)) % (2 + ks3)))) > (0)))) < ((-1) + ks3)))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 3.0
        tmp5 = tmp3 + tmp4
        tmp6 = 0.0
        tmp7 = triton_helpers.maximum(tmp5, tmp6)
        tmp8 = 6.0
        tmp9 = triton_helpers.minimum(tmp7, tmp8)
        tmp10 = tmp3 * tmp9
        tmp11 = 0.16666666666666666
        tmp12 = tmp10 * tmp11
        tmp13 = tl.sigmoid(tmp12)
        tmp14 = tl_math.abs(tmp13)
        tmp15 = 1.0
        tmp16 = tmp14 < tmp15
        tmp17 = 0.5
        tmp18 = tmp14 * tmp17
        tmp19 = tmp18 * tmp14
        tmp20 = tmp14 - tmp17
        tmp21 = tmp20 * tmp15
        tmp22 = tl.where(tmp16, tmp19, tmp21)
        tmp23 = tl.full(tmp22.shape, 0, tmp22.dtype)
        tmp24 = tl.where(tmp2, tmp22, tmp23)
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(r0_mask & xmask, tmp27, _tmp26)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp26, xmask)


# kernel path: /tmp/torchinductor_sahanp/hq/chqtpbjt4qdporqhehikm3hq5pbx4kaflhxukckkqoga6ya2osgp.py
# Topologically Sorted Source Nodes: [x, x_1, loss], Original ATen: [aten.replication_pad3d, aten.hardswish, aten.huber_loss]
# Source node to ATen node mapping:
#   loss => abs_1, lt_8, mean, mul_24, mul_25, mul_26, sigmoid, sub_26, where
#   x => _unsafe_index, _unsafe_index_1, _unsafe_index_2
#   x_1 => add_11, clamp_max_3, clamp_min_3, div, mul_6
# Graph fragment:
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg4_1, [None, None, %clamp_max, None, None]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index, [None, None, None, %clamp_max_1, None]), kwargs = {})
#   %_unsafe_index_2 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_1, [None, None, None, None, %clamp_max_2]), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_2, 3), kwargs = {})
#   %clamp_min_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_11, 0), kwargs = {})
#   %clamp_max_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_3, 6), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_2, %clamp_max_3), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_6, 6), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%div,), kwargs = {})
#   %abs_1 : [num_users=4] = call_function[target=torch.ops.aten.abs.default](args = (%sigmoid,), kwargs = {})
#   %lt_8 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%abs_1, 1.0), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%abs_1, 0.5), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_24, %abs_1), kwargs = {})
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%abs_1, 0.5), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_26, 1.0), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%lt_8, %mul_25, %mul_26), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%where,), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_hardswish_huber_loss_replication_pad3d_1(in_out_ptr0, in_ptr0, ks0, ks1, ks2, ks3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 15
    R0_BLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_mask = r0_index < r0_numel
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 8*ks0 + 4*ks0*ks1 + 4*ks0*ks2 + 4*ks0*ks3 + 2*ks0*ks1*ks2 + 2*ks0*ks1*ks3 + 2*ks0*ks2*ks3 + ks0*ks1*ks2*ks3
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp4 / tmp6
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp7, None)


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
        buf0 = empty_strided_cuda((15, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_1, loss], Original ATen: [aten.replication_pad3d, aten.hardswish, aten.huber_loss]
        (14 + 8*s0 + 4*s0*s1 + 4*s0*s2 + 4*s0*s3 + 2*s0*s1*s2 + 2*s0*s1*s3 + 2*s0*s2*s3 + s0*s1*s2*s3) // 15
        get_raw_stream(0)
        triton_red_fused_hardswish_huber_loss_replication_pad3d_0[grid(15)](arg4_1, buf0, 3, 32, 32, 32, 15, 7861, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del arg4_1
        buf1 = empty_strided_cuda((), (), torch.float32)
        buf2 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [x, x_1, loss], Original ATen: [aten.replication_pad3d, aten.hardswish, aten.huber_loss]
        get_raw_stream(0)
        triton_per_fused_hardswish_huber_loss_replication_pad3d_1[grid(1)](buf2, buf0, 3, 32, 32, 32, 1, 15, XBLOCK=1, num_warps=2, num_stages=1)
        del buf0
    return (buf2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 3
    arg1_1 = 32
    arg2_1 = 32
    arg3_1 = 32
    arg4_1 = rand_strided((1, 3, 32, 32, 32), (98304, 32768, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
