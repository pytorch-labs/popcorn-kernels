# AOT ID: ['172_inference']
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


# kernel path: /tmp/torchinductor_sahanp/sp/csp4ooz44ozbwg5ivbpninkybzfhffrzwvtvlhpdk65koskikvg7.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   x => var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%arg0_1, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_0(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 3
    r0_numel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(r0_mask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(r0_mask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(r0_mask & xmask, tmp2_weight_next, tmp2_weight)
    tmp5, tmp6, tmp7 = triton_helpers.welford(tmp2_mean, tmp2_m2, tmp2_weight, 1)
    tmp2 = tmp5[:, None]
    tmp3 = tmp6[:, None]
    tmp7[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tl.store(out_ptr1 + (x0), tmp3, xmask)


# kernel path: /tmp/torchinductor_sahanp/ei/ceitc7m35nof5zwhonsxslmjbebd5lyupu5wokh4afsilu6zoiii.py
# Topologically Sorted Source Nodes: [randn_like, target], Original ATen: [aten.randn_like, aten._softmax]
# Source node to ATen node mapping:
#   randn_like => inductor_lookup_seed_default, inductor_random_default
#   target => amax
# Graph fragment:
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default : [num_users=2] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 12288], %inductor_lookup_seed_default, randn), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%inductor_random_default, [1], True), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__softmax_randn_like_1(in_ptr0, out_ptr0, out_ptr1, load_seed_offset, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2
    r0_numel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + load_seed_offset)
        tmp1 = r0_1 + 6144*x0
        tmp2 = tl.randn(tmp0, (tmp1).to(tl.uint32))
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = triton_helpers.maximum(_tmp4, tmp3)
        _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
        tl.store(out_ptr0 + (r0_1 + 6144*x0), tmp2, r0_mask & xmask)
    tmp4 = triton_helpers.max2(_tmp4, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp4, xmask)


# kernel path: /tmp/torchinductor_sahanp/qg/cqgz55yxe5nimyn6wj5goyafmctv5v2wytoci43qtylp3sz75adj.py
# Topologically Sorted Source Nodes: [target], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   target => amax
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%inductor_random_default, [1], True), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__softmax_2(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    R0_BLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = triton_helpers.max2(tmp1, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp3, None)


# kernel path: /tmp/torchinductor_sahanp/bf/cbfm42ntxghe4yyzyvjigjcu343qixosiqg2azvc4mivtikmbvdf.py
# Topologically Sorted Source Nodes: [target], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   target => exp, sub_1, sum_1
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%inductor_random_default, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_1,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__softmax_3(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2
    r0_numel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    _tmp6 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 6144*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp0 - tmp2
        tmp4 = tl_math.exp(tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)


# kernel path: /tmp/torchinductor_sahanp/fl/cfld6meghqpsiqwxeecapmtywsr2joejckn2hykwt3uk7dgf3cso.py
# Topologically Sorted Source Nodes: [target], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   target => exp, sub_1, sum_1
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%inductor_random_default, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_1,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__softmax_4(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    R0_BLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp3, None)


# kernel path: /tmp/torchinductor_sahanp/z6/cz6vke3ojkwuqcbav6xpv5mpnz643amzwr3qppehynyvhasc22qc.py
# Topologically Sorted Source Nodes: [log_softmax], Original ATen: [aten._log_softmax]
# Source node to ATen node mapping:
#   log_softmax => amax_1
# Graph fragment:
#   %amax_1 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_2, [1], True), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__log_softmax_5(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2
    r0_numel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    _tmp11 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 6144*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + ((r0_1 + 6144*x0) // 4096), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + ((r0_1 + 6144*x0) // 4096), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 - tmp1
        tmp4 = 4096.0
        tmp5 = tmp3 / tmp4
        tmp6 = 1e-05
        tmp7 = tmp5 + tmp6
        tmp8 = libdevice.rsqrt(tmp7)
        tmp9 = tmp2 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp12 = triton_helpers.maximum(_tmp11, tmp10)
        _tmp11 = tl.where(r0_mask & xmask, tmp12, _tmp11)
    tmp11 = triton_helpers.max2(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp11, xmask)


# kernel path: /tmp/torchinductor_sahanp/zy/czyeigr7hvnp7vbazqqx3eaawnkntkoncnkflwtuei4vzt3k3w5r.py
# Topologically Sorted Source Nodes: [log_softmax], Original ATen: [aten._log_softmax]
# Source node to ATen node mapping:
#   log_softmax => exp_1, sub_2, sum_2
# Graph fragment:
#   %sub_2 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_2, %amax_1), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_2,), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_1, [1], True), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__log_softmax_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2
    r0_numel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    tmp10 = tl.load(in_ptr3 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    _tmp15 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 6144*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + ((r0_1 + 6144*x0) // 4096), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + ((r0_1 + 6144*x0) // 4096), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 - tmp1
        tmp4 = 4096.0
        tmp5 = tmp3 / tmp4
        tmp6 = 1e-05
        tmp7 = tmp5 + tmp6
        tmp8 = libdevice.rsqrt(tmp7)
        tmp9 = tmp2 * tmp8
        tmp12 = tmp9 - tmp11
        tmp13 = tl_math.exp(tmp12)
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(r0_mask & xmask, tmp16, _tmp15)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp15, xmask)


# kernel path: /tmp/torchinductor_sahanp/l7/cl7ls2oftzlmub2riiydsgbczresbvv4tlcx6dr5s62iqyyozt44.py
# Topologically Sorted Source Nodes: [target, loss, log_softmax], Original ATen: [aten._softmax, aten.xlogy, aten._log_softmax, aten.mul, aten.sub, aten.sum]
# Source node to ATen node mapping:
#   log_softmax => log, sub_2, sub_3
#   loss => eq, full_default, full_default_1, isnan, log_1, mul_1, mul_2, sub_4, sum_3, where, where_1
#   target => div, exp, sub_1
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%inductor_random_default, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_1,), kwargs = {})
#   %div : [num_users=5] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
#   %isnan : [num_users=1] = call_function[target=torch.ops.aten.isnan.default](args = (%div,), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], nan), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%div, 0), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %log_1 : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%div,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %log_1), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default, %mul_2), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%isnan, %full_default_1, %where), kwargs = {})
#   %sub_2 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_2, %amax_1), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_2,), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_2, %log), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %sub_3), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_1, %mul_1), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%sub_4,), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__log_softmax__softmax_mul_sub_sum_xlogy_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2
    r0_numel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    tmp1 = tl.load(in_ptr0 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.load(in_ptr1 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp26 = tl.load(in_ptr5 + (0))
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK, R0_BLOCK])
    tmp29 = tl.load(in_ptr6 + (0))
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, R0_BLOCK])
    _tmp36 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 6144*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr2 + (r0_1 + 6144*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tl.load(in_ptr3 + ((r0_1 + 6144*x0) // 4096), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp19 = tl.load(in_ptr4 + ((r0_1 + 6144*x0) // 4096), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp0 - tmp2
        tmp4 = tl_math.exp(tmp3)
        tmp7 = tmp4 / tmp6
        tmp8 = libdevice.isnan(tmp7).to(tl.int1)
        tmp9 = 0.0
        tmp10 = tmp7 == tmp9
        tmp11 = tl_math.log(tmp7)
        tmp12 = tmp7 * tmp11
        tmp13 = tl.where(tmp10, tmp9, tmp12)
        tmp14 = float("nan")
        tmp15 = tl.where(tmp8, tmp14, tmp13)
        tmp18 = tmp16 - tmp17
        tmp20 = 4096.0
        tmp21 = tmp19 / tmp20
        tmp22 = 1e-05
        tmp23 = tmp21 + tmp22
        tmp24 = libdevice.rsqrt(tmp23)
        tmp25 = tmp18 * tmp24
        tmp28 = tmp25 - tmp27
        tmp31 = tl_math.log(tmp30)
        tmp32 = tmp28 - tmp31
        tmp33 = tmp7 * tmp32
        tmp34 = tmp15 - tmp33
        tmp35 = tl.broadcast_to(tmp34, [XBLOCK, R0_BLOCK])
        tmp37 = _tmp36 + tmp35
        _tmp36 = tl.where(r0_mask & xmask, tmp37, _tmp36)
    tmp36 = tl.sum(_tmp36, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp36, xmask)


# kernel path: /tmp/torchinductor_sahanp/xj/cxj2p6xbnzyplsnq2nmf7ivug37jebskiqa7kif5anqxktzqh3km.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten.sum, aten.div]
# Source node to ATen node mapping:
#   loss => div_1, sum_3
# Graph fragment:
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%sub_4,), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_3, 1), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_div_sum_8(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    R0_BLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None]
    tmp4 = 1.0
    tmp5 = tmp3 * tmp4
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp5, None)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (1, 3, 64, 64), (12288, 4096, 64, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 3, 1, 1), (3, 1, 3, 3), torch.float32)
        buf1 = empty_strided_cuda((1, 3, 1, 1), (3, 1, 3, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten._native_batch_norm_legit]
        get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_0[grid(3)](arg0_1, buf0, buf1, 3, 4096, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        buf3 = empty_strided_cuda((1, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf3)
        buf4 = empty_strided_cuda((1, 12288), (12288, 1), torch.float32)
        buf5 = empty_strided_cuda((1, 1, 2), (2, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [randn_like, target], Original ATen: [aten.randn_like, aten._softmax]
        get_raw_stream(0)
        triton_red_fused__softmax_randn_like_1[grid(2)](buf3, buf4, buf5, 0, 2, 6144, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del buf3
        buf6 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [target], Original ATen: [aten._softmax]
        get_raw_stream(0)
        triton_per_fused__softmax_2[grid(1)](buf5, buf6, 1, 2, XBLOCK=1, num_warps=2, num_stages=1)
        buf7 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [target], Original ATen: [aten._softmax]
        get_raw_stream(0)
        triton_red_fused__softmax_3[grid(2)](buf4, buf6, buf7, 2, 6144, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        buf8 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [target], Original ATen: [aten._softmax]
        get_raw_stream(0)
        triton_per_fused__softmax_4[grid(1)](buf7, buf8, 1, 2, XBLOCK=1, num_warps=2, num_stages=1)
        buf9 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [log_softmax], Original ATen: [aten._log_softmax]
        get_raw_stream(0)
        triton_red_fused__log_softmax_5[grid(2)](arg0_1, buf0, buf1, buf9, 2, 6144, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        buf10 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [log_softmax], Original ATen: [aten._log_softmax]
        get_raw_stream(0)
        triton_per_fused__softmax_2[grid(1)](buf9, buf10, 1, 2, XBLOCK=1, num_warps=2, num_stages=1)
        buf11 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [log_softmax], Original ATen: [aten._log_softmax]
        get_raw_stream(0)
        triton_red_fused__log_softmax_6[grid(2)](arg0_1, buf0, buf1, buf10, buf11, 2, 6144, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        buf12 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [log_softmax], Original ATen: [aten._log_softmax]
        get_raw_stream(0)
        triton_per_fused__softmax_4[grid(1)](buf11, buf12, 1, 2, XBLOCK=1, num_warps=2, num_stages=1)
        buf13 = buf4; del buf4  # reuse
        buf14 = reinterpret_tensor(buf11, (2, ), (1, ), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [target, loss, log_softmax], Original ATen: [aten._softmax, aten.xlogy, aten._log_softmax, aten.mul, aten.sub, aten.sum]
        get_raw_stream(0)
        triton_red_fused__log_softmax__softmax_mul_sub_sum_xlogy_7[grid(2)](buf13, buf6, buf8, arg0_1, buf0, buf1, buf10, buf12, buf14, 2, 6144, XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)
        del arg0_1
        del buf0
        del buf1
        del buf10
        del buf12
        del buf13
        del buf6
        buf15 = reinterpret_tensor(buf8, (), (), 0); del buf8  # reuse
        buf16 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [loss], Original ATen: [aten.sum, aten.div]
        get_raw_stream(0)
        triton_per_fused_div_sum_8[grid(1)](buf16, buf14, 1, 2, XBLOCK=1, num_warps=2, num_stages=1)
        del buf14
    return (buf16, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
