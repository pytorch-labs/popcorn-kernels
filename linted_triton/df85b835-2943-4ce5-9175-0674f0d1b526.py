# AOT ID: ['181_inference']
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


# kernel path: /tmp/torchinductor_sahanp/yf/cyf5apwcjugkzd4z6tnthoho7cn4a2lywwe2w2t63dq22kdcamik.py
# Topologically Sorted Source Nodes: [pad, x_2], Original ATen: [aten.replication_pad3d, aten._unsafe_index]
# Source node to ATen node mapping:
#   pad => _unsafe_index_1, _unsafe_index_2, _unsafe_index_3
#   x_2 => _unsafe_index_4
# Graph fragment:
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%unsqueeze_1, [None, None, %clamp_max, None, None]), kwargs = {})
#   %_unsafe_index_2 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_1, [None, None, None, %clamp_max_1, None]), kwargs = {})
#   %_unsafe_index_3 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_2, [None, None, None, None, %clamp_max_2]), kwargs = {})
#   %_unsafe_index_4 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%squeeze, [None, None, %unsqueeze_3, %unsqueeze_4, %convert_element_type_9]), kwargs = {})

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__unsafe_index_replication_pad3d_0(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // ks0) % 6)
    x1 = ((xindex // ks2) % ks3)
    x0 = (xindex % ks2)
    x7 = xindex // ks5
    x4 = xindex
    tmp0 = x2
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp3.to(tl.int32)
    tmp5 = 2.0
    tmp6 = ks1
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 * tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tmp9.to(tl.float64)
    tmp11 = tl.full([1], 2.0, tl.float64)
    tmp12 = tmp11 * tmp10
    tmp13 = tmp10 / tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = x1
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp16 * tmp14
    tmp18 = tmp17.to(tl.int64)
    tmp19 = 2 + 2*ks1
    tmp20 = tmp18 + tmp19
    tmp21 = tmp18 < 0
    tmp22 = tl.where(tmp21, tmp20, tmp18)
    tmp23 = ks4
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp5 * tmp24
    tmp26 = tmp5 + tmp25
    tmp27 = tmp26.to(tl.float64)
    tmp28 = tmp11 * tmp27
    tmp29 = tmp27 / tmp28
    tmp30 = tmp29.to(tl.float32)
    tmp31 = x0
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp32 * tmp30
    tmp34 = tmp33.to(tl.int64)
    tmp35 = 2 + 2*ks4
    tmp36 = tmp34 + tmp35
    tmp37 = tmp34 < 0
    tmp38 = tl.where(tmp37, tmp36, tmp34)
    tmp39 = tmp6.to(tl.float64)
    tmp40 = tmp11 * tmp39
    tmp41 = tmp39 / tmp40
    tmp42 = tmp41.to(tl.float32)
    tmp43 = (((-1) + 2*ks1) * (((-1) + 2*ks1) <= (((0) * ((0) >= ((-1) + tmp22)) + ((-1) + tmp22) * (((-1) + tmp22) > (0))))) + (((0) * ((0) >= ((-1) + tmp22)) + ((-1) + tmp22) * (((-1) + tmp22) > (0)))) * ((((0) * ((0) >= ((-1) + tmp22)) + ((-1) + tmp22) * (((-1) + tmp22) > (0)))) < ((-1) + 2*ks1)))
    tmp44 = tmp43.to(tl.float32)
    tmp45 = tmp44 * tmp42
    tmp46 = tmp45.to(tl.int64)
    tmp47 = tmp46 + tmp6
    tmp48 = tmp46 < 0
    tmp49 = tl.where(tmp48, tmp47, tmp46)
    tmp50 = tmp23.to(tl.float64)
    tmp51 = tmp11 * tmp50
    tmp52 = tmp50 / tmp51
    tmp53 = tmp52.to(tl.float32)
    tmp54 = (((-1) + 2*ks4) * (((-1) + 2*ks4) <= (((0) * ((0) >= ((-1) + tmp38)) + ((-1) + tmp38) * (((-1) + tmp38) > (0))))) + (((0) * ((0) >= ((-1) + tmp38)) + ((-1) + tmp38) * (((-1) + tmp38) > (0)))) * ((((0) * ((0) >= ((-1) + tmp38)) + ((-1) + tmp38) * (((-1) + tmp38) > (0)))) < ((-1) + 2*ks4)))
    tmp55 = tmp54.to(tl.float32)
    tmp56 = tmp55 * tmp53
    tmp57 = tmp56.to(tl.int64)
    tmp58 = tmp57 + tmp23
    tmp59 = tmp57 < 0
    tmp60 = tl.where(tmp59, tmp58, tmp57)
    tmp61 = tl.load(in_ptr0 + (tmp60 + ks4*tmp49 + ks1*ks4*x7), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x4), tmp61, xmask)


def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        16 + 16*s1 + 16*s2 + 16*s1*s2
        4 + 4*s2
        4 + 4*s1
        96 + 96*s1 + 96*s2 + 96*s1*s2
        buf0 = empty_strided_cuda((1, s0, 6, 4 + 4*s1, 4 + 4*s2), (96*s0 + 96*s0*s1 + 96*s0*s2 + 96*s0*s1*s2, 96 + 96*s1 + 96*s2 + 96*s1*s2, 16 + 16*s1 + 16*s2 + 16*s1*s2, 4 + 4*s2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pad, x_2], Original ATen: [aten.replication_pad3d, aten._unsafe_index]
        triton_poi_fused__unsafe_index_replication_pad3d_0_xnumel = 96*s0 + 96*s0*s1 + 96*s0*s2 + 96*s0*s1*s2
        get_raw_stream(0)
        triton_poi_fused__unsafe_index_replication_pad3d_0[grid(triton_poi_fused__unsafe_index_replication_pad3d_0_xnumel)](arg3_1, buf0, 17424, 32, 132, 132, 32, 104544, 313632, XBLOCK=1024, num_warps=4, num_stages=1)
        del arg3_1
    return (reinterpret_tensor(buf0, (1, s0, 1, 6, 4 + 4*s1, 4 + 4*s2), (96*s0 + 96*s0*s1 + 96*s0*s2 + 96*s0*s1*s2, 96 + 96*s1 + 96*s2 + 96*s1*s2, 96 + 96*s1 + 96*s2 + 96*s1*s2, 16 + 16*s1 + 16*s2 + 16*s1*s2, 4 + 4*s2, 1), 0), )


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
