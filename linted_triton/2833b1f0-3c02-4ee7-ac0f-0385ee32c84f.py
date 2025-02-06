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


# kernel path: /tmp/torchinductor_sahanp/nu/cnu7y5btmkamkkj35xqncna2e7i5ep75okiycneyuo4hd2tygeoa.py
# Topologically Sorted Source Nodes: [linear, x_1], Original ATen: [aten.addmm, aten.sigmoid]
# Source node to ATen node mapping:
#   linear => add_tensor_3
#   x_1 => sigmoid
# Graph fragment:
#   %add_tensor_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_3, %primals_3), kwargs = {})
#   %sigmoid : [num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_tensor_3,), kwargs = {})

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_sigmoid_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tl.store(in_out_ptr0 + (x0), tmp3, xmask)


# kernel path: /tmp/torchinductor_sahanp/hy/chy5mtt5gzq77nwrfi36rf7fxahuohpxiw7h2obj5c7ot33np6wn.py
# Topologically Sorted Source Nodes: [linear_1, x_2], Original ATen: [aten.addmm, aten.sigmoid]
# Source node to ATen node mapping:
#   linear_1 => add_tensor_2
#   x_2 => sigmoid_1
# Graph fragment:
#   %add_tensor_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_2, %primals_5), kwargs = {})
#   %sigmoid_1 : [num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_tensor_2,), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_sigmoid_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tl.store(in_out_ptr0 + (x0), tmp3, xmask)


# kernel path: /tmp/torchinductor_sahanp/is/cisaxkmhcffj7fhhr3hjomlgjxe52dpiucrefanf2cfjj4ywvykl.py
# Topologically Sorted Source Nodes: [linear_2, x_3], Original ATen: [aten.addmm, aten.sigmoid]
# Source node to ATen node mapping:
#   linear_2 => add_tensor_1
#   x_3 => sigmoid_2
# Graph fragment:
#   %add_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_1, %primals_7), kwargs = {})
#   %sigmoid_2 : [num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_tensor_1,), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_sigmoid_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tl.store(in_out_ptr0 + (x0), tmp3, xmask)


# kernel path: /tmp/torchinductor_sahanp/pe/cpe7tlekzlgzkfslhmhnft5izvmdaunvwszlnkzz5pwc7p24fnad.py
# Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.addmm, aten.sigmoid]
# Source node to ATen node mapping:
#   input_1 => add_tensor
#   input_2 => sigmoid_3
# Graph fragment:
#   %add_tensor : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default, %primals_9), kwargs = {})
#   %sigmoid_3 : [num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_tensor,), kwargs = {})
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_sigmoid_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tl.store(in_out_ptr0 + (x0), tmp3, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11 = args
    args.clear()
    assert_size_stride(primals_1, (1, 3, 32, 32), (3072, 1024, 32, 1))
    assert_size_stride(primals_2, (128, 3072), (3072, 1))
    assert_size_stride(primals_3, (128, ), (1, ))
    assert_size_stride(primals_4, (64, 128), (128, 1))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (32, 64), (64, 1))
    assert_size_stride(primals_7, (32, ), (1, ))
    assert_size_stride(primals_8, (16, 32), (32, 1))
    assert_size_stride(primals_9, (16, ), (1, ))
    assert_size_stride(primals_10, (8, 16), (16, 1))
    assert_size_stride(primals_11, (8, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (1, 3072), (3072, 1), 0), reinterpret_tensor(primals_2, (3072, 128), (1, 3072), 0), out=buf0)
        del primals_2
        buf1 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [linear, x_1], Original ATen: [aten.addmm, aten.sigmoid]
        get_raw_stream(0)
        triton_poi_fused_addmm_sigmoid_0[grid(128)](buf1, primals_3, 128, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_3
        buf2 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.addmm]
        extern_kernels.mm(buf1, reinterpret_tensor(primals_4, (128, 64), (1, 128), 0), out=buf2)
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [linear_1, x_2], Original ATen: [aten.addmm, aten.sigmoid]
        get_raw_stream(0)
        triton_poi_fused_addmm_sigmoid_1[grid(64)](buf3, primals_5, 64, XBLOCK=64, num_warps=1, num_stages=1)
        del primals_5
        buf4 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.addmm]
        extern_kernels.mm(buf3, reinterpret_tensor(primals_6, (64, 32), (1, 64), 0), out=buf4)
        buf5 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [linear_2, x_3], Original ATen: [aten.addmm, aten.sigmoid]
        get_raw_stream(0)
        triton_poi_fused_addmm_sigmoid_2[grid(32)](buf5, primals_7, 32, XBLOCK=32, num_warps=1, num_stages=1)
        del primals_7
        buf6 = empty_strided_cuda((1, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.addmm]
        extern_kernels.mm(buf5, reinterpret_tensor(primals_8, (32, 16), (1, 32), 0), out=buf6)
        buf7 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.addmm, aten.sigmoid]
        get_raw_stream(0)
        triton_poi_fused_addmm_sigmoid_3[grid(16)](buf7, primals_9, 16, XBLOCK=16, num_warps=1, num_stages=1)
        del primals_9
        buf8 = empty_strided_cuda((1, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, buf7, reinterpret_tensor(primals_10, (16, 8), (1, 16), 0), alpha=1, beta=1, out=buf8)
        del primals_11
    return (buf8, reinterpret_tensor(primals_1, (1, 3072), (3072, 1), 0), buf1, buf3, buf5, buf7, primals_10, primals_8, primals_6, primals_4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 3, 32, 32), (3072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((32, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((16, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((8, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    def fn():
        return call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
