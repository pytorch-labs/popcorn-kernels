# AOT ID: ['4_forward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import (
    grid,
    split_scan_grid,
    grid_combo_kernels,
    start_graph,
    end_graph,
    cooperative_reduction_grid,
)
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
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


# kernel path: /tmp/torchinductor_sahanp/vd/cvdhk2pc7lxkdatu72i5wpslrufudmeqsmwy5zqprl4gucq4i7pe.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x => constant_pad_nd
# Graph fragment:
#   %constant_pad_nd : [num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%primals_1, [2, 2], 0.0), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_constant_pad_nd_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 162
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 54)
    x1 = xindex // 54
    x2 = xindex
    tmp0 = (-2) + x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 50, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + ((-2) + x0 + 50*x1), tmp5 & xmask, other=0.0)
    tl.store(out_ptr0 + (x2), tmp6, xmask)




# kernel path: /tmp/torchinductor_sahanp/rq/crqayxt6ncpbadd7jvnb673rz7rbbd4t5mthcg4nihz7coyzhpwq.py
# Topologically Sorted Source Nodes: [hx], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   hx => full_default
# Graph fragment:
#   %full_default : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([1, 32], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__to_copy_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/v4/cv4tmx57xidoa3zgc2k5sgd4mkszkb2jpjpxln2rd7gebjmwhyon.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_1 => convolution
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd, %primals_2, %primals_3, [1], [0], [1], False, [0], 1), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 832
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 52
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)




# kernel path: /tmp/torchinductor_sahanp/uh/cuhhhrqojzfxtqd44dsnijh4kprrzhxl4gwesxo4xmqfs6umvf5b.py
# Topologically Sorted Source Nodes: [ret], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret => mm_default_103
# Graph fragment:
#   %mm_default_103 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/cm/ccmaufbxwonbejn67lpn4wfqoounqq3ia6wmp7agvbq4njtl4pdf.py
# Topologically Sorted Source Nodes: [ret], Original ATen: [aten.addmm, aten.add, aten.tanh]
# Source node to ATen node mapping:
#   ret => add, add_tensor_103, add_tensor_104, tanh
# Graph fragment:
#   %add_tensor_104 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_104, %primals_7), kwargs = {})
#   %add_tensor_103 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_103, %primals_6), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_tensor_104, %add_tensor_103), kwargs = {})
#   %tanh : [num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%add,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_addmm_tanh_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp4 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = libdevice.tanh(tmp6)
    tl.store(in_out_ptr0 + (x0), tmp7, xmask)




# kernel path: /tmp/torchinductor_sahanp/jg/cjgtdw7t6c3hb47tj3dkejxlhjzes6hl5t35vypxw3ibftwkha3o.py
# Topologically Sorted Source Nodes: [ret_1], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_1 => mm_default_101
# Graph fragment:
#   %mm_default_101 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_1, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (1 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/rf/crfffyka2we372qswkcqb43z5f2vpnptn5l7zx632p4tt5ym6pyy.py
# Topologically Sorted Source Nodes: [ret_2], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_2 => mm_default_99
# Graph fragment:
#   %mm_default_99 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_2, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/cw/ccwtpufbicj2ljc3jwlqvaz23jzosnfobruz4asqvrtnf6dcrve2.py
# Topologically Sorted Source Nodes: [ret_3], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_3 => mm_default_97
# Graph fragment:
#   %mm_default_97 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_3, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (3 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/tt/cttm2zn5rk2zruqf72wh6hebu7xgmkq65d6fm3mgs5cgu5eucw3w.py
# Topologically Sorted Source Nodes: [ret_4], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_4 => mm_default_95
# Graph fragment:
#   %mm_default_95 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_4, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/cd/ccdqt3fs3bpk2dev3znc4iwv3ru547gm7hsyj7o2h4274wdxd3za.py
# Topologically Sorted Source Nodes: [ret_5], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_5 => mm_default_93
# Graph fragment:
#   %mm_default_93 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_5, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (5 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/eg/ceguhzqmacoqdopu55cqwokkb6idqkikehbcn2wicr6fvucjldet.py
# Topologically Sorted Source Nodes: [ret_6], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_6 => mm_default_91
# Graph fragment:
#   %mm_default_91 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_6, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (6 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/zi/czikgbtuyyolwgnris4gcbgx6g7kshlrheptwbuebick4re3vovy.py
# Topologically Sorted Source Nodes: [ret_7], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_7 => mm_default_89
# Graph fragment:
#   %mm_default_89 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_7, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_11(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (7 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/zk/czklydjvrfmg5g2wqg6cgghbdk54akz7mmi2w4fdqmdxhtlw26dv.py
# Topologically Sorted Source Nodes: [ret_8], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_8 => mm_default_87
# Graph fragment:
#   %mm_default_87 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_8, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (8 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/qx/cqxh4igbecbbinq6b2vxxncxwghjm2cccjuxwzo4mssyij4zwwqn.py
# Topologically Sorted Source Nodes: [ret_9], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_9 => mm_default_85
# Graph fragment:
#   %mm_default_85 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_9, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_13(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (9 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/dm/cdmxlpxjz7rezjnf5fk7kxmbutjfw3v6uaxadmf444zydvtyeopo.py
# Topologically Sorted Source Nodes: [ret_10], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_10 => mm_default_83
# Graph fragment:
#   %mm_default_83 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_10, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_14(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (10 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/gz/cgzw2swoibmuteita2dyhc7mxe5ogdvzvsfx5gm6wmhwfmvjbbju.py
# Topologically Sorted Source Nodes: [ret_11], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_11 => mm_default_81
# Graph fragment:
#   %mm_default_81 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_11, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_15(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (11 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/im/cimvfejqfmxqywhrfbfn7qaxqcjcwgsd36y5hahgbfxq52iyclmo.py
# Topologically Sorted Source Nodes: [ret_12], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_12 => mm_default_79
# Graph fragment:
#   %mm_default_79 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_12, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_16(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (12 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/ib/cibkpabru6hkmshg2dpwb2bhnj2klo4epxrcw5qgofxpvd3ctb5x.py
# Topologically Sorted Source Nodes: [ret_13], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_13 => mm_default_77
# Graph fragment:
#   %mm_default_77 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_13, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_17(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (13 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/rf/crfllorjpxb6qeyoepcopsg6bdolca2n2rgmbfg5hx6qry4isas7.py
# Topologically Sorted Source Nodes: [ret_14], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_14 => mm_default_75
# Graph fragment:
#   %mm_default_75 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_14, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_18(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (14 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/xz/cxzif3msmr6h76v5zn2co26wnpkn3bf6jidayeu3fps5mp7aezi3.py
# Topologically Sorted Source Nodes: [ret_15], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_15 => mm_default_73
# Graph fragment:
#   %mm_default_73 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_15, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_19(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (15 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/7z/c7zcaekdvzi2hjpsucdsduds7zlrv5tydkscgxoixukiozxvqnsh.py
# Topologically Sorted Source Nodes: [ret_16], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_16 => mm_default_71
# Graph fragment:
#   %mm_default_71 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_16, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_20(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (16 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/77/c77ppvhxamgqaey7jcbsjay2eb7zxkrn7elldhc47kd33kjhogia.py
# Topologically Sorted Source Nodes: [ret_17], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_17 => mm_default_69
# Graph fragment:
#   %mm_default_69 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_17, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_21(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (17 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/ui/cuioexov3it2jrhjdkcegfr4hmpm56m3hkbiuge4kst52a5z25tc.py
# Topologically Sorted Source Nodes: [ret_18], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_18 => mm_default_67
# Graph fragment:
#   %mm_default_67 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_18, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_22(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (18 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/7x/c7xqjmedvnssfzwrh4gxrfyrjrfcxswcmojczwql5fvqbus43fi6.py
# Topologically Sorted Source Nodes: [ret_19], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_19 => mm_default_65
# Graph fragment:
#   %mm_default_65 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_19, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_23(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (19 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/e2/ce2clfojtmkzflbvtmtp7lyqpq2fc6cj64vzsgbasduthsh2lc36.py
# Topologically Sorted Source Nodes: [ret_20], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_20 => mm_default_63
# Graph fragment:
#   %mm_default_63 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_20, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_24(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (20 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/n2/cn2hw3bwkgufxbz2pe5hktz5ccgfsmflg5l3nfgsod3ukkc4s3ko.py
# Topologically Sorted Source Nodes: [ret_21], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_21 => mm_default_61
# Graph fragment:
#   %mm_default_61 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_21, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_25(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (21 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/uu/cuuz636j2jbxddlnicu5dartkqybikgn36mqak7fnccuhl7gtkjt.py
# Topologically Sorted Source Nodes: [ret_22], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_22 => mm_default_59
# Graph fragment:
#   %mm_default_59 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_22, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_26(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (22 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/hq/chqr7heof74qnwumjstckrij56xqobe7axzyap2q5opaixa2dcka.py
# Topologically Sorted Source Nodes: [ret_23], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_23 => mm_default_57
# Graph fragment:
#   %mm_default_57 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_23, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_27(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (23 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/zh/czhhctyy6smgv5kh7hv7ad5fywdipcuzpj7erqawnfnoooy4zg2j.py
# Topologically Sorted Source Nodes: [ret_24], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_24 => mm_default_55
# Graph fragment:
#   %mm_default_55 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_24, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_28(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (24 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/og/cogpqcqtpfg3v2kq5nqvacrnnkqatkx5mpkqg7dmfltuifbknpr4.py
# Topologically Sorted Source Nodes: [ret_25], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_25 => mm_default_53
# Graph fragment:
#   %mm_default_53 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_25, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_29(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (25 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/qr/cqrhfq7ltl4244bhdrfqebi4c44hnqc2si5c4bimftvfhte3rjhl.py
# Topologically Sorted Source Nodes: [ret_26], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_26 => mm_default_51
# Graph fragment:
#   %mm_default_51 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_26, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_30(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (26 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/vj/cvj7lnbnlpm6mbfe5uumwtzxost6ff4ohfxo4ja45cnknreuekpp.py
# Topologically Sorted Source Nodes: [ret_27], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_27 => mm_default_49
# Graph fragment:
#   %mm_default_49 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_27, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_31(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (27 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/6o/c6om7nbmpmfuk7hdx73xrgu345shtjdhnw2uwb6ipmqpycwx7zrd.py
# Topologically Sorted Source Nodes: [ret_28], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_28 => mm_default_47
# Graph fragment:
#   %mm_default_47 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_28, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_32(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (28 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/ds/cdssrbc5lm6flvof5yf6jhvngeag4bajpjokhh7aumhf3fmcjm2v.py
# Topologically Sorted Source Nodes: [ret_29], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_29 => mm_default_45
# Graph fragment:
#   %mm_default_45 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_29, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_33(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (29 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/fj/cfjkdu3aso3sd53fqvivq622ybivbljkgn6okjnxxcerosszb4zb.py
# Topologically Sorted Source Nodes: [ret_30], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_30 => mm_default_43
# Graph fragment:
#   %mm_default_43 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_30, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_34(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (30 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/32/c32c5xruygjs3fcylezsv5hvx4yhzttmtymifhb2woxihi747z7o.py
# Topologically Sorted Source Nodes: [ret_31], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_31 => mm_default_41
# Graph fragment:
#   %mm_default_41 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_31, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_35(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (31 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/6g/c6gopi4u5yhltsj4w4tvlvflsb6e45xwshum2kzbgr4scmiaf2av.py
# Topologically Sorted Source Nodes: [ret_32], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_32 => mm_default_39
# Graph fragment:
#   %mm_default_39 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_32, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_36(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (32 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/hw/chw66cv66brqbcbmi5od2e2znv62ako7jrsxnfwcgwr6xpapjwmt.py
# Topologically Sorted Source Nodes: [ret_33], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_33 => mm_default_37
# Graph fragment:
#   %mm_default_37 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_33, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_37(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (33 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/sq/csqod4hnwqffe2viuticav5vaq326iveqfnyjuadqd2sr55ywm3o.py
# Topologically Sorted Source Nodes: [ret_34], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_34 => mm_default_35
# Graph fragment:
#   %mm_default_35 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_34, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_38(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (34 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/pr/cprb7q6kignsofmqqy2aekkxq5k2mjucx6kb6ce7mbx3q5ub5qb6.py
# Topologically Sorted Source Nodes: [ret_35], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_35 => mm_default_33
# Graph fragment:
#   %mm_default_33 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_35, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_39(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (35 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/ib/cib5qvbueh42blygoi55j45uyvnnrihif7svvuwxsj2e5apvymin.py
# Topologically Sorted Source Nodes: [ret_36], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_36 => mm_default_31
# Graph fragment:
#   %mm_default_31 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_36, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_40(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (36 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/zk/czk7mw4mwgvxxox6bx3vctbyp5weff6ztspxozbiax3bg522d3ru.py
# Topologically Sorted Source Nodes: [ret_37], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_37 => mm_default_29
# Graph fragment:
#   %mm_default_29 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_37, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_41(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (37 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/rw/crwa6v2zbg76djil37ccnbyc6isn3xf6siv3klhdy4vugra6pq7r.py
# Topologically Sorted Source Nodes: [ret_38], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_38 => mm_default_27
# Graph fragment:
#   %mm_default_27 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_38, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_42(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (38 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/t3/ct3au4ld3h54rhjm7d56no4su2dtlwe6rso4yomibwiuzgirin6t.py
# Topologically Sorted Source Nodes: [ret_39], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_39 => mm_default_25
# Graph fragment:
#   %mm_default_25 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_39, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_43(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (39 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/4l/c4ldyfpd3mt4k4xacnubkvbvpvpnvgqunsnlwcxmyrtrtsanlzwh.py
# Topologically Sorted Source Nodes: [ret_40], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_40 => mm_default_23
# Graph fragment:
#   %mm_default_23 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_40, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_44(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (40 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/lq/clql3n5zeofm2fws3e7gq3mig2b7wrimlszzrgpiox3woxb6zbm7.py
# Topologically Sorted Source Nodes: [ret_41], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_41 => mm_default_21
# Graph fragment:
#   %mm_default_21 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_41, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_45(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (41 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/ln/cln5hcejwtyezmtvdtrimq6shxzhuw4zet3ukd353tongxvhxoj4.py
# Topologically Sorted Source Nodes: [ret_42], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_42 => mm_default_19
# Graph fragment:
#   %mm_default_19 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_42, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_46(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (42 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/rc/crc4vdvvswnx54pbpwvkplfta34bko75hh5x666zvo76zex67i4j.py
# Topologically Sorted Source Nodes: [ret_43], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_43 => mm_default_17
# Graph fragment:
#   %mm_default_17 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_43, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_47(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (43 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/us/cusy2e7fsncx3pjzpaj6vjya3mnglzdhplmrjzdmloncshjnjjxt.py
# Topologically Sorted Source Nodes: [ret_44], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_44 => mm_default_15
# Graph fragment:
#   %mm_default_15 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_44, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_48(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (44 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/tb/ctbxmqpxmot3zq24wk6xm37khm5e2bczot5rwm3yjd6lftm6c6fv.py
# Topologically Sorted Source Nodes: [ret_45], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_45 => mm_default_13
# Graph fragment:
#   %mm_default_13 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_45, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_49(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (45 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/x7/cx7ydz7xmudmnrjenh4bdszrzy7qhapfod3n6wzl7gwbpayez4yx.py
# Topologically Sorted Source Nodes: [ret_46], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_46 => mm_default_11
# Graph fragment:
#   %mm_default_11 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_46, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_50(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (46 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/rj/crjxupoyvtsrqikg4h36eikppwmxdaoxjp6qctll4qqdx2ia7sn5.py
# Topologically Sorted Source Nodes: [ret_47], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_47 => mm_default_9
# Graph fragment:
#   %mm_default_9 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_47, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_51(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (47 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/iq/ciq3clxxw33wpl5n652e4jvbvzkqvwcc4iukvjaq7xx7kim7l6ea.py
# Topologically Sorted Source Nodes: [ret_48], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_48 => mm_default_7
# Graph fragment:
#   %mm_default_7 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_48, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_52(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (48 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/lv/clvacoj4yihzeog6vcvd5cuc7m3ga6kcjopthbp77h5dc2uonj4g.py
# Topologically Sorted Source Nodes: [ret_49], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_49 => mm_default_5
# Graph fragment:
#   %mm_default_5 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_49, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_53(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (49 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/qm/cqmxvdit4obanmxvqfbz5edkudlhz2n2xgjufigaha6q4qpmh5l6.py
# Topologically Sorted Source Nodes: [ret_50], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_50 => mm_default_3
# Graph fragment:
#   %mm_default_3 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_50, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_54(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (50 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/62/c62o6kf2wps6wuy4ji6kr5vviiocyjqpkfbhyahqhfnapnxmibv5.py
# Topologically Sorted Source Nodes: [ret_51], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   ret_51 => mm_default_1
# Graph fragment:
#   %mm_default_1 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_51, %permute_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_55(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (51 + 52*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/3m/c3mn5ewzv5co6ckbkrftkgl6kzi3nsogcw3ydswwkc3m4utmennz.py
# Topologically Sorted Source Nodes: [ret_51, x_3, x_5], Original ATen: [aten.addmm, aten.add, aten.tanh, aten.neg, aten._softmax, aten.bernoulli, aten._to_copy, aten.div, aten.mul]
# Source node to ATen node mapping:
#   ret_51 => add_51, add_tensor_1, add_tensor_2, tanh_51
#   x_3 => amax, exp, neg, sub, sum_1
#   x_5 => convert_element_type_1, div_1, inductor_lookup_seed_default, inductor_random_default_1, lt, mul
# Graph fragment:
#   %add_tensor_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_2, %primals_7), kwargs = {})
#   %add_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_1, %primals_6), kwargs = {})
#   %add_51 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_tensor_2, %add_tensor_1), kwargs = {})
#   %tanh_51 : [num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%add_51,), kwargs = {})
#   %neg : [num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%tanh_51,), kwargs = {})
#   %amax : [num_users=2] = call_function[target=torch.ops.aten.amax.default](args = (%neg, [1], True), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%neg, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %sum_1 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 1, 1], %inductor_lookup_seed_default, rand), kwargs = {})
#   %lt : [num_users=2] = call_function[target=torch.ops.aten.lt.Scalar](args = (%inductor_random_default_1, 0.5), kwargs = {})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt, torch.float32), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%convert_element_type_1, 0.5), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze, %div_1), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__softmax__to_copy_add_addmm_bernoulli_div_mul_neg_tanh_56(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, out_ptr3, out_ptr4, load_seed_offset, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 32
    R0_BLOCK: tl.constexpr = 32
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp5 = tl.load(in_out_ptr0 + (r0_0), None)
    tmp6 = tl.load(in_ptr1 + (r0_0), None)
    tmp8 = tl.load(in_ptr2 + (r0_0), None)
    tmp9 = tl.load(in_ptr3 + (r0_0), None)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = tl.full([1, 1], 0, tl.int32)
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.5
    tmp4 = tmp2 < tmp3
    tmp7 = tmp5 + tmp6
    tmp10 = tmp8 + tmp9
    tmp11 = tmp7 + tmp10
    tmp12 = libdevice.tanh(tmp11)
    tmp13 = -tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = triton_helpers.max2(tmp14, 1)[:, None]
    tmp17 = tmp13 - tmp16
    tmp18 = tl_math.exp(tmp17)
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
    tmp21 = tl.sum(tmp19, 1)[:, None]
    tmp22 = tmp18 / tmp21
    tmp23 = tmp4.to(tl.float32)
    tmp24 = 2.0
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 * tmp25
    tl.store(out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp4, None)
    tl.store(in_out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp12, None)
    tl.store(out_ptr4 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp26, None)
    tl.store(out_ptr2 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp16, None)
    tl.store(out_ptr3 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp21, None)




# kernel path: /tmp/torchinductor_sahanp/jv/cjv5q7nuc4bxwj3sthhqwgyr23aocfqr2urfueq4gywkmx2g3prx.py
# Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.addmm, aten.relu]
# Source node to ATen node mapping:
#   input_1 => add_tensor
#   input_2 => relu
# Graph fragment:
#   %add_tensor : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default, %primals_9), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_relu_57(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)




#include "/tmp/torchinductor_sahanp/3b/c3bi5gk6mslf6u4iaqafhxm64z6u65e3eain4xlary5blqnvv6xx.h"
extern "C"  void kernel(int64_t* in_out_ptr0)
{
    {
        {
            {
                auto tmp0 = in_out_ptr0[static_cast<int64_t>(0L)];
                auto tmp1 = static_cast<int32_t>(0);
                auto tmp2 = static_cast<int64_t>(0);
                auto tmp3 = static_cast<int64_t>(10);
                auto tmp4 = randint64_cpu(tmp0, tmp1, tmp2, tmp3);
                in_out_ptr0[static_cast<int64_t>(0L)] = tmp4;
            }
        }
    }
}
''')


# kernel path: /tmp/torchinductor_sahanp/ac/cac6rzmlhgqno4o27jtsulip2pfodpvy5xz4laa2h7kkleyynpep.py
# Topologically Sorted Source Nodes: [ones_like, target_poisson], Original ATen: [aten.ones_like, aten.poisson]
# Source node to ATen node mapping:
#   ones_like => full_default_1
#   target_poisson => poisson
# Graph fragment:
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 10], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %poisson : [num_users=2] = call_function[target=torch.ops.aten.poisson.default](args = (%full_default_1,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_ones_like_poisson_59(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 1.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)




# kernel path: /tmp/torchinductor_sahanp/q2/cq2qinfdgrkelllethylawcdgglfbao5i7nvimmoaitrvgdzpjuu.py
# Topologically Sorted Source Nodes: [target_l1, l1_loss, ce_loss, poisson_loss], Original ATen: [aten.randn_like, aten.sub, aten.abs, aten.mean, aten._log_softmax, aten.nll_loss_forward, aten.mul, aten.exp]
# Source node to ATen node mapping:
#   ce_loss => amax_1, convert_element_type_3, div_2, exp_1, full_default_3, log, ne, neg_1, sub_2, sum_2, sum_3, sum_4, where_1
#   l1_loss => abs_1, mean, sub_1
#   poisson_loss => exp_2, mean_1, mul_1, sub_4
#   target_l1 => inductor_lookup_seed_default_1, inductor_random_default
# Graph fragment:
#   %inductor_lookup_seed_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 1), kwargs = {})
#   %inductor_random_default : [num_users=2] = call_function[target=torch.ops.prims.inductor_random.default](args = ([1, 10], %inductor_lookup_seed_default_1, randn), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%addmm_105, %inductor_random_default), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_1,), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%abs_1,), kwargs = {})
#   %amax_1 : [num_users=2] = call_function[target=torch.ops.aten.amax.default](args = (%addmm_105, [1], True), kwargs = {})
#   %sub_2 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%addmm_105, %amax_1), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_2,), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_1, [1], True), kwargs = {})
#   %log : [num_users=2] = call_function[target=torch.ops.aten.log.default](args = (%sum_2,), kwargs = {})
#   %ne : [num_users=4] = call_function[target=torch.ops.aten.ne.Scalar](args = (%device_put_1, -100), kwargs = {})
#   %neg_1 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze_1,), kwargs = {})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne, %neg_1, %full_default_3), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne,), kwargs = {})
#   %convert_element_type_3 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_3, torch.float32), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_1,), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_4, %convert_element_type_3), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%poisson, %addmm_105), kwargs = {})
#   %exp_2 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%addmm_105,), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%exp_2, %mul_1), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sub_4,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__log_softmax_abs_exp_mean_mul_nll_loss_forward_randn_like_sub_60(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3, load_seed_offset, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 10
    R0_BLOCK: tl.constexpr = 16
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp3 = tl.load(in_ptr1 + (r0_0), r0_mask, other=0.0)
    tmp21 = tl.load(in_ptr2 + (r0_0), r0_mask, other=0.0)
    tmp29 = tl.load(in_ptr3 + (0))
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, 1])
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = r0_0
    tmp2 = tl.randn(tmp0, (tmp1).to(tl.uint32))
    tmp4 = tmp3 - tmp2
    tmp5 = tl_math.abs(tmp4)
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp8 = tl.where(r0_mask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp12 = tl.where(r0_mask, tmp10, float("-inf"))
    tmp13 = triton_helpers.max2(tmp12, 1)[:, None]
    tmp14 = tmp3 - tmp13
    tmp15 = tl_math.exp(tmp14)
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
    tmp18 = tl.where(r0_mask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tmp20 = tl_math.exp(tmp3)
    tmp22 = tmp21 * tmp3
    tmp23 = tmp20 - tmp22
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK, R0_BLOCK])
    tmp26 = tl.where(r0_mask, tmp24, 0)
    tmp27 = tl.sum(tmp26, 1)[:, None]
    tmp28 = tl_math.log(tmp19)
    tmp31 = tl.full([1, 1], -100, tl.int64)
    tmp32 = tmp30 != tmp31
    tmp33 = tl.full([1, 1], 0, tl.int64)
    tmp34 = tl.where(tmp32, tmp30, tmp33)
    tmp35 = tl.full([XBLOCK, 1], 10, tl.int32)
    tmp36 = tmp34 + tmp35
    tmp37 = tmp34 < 0
    tmp38 = tl.where(tmp37, tmp36, tmp34)
    tl.device_assert((0 <= tmp38) & (tmp38 < 10), "index out of bounds: 0 <= tmp38 < 10")
    tmp40 = tl.load(in_ptr1 + (tmp38), None, eviction_policy='evict_last')
    tmp41 = tmp40 - tmp13
    tmp42 = tmp41 - tmp28
    tmp43 = -tmp42
    tmp44 = 0.0
    tmp45 = tl.where(tmp32, tmp43, tmp44)
    tmp46 = tmp32.to(tl.int64)
    tmp47 = tmp46.to(tl.float32)
    tmp48 = tmp45 / tmp47
    tmp49 = 10.0
    tmp50 = tmp27 / tmp49
    tmp51 = tmp9 / tmp49
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp2, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp28, None)
    tl.store(out_ptr2 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp32, None)
    tl.store(out_ptr3 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp48, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp50, None)
    tl.debug_barrier()
    tl.store(in_out_ptr2 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp51, None)
    tl.store(out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp13, None)







def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11 = args
    args.clear()
    assert_size_stride(primals_1, (1, 3, 50), (150, 50, 1))
    assert_size_stride(primals_2, (16, 3, 3), (9, 3, 1))
    assert_size_stride(primals_3, (16, ), (1, ))
    assert_size_stride(primals_4, (32, 16), (16, 1))
    assert_size_stride(primals_5, (32, 32), (32, 1))
    assert_size_stride(primals_6, (32, ), (1, ))
    assert_size_stride(primals_7, (32, ), (1, ))
    assert_size_stride(primals_8, (64, 32), (32, 1))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_10, (10, 64), (64, 1))
    assert_size_stride(primals_11, (10, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 3, 54), (162, 54, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_0[grid(162)](primals_1, buf0, 162, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_1
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(buf0, primals_2, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf1, (1, 16, 52), (832, 52, 1))
        buf2 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1[grid(32)](buf2, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf3 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_2[grid(832)](buf3, primals_3, 832, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_3
        buf4 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret], Original ATen: [aten.addmm]
        extern_kernels.mm(buf2, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf4)
        buf5 = empty_strided_cuda((1, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_3[grid(16)](buf3, buf5, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf6 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret], Original ATen: [aten.addmm]
        extern_kernels.mm(buf5, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf6)
        buf7 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [ret], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf7, primals_7, buf6, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf8 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [ret_1], Original ATen: [aten.addmm]
        extern_kernels.mm(buf7, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf8)
        buf9 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [ret_1], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_5[grid(16)](buf3, buf9, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf10 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_1], Original ATen: [aten.addmm]
        extern_kernels.mm(buf9, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf10)
        buf11 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [ret_1], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf11, primals_7, buf10, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf12 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [ret_2], Original ATen: [aten.addmm]
        extern_kernels.mm(buf11, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf12)
        buf13 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [ret_2], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_6[grid(16)](buf3, buf13, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf14 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_2], Original ATen: [aten.addmm]
        extern_kernels.mm(buf13, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf14)
        buf15 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [ret_2], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf15, primals_7, buf14, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf16 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [ret_3], Original ATen: [aten.addmm]
        extern_kernels.mm(buf15, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf16)
        buf17 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [ret_3], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_7[grid(16)](buf3, buf17, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf18 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_3], Original ATen: [aten.addmm]
        extern_kernels.mm(buf17, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf18)
        buf19 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [ret_3], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf19, primals_7, buf18, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf20 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [ret_4], Original ATen: [aten.addmm]
        extern_kernels.mm(buf19, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf20)
        buf21 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [ret_4], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_8[grid(16)](buf3, buf21, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf22 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_4], Original ATen: [aten.addmm]
        extern_kernels.mm(buf21, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf22)
        buf23 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [ret_4], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf23, primals_7, buf22, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf24 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [ret_5], Original ATen: [aten.addmm]
        extern_kernels.mm(buf23, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf24)
        buf25 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [ret_5], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_9[grid(16)](buf3, buf25, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf26 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_5], Original ATen: [aten.addmm]
        extern_kernels.mm(buf25, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf26)
        buf27 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [ret_5], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf27, primals_7, buf26, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf28 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [ret_6], Original ATen: [aten.addmm]
        extern_kernels.mm(buf27, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf28)
        buf29 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [ret_6], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_10[grid(16)](buf3, buf29, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf30 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_6], Original ATen: [aten.addmm]
        extern_kernels.mm(buf29, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf30)
        buf31 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [ret_6], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf31, primals_7, buf30, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf32 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [ret_7], Original ATen: [aten.addmm]
        extern_kernels.mm(buf31, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf32)
        buf33 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [ret_7], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_11[grid(16)](buf3, buf33, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf34 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_7], Original ATen: [aten.addmm]
        extern_kernels.mm(buf33, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf34)
        buf35 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [ret_7], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf35, primals_7, buf34, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf36 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [ret_8], Original ATen: [aten.addmm]
        extern_kernels.mm(buf35, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf36)
        buf37 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [ret_8], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_12[grid(16)](buf3, buf37, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf38 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_8], Original ATen: [aten.addmm]
        extern_kernels.mm(buf37, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf38)
        buf39 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [ret_8], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf39, primals_7, buf38, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf40 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [ret_9], Original ATen: [aten.addmm]
        extern_kernels.mm(buf39, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf40)
        buf41 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [ret_9], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_13[grid(16)](buf3, buf41, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf42 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_9], Original ATen: [aten.addmm]
        extern_kernels.mm(buf41, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf42)
        buf43 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [ret_9], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf43, primals_7, buf42, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf44 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [ret_10], Original ATen: [aten.addmm]
        extern_kernels.mm(buf43, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf44)
        buf45 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [ret_10], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_14[grid(16)](buf3, buf45, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf46 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_10], Original ATen: [aten.addmm]
        extern_kernels.mm(buf45, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf46)
        buf47 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [ret_10], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf47, primals_7, buf46, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf48 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [ret_11], Original ATen: [aten.addmm]
        extern_kernels.mm(buf47, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf48)
        buf49 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [ret_11], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_15[grid(16)](buf3, buf49, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf50 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_11], Original ATen: [aten.addmm]
        extern_kernels.mm(buf49, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf50)
        buf51 = buf48; del buf48  # reuse
        # Topologically Sorted Source Nodes: [ret_11], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf51, primals_7, buf50, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf52 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [ret_12], Original ATen: [aten.addmm]
        extern_kernels.mm(buf51, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf52)
        buf53 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [ret_12], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_16[grid(16)](buf3, buf53, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf54 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_12], Original ATen: [aten.addmm]
        extern_kernels.mm(buf53, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf54)
        buf55 = buf52; del buf52  # reuse
        # Topologically Sorted Source Nodes: [ret_12], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf55, primals_7, buf54, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf56 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [ret_13], Original ATen: [aten.addmm]
        extern_kernels.mm(buf55, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf56)
        buf57 = buf53; del buf53  # reuse
        # Topologically Sorted Source Nodes: [ret_13], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_17[grid(16)](buf3, buf57, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf58 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_13], Original ATen: [aten.addmm]
        extern_kernels.mm(buf57, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf58)
        buf59 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [ret_13], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf59, primals_7, buf58, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf60 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [ret_14], Original ATen: [aten.addmm]
        extern_kernels.mm(buf59, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf60)
        buf61 = buf57; del buf57  # reuse
        # Topologically Sorted Source Nodes: [ret_14], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_18[grid(16)](buf3, buf61, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf62 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_14], Original ATen: [aten.addmm]
        extern_kernels.mm(buf61, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf62)
        buf63 = buf60; del buf60  # reuse
        # Topologically Sorted Source Nodes: [ret_14], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf63, primals_7, buf62, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf64 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [ret_15], Original ATen: [aten.addmm]
        extern_kernels.mm(buf63, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf64)
        buf65 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [ret_15], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_19[grid(16)](buf3, buf65, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf66 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_15], Original ATen: [aten.addmm]
        extern_kernels.mm(buf65, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf66)
        buf67 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [ret_15], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf67, primals_7, buf66, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf68 = buf66; del buf66  # reuse
        # Topologically Sorted Source Nodes: [ret_16], Original ATen: [aten.addmm]
        extern_kernels.mm(buf67, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf68)
        buf69 = buf65; del buf65  # reuse
        # Topologically Sorted Source Nodes: [ret_16], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_20[grid(16)](buf3, buf69, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf70 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_16], Original ATen: [aten.addmm]
        extern_kernels.mm(buf69, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf70)
        buf71 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [ret_16], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf71, primals_7, buf70, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf72 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [ret_17], Original ATen: [aten.addmm]
        extern_kernels.mm(buf71, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf72)
        buf73 = buf69; del buf69  # reuse
        # Topologically Sorted Source Nodes: [ret_17], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_21[grid(16)](buf3, buf73, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf74 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_17], Original ATen: [aten.addmm]
        extern_kernels.mm(buf73, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf74)
        buf75 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [ret_17], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf75, primals_7, buf74, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf76 = buf74; del buf74  # reuse
        # Topologically Sorted Source Nodes: [ret_18], Original ATen: [aten.addmm]
        extern_kernels.mm(buf75, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf76)
        buf77 = buf73; del buf73  # reuse
        # Topologically Sorted Source Nodes: [ret_18], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_22[grid(16)](buf3, buf77, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf78 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_18], Original ATen: [aten.addmm]
        extern_kernels.mm(buf77, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf78)
        buf79 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [ret_18], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf79, primals_7, buf78, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf80 = buf78; del buf78  # reuse
        # Topologically Sorted Source Nodes: [ret_19], Original ATen: [aten.addmm]
        extern_kernels.mm(buf79, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf80)
        buf81 = buf77; del buf77  # reuse
        # Topologically Sorted Source Nodes: [ret_19], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_23[grid(16)](buf3, buf81, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf82 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_19], Original ATen: [aten.addmm]
        extern_kernels.mm(buf81, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf82)
        buf83 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [ret_19], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf83, primals_7, buf82, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf84 = buf82; del buf82  # reuse
        # Topologically Sorted Source Nodes: [ret_20], Original ATen: [aten.addmm]
        extern_kernels.mm(buf83, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf84)
        buf85 = buf81; del buf81  # reuse
        # Topologically Sorted Source Nodes: [ret_20], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_24[grid(16)](buf3, buf85, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf86 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_20], Original ATen: [aten.addmm]
        extern_kernels.mm(buf85, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf86)
        buf87 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [ret_20], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf87, primals_7, buf86, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf88 = buf86; del buf86  # reuse
        # Topologically Sorted Source Nodes: [ret_21], Original ATen: [aten.addmm]
        extern_kernels.mm(buf87, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf88)
        buf89 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [ret_21], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_25[grid(16)](buf3, buf89, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf90 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_21], Original ATen: [aten.addmm]
        extern_kernels.mm(buf89, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf90)
        buf91 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [ret_21], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf91, primals_7, buf90, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf92 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [ret_22], Original ATen: [aten.addmm]
        extern_kernels.mm(buf91, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf92)
        buf93 = buf89; del buf89  # reuse
        # Topologically Sorted Source Nodes: [ret_22], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_26[grid(16)](buf3, buf93, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf94 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_22], Original ATen: [aten.addmm]
        extern_kernels.mm(buf93, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf94)
        buf95 = buf92; del buf92  # reuse
        # Topologically Sorted Source Nodes: [ret_22], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf95, primals_7, buf94, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf96 = buf94; del buf94  # reuse
        # Topologically Sorted Source Nodes: [ret_23], Original ATen: [aten.addmm]
        extern_kernels.mm(buf95, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf96)
        buf97 = buf93; del buf93  # reuse
        # Topologically Sorted Source Nodes: [ret_23], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_27[grid(16)](buf3, buf97, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf98 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_23], Original ATen: [aten.addmm]
        extern_kernels.mm(buf97, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf98)
        buf99 = buf96; del buf96  # reuse
        # Topologically Sorted Source Nodes: [ret_23], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf99, primals_7, buf98, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf100 = buf98; del buf98  # reuse
        # Topologically Sorted Source Nodes: [ret_24], Original ATen: [aten.addmm]
        extern_kernels.mm(buf99, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf100)
        buf101 = buf97; del buf97  # reuse
        # Topologically Sorted Source Nodes: [ret_24], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_28[grid(16)](buf3, buf101, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf102 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_24], Original ATen: [aten.addmm]
        extern_kernels.mm(buf101, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf102)
        buf103 = buf100; del buf100  # reuse
        # Topologically Sorted Source Nodes: [ret_24], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf103, primals_7, buf102, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf104 = buf102; del buf102  # reuse
        # Topologically Sorted Source Nodes: [ret_25], Original ATen: [aten.addmm]
        extern_kernels.mm(buf103, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf104)
        buf105 = buf101; del buf101  # reuse
        # Topologically Sorted Source Nodes: [ret_25], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_29[grid(16)](buf3, buf105, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf106 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_25], Original ATen: [aten.addmm]
        extern_kernels.mm(buf105, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf106)
        buf107 = buf104; del buf104  # reuse
        # Topologically Sorted Source Nodes: [ret_25], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf107, primals_7, buf106, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf108 = buf106; del buf106  # reuse
        # Topologically Sorted Source Nodes: [ret_26], Original ATen: [aten.addmm]
        extern_kernels.mm(buf107, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf108)
        buf109 = buf105; del buf105  # reuse
        # Topologically Sorted Source Nodes: [ret_26], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_30[grid(16)](buf3, buf109, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf110 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_26], Original ATen: [aten.addmm]
        extern_kernels.mm(buf109, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf110)
        buf111 = buf108; del buf108  # reuse
        # Topologically Sorted Source Nodes: [ret_26], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf111, primals_7, buf110, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf112 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [ret_27], Original ATen: [aten.addmm]
        extern_kernels.mm(buf111, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf112)
        buf113 = buf109; del buf109  # reuse
        # Topologically Sorted Source Nodes: [ret_27], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_31[grid(16)](buf3, buf113, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf114 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_27], Original ATen: [aten.addmm]
        extern_kernels.mm(buf113, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf114)
        buf115 = buf112; del buf112  # reuse
        # Topologically Sorted Source Nodes: [ret_27], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf115, primals_7, buf114, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf116 = buf114; del buf114  # reuse
        # Topologically Sorted Source Nodes: [ret_28], Original ATen: [aten.addmm]
        extern_kernels.mm(buf115, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf116)
        buf117 = buf113; del buf113  # reuse
        # Topologically Sorted Source Nodes: [ret_28], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_32[grid(16)](buf3, buf117, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf118 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_28], Original ATen: [aten.addmm]
        extern_kernels.mm(buf117, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf118)
        buf119 = buf116; del buf116  # reuse
        # Topologically Sorted Source Nodes: [ret_28], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf119, primals_7, buf118, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf120 = buf118; del buf118  # reuse
        # Topologically Sorted Source Nodes: [ret_29], Original ATen: [aten.addmm]
        extern_kernels.mm(buf119, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf120)
        buf121 = buf117; del buf117  # reuse
        # Topologically Sorted Source Nodes: [ret_29], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_33[grid(16)](buf3, buf121, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf122 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_29], Original ATen: [aten.addmm]
        extern_kernels.mm(buf121, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf122)
        buf123 = buf120; del buf120  # reuse
        # Topologically Sorted Source Nodes: [ret_29], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf123, primals_7, buf122, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf124 = buf122; del buf122  # reuse
        # Topologically Sorted Source Nodes: [ret_30], Original ATen: [aten.addmm]
        extern_kernels.mm(buf123, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf124)
        buf125 = buf121; del buf121  # reuse
        # Topologically Sorted Source Nodes: [ret_30], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_34[grid(16)](buf3, buf125, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf126 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_30], Original ATen: [aten.addmm]
        extern_kernels.mm(buf125, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf126)
        buf127 = buf124; del buf124  # reuse
        # Topologically Sorted Source Nodes: [ret_30], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf127, primals_7, buf126, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf128 = buf126; del buf126  # reuse
        # Topologically Sorted Source Nodes: [ret_31], Original ATen: [aten.addmm]
        extern_kernels.mm(buf127, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf128)
        buf129 = buf125; del buf125  # reuse
        # Topologically Sorted Source Nodes: [ret_31], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_35[grid(16)](buf3, buf129, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf130 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_31], Original ATen: [aten.addmm]
        extern_kernels.mm(buf129, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf130)
        buf131 = buf128; del buf128  # reuse
        # Topologically Sorted Source Nodes: [ret_31], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf131, primals_7, buf130, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf132 = buf130; del buf130  # reuse
        # Topologically Sorted Source Nodes: [ret_32], Original ATen: [aten.addmm]
        extern_kernels.mm(buf131, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf132)
        buf133 = buf129; del buf129  # reuse
        # Topologically Sorted Source Nodes: [ret_32], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_36[grid(16)](buf3, buf133, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf134 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_32], Original ATen: [aten.addmm]
        extern_kernels.mm(buf133, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf134)
        buf135 = buf132; del buf132  # reuse
        # Topologically Sorted Source Nodes: [ret_32], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf135, primals_7, buf134, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf136 = buf134; del buf134  # reuse
        # Topologically Sorted Source Nodes: [ret_33], Original ATen: [aten.addmm]
        extern_kernels.mm(buf135, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf136)
        buf137 = buf133; del buf133  # reuse
        # Topologically Sorted Source Nodes: [ret_33], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_37[grid(16)](buf3, buf137, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf138 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_33], Original ATen: [aten.addmm]
        extern_kernels.mm(buf137, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf138)
        buf139 = buf136; del buf136  # reuse
        # Topologically Sorted Source Nodes: [ret_33], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf139, primals_7, buf138, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf140 = buf138; del buf138  # reuse
        # Topologically Sorted Source Nodes: [ret_34], Original ATen: [aten.addmm]
        extern_kernels.mm(buf139, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf140)
        buf141 = buf137; del buf137  # reuse
        # Topologically Sorted Source Nodes: [ret_34], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_38[grid(16)](buf3, buf141, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf142 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_34], Original ATen: [aten.addmm]
        extern_kernels.mm(buf141, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf142)
        buf143 = buf140; del buf140  # reuse
        # Topologically Sorted Source Nodes: [ret_34], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf143, primals_7, buf142, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf144 = buf142; del buf142  # reuse
        # Topologically Sorted Source Nodes: [ret_35], Original ATen: [aten.addmm]
        extern_kernels.mm(buf143, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf144)
        buf145 = buf141; del buf141  # reuse
        # Topologically Sorted Source Nodes: [ret_35], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_39[grid(16)](buf3, buf145, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf146 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_35], Original ATen: [aten.addmm]
        extern_kernels.mm(buf145, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf146)
        buf147 = buf144; del buf144  # reuse
        # Topologically Sorted Source Nodes: [ret_35], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf147, primals_7, buf146, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf148 = buf146; del buf146  # reuse
        # Topologically Sorted Source Nodes: [ret_36], Original ATen: [aten.addmm]
        extern_kernels.mm(buf147, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf148)
        buf149 = buf145; del buf145  # reuse
        # Topologically Sorted Source Nodes: [ret_36], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_40[grid(16)](buf3, buf149, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf150 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_36], Original ATen: [aten.addmm]
        extern_kernels.mm(buf149, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf150)
        buf151 = buf148; del buf148  # reuse
        # Topologically Sorted Source Nodes: [ret_36], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf151, primals_7, buf150, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf152 = buf150; del buf150  # reuse
        # Topologically Sorted Source Nodes: [ret_37], Original ATen: [aten.addmm]
        extern_kernels.mm(buf151, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf152)
        buf153 = buf149; del buf149  # reuse
        # Topologically Sorted Source Nodes: [ret_37], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_41[grid(16)](buf3, buf153, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf154 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_37], Original ATen: [aten.addmm]
        extern_kernels.mm(buf153, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf154)
        buf155 = buf152; del buf152  # reuse
        # Topologically Sorted Source Nodes: [ret_37], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf155, primals_7, buf154, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf156 = buf154; del buf154  # reuse
        # Topologically Sorted Source Nodes: [ret_38], Original ATen: [aten.addmm]
        extern_kernels.mm(buf155, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf156)
        buf157 = buf153; del buf153  # reuse
        # Topologically Sorted Source Nodes: [ret_38], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_42[grid(16)](buf3, buf157, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf158 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_38], Original ATen: [aten.addmm]
        extern_kernels.mm(buf157, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf158)
        buf159 = buf156; del buf156  # reuse
        # Topologically Sorted Source Nodes: [ret_38], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf159, primals_7, buf158, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf160 = buf158; del buf158  # reuse
        # Topologically Sorted Source Nodes: [ret_39], Original ATen: [aten.addmm]
        extern_kernels.mm(buf159, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf160)
        buf161 = buf157; del buf157  # reuse
        # Topologically Sorted Source Nodes: [ret_39], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_43[grid(16)](buf3, buf161, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf162 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_39], Original ATen: [aten.addmm]
        extern_kernels.mm(buf161, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf162)
        buf163 = buf160; del buf160  # reuse
        # Topologically Sorted Source Nodes: [ret_39], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf163, primals_7, buf162, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf164 = buf162; del buf162  # reuse
        # Topologically Sorted Source Nodes: [ret_40], Original ATen: [aten.addmm]
        extern_kernels.mm(buf163, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf164)
        buf165 = buf161; del buf161  # reuse
        # Topologically Sorted Source Nodes: [ret_40], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_44[grid(16)](buf3, buf165, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf166 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_40], Original ATen: [aten.addmm]
        extern_kernels.mm(buf165, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf166)
        buf167 = buf164; del buf164  # reuse
        # Topologically Sorted Source Nodes: [ret_40], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf167, primals_7, buf166, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf168 = buf166; del buf166  # reuse
        # Topologically Sorted Source Nodes: [ret_41], Original ATen: [aten.addmm]
        extern_kernels.mm(buf167, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf168)
        buf169 = buf165; del buf165  # reuse
        # Topologically Sorted Source Nodes: [ret_41], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_45[grid(16)](buf3, buf169, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf170 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_41], Original ATen: [aten.addmm]
        extern_kernels.mm(buf169, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf170)
        buf171 = buf168; del buf168  # reuse
        # Topologically Sorted Source Nodes: [ret_41], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf171, primals_7, buf170, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf172 = buf170; del buf170  # reuse
        # Topologically Sorted Source Nodes: [ret_42], Original ATen: [aten.addmm]
        extern_kernels.mm(buf171, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf172)
        buf173 = buf169; del buf169  # reuse
        # Topologically Sorted Source Nodes: [ret_42], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_46[grid(16)](buf3, buf173, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf174 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_42], Original ATen: [aten.addmm]
        extern_kernels.mm(buf173, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf174)
        buf175 = buf172; del buf172  # reuse
        # Topologically Sorted Source Nodes: [ret_42], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf175, primals_7, buf174, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf176 = buf174; del buf174  # reuse
        # Topologically Sorted Source Nodes: [ret_43], Original ATen: [aten.addmm]
        extern_kernels.mm(buf175, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf176)
        buf177 = buf173; del buf173  # reuse
        # Topologically Sorted Source Nodes: [ret_43], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_47[grid(16)](buf3, buf177, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf178 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_43], Original ATen: [aten.addmm]
        extern_kernels.mm(buf177, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf178)
        buf179 = buf176; del buf176  # reuse
        # Topologically Sorted Source Nodes: [ret_43], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf179, primals_7, buf178, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf180 = buf178; del buf178  # reuse
        # Topologically Sorted Source Nodes: [ret_44], Original ATen: [aten.addmm]
        extern_kernels.mm(buf179, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf180)
        buf181 = buf177; del buf177  # reuse
        # Topologically Sorted Source Nodes: [ret_44], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_48[grid(16)](buf3, buf181, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf182 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_44], Original ATen: [aten.addmm]
        extern_kernels.mm(buf181, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf182)
        buf183 = buf180; del buf180  # reuse
        # Topologically Sorted Source Nodes: [ret_44], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf183, primals_7, buf182, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf184 = buf182; del buf182  # reuse
        # Topologically Sorted Source Nodes: [ret_45], Original ATen: [aten.addmm]
        extern_kernels.mm(buf183, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf184)
        buf185 = buf181; del buf181  # reuse
        # Topologically Sorted Source Nodes: [ret_45], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_49[grid(16)](buf3, buf185, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf186 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_45], Original ATen: [aten.addmm]
        extern_kernels.mm(buf185, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf186)
        buf187 = buf184; del buf184  # reuse
        # Topologically Sorted Source Nodes: [ret_45], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf187, primals_7, buf186, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf188 = buf186; del buf186  # reuse
        # Topologically Sorted Source Nodes: [ret_46], Original ATen: [aten.addmm]
        extern_kernels.mm(buf187, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf188)
        buf189 = buf185; del buf185  # reuse
        # Topologically Sorted Source Nodes: [ret_46], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_50[grid(16)](buf3, buf189, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf190 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_46], Original ATen: [aten.addmm]
        extern_kernels.mm(buf189, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf190)
        buf191 = buf188; del buf188  # reuse
        # Topologically Sorted Source Nodes: [ret_46], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf191, primals_7, buf190, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf192 = buf190; del buf190  # reuse
        # Topologically Sorted Source Nodes: [ret_47], Original ATen: [aten.addmm]
        extern_kernels.mm(buf191, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf192)
        buf193 = buf189; del buf189  # reuse
        # Topologically Sorted Source Nodes: [ret_47], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_51[grid(16)](buf3, buf193, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf194 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_47], Original ATen: [aten.addmm]
        extern_kernels.mm(buf193, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf194)
        buf195 = buf192; del buf192  # reuse
        # Topologically Sorted Source Nodes: [ret_47], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf195, primals_7, buf194, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf196 = buf194; del buf194  # reuse
        # Topologically Sorted Source Nodes: [ret_48], Original ATen: [aten.addmm]
        extern_kernels.mm(buf195, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf196)
        buf197 = buf193; del buf193  # reuse
        # Topologically Sorted Source Nodes: [ret_48], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_52[grid(16)](buf3, buf197, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf198 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_48], Original ATen: [aten.addmm]
        extern_kernels.mm(buf197, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf198)
        buf199 = buf196; del buf196  # reuse
        # Topologically Sorted Source Nodes: [ret_48], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf199, primals_7, buf198, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf200 = buf198; del buf198  # reuse
        # Topologically Sorted Source Nodes: [ret_49], Original ATen: [aten.addmm]
        extern_kernels.mm(buf199, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf200)
        buf201 = buf197; del buf197  # reuse
        # Topologically Sorted Source Nodes: [ret_49], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_53[grid(16)](buf3, buf201, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf202 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_49], Original ATen: [aten.addmm]
        extern_kernels.mm(buf201, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf202)
        buf203 = buf200; del buf200  # reuse
        # Topologically Sorted Source Nodes: [ret_49], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf203, primals_7, buf202, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf204 = buf202; del buf202  # reuse
        # Topologically Sorted Source Nodes: [ret_50], Original ATen: [aten.addmm]
        extern_kernels.mm(buf203, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf204)
        buf205 = buf201; del buf201  # reuse
        # Topologically Sorted Source Nodes: [ret_50], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_54[grid(16)](buf3, buf205, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf206 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_50], Original ATen: [aten.addmm]
        extern_kernels.mm(buf205, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf206)
        buf207 = buf204; del buf204  # reuse
        # Topologically Sorted Source Nodes: [ret_50], Original ATen: [aten.addmm, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_tanh_4[grid(32)](buf207, primals_7, buf206, primals_6, 32, XBLOCK=32, num_warps=1, num_stages=1)
        buf208 = buf206; del buf206  # reuse
        # Topologically Sorted Source Nodes: [ret_51], Original ATen: [aten.addmm]
        extern_kernels.mm(buf207, reinterpret_tensor(primals_5, (32, 32), (1, 32), 0), out=buf208)
        buf209 = buf205; del buf205  # reuse
        # Topologically Sorted Source Nodes: [ret_51], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_55[grid(16)](buf3, buf209, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf210 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_51], Original ATen: [aten.addmm]
        extern_kernels.mm(buf209, reinterpret_tensor(primals_4, (16, 32), (1, 16), 0), out=buf210)
        del buf209
        buf214 = empty_strided_cuda((2, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [2], out=buf214)
        buf216 = empty_strided_cuda((1, 1, 1), (1, 1, 1), torch.bool)
        buf211 = buf208; del buf208  # reuse
        buf212 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        buf213 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        buf217 = empty_strided_cuda((1, 1, 32), (32, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret_51, x_3, x_5], Original ATen: [aten.addmm, aten.add, aten.tanh, aten.neg, aten._softmax, aten.bernoulli, aten._to_copy, aten.div, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax__to_copy_add_addmm_bernoulli_div_mul_neg_tanh_56[grid(1)](buf211, buf214, primals_7, buf210, primals_6, buf216, buf212, buf213, buf217, 0, 1, 32, XBLOCK=1, num_warps=2, num_stages=1)
        del buf210
        del primals_6
        del primals_7
        buf218 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf217, (1, 32), (0, 1), 0), reinterpret_tensor(primals_8, (32, 64), (1, 32), 0), out=buf218)
        buf219 = buf218; del buf218  # reuse
        # Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_57[grid(64)](buf219, primals_9, 64, XBLOCK=64, num_warps=1, num_stages=1)
        del primals_9
        buf220 = empty_strided_cuda((1, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, buf219, reinterpret_tensor(primals_10, (64, 10), (1, 64), 0), alpha=1, beta=1, out=buf220)
        del primals_11
    buf222 = empty_strided_cpu((1, ), (1, ), torch.int64)
    # Topologically Sorted Source Nodes: [], Original ATen: []
    aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf222)
    buf223 = buf222; del buf222  # reuse
    cpp_fused_randint_58(buf223)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf224 = empty_strided_cuda((1, ), (1, ), torch.int64)
        buf224.copy_(buf223, False)
        del buf223
        buf225 = empty_strided_cuda((1, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ones_like, target_poisson], Original ATen: [aten.ones_like, aten.poisson]
        stream0 = get_raw_stream(0)
        triton_poi_fused_ones_like_poisson_59[grid(10)](buf225, 10, XBLOCK=16, num_warps=1, num_stages=1)
        # Topologically Sorted Source Nodes: [ones_like, target_poisson], Original ATen: [aten.ones_like, aten.poisson]
        buf226 = torch.ops.aten.poisson.default(buf225)
        buf227 = buf226
        del buf226
        buf221 = buf225; del buf225  # reuse
        buf228 = empty_strided_cuda((), (), torch.float32)
        buf229 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        buf230 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        buf233 = empty_strided_cuda((), (), torch.float32)
        buf231 = buf230; del buf230  # reuse
        buf232 = empty_strided_cuda((1, ), (1, ), torch.bool)
        buf235 = empty_strided_cuda((), (), torch.float32)
        buf236 = buf233; del buf233  # reuse
        buf234 = buf228; del buf228  # reuse
        # Topologically Sorted Source Nodes: [target_l1, l1_loss, ce_loss, poisson_loss], Original ATen: [aten.randn_like, aten.sub, aten.abs, aten.mean, aten._log_softmax, aten.nll_loss_forward, aten.mul, aten.exp]
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax_abs_exp_mean_mul_nll_loss_forward_randn_like_sub_60[grid(1)](buf231, buf236, buf234, buf214, buf220, buf227, buf224, buf221, buf229, buf232, buf235, 1, 1, 10, XBLOCK=1, num_warps=2, num_stages=1)
        del buf214
    return (buf220, buf234, buf235, buf236, primals_2, buf0, buf2, reinterpret_tensor(buf3, (1, 16), (832, 52), 0), buf7, reinterpret_tensor(buf3, (1, 16), (832, 52), 1), buf11, reinterpret_tensor(buf3, (1, 16), (832, 52), 2), buf15, reinterpret_tensor(buf3, (1, 16), (832, 52), 3), buf19, reinterpret_tensor(buf3, (1, 16), (832, 52), 4), buf23, reinterpret_tensor(buf3, (1, 16), (832, 52), 5), buf27, reinterpret_tensor(buf3, (1, 16), (832, 52), 6), buf31, reinterpret_tensor(buf3, (1, 16), (832, 52), 7), buf35, reinterpret_tensor(buf3, (1, 16), (832, 52), 8), buf39, reinterpret_tensor(buf3, (1, 16), (832, 52), 9), buf43, reinterpret_tensor(buf3, (1, 16), (832, 52), 10), buf47, reinterpret_tensor(buf3, (1, 16), (832, 52), 11), buf51, reinterpret_tensor(buf3, (1, 16), (832, 52), 12), buf55, reinterpret_tensor(buf3, (1, 16), (832, 52), 13), buf59, reinterpret_tensor(buf3, (1, 16), (832, 52), 14), buf63, reinterpret_tensor(buf3, (1, 16), (832, 52), 15), buf67, reinterpret_tensor(buf3, (1, 16), (832, 52), 16), buf71, reinterpret_tensor(buf3, (1, 16), (832, 52), 17), buf75, reinterpret_tensor(buf3, (1, 16), (832, 52), 18), buf79, reinterpret_tensor(buf3, (1, 16), (832, 52), 19), buf83, reinterpret_tensor(buf3, (1, 16), (832, 52), 20), buf87, reinterpret_tensor(buf3, (1, 16), (832, 52), 21), buf91, reinterpret_tensor(buf3, (1, 16), (832, 52), 22), buf95, reinterpret_tensor(buf3, (1, 16), (832, 52), 23), buf99, reinterpret_tensor(buf3, (1, 16), (832, 52), 24), buf103, reinterpret_tensor(buf3, (1, 16), (832, 52), 25), buf107, reinterpret_tensor(buf3, (1, 16), (832, 52), 26), buf111, reinterpret_tensor(buf3, (1, 16), (832, 52), 27), buf115, reinterpret_tensor(buf3, (1, 16), (832, 52), 28), buf119, reinterpret_tensor(buf3, (1, 16), (832, 52), 29), buf123, reinterpret_tensor(buf3, (1, 16), (832, 52), 30), buf127, reinterpret_tensor(buf3, (1, 16), (832, 52), 31), buf131, reinterpret_tensor(buf3, (1, 16), (832, 52), 32), buf135, reinterpret_tensor(buf3, (1, 16), (832, 52), 33), buf139, reinterpret_tensor(buf3, (1, 16), (832, 52), 34), buf143, reinterpret_tensor(buf3, (1, 16), (832, 52), 35), buf147, reinterpret_tensor(buf3, (1, 16), (832, 52), 36), buf151, reinterpret_tensor(buf3, (1, 16), (832, 52), 37), buf155, reinterpret_tensor(buf3, (1, 16), (832, 52), 38), buf159, reinterpret_tensor(buf3, (1, 16), (832, 52), 39), buf163, reinterpret_tensor(buf3, (1, 16), (832, 52), 40), buf167, reinterpret_tensor(buf3, (1, 16), (832, 52), 41), buf171, reinterpret_tensor(buf3, (1, 16), (832, 52), 42), buf175, reinterpret_tensor(buf3, (1, 16), (832, 52), 43), buf179, reinterpret_tensor(buf3, (1, 16), (832, 52), 44), buf183, reinterpret_tensor(buf3, (1, 16), (832, 52), 45), buf187, reinterpret_tensor(buf3, (1, 16), (832, 52), 46), buf191, reinterpret_tensor(buf3, (1, 16), (832, 52), 47), buf195, reinterpret_tensor(buf3, (1, 16), (832, 52), 48), buf199, reinterpret_tensor(buf3, (1, 16), (832, 52), 49), buf203, reinterpret_tensor(buf3, (1, 16), (832, 52), 50), buf207, reinterpret_tensor(buf3, (1, 16), (832, 52), 51), buf211, buf212, buf213, buf216, reinterpret_tensor(buf217, (1, 32), (32, 1), 0), buf219, buf220, buf221, buf224, buf227, buf229, buf231, buf232, primals_10, primals_8, primals_4, primals_5, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 3, 50), (150, 50, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((16, 3, 3), (9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((32, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((10, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
