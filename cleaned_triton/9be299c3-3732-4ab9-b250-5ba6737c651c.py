
import torch 
import triton 
import triton .language as tl 
from torch ._inductor .runtime .triton_heuristics import (
grid ,
)
from torch ._C import _cuda_getCurrentRawStream as get_raw_stream 
from torch ._C import _cuda_getCurrentRawStream as get_raw_stream 

aten =torch .ops .aten 
inductor_ops =torch .ops .inductor 
_quantized =torch .ops ._quantized 
assert_size_stride =torch ._C ._dynamo .guards .assert_size_stride 
empty_strided_cpu =torch ._C ._dynamo .guards ._empty_strided_cpu 
empty_strided_cuda =torch ._C ._dynamo .guards ._empty_strided_cuda 
empty_strided_xpu =torch ._C ._dynamo .guards ._empty_strided_xpu 
reinterpret_tensor =torch ._C ._dynamo .guards ._reinterpret_tensor 
alloc_from_pool =torch .ops .inductor ._alloc_from_pool 

empty_strided_p2p =torch ._C ._distributed_c10d ._SymmetricMemory .empty_strided_p2p 

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_celu_elu_hardsigmoid_0 (in_ptr0 ,out_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =xindex 
    tmp0 =tl .load (in_ptr0 +(x0 ),xmask )
    tmp1 =0.0 
    tmp2 =tmp0 >tmp1 
    tmp3 =1.0507009873554805 
    tmp4 =tmp0 *tmp3 
    tmp5 =1.0 
    tmp6 =tmp0 *tmp5 
    tmp7 =libdevice .expm1 (tmp6 )
    tmp8 =1.7580993408473766 
    tmp9 =tmp7 *tmp8 
    tmp10 =tl .where (tmp2 ,tmp4 ,tmp9 )
    tmp11 =tmp10 >tmp1 
    tmp12 =libdevice .expm1 (tmp10 )
    tmp13 =tl .where (tmp11 ,tmp10 ,tmp12 )
    tmp14 =3.0 
    tmp15 =tmp13 +tmp14 
    tmp16 =triton_helpers .maximum (tmp15 ,tmp1 )
    tmp17 =6.0 
    tmp18 =triton_helpers .minimum (tmp16 ,tmp17 )
    tmp19 =0.16666666666666666 
    tmp20 =tmp18 *tmp19 
    tmp21 =tmp20 >tmp1 
    tmp22 =tmp20 *tmp3 
    tmp23 =tmp20 *tmp5 
    tmp24 =libdevice .expm1 (tmp23 )
    tmp25 =tmp24 *tmp8 
    tmp26 =tl .where (tmp21 ,tmp22 ,tmp25 )
    tmp27 =tmp26 >tmp1 
    tmp28 =libdevice .expm1 (tmp26 )
    tmp29 =tl .where (tmp27 ,tmp26 ,tmp28 )
    tl .store (out_ptr0 +(x0 ),tmp29 ,xmask )

def call (args ):
    arg0_1 ,arg1_1 ,arg2_1 ,arg3_1 =args 
    args .clear ()
    s0 =arg0_1 
    s1 =arg1_1 
    s2 =arg2_1 
    assert_size_stride (arg3_1 ,(1 ,s0 ,s1 ,s2 ),(s0 *s1 *s2 ,s1 *s2 ,s2 ,1 ))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        buf0 =empty_strided_cuda ((1 ,s0 ,s1 ,s2 ),(s0 *s1 *s2 ,s1 *s2 ,s2 ,1 ),torch .float32 )

        triton_poi_fused_celu_elu_hardsigmoid_0_xnumel =s0 *s1 *s2 
        get_raw_stream (0 )
        triton_poi_fused_celu_elu_hardsigmoid_0 [grid (triton_poi_fused_celu_elu_hardsigmoid_0_xnumel )](arg3_1 ,buf0 ,12288 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        del arg3_1 
    return (reinterpret_tensor (buf0 ,(1 ,s0 *s1 *s2 ),(s0 *s1 *s2 ,1 ),0 ),)

def benchmark_compiled_module (times =10 ,repeat =10 ):
    from torch ._dynamo .testing import rand_strided 
    from torch ._inductor .utils import print_performance 
    arg0_1 =3 
    arg1_1 =64 
    arg2_1 =64 
    arg3_1 =rand_strided ((1 ,3 ,64 ,64 ),(12288 ,4096 ,64 ,1 ),device ='cuda:0',dtype =torch .float32 )
    fn =lambda :call ([arg0_1 ,arg1_1 ,arg2_1 ,arg3_1 ])
    return print_performance (fn ,times =times ,repeat =repeat )

if __name__ =="__main__":
    from torch ._inductor .wrapper_benchmark import compiled_module_main 
    compiled_module_main ('None',benchmark_compiled_module )
