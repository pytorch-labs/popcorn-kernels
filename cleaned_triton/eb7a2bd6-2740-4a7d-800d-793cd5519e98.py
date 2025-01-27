
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
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_constant_pad_nd_0 (in_ptr0 ,out_ptr0 ,ks0 ,ks1 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =(xindex %ks0 )
    x1 =xindex //ks0 
    x2 =xindex 
    tmp0 =(-2 )+x0 
    tmp1 =tl .full ([1 ],0 ,tl .int64 )
    tmp2 =tmp0 >=tmp1 
    tmp3 =ks1 
    tmp4 =tmp0 <tmp3 
    tmp5 =tmp2 &tmp4 
    tmp6 =tl .load (in_ptr0 +((-2 )+x0 +ks1 *x1 ),tmp5 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tl .store (out_ptr0 +(x2 ),tmp6 ,xmask )

def call (args ):
    arg0_1 ,arg1_1 ,arg2_1 =args 
    args .clear ()
    s0 =arg0_1 
    s1 =arg1_1 
    assert_size_stride (arg2_1 ,(1 ,s0 ,s1 ),(s0 *s1 ,s1 ,1 ))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        4 +s1 
        buf0 =empty_strided_cuda ((1 ,s0 ,4 +s1 ),(4 *s0 +s0 *s1 ,4 +s1 ,1 ),torch .float32 )

        triton_poi_fused_constant_pad_nd_0_xnumel =4 *s0 +s0 *s1 
        get_raw_stream (0 )
        triton_poi_fused_constant_pad_nd_0 [grid (triton_poi_fused_constant_pad_nd_0_xnumel )](arg2_1 ,buf0 ,68 ,64 ,680 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        del arg2_1 
    return (reinterpret_tensor (buf0 ,(1 ,4 +s1 ,s0 ),(4 *s0 +s0 *s1 ,1 ,4 +s1 ),0 ),)

def benchmark_compiled_module (times =10 ,repeat =10 ):
    from torch ._dynamo .testing import rand_strided 
    from torch ._inductor .utils import print_performance 
    arg0_1 =10 
    arg1_1 =64 
    arg2_1 =rand_strided ((1 ,10 ,64 ),(640 ,64 ,1 ),device ='cuda:0',dtype =torch .float32 )
    fn =lambda :call ([arg0_1 ,arg1_1 ,arg2_1 ])
    return print_performance (fn ,times =times ,repeat =repeat )

if __name__ =="__main__":
    from torch ._inductor .wrapper_benchmark import compiled_module_main 
    compiled_module_main ('None',benchmark_compiled_module )
