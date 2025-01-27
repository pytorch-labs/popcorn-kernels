
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
def triton_poi_fused_pixel_shuffle_0 (in_ptr0 ,out_ptr0 ,ks0 ,ks1 ,ks2 ,ks3 ,ynumel ,xnumel ,YBLOCK :tl .constexpr ,XBLOCK :tl .constexpr ):
    xnumel =2 
    yoffset =(tl .program_id (1 )+tl .program_id (2 )*tl .num_programs (1 ))*YBLOCK 
    yindex =yoffset +tl .arange (0 ,YBLOCK )[None ,:]
    ymask =yindex <ynumel 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    xmask =xindex <xnumel 
    x4 =xindex 
    y0 =(yindex %ks0 )
    y1 =((yindex //ks0 )%2 )
    y2 =((yindex //ks1 )%ks2 )
    y3 =yindex //ks3 
    y5 =yindex 
    tmp0 =tl .load (in_ptr0 +(y0 +ks0 *y2 +ks0 *ks2 *x4 +2 *ks0 *ks2 *y1 +4 *ks0 *ks2 *y3 ),xmask &ymask ,eviction_policy ='evict_last')
    tl .store (out_ptr0 +(x4 +2 *y5 ),tmp0 ,xmask &ymask )

def call (args ):
    arg0_1 ,arg1_1 ,arg2_1 ,arg3_1 =args 
    args .clear ()
    s0 =arg0_1 
    s1 =arg1_1 
    s2 =arg2_1 
    assert_size_stride (arg3_1 ,(1 ,s0 ,s1 ,s2 ),(s0 *s1 *s2 ,s1 *s2 ,s2 ,1 ))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        2 *s2 
        2 *s1 *s2 
        buf0 =empty_strided_cuda ((1 ,s0 //4 ,s1 ,2 ,s2 ,2 ),(4 *s1 *s2 *(s0 //4 ),4 *s1 *s2 ,4 *s2 ,2 *s2 ,2 ,1 ),torch .float32 )

        triton_poi_fused_pixel_shuffle_0_ynumel =2 *s1 *s2 *(s0 //4 )
        get_raw_stream (0 )
        triton_poi_fused_pixel_shuffle_0 [grid (triton_poi_fused_pixel_shuffle_0_ynumel ,2 )](arg3_1 ,buf0 ,64 ,128 ,64 ,8192 ,32768 ,2 ,XBLOCK =2 ,YBLOCK =512 ,num_warps =4 ,num_stages =1 )
        del arg3_1 
    return (reinterpret_tensor (buf0 ,(1 ,s0 //4 ,2 *s1 ,2 *s2 ),(2 *s1 *s2 *(s0 //4 )*(s0 //(2 *(s0 //4 ))),4 *s1 *s2 ,2 *s2 ,1 ),0 ),)

def benchmark_compiled_module (times =10 ,repeat =10 ):
    from torch ._dynamo .testing import rand_strided 
    from torch ._inductor .utils import print_performance 
    arg0_1 =16 
    arg1_1 =64 
    arg2_1 =64 
    arg3_1 =rand_strided ((1 ,16 ,64 ,64 ),(65536 ,4096 ,64 ,1 ),device ='cuda:0',dtype =torch .float32 )
    fn =lambda :call ([arg0_1 ,arg1_1 ,arg2_1 ,arg3_1 ])
    return print_performance (fn ,times =times ,repeat =repeat )

if __name__ =="__main__":
    from torch ._inductor .wrapper_benchmark import compiled_module_main 
    compiled_module_main ('None',benchmark_compiled_module )
