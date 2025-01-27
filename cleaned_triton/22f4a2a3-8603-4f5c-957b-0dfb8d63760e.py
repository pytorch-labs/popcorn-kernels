
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
def triton_poi_fused_max_pool2d_with_indices_0 (in_ptr0 ,out_ptr0 ,ks0 ,ks1 ,ks2 ,ks3 ,ks4 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =(xindex %ks0 )
    x1 =((xindex //ks0 )%ks1 )
    x2 =xindex //ks2 
    x3 =xindex 
    tmp0 =tl .load (in_ptr0 +(2 *x0 +2 *ks4 *x1 +ks3 *ks4 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp1 =tl .load (in_ptr0 +(1 +2 *x0 +2 *ks4 *x1 +ks3 *ks4 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp3 =tl .load (in_ptr0 +(ks4 +2 *x0 +2 *ks4 *x1 +ks3 *ks4 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp5 =tl .load (in_ptr0 +(1 +ks4 +2 *x0 +2 *ks4 *x1 +ks3 *ks4 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp2 =triton_helpers .maximum (tmp1 ,tmp0 )
    tmp4 =triton_helpers .maximum (tmp3 ,tmp2 )
    tmp6 =triton_helpers .maximum (tmp5 ,tmp4 )
    tl .store (out_ptr0 +(x3 ),tmp6 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_max_pool2d_with_indices_view_1 (in_ptr0 ,out_ptr0 ,ks0 ,ks1 ,ks2 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =(xindex %64 )
    x1 =xindex //64 
    x2 =xindex 
    tmp0 =tl .load (in_ptr0 +(((x0 +64 *x1 )%(ks0 *ks1 *ks2 ))),xmask ,eviction_policy ='evict_last')
    tl .store (out_ptr0 +(x2 ),tmp0 ,xmask )

def call (args ):
    arg0_1 ,arg1_1 ,arg2_1 ,arg3_1 =args 
    args .clear ()
    s0 =arg0_1 
    s1 =arg1_1 
    s2 =arg2_1 
    assert_size_stride (arg3_1 ,(1 ,s0 ,s1 ,s2 ),(s0 *s1 *s2 ,s1 *s2 ,s2 ,1 ))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        s2 //2 
        s1 //2 
        (s1 //2 )*(s2 //2 )
        buf0 =empty_strided_cuda ((1 ,s0 ,s1 //2 ,s2 //2 ),(s0 *(s1 //2 )*(s2 //2 ),(s1 //2 )*(s2 //2 ),s2 //2 ,1 ),torch .float32 )

        triton_poi_fused_max_pool2d_with_indices_0_xnumel =s0 *(s1 //2 )*(s2 //2 )
        get_raw_stream (0 )
        triton_poi_fused_max_pool2d_with_indices_0 [grid (triton_poi_fused_max_pool2d_with_indices_0_xnumel )](arg3_1 ,buf0 ,32 ,32 ,1024 ,64 ,64 ,3072 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        del arg3_1 
        buf1 =empty_strided_cuda ((1 ,(s0 *(s1 //2 )*(s2 //2 ))//64 ,64 ),(64 *((s0 *(s1 //2 )*(s2 //2 ))//64 ),64 ,1 ),torch .float32 )

        triton_poi_fused_max_pool2d_with_indices_view_1_xnumel =64 *((s0 *(s1 //2 )*(s2 //2 ))//64 )
        get_raw_stream (0 )
        triton_poi_fused_max_pool2d_with_indices_view_1 [grid (triton_poi_fused_max_pool2d_with_indices_view_1_xnumel )](buf0 ,buf1 ,32 ,32 ,3 ,3072 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        del buf0 
    return (buf1 ,s0 ,s1 ,s2 ,)

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
