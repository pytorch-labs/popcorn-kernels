
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
def triton_poi_fused_adaptive_max_pool2d_0 (in_ptr0 ,out_ptr0 ,ks0 ,ks1 ,ks2 ,ks3 ,ks4 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =(xindex %ks0 )
    x1 =((xindex //ks0 )%ks1 )
    x2 =xindex //ks2 
    x3 =xindex 
    tmp0 =tl .load (in_ptr0 +(4 *x0 +4 *ks4 *x1 +ks3 *ks4 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp1 =tl .load (in_ptr0 +(1 +4 *x0 +4 *ks4 *x1 +ks3 *ks4 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp3 =tl .load (in_ptr0 +(2 +4 *x0 +4 *ks4 *x1 +ks3 *ks4 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp5 =tl .load (in_ptr0 +(3 +4 *x0 +4 *ks4 *x1 +ks3 *ks4 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp7 =tl .load (in_ptr0 +(ks4 +4 *x0 +4 *ks4 *x1 +ks3 *ks4 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp9 =tl .load (in_ptr0 +(1 +ks4 +4 *x0 +4 *ks4 *x1 +ks3 *ks4 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp11 =tl .load (in_ptr0 +(2 +ks4 +4 *x0 +4 *ks4 *x1 +ks3 *ks4 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp13 =tl .load (in_ptr0 +(3 +ks4 +4 *x0 +4 *ks4 *x1 +ks3 *ks4 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp15 =tl .load (in_ptr0 +(2 *ks4 +4 *x0 +4 *ks4 *x1 +ks3 *ks4 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp17 =tl .load (in_ptr0 +(1 +2 *ks4 +4 *x0 +4 *ks4 *x1 +ks3 *ks4 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp19 =tl .load (in_ptr0 +(2 +2 *ks4 +4 *x0 +4 *ks4 *x1 +ks3 *ks4 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp21 =tl .load (in_ptr0 +(3 +2 *ks4 +4 *x0 +4 *ks4 *x1 +ks3 *ks4 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp23 =tl .load (in_ptr0 +(3 *ks4 +4 *x0 +4 *ks4 *x1 +ks3 *ks4 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp25 =tl .load (in_ptr0 +(1 +3 *ks4 +4 *x0 +4 *ks4 *x1 +ks3 *ks4 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp27 =tl .load (in_ptr0 +(2 +3 *ks4 +4 *x0 +4 *ks4 *x1 +ks3 *ks4 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp29 =tl .load (in_ptr0 +(3 +3 *ks4 +4 *x0 +4 *ks4 *x1 +ks3 *ks4 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp2 =triton_helpers .maximum (tmp1 ,tmp0 )
    tmp4 =triton_helpers .maximum (tmp3 ,tmp2 )
    tmp6 =triton_helpers .maximum (tmp5 ,tmp4 )
    tmp8 =triton_helpers .maximum (tmp7 ,tmp6 )
    tmp10 =triton_helpers .maximum (tmp9 ,tmp8 )
    tmp12 =triton_helpers .maximum (tmp11 ,tmp10 )
    tmp14 =triton_helpers .maximum (tmp13 ,tmp12 )
    tmp16 =triton_helpers .maximum (tmp15 ,tmp14 )
    tmp18 =triton_helpers .maximum (tmp17 ,tmp16 )
    tmp20 =triton_helpers .maximum (tmp19 ,tmp18 )
    tmp22 =triton_helpers .maximum (tmp21 ,tmp20 )
    tmp24 =triton_helpers .maximum (tmp23 ,tmp22 )
    tmp26 =triton_helpers .maximum (tmp25 ,tmp24 )
    tmp28 =triton_helpers .maximum (tmp27 ,tmp26 )
    tmp30 =triton_helpers .maximum (tmp29 ,tmp28 )
    tl .store (out_ptr0 +(x3 ),tmp30 ,xmask )

def call (args ):
    arg0_1 ,arg1_1 ,arg2_1 ,arg3_1 =args 
    args .clear ()
    s0 =arg0_1 
    s1 =arg1_1 
    s2 =arg2_1 
    assert_size_stride (arg3_1 ,(1 ,s0 ,s1 ,s2 ),(s0 *s1 *s2 ,s1 *s2 ,s2 ,1 ))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        s2 //4 
        s1 //4 
        (s1 //4 )*(s2 //4 )
        buf0 =empty_strided_cuda ((1 ,s0 ,s1 //4 ,s2 //4 ),(s0 *(s1 //4 )*(s2 //4 ),(s1 //4 )*(s2 //4 ),s2 //4 ,1 ),torch .float32 )

        triton_poi_fused_adaptive_max_pool2d_0_xnumel =s0 *(s1 //4 )*(s2 //4 )
        get_raw_stream (0 )
        triton_poi_fused_adaptive_max_pool2d_0 [grid (triton_poi_fused_adaptive_max_pool2d_0_xnumel )](arg3_1 ,buf0 ,16 ,16 ,256 ,64 ,64 ,768 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        del arg3_1 
    return (reinterpret_tensor (buf0 ,(1 ,(s1 //4 )*(s2 //4 ),s0 ),(s0 *(s1 //4 )*(s2 //4 ),1 ,(s1 //4 )*(s2 //4 )),0 ),)

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
