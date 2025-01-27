
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
def triton_poi_fused_constant_pad_nd_0 (in_ptr0 ,out_ptr0 ,ks0 ,ks1 ,ks2 ,ks3 ,ks4 ,ks5 ,ks6 ,ks7 ,ks8 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x6 =((xindex //ks0 )%ks1 )
    x1 =((xindex //ks3 )%ks4 )
    x0 =(xindex %ks3 )
    x2 =((xindex //ks7 )%ks1 )
    x3 =xindex //ks8 
    x10 =xindex 
    tmp0 =(-1 )+x6 
    tmp1 =tl .full ([1 ],0 ,tl .int64 )
    tmp2 =tmp0 >=tmp1 
    tmp3 =ks2 
    tmp4 =tmp0 <tmp3 
    tmp5 =(-1 )+x1 
    tmp6 =tmp5 >=tmp1 
    tmp7 =ks5 
    tmp8 =tmp5 <tmp7 
    tmp9 =(-1 )+x0 
    tmp10 =tmp9 >=tmp1 
    tmp11 =ks6 
    tmp12 =tmp9 <tmp11 
    tmp13 =tmp2 &tmp4 
    tmp14 =tmp13 &tmp6 
    tmp15 =tmp14 &tmp8 
    tmp16 =tmp15 &tmp10 
    tmp17 =tmp16 &tmp12 
    tmp18 =tl .load (in_ptr0 +((-1 )+x0 +((-1 )*ks6 )+ks6 *x1 +((-1 )*ks5 *ks6 )+ks5 *ks6 *x2 +ks2 *ks5 *ks6 *x3 ),tmp17 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tl .store (out_ptr0 +(x10 ),tmp18 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_constant_pad_nd_view_1 (in_ptr0 ,out_ptr0 ,ks0 ,ks1 ,ks2 ,ks3 ,ks4 ,ks5 ,ks6 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =(xindex %ks0 )
    x1 =xindex //ks0 
    x2 =xindex 
    tmp0 =tl .load (in_ptr0 +(2 *(((x0 //ks2 )%ks3 ))+4 *(x0 //ks1 )+8 *x1 +ks6 *(((x0 //ks2 )%ks3 ))+2 *ks5 *(x0 //ks1 )+2 *ks6 *(x0 //ks1 )+4 *ks4 *x1 +4 *ks5 *x1 +4 *ks6 *x1 +ks5 *ks6 *(x0 //ks1 )+2 *ks4 *ks5 *x1 +2 *ks4 *ks6 *x1 +2 *ks5 *ks6 *x1 +ks4 *ks5 *ks6 *x1 +((x0 %ks2 ))),xmask ,eviction_policy ='evict_last')
    tl .store (out_ptr0 +(x2 ),tmp0 ,xmask )

def call (args ):
    arg0_1 ,arg1_1 ,arg2_1 ,arg3_1 ,arg4_1 =args 
    args .clear ()
    s0 =arg0_1 
    s1 =arg1_1 
    s2 =arg2_1 
    s3 =arg3_1 
    assert_size_stride (arg4_1 ,(1 ,s0 ,s1 ,s2 ,s3 ),(s0 *s1 *s2 *s3 ,s1 *s2 *s3 ,s2 *s3 ,s3 ,1 ))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        4 +2 *s2 +2 *s3 +s2 *s3 
        2 +s1 
        2 +s3 
        2 +s2 
        4 +2 *s2 +2 *s3 +s2 *s3 
        8 +4 *s1 +4 *s2 +4 *s3 +2 *s1 *s2 +2 *s1 *s3 +2 *s2 *s3 +s1 *s2 *s3 
        buf0 =empty_strided_cuda ((1 ,s0 ,2 +s1 ,2 +s2 ,2 +s3 ),(8 *s0 +4 *s0 *s1 +4 *s0 *s2 +4 *s0 *s3 +2 *s0 *s1 *s2 +2 *s0 *s1 *s3 +2 *s0 *s2 *s3 +s0 *s1 *s2 *s3 ,8 +4 *s1 +4 *s2 +4 *s3 +2 *s1 *s2 +2 *s1 *s3 +2 *s2 *s3 +s1 *s2 *s3 ,4 +2 *s2 +2 *s3 +s2 *s3 ,2 +s3 ,1 ),torch .float32 )

        triton_poi_fused_constant_pad_nd_0_xnumel =8 *s0 +4 *s0 *s1 +4 *s0 *s2 +4 *s0 *s3 +2 *s0 *s1 *s2 +2 *s0 *s1 *s3 +2 *s0 *s2 *s3 +s0 *s1 *s2 *s3 
        get_raw_stream (0 )
        triton_poi_fused_constant_pad_nd_0 [grid (triton_poi_fused_constant_pad_nd_0_xnumel )](arg4_1 ,buf0 ,144 ,12 ,10 ,12 ,12 ,10 ,10 ,144 ,1728 ,5184 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        del arg4_1 
        8 +4 *s1 +4 *s2 +4 *s3 +2 *s1 *s2 +2 *s1 *s3 +2 *s2 *s3 +s1 *s2 *s3 
        buf1 =empty_strided_cuda ((1 ,s0 ,8 +4 *s1 +4 *s2 +4 *s3 +2 *s1 *s2 +2 *s1 *s3 +2 *s2 *s3 +s1 *s2 *s3 ),(8 *s0 +4 *s0 *s1 +4 *s0 *s2 +4 *s0 *s3 +2 *s0 *s1 *s2 +2 *s0 *s1 *s3 +2 *s0 *s2 *s3 +s0 *s1 *s2 *s3 ,8 +4 *s1 +4 *s2 +4 *s3 +2 *s1 *s2 +2 *s1 *s3 +2 *s2 *s3 +s1 *s2 *s3 ,1 ),torch .float32 )

        triton_poi_fused_constant_pad_nd_view_1_xnumel =8 *s0 +4 *s0 *s1 +4 *s0 *s2 +4 *s0 *s3 +2 *s0 *s1 *s2 +2 *s0 *s1 *s3 +2 *s0 *s2 *s3 +s0 *s1 *s2 *s3 
        get_raw_stream (0 )
        triton_poi_fused_constant_pad_nd_view_1 [grid (triton_poi_fused_constant_pad_nd_view_1_xnumel )](buf0 ,buf1 ,1728 ,144 ,12 ,12 ,10 ,10 ,10 ,5184 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        del buf0 
    return (buf1 ,)

def benchmark_compiled_module (times =10 ,repeat =10 ):
    from torch ._dynamo .testing import rand_strided 
    from torch ._inductor .utils import print_performance 
    arg0_1 =3 
    arg1_1 =10 
    arg2_1 =10 
    arg3_1 =10 
    arg4_1 =rand_strided ((1 ,3 ,10 ,10 ,10 ),(3000 ,1000 ,100 ,10 ,1 ),device ='cuda:0',dtype =torch .float32 )
    fn =lambda :call ([arg0_1 ,arg1_1 ,arg2_1 ,arg3_1 ,arg4_1 ])
    return print_performance (fn ,times =times ,repeat =repeat )

if __name__ =="__main__":
    from torch ._inductor .wrapper_benchmark import compiled_module_main 
    compiled_module_main ('None',benchmark_compiled_module )
